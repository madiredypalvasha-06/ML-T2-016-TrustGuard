"""Deep ensemble uncertainty (approach C) and the trust-aware classifier.

A deep ensemble averages the softmax distributions of independently trained
models; the spread of their predictions is itself a powerful uncertainty
signal. `TrustAwareClassifier` is the deployable wrapper: it turns a raw
prediction into (label, calibrated confidence, uncertainty, trust verdict)
along with a human-readable *reason* for why the model should or should not
be trusted — the core of problem ML-T2-016.
"""
import os

import numpy as np
import torch

from src import config, uncertainty as uc
from src.models import CIFAR10CNN


def load_members(tags=None, device="cpu"):
    """Load ensemble member models (member_0..member_{ENSEMBLE_MEMBERS-1})."""
    if tags is None:
        tags = [f"member_{i}" for i in range(config.ENSEMBLE_MEMBERS)]
    models = []
    for tag in tags:
        path = config.ENSEMBLE_PATH(tag)
        m = CIFAR10CNN(config.DROPOUT_P)
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device).eval()
        models.append(m)
    return models


def missing_members(tags=None):
    if tags is None:
        tags = [f"member_{i}" for i in range(config.ENSEMBLE_MEMBERS)]
    return [t for t in tags
            if not os.path.exists(config.ENSEMBLE_PATH(t))]


def ensemble_predict_loader(models, loader, temperature=1.0, device="cpu"):
    """Averaged probs + scores over an entire loader.

    Returns dict:
      probs          : (N, C) mean softmax probs
      logits_n        : (M, N, C) per-member logits
      scores          : per-instance uncertainty scores
    """
    per = []
    for xb, _ in loader:
        per.append(ensemble_predict_batch(models, xb, temperature, device))
    # stack along N
    probs = np.concatenate([p["probs"] for p in per], axis=0)
    scores = {}
    for k in per[0]["scores"]:
        scores[k] = np.concatenate([p["scores"][k] for p in per], axis=0)
    return {"probs": probs, "scores": scores}


def ensemble_predict_batch(models, xb, temperature=1.0, device="cpu"):
    """Single-batch ensemble forward pass.

    Returns dict:
      probs      : (B, C) mean softmax probs
      logits_n   : (M, B, C) per-member logits
      scores     : dict of per-instance uncertainty scores
    """
    x = xb.to(device)
    per_member_logits = []
    with torch.no_grad():
        for m in models:
            per_member_logits.append(m(x).detach().cpu())
    logits_n = torch.stack(per_member_logits)                 # (M, B, C)
    probs = torch.softmax(logits_n / float(temperature), dim=-1).numpy()  # (M,B,C)
    mean_probs = probs.mean(axis=0)
    scores = {
        "ens_entropy": uc.predictive_entropy(mean_probs),              # total
        "ens_variance": probs.var(axis=0).mean(axis=-1),               # disagreement
        "ens_mutual_info": uc.predictive_entropy(mean_probs)
                           - np.mean(uc.predictive_entropy(probs), axis=0),
        "ens_fraction_disagree": np.mean(
            probs.argmax(axis=-1) != probs[0].argmax(axis=-1), axis=0),
    }
    return {"probs": mean_probs, "scores": scores, "logits_n": logits_n}


class TrustAwareClassifier:
    """Production wrapper: `predict(image) -> verdict`.

    Combines temperature-scaled confidence, ensemble uncertainty and
    feature-space familiarity into a single, explainable trust verdict.

    Thresholds are learned on the held-out calibration split at a chosen
    coverage target (fraction of predictions we agree to accept).
    """

    def __init__(self, models, temperature=1.0, coverage=0.9, device="cpu",
                 mahalanobis=None):
        self.models = models
        self.temperature = float(temperature)
        self.coverage = float(coverage)
        self.device = device
        self.mahalanobis = mahalanobis
        self.thresholds = {}

    # ---- threshold learning (evidence-based, on calibration split) --------
    def fit_thresholds(self, cal_x, cal_y, batch_size=256):
        import torch
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                cal_x, torch.zeros(len(cal_x), dtype=torch.long)),
            batch_size=batch_size, shuffle=False)
        r = ensemble_predict_loader(self.models, loader, self.temperature, self.device)
        scores = dict(r["scores"])
        if self.mahalanobis is not None:
            feats, _ = self.models[0].forward_features(cal_x.to(self.device))
            scores["mahalanobis"] = self.mahalanobis.distance(feats.cpu().numpy())
        self.thresholds = _coverage_thresholds(scores, self.coverage)
        return self.thresholds

    # ---- inference --------------------------------------------------------
    @torch.no_grad()
    def predict_batch(self, xb):
        """Vectorized verdicts for a batch (B,3,H,W) in [0,1].

        Returns dict with preds, calibrated confidence, uncertainty scores,
        trust verdicts and per-item reason summaries."""
        xb = xb.to(self.device)
        per_member = [m(xb).detach().cpu() for m in self.models]
        logits_n = torch.stack(per_member)                        # (M,B,C)
        probs = torch.softmax(logits_n / float(self.temperature), dim=-1).numpy()
        mean_probs = probs.mean(axis=0)                            # (B,C)
        scores = {
            "cal_conf": mean_probs.max(axis=-1),
            "ens_entropy": uc.predictive_entropy(mean_probs),
            "ens_fraction_disagree": np.mean(
                probs.argmax(-1) != probs[0].argmax(-1), axis=0),
        }
        preds = mean_probs.argmax(axis=-1)

        ok_conf = scores["cal_conf"] >= self.thresholds.get("cal_conf", 0)
        ok_unc = scores["ens_entropy"] <= self.thresholds.get("ens_entropy", np.inf)
        if self.mahalanobis is not None:
            feats, _ = self.models[0].forward_features(xb)
            dist = self.mahalanobis.distance(feats.cpu().numpy())
            ok_ood = dist <= self.thresholds.get("mahalanobis", np.inf)
        else:
            dist = np.full(len(xb), np.nan)
            ok_ood = np.ones(len(xb), dtype=bool)

        verdicts = np.where(
            ok_conf & ok_unc & ok_ood, "TRUST",
            np.where(~ok_conf, "ABSTAIN", "REVIEW"))

        reasons = []
        for a, b, c, d in zip(ok_conf, ok_unc, ok_ood, dist):
            parts = []
            parts.append("confidence=" + ("pass" if a else "LOW"))
            parts.append("uncertainty=" + ("pass" if b else "HIGH"))
            parts.append("familiarity=" + ("pass" if c else f"OOD(d={d:.1f})"))
            reasons.append(" · ".join(parts))

        return {
            "preds": preds,
            "probs": mean_probs,
            "scores": scores,
            "verdicts": verdicts,
            "reasons": reasons,
            "mahalanobis_dist": dist,
        }

    @torch.no_grad()
    def predict(self, x):
        """Single-image convenience view (x: (3,H,W) tensor in [0,1])."""
        r = self.predict_batch(x.unsqueeze(0))
        return {
            "pred": int(r["preds"][0]),
            "probs": r["probs"][0],
            "scores": {k: float(np.asarray(v)[0]) for k, v in r["scores"].items()},
            "checks": [part.split("=", 1) for part in r["reasons"][0].split(" · ")],
            "verdict": str(r["verdicts"][0]),
            "mahalanobis_dist": float(r["mahalanobis_dist"][0]),
        }


def _coverage_thresholds(scores, coverage):
    """Element-wise thresholds that keep `coverage` fraction of a set."""
    thr = {}
    for name, arr in scores.items():
        if arr.ndim != 1:
            continue
        k = max(1, int(round(coverage * len(arr))))
        if name in ("cal_conf",):
            thr[name] = float(np.sort(arr)[len(arr) - k])
        else:
            thr[name] = float(np.sort(arr)[k - 1])
    return thr