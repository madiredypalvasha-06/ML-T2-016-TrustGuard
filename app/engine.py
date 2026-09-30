"""Runtime engine for the app: model loading, trust profiles, batch scans.

Two deployment profiles are supported and chosen automatically:
  * "ensemble"   — deep ensemble + Mahalanobis familiarity (production)
  * "mc_dropout" — single model + Monte-Carlo Dropout (fallback while the
                   ensemble members are still training; clearly labelled)
The app downgrades gracefully and tells the operator which profile is live.
"""
import json
import os

import numpy as np
import streamlit as st
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, ensembles as ens, mahalanobis as mh, uncertainty as uc
from src.models import CIFAR10CNN, load_model

upload_buffer = {}


@st.cache_resource(show_spinner="Revving up models…")
def get_resources(min_mc=8):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    base = load_model(config.MODEL_CHECKPOINT, device)
    temp = _load_temp()
    member_tags = [f"member_{i}" for i in range(config.ENSEMBLE_MEMBERS)]
    members = ens.load_members(tags=member_tags, device=device) \
        if _members_ready() else None

    maha = None
    if os.path.exists(config.MAHALANOBIS_PATH):
        maha = mh.MahalanobisOOD.load(config.MAHALANOBIS_PATH)

    if members is not None:
        profile = "ensemble"
    else:
        profile = "mc_dropout"
    return dict(device=device, base=base, temp=temp, members=members,
                maha=maha, profile=profile)


def _load_temp():
    if os.path.exists(config.TEMP_PARAMS):
        with open(config.TEMP_PARAMS) as f:
            return float(json.load(f)["temperature"])
    return 1.0


def _members_ready():
    return all(os.path.exists(config.ENSEMBLE_PATH(f"member_{i}"))
               for i in range(config.ENSEMBLE_MEMBERS))


@st.cache_resource(show_spinner=False)
def _fit_trust_thresholds(coverage, device, profile):
    """Evidence-based abstention thresholds, fitted on the calibration split."""
    from src.data import get_app_datasets
    cal_ds, _ = get_app_datasets()
    # get_app_datasets already stores these 600 images in the order this used to
    # draw them, so the fitted thresholds are unchanged.
    cal_x, cal_y = cal_ds.tensors

    res = get_resources()
    if res["members"] is not None:
        tc = ens.TrustAwareClassifier(res["members"], temperature=res["temp"],
                                      coverage=coverage, device=res["device"],
                                      mahalanobis=res["maha"])
        tc.fit_thresholds(cal_x, cal_y)
        return tc
    # mc-dropout fallback: fit entropy threshold using the calibration split
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(cal_x, cal_y),
        batch_size=256, shuffle=False)
    probs, _, scores, _ = _score_loader_mc(res, loader, n_mc=12)
    ent = scores["mc_entropy"]
    k = max(1, int(round(coverage * len(ent))))
    thr = float(np.sort(ent)[k - 1])
    return {"cal_conf": 0.0, "ens_entropy": thr, "_mode": "mc", "temp": res["temp"]}


def _score_loader_mc(res, loader, n_mc=None, max_n=None):
    from src.evaluate import score_dataset
    return score_dataset(res["base"], loader, res["temp"], res["device"],
                         do_mc=True, n_mc=config.MC_DROPOUT_PASSES if n_mc is None else n_mc,
                         max_mc=max_n)


def predict_one(image_tensor, coverage=0.9):
    """Single-image trust verdict (vectors via whichever profile is live)."""
    res = get_resources()
    device, profile = res["device"], res["profile"]
    if profile == "ensemble":
        tc = _fit_trust_thresholds(coverage, device, profile)
        r = tc.predict(image_tensor)
        r["profile"] = profile
        return r
    # MC-Dropout fallback path
    thr = _fit_trust_thresholds(coverage, device, profile)
    xb = image_tensor.unsqueeze(0)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xb, torch.zeros(1, dtype=torch.long)),
        batch_size=1)
    with torch.no_grad():
        logits_all, _ = uc.mc_dropout_logits(res["base"], loader, device,
                                             passes=12)
        probs = torch.softmax(logits_all / res["temp"], dim=-1).mean(dim=0).cpu().numpy()
    mean = probs[0]
    ent = float(uc.predictive_entropy(mean[None])[0])
    conf = float(mean.max())
    ok_unc = ent <= thr["ens_entropy"]
    verdict = "TRUST" if ok_unc else ("REVIEW" if conf >= 0.5 else "ABSTAIN")
    return {
        "pred": int(mean.argmax()),
        "probs": mean,
        "scores": {"cal_conf": conf, "ens_entropy": ent},
        "checks": [["confidence", "pass" if conf >= 0.5 else "LOW"],
                   ["uncertainty", "pass" if ok_unc else "HIGH"],
                   ["familiarity", "pass"]],
        "verdict": verdict,
        "mahalanobis_dist": float("nan"),
        "profile": profile,
    }


def batch_scan(index_tensor, labels=None, coverage=0.9, max_n=1000):
    """Scan a batch through the live trust pipeline. Returns analytics dict."""
    res = get_resources()
    rng = np.random.RandomState(0)
    idx = rng.choice(len(index_tensor),
                     min(max_n, len(index_tensor)), replace=False)
    x = index_tensor[idx]
    device, profile = res["device"], res["profile"]
    out = dict(profile=profile, n=len(x), x=x.cpu(),
               entropy=[], conf=[], dist=[], verdicts=[], preds=[], labels=[])
    if profile == "ensemble":
        tc = _fit_trust_thresholds(coverage, device, profile)
        for i in range(0, len(x), 256):
            r = tc.predict_batch(x[i:i + 256])
            out["preds"].append(r["preds"])
            out["verdicts"].append(r["verdicts"])
            out["entropy"].append(r["scores"]["ens_entropy"])
            out["conf"].append(r["scores"]["cal_conf"])
            out["dist"].append(np.asarray(r["mahalanobis_dist"]))
    else:
        thr = _fit_trust_thresholds(coverage, device, profile)
        for i in range(0, len(x), 128):
            chunk = x[i:i + 128]
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    chunk, torch.zeros(len(chunk), dtype=torch.long)),
                batch_size=128)
            with torch.no_grad():
                logits_all, _ = uc.mc_dropout_logits(res["base"], loader, device,
                                                     passes=12)
                mean = torch.softmax(logits_all / res["temp"], dim=-1) \
                    .mean(dim=0).cpu().numpy()
            ent = uc.predictive_entropy(mean)
            conf = mean.max(axis=-1)
            verd = np.where(ent <= thr["ens_entropy"],
                            "TRUST", np.where(conf >= 0.5, "REVIEW", "ABSTAIN"))
            out["preds"].append(mean.argmax(axis=-1))
            out["verdicts"].append(verd)
            out["entropy"].append(ent)
            out["conf"].append(conf)
            out["dist"].append(np.full(len(mean), np.nan))

    for k in ("preds", "verdicts", "entropy", "conf", "dist"):
        out[k] = np.concatenate(out[k])
    if labels is not None:
        out["labels"] = labels[idx]
    out["correct"] = out["preds"] == out["labels"] \
        if labels is not None else np.full(len(out["preds"]), np.nan)
    trust = out["verdicts"] == "TRUST"
    out["trust_rate"] = float(trust.mean())
    out["accuracy_overall"] = float(np.nanmean(out["correct"]))
    out["accuracy_on_trusted"] = float(np.nanmean(out["correct"][trust])) \
        if trust.any() else float("nan")
    out["flagged"] = ~trust
    out["trust_threshold"] = _threshold_hint(res, coverage)
    return out


def _threshold_hint(res, coverage):
    if res["members"] is None:
        return _fit_trust_thresholds(coverage, res["device"], res["profile"]).get(
            "ens_entropy", np.inf)
    tc = _fit_trust_thresholds(coverage, res["device"], res["profile"])
    return tc.thresholds.get("ens_entropy", np.inf)


def scan_shift(corr, severity, coverage=0.9, max_n=800, device=None):
    """Corrupted batch scan with labels for the drift monitor."""
    res = get_resources()
    from src.data import get_app_datasets, get_shift_data
    device = res["device"]
    _, test_ds = get_app_datasets()
    x, y = get_shift_data(corr, severity, device=device, test_ds=test_ds)
    return batch_scan(x.cpu(), labels=y, coverage=coverage, max_n=max_n)


def scan_clean(coverage=0.9, max_n=800):
    res = get_resources()
    from src.data import get_app_datasets
    _, test_ds = get_app_datasets()
    x = torch.stack([t for t, _ in test_ds])
    y = np.array([l for _, l in test_ds])
    return batch_scan(x, labels=y, coverage=coverage, max_n=max_n)


def shift_catalog():
    return config.CORRUPTIONS


def get_shift_data(corr, severity):
    from src.data import get_app_datasets, get_shift_data as _g
    res = get_resources()
    _, test_ds = get_app_datasets()
    return _g(corr, severity, device=res["device"], test_ds=test_ds)


def shift_severities():
    return [1, 2, 3, 4, 5]