"""Uncertainty / reliability methods: softmax, temperature scaling, MC Dropout.

Each method produces, per instance, a predictive distribution and a scalar
uncertainty score. Scores are defined so that HIGHER uncertainty = LESS trust.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

from src import config


@torch.no_grad()
def predict_logits(model, loader, device):
    """Single deterministic forward pass -> concatenated logits (N, C)."""
    model.eval()
    logits = []
    for xb, _ in loader:
        out = model(xb.to(device))
        logits.append(out.detach().cpu())
    return torch.cat(logits)


class TemperatureScaling:
    """Post-hoc temperature scaling fitted on a held-out calibration set."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, labels):
        logits = torch.as_tensor(logits, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long)
        temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
        opt = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

        def nll_loss():
            out = logits / temperature
            return F.cross_entropy(out, labels)

        def closure():
            opt.zero_grad()
            loss = nll_loss()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            temperature.clamp_(min=1e-3)
        self.temperature = float(temperature.item())
        return self.temperature

    def calibrated_probs(self, logits):
        logits = torch.as_tensor(logits, dtype=torch.float32)
        return torch.softmax(logits / self.temperature, dim=-1).numpy()

    @staticmethod
    def save(temperature, path):
        with open(path, "w") as f:
            json.dump({"temperature": float(temperature)}, f)

    @staticmethod
    def load(path):
        with open(path) as f:
            data = json.load(f)
        ts = TemperatureScaling()
        ts.temperature = float(data["temperature"])
        return ts


@torch.no_grad()
def mc_dropout_logits(model, loader, device, passes=config.MC_DROPOUT_PASSES):
    """Run `passes` stochastic forward passes with dropout enabled.

    Returns (logits_all, n) with logits_all shape (passes, N, C).
    """
    model.train()
    model.apply(lambda m: _enable_dropout(m))
    all_logits = []
    examples = 0
    for _ in range(passes):
        pass_logits = []
        for xb, _ in loader:
            out = model(xb.to(device))
            pass_logits.append(out.detach().cpu())
        all_logits.append(torch.cat(pass_logits))
        examples = all_logits[-1].shape[0]
    return torch.stack(all_logits), examples


def _enable_dropout(m):
    if isinstance(m, torch.nn.Dropout):
        m.train()


def probs_from_logits(logits):
    return torch.softmax(logits, dim=-1)


def predictive_entropy(probs):
    """Entropy of the (averaged) predictive distribution. Higher = more uncertain."""
    eps = 1e-12
    return -np.sum(probs * np.log(probs + eps), axis=-1)


def softmax_uncertainty(logits):
    """1 - max softmax probability (baseline MSP-based score)."""
    probs = probs_from_logits(logits).numpy()
    return 1.0 - probs.max(axis=-1)


def entropy_uncertainty(logits):
    return predictive_entropy(probs_from_logits(logits).numpy())


def mc_dropout_stats(logits_all, temperature=1.0):
    """Summarise MC-Dropout Monte-Carlo logits.

    Returns dict with:
      probs_mean     : (N, C) mean softmax across passes (temperature-scaled)
      entropy        : predictive entropy of the mean distribution (N,)
      variance      : mean per-class variance of softmax probs (N,)
      mutual_info   : predictive entropy - mean entropy (epistemic) (N,)
    """
    probs_t = torch.softmax(logits_all / float(temperature), dim=-1).numpy()  # (P,N,C)
    probs_mean = probs_t.mean(axis=0)
    entropy = predictive_entropy(probs_mean)
    mean_entropy = np.mean(predictive_entropy(probs_t), axis=0)
    variance = probs_t.var(axis=0).mean(axis=-1)
    mutual_info = entropy - mean_entropy
    return {
        "probs_mean": probs_mean,
        "entropy": entropy,
        "variance": variance,
        "mutual_info": mutual_info,
    }


def make_uncertainty_scores(probs, mc_stats=None):
    """Produce all scalar uncertainty scores aligned with one dataset.

    probs must be the final (temperature-scaled) predicted probabilities.
    scores: softmax (1 - max prob), entropy; plus mc_* if mc_stats given.
    Also returns model confidence (max prob) per method for trust thresholds.
    """
    probs = np.asarray(probs, dtype=np.float64)
    conf = probs.max(axis=-1)
    scores = {
        "softmax": 1.0 - conf,
        "entropy": predictive_entropy(probs),
        "msp_conf": conf,
    }
    if mc_stats is not None:
        scores["mc_entropy"] = mc_stats["entropy"]
        scores["mc_variance"] = mc_stats["variance"]
        scores["mc_mutual_info"] = mc_stats["mutual_info"]
    return scores