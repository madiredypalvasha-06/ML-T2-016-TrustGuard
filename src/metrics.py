"""Evaluation metrics: calibration, Brier, risk-coverage, OOD separation."""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config


def expected_calibration_error(probs, labels, n_bins=15):
    """Standard binning-based ECE (Guo et al. style). Confidence vs accuracy."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    pred_class = probs.argmax(axis=-1)
    conf = probs.max(axis=-1)
    correct = (pred_class == labels).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, accs, confs, densities, bins = 0.0, [], [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            hi = 1.0001
        idx = (conf >= lo) & (conf < hi)
        n = idx.sum()
        if n == 0:
            accs.append(np.nan)
            confs.append(np.nan)
            densities.append(0.0)
            bins.append((lo + hi) / 2)
            continue
        acc = correct[idx].mean()
        c = conf[idx].mean()
        accs.append(acc)
        confs.append(c)
        densities.append(n / len(conf))
        bins.append((lo + hi) / 2)
        ece += (n / len(conf)) * np.abs(acc - c)
    return {
        "ece": float(ece),
        "accs": np.array(accs),
        "confs": np.array(confs),
        "densities": np.array(densities),
        "bins": np.array(bins),
    }


def brier_score(probs, labels, n_classes=None):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    if n_classes is None:
        n_classes = probs.shape[-1]
    onehot = np.eye(n_classes)[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=-1)))


def accuracy(probs, labels):
    return float((np.asarray(probs).argmax(axis=-1) == np.asarray(labels)).mean())


def correct_mask(probs, labels):
    return np.asarray(probs).argmax(axis=-1) == np.asarray(labels)


def risk_coverage_curve(probs, labels, uncertainty, coverages=None):
    """Accuracy (and risk) as a function of the covered fraction.

    Keeps the predictions with the LOWEST uncertainty first (i.e. the most
    confident / trustworthy ones). `uncertainty` must be higher=less trusted.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    uncertainty = np.asarray(uncertainty)
    if coverages is None:
        coverages = np.round(np.linspace(0.05, 1.0, 20), 3)
    order = np.argsort(uncertainty, kind="stable")
    n = len(uncertainty)
    accs, risks = [], []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        sel = order[:k]
        acc = (probs[sel].argmax(axis=-1) == labels[sel]).mean()
        accs.append(acc)
        risks.append(1.0 - acc)
    return np.array(coverages), np.array(accs), np.array(risks)


def accuracy_at_coverage(probs, labels, uncertainty, coverage):
    covs, accs, _ = risk_coverage_curve(probs, labels, uncertainty)
    i = int(np.argmin(np.abs(covs - coverage)))
    return float(accs[i]), float(covs[i])


def auroc_score(scores_pos, scores_neg):
    """AUROC for separating positive (in-distribution) vs negative (shifted).

    scores_pos should be high for in-distribution samples.
    """
    scores = np.concatenate([scores_pos, scores_neg])
    y = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    auroc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auroc)


def error_detection_auroc(uncertainty, correct):
    """AUROC that a random misclassified example has HIGHER uncertainty
    than a random correctly-classified example. Directly measures whether the
    uncertainty signal tells you when *not* to trust the prediction."""
    unc = np.asarray(uncertainty)
    cor = np.asarray(correct, dtype=bool)
    return auroc_score(unc[~cor], unc[cor])


# ---------------------------------------------------------------- plotting ---


def reliability_diagram(probs, labels, path, title="Reliability diagram"):
    res = expected_calibration_error(probs, labels)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax[0].plot(res["bins"], res["accs"], "o-", color="#1f77b4", lw=2)
    ax[0].set_xlabel("Confidence")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title(f"{title} (ECE={res['ece']:.4f})")
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[1].bar(res["bins"], res["densities"], width=1.0 / len(res["bins"]),
              color="#2ca02c", alpha=0.7)
    ax[1].set_xlabel("Confidence bin")
    ax[1].set_ylabel("Fraction of data")
    ax[1].set_title("Confidence histogram")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return res


def plot_risk_coverage(probs, labels, scores, labels_names, path, title=""):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, unc in scores.items():
        covs, accs, risks = risk_coverage_curve(probs, labels, unc)
        ax.plot(covs, risks, lw=2, label=labels_names.get(name, name))
    ax.set_xlabel("Coverage (fraction of predictions kept)")
    ax.set_ylabel("Risk (1 - accuracy on kept predictions)")
    ax.set_title(title or "Risk-Coverage curves (lower is better)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def risk_coverage_summary(probs, labels, scores):
    """Table of accuracy at coverage levels for every method."""
    rows = {}
    for name, unc in scores.items():
        row = {}
        for cov in (1.0, 0.95, 0.9, 0.8, 0.7, 0.5):
            acc, cov_reached = accuracy_at_coverage(probs, labels, unc, cov)
            row[cov] = acc
        rows[name] = row
    return rows


def savefig(figure, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return path