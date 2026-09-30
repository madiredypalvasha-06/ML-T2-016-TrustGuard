"""Mahalanobis distance in feature space for OOD / shift detection.

Approach D from the research brief: an input is "unfamiliar" when its
penultimate-layer features lie far (in a whitened, class-conditioned metric)
from every class centroid learned on the training distribution. This signal
complements confidence calibration and is robust to the "overconfident on
noise" failure mode of softmax scores.
"""
from typing import Optional

import numpy as np
import torch
from sklearn.covariance import LedoitWolf


class MahalanobisOOD:
    """Class-conditional Mahalanobis distance (Lee et al. 2018, simplified)."""

    def __init__(self):
        self.means = None        # (C, D)
        self.cov_inv = None      # (D, D) pooled precision matrix
        self.centroids = None    # fitted class centroids used for CIFAR-10
        self.classes = None

    def fit(self, features, labels):
        """features (N, D), labels (N,) int. Fits per-class means + pooled
        shrunk covariance (Ledoit-Wolf) over all classes."""
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        self.classes = np.unique(labels)
        eps = 1e-6
        self.means = np.stack([
            features[labels == c].mean(axis=0)
            if (labels == c).any() else np.zeros(features.shape[1])
            for c in self.classes
        ])
        residuals = features - self.means[labels]
        cov = LedoitWolf().fit(residuals).covariance_
        self.cov_inv = np.linalg.pinv(cov + eps * np.eye(cov.shape[0]))
        return self

    def distance(self, features):
        """Per-instance *minimum* Mahalanobis distance over all class means.
        Higher = more unlike the training distribution."""
        features = np.asarray(features, dtype=np.float64)
        out = np.full(len(features), np.inf)
        if self.means is None:
            return out
        for i in range(len(features)):
            diffs = features[i] - self.means          # (C, D)
            d2 = np.einsum("cd,de,ce->c", diffs, self.cov_inv, diffs)
            out[i] = np.sqrt(max(d2.min(), 0.0))
        return out

    def save(self, path):
        np.savez_compressed(path,
                            means=self.means,
                            cov_inv=self.cov_inv,
                            classes=self.classes)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        m = cls()
        m.means, m.cov_inv, m.classes = data["means"], data["cov_inv"], data["classes"]
        return m


def fit_from_dataloader(model, loader, device):
    """Extract penultimate features from a loader and fit the estimator
    on the *labels of the calibration + a slice of the training set*."""
    feats, labels = [], []
    with torch.no_grad():
        model.eval()
        for xb, yb in loader:
            f, _ = model.forward_features(xb.to(device))
            feats.append(f.cpu().numpy())
            labels.append(yb.numpy())
    return MahalanobisOOD().fit(np.concatenate(feats),
                                np.concatenate(labels))