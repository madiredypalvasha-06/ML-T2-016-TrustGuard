"""Data loading and preparation for CIFAR-10 and controlled distribution shift.

Distribution shift is produced locally by reproducing the standard CIFAR-10-C
corruption recipes (see src/corruptions.py) applied to the clean test set, so
the study is fully self-contained and reproducible.
"""
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset

from src import config, corruptions

_CLEAN_CACHE = None


def get_clean_datasets():
    """Clean CIFAR-10 datasets as (train_ds, calibration_ds, test_ds).

      train_ds       : (45000, 3, 32, 32) float + int labels
      calibration_ds : (5000, 3, 32, 32) float + int labels  (never used for
                       gradient training; only temperature/threshold fitting)
      test_ds        : (10000, 3, 32, 32) float + int labels (fully held out)
    """
    global _CLEAN_CACHE
    if _CLEAN_CACHE is not None:
        return _CLEAN_CACHE
    from torchvision import datasets, transforms

    train = datasets.CIFAR10(
        config.CIFAR_DIR, train=True, download=True,
        transform=transforms.ToTensor(),
    )
    test = datasets.CIFAR10(
        config.CIFAR_DIR, train=False, download=True,
        transform=transforms.ToTensor(),
    )

    x_tr = torch.stack([train[i][0] for i in range(len(train))])
    y_tr = torch.tensor([train[i][1] for i in range(len(train))])
    x_te = torch.stack([test[i][0] for i in range(len(test))])
    y_te = torch.tensor([test[i][1] for i in range(len(test))])

    rng = np.random.RandomState(config.SEED)
    n_cal = int(len(x_tr) * (1.0 - config.TRAIN_FRAC))
    perm = rng.permutation(len(x_tr))
    cal_idx, tr_idx = perm[:n_cal], perm[n_cal:]

    train_ds = TensorDataset(x_tr[tr_idx], y_tr[tr_idx])
    calibration_ds = TensorDataset(x_tr[cal_idx], y_tr[cal_idx])
    test_ds = TensorDataset(x_te, y_te)
    _CLEAN_CACHE = (train_ds, calibration_ds, test_ds)
    return _CLEAN_CACHE


def get_shift_data(corruption, severity, device=None):
    """Return (x_shifted, y) with x_shifted the corrupted form of the clean
    test set, y the true test labels (unchanged by corruption).

    x_shifted is a (10000, 3, 32, 32) float tensor; y is (10000,) int.
    """
    _, _, test_ds = get_clean_datasets()
    x_all = torch.stack([t for t, _ in test_ds])
    y_all = torch.tensor([lbl for _, lbl in test_ds]).numpy()
    x = x_all.to(device) if device is not None else x_all
    xc = corruptions.corrupt(x, corruption, severity)
    return xc, y_all


def dataloader(ds, batch_size=256, shuffle=False, drop_last=False):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
    )