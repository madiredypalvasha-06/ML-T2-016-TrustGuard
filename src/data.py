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
_APP_CACHE = None


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


def get_app_datasets():
    """The two slices the deployed app needs, as (calibration_ds, test_ds).

    The app never trains, so it has no use for the 45,001-image train split.
    Reading the committed subset in assets/app_images.npz avoids downloading
    and unpacking all of CIFAR-10 (170 MB) on every cold start, and avoids
    materialising the train split in memory at all.

    Values are bit-identical to get_clean_datasets(): the archive stores uint8
    and is scaled by the same 1/255 that transforms.ToTensor applies. The test
    set keeps its original ordering, so sample index N still refers to the same
    image, and the calibration images are stored pre-sampled in the order
    app/engine.py would have drawn them.

    Falls back to a full download if the archive is absent, so a fresh clone
    without it still runs.
    """
    if not os.path.exists(config.APP_IMAGES):
        _, cal_ds, test_ds = get_clean_datasets()
        return cal_ds, test_ds

    global _APP_CACHE
    if _APP_CACHE is None:
        z = np.load(config.APP_IMAGES)
        to_tensor = lambda a: torch.from_numpy(
            a.astype(np.float32) / 255.0)
        cal_ds = TensorDataset(to_tensor(z["calib_x"]),
                               torch.from_numpy(z["calib_y"].astype(np.int64)))
        test_ds = TensorDataset(to_tensor(z["test_x"]),
                                torch.from_numpy(z["test_y"].astype(np.int64)))
        _APP_CACHE = (cal_ds, test_ds)
    return _APP_CACHE


def get_shift_data(corruption, severity, device=None, test_ds=None):
    """Return (x_shifted, y) with x_shifted the corrupted form of the clean
    test set, y the true test labels (unchanged by corruption).

    x_shifted is a (10000, 3, 32, 32) float tensor; y is (10000,) int.

    test_ds lets a caller supply the split, so the app can pass the committed
    subset instead of triggering a download. It must contain the full test set:
    the corruption recipes seed one generator per batch, so corrupting a
    smaller batch would change the noise each image receives.
    """
    if test_ds is None:
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