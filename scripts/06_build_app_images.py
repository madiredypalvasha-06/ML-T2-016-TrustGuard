"""Build the image subset the deployed app needs, so it never downloads CIFAR-10.

The app only ever touches two slices of CIFAR-10:

  * 600 calibration images, sampled exactly as app/engine.py samples them, used
    to fit the abstention thresholds.
  * the full 10,000-image test set, because the corruption recipes in
    src/corruptions.py seed a single generator per *batch* (``torch.randn(x.shape,
    generator=g)``). Corrupting a subset would hand each image a different noise
    draw than corrupting the full set, so the committed artefacts would stop
    matching the ones in the paper and the demo screenshots. The whole test set
    is therefore kept, and stored in its original order so sample index N still
    means the same image.

Images are stored as uint8 rather than float32, which is a 4x saving over the
in-memory tensors, and converted back with the same /255 that
``transforms.ToTensor`` applies, so the values are bit-identical.

Run:  python scripts/06_build_app_images.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config

OUT_PATH = os.path.join(config.BASE_DIR, "assets", "app_images.npz")

# Must match app/engine.py::_fit_trust_thresholds.
CALIB_SEED = 42
N_CALIB = 600


def main():
    from torchvision import datasets

    train = datasets.CIFAR10(config.CIFAR_DIR, train=True, download=True)
    test = datasets.CIFAR10(config.CIFAR_DIR, train=False, download=True)

    # Reproduce the calibration split exactly as src/data.py defines it, then
    # take the same 600 indices app/engine.py would have drawn.
    rng = np.random.RandomState(config.SEED)
    n_cal = int(len(train) * (1.0 - config.TRAIN_FRAC))
    cal_idx = rng.permutation(len(train))[:n_cal]
    picked = np.random.RandomState(CALIB_SEED).choice(n_cal, N_CALIB, replace=False)
    src_idx = cal_idx[picked]

    # (N, 32, 32, 3) uint8 -> (N, 3, 32, 32) uint8, matching ToTensor layout.
    calib_x = np.ascontiguousarray(train.data[src_idx].transpose(0, 3, 1, 2))
    calib_y = np.asarray(train.targets, dtype=np.int64)[src_idx]
    test_x = np.ascontiguousarray(test.data.transpose(0, 3, 1, 2))
    test_y = np.asarray(test.targets, dtype=np.int64)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        calib_x=calib_x, calib_y=calib_y,
        test_x=test_x, test_y=test_y,
        n_calib_total=np.int64(n_cal),
        calib_seed=np.int64(CALIB_SEED),
    )

    size = os.path.getsize(OUT_PATH)
    print(f"wrote {os.path.relpath(OUT_PATH, config.BASE_DIR)}  ({size/1e6:.1f} MB)")
    print(f"  calib_x {calib_x.shape} {calib_x.dtype}   calib_y {calib_y.shape}")
    print(f"  test_x  {test_x.shape} {test_x.dtype}    test_y  {test_y.shape}")
    print("  replaces a 170 MB download on every cold start")


if __name__ == "__main__":
    main()
