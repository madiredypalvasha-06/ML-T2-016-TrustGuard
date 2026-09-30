"""Global configuration for the ML-T2-016 uncertainty project."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULT_DIR = os.path.join(BASE_DIR, "results")
FIGURE_DIR = os.path.join(RESULT_DIR, "figures")
APP_DIR = os.path.join(BASE_DIR, "app")

for _d in (DATA_DIR, CHECKPOINT_DIR, RESULT_DIR, FIGURE_DIR):
    os.makedirs(_d, exist_ok=True)

SEED = 42

# --- data ---
CIFAR_DIR = os.path.join(DATA_DIR, "cifar-10")
CIFAR10C_DIR = os.path.join(DATA_DIR, "CIFAR-10-C")
CIFAR10C_URL = (
    "https://github.com/hendrycks/robustness/releases/download/1.0/cifar-10-c.tar"
)
CIFAR10C_LABELS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost",
    "fog", "brightness", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression",
]
# subset of corruptions used in the study
CORRUPTIONS = ["gaussian_noise", "motion_blur", "fog", "brightness", "contrast"]
SEVERITIES = [1, 2, 3, 4, 5]

# train/calibration/test split on clean data
TRAIN_FRAC = 0.9          # remaining 10% of train split -> calibration
MC_DROPOUT_PASSES = 30

# --- model / training ---
DEVICE = None  # resolved lazily
EPOCHS = 24
BATCH_SIZE = 128
LR = 3e-3
WEIGHT_DECAY = 5e-4
DROPOUT_P = 0.2
MODEL_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "baseline_cnn.pt")
TEMP_PARAMS = os.path.join(CHECKPOINT_DIR, "temp_scale.json")

# --- deep ensemble ---
ENSEMBLE_MEMBERS = 4
ENSEMBLE_PATH = lambda tag: os.path.join(CHECKPOINT_DIR, f"{tag}.pt")
ENSEMBLE_TAGS = [f"member_{i}" for i in range(1, ENSEMBLE_MEMBERS)]
MAHALANOBIS_PATH = os.path.join(CHECKPOINT_DIR, "mahalanobis_stats.npz")

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_device(force_cpu=False) -> str:
    global DEVICE
    if force_cpu:
        return "cpu"
    if DEVICE is None:
        import torch
        if torch.backends.mps.is_available():
            DEVICE = "mps"
        else:
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DEVICE