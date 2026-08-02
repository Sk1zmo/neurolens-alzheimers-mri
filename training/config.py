"""Central configuration for the Alzheimer's MRI classification pipeline."""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SPLITS_DIR = ARTIFACTS_DIR / "splits"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

DATA_ROOT = Path(
    r"F:\Alzheimers folder\Alzheimers disease dataset"
    r"\Alzheimers disease dataset\Alzheimer's dataset"
)
ORIGINAL_DIR = DATA_ROOT / "OriginalDataset"
AUGMENTED_DIR = DATA_ROOT / "AugmentedAlzheimerDataset"

# Scans contributed through the web app and pulled down for retraining.
RETRAIN_DIR = PROJECT_ROOT / "artifacts" / "retrain_pool"

# --------------------------------------------------------------------------
# Label space
# --------------------------------------------------------------------------
# Order is fixed forever: it is baked into the ONNX model and the frontend.
CLASS_DIRS = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented",
]
CLASS_LABELS = [
    "Non Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented",
]
NUM_CLASSES = len(CLASS_DIRS)
DIR_TO_INDEX = {name: i for i, name in enumerate(CLASS_DIRS)}

# --------------------------------------------------------------------------
# Preprocessing — must stay in sync with web/api/_inference.py
# --------------------------------------------------------------------------
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------
SEED = 1337
# The clean evaluation set is carved out of OriginalDataset only. Augmented
# images are training-side extras, and any augmented frame that is a
# near-duplicate of a val/test original gets dropped (see prepare_split.py).
VAL_FRACTION = 0.15
TEST_FRACTION = 0.20
# Cosine similarity in pretrained-embedding space above which an augmented
# image is treated as derived from a held-out original.
LEAK_SIMILARITY_THRESHOLD = 0.92

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
# Sized for an 8 GB laptop GPU that is also driving the desktop. Larger values
# train fine on a dedicated card but hit allocation failures here once a
# browser or design tool is holding VRAM.
BATCH_SIZE = 40
NUM_WORKERS = 4
EPOCHS_HEAD = 3          # warm-up with the backbone frozen
EPOCHS_FINETUNE = 22     # full fine-tune with cosine decay
LR_HEAD = 3e-3
LR_BACKBONE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 6
AMP = True

# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
ONNX_OPSET = 17
MODEL_NAME = "alzheimer_effnetb0"


def ensure_dirs() -> None:
    for d in (ARTIFACTS_DIR, SPLITS_DIR, CHECKPOINT_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
