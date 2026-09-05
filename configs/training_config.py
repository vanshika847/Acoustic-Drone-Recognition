"""Final-round training configuration for the acoustic drone detector."""

from __future__ import annotations

from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_MANIFEST = PROJECT_ROOT / "outputs" / "features" / "train_shard_manifest.csv"
VALIDATION_MANIFEST = PROJECT_ROOT / "outputs" / "features" / "validation_shard_manifest.csv"
TEST_MANIFEST = PROJECT_ROOT / "outputs" / "features" / "test_shard_manifest.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
TRAINING_HISTORY_PATH = LOG_DIR / "final_training_history.csv"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Final training run
# ---------------------------------------------------------------------------
# The existing feature shards were already built for a batch size of small
# samples. Gradient accumulation gives a larger effective batch without
# changing the stored feature data.
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 70

# Conservative learning rate for the already strong temporal CNN.
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
GRADIENT_CLIP = 1.0
LABEL_SMOOTHING = 0.0

# Scheduler: linear warm-up followed by cosine decay.
WARMUP_EPOCHS = 3
SCHEDULER_T_MAX = NUM_EPOCHS
SCHEDULER_MIN_LR = 1.0e-6

# Early stopping prevents the final run from overfitting the fixed validation
# set. The saved best checkpoint is still selected by validation drone F1.
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 1.0e-4

# Checkpoints
BEST_MODEL_NAME = "best_model.pt"
LAST_MODEL_NAME = "last_model.pt"
SAVE_EVERY = 1

# IMPORTANT: final run starts from scratch. This avoids accidentally carrying
# optimizer state, hardness state, or weights from training round 1-4.
RESUME_TRAINING = False

# Reproducibility
RANDOM_SEED = 42

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Sampling / hard-example mining
# ---------------------------------------------------------------------------
# The dataset is strongly background-heavy. Balanced sampling is retained,
# but hard mining is deliberately mild so a few difficult background
# recordings cannot dominate the final model.
HARD_MINING_WARMUP_EPOCHS = 5
HARD_MINING_STRENGTH = 0.25
BALANCED_SAMPLING_RATIO = 0.90
MIN_HARD_WEIGHT = 0.50
MAX_HARD_WEIGHT = 2.0
HARDNESS_MOMENTUM = 0.85
EPOCH_SAMPLE_MULTIPLIER = 1.0
SAMPLE_WITH_REPLACEMENT = True

# Feature-space augmentation. Validation/test data are never augmented.
AUGMENT_PROBABILITY = 0.10
TIME_MASK_MAX_FRAMES = 8
FEATURE_DROPOUT_PROBABILITY = 0.0

# Threshold is learned from validation drone F1 and stored in the best
# checkpoint. This is only the fallback before the first best checkpoint.
DEFAULT_DECISION_THRESHOLD = 0.50

# EMA is used for validation/export; slightly faster decay adapts sooner in early epochs.
EMA_DECAY = 0.98

# DataLoader settings. Keep workers at 0 for Windows stability.
NUM_WORKERS = 0
PIN_MEMORY = DEVICE.type == "cuda"
