"""Training configuration for the acoustic drone detector."""

from __future__ import annotations

from pathlib import Path

import torch


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# RANDOMNESS / REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42


# ============================================================
# DATA
# ============================================================

# The project already contains the generated shard-based
# feature manifests here.
FEATURE_MANIFEST_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "features"
)

TRAIN_MANIFEST = (
    FEATURE_MANIFEST_DIR
    / "train_shard_manifest.csv"
)

VALIDATION_MANIFEST = (
    FEATURE_MANIFEST_DIR
    / "validation_shard_manifest.csv"
)

TEST_MANIFEST = (
    FEATURE_MANIFEST_DIR
    / "test_shard_manifest.csv"
)


# ============================================================
# DATALOADER
# ============================================================

BATCH_SIZE = 8

NUM_WORKERS = 0

PIN_MEMORY = torch.cuda.is_available()


# ============================================================
# TRAINING LENGTH
# ============================================================

NUM_EPOCHS = 50

EARLY_STOPPING_PATIENCE = 7

EARLY_STOPPING_MIN_DELTA = 0.0005


# ============================================================
# OPTIMIZER
# ============================================================

LEARNING_RATE = 1.0e-4

WEIGHT_DECAY = 1.0e-4


# ============================================================
# LEARNING-RATE SCHEDULER
# ============================================================

SCHEDULER_T_MAX = NUM_EPOCHS

SCHEDULER_MIN_LR = 1.0e-6


# ============================================================
# GRADIENT CONTROL
# ============================================================

GRADIENT_CLIP = 1.0


# ============================================================
# DATA AUGMENTATION
# ============================================================

AUGMENT_PROBABILITY = 0.35

TIME_MASK_MAX_FRAMES = 12

FEATURE_DROPOUT_PROBABILITY = 0.08


# ============================================================
# BALANCED SAMPLING
# ============================================================

BALANCED_SAMPLING_RATIO = 0.70

EPOCH_SAMPLE_MULTIPLIER = 1.0

SAMPLE_WITH_REPLACEMENT = True


# ============================================================
# HARD-EXAMPLE MINING
# ============================================================

HARD_MINING_WARMUP_EPOCHS = 5

HARD_MINING_STRENGTH = 0.35

HARDNESS_MOMENTUM = 0.90

MIN_HARD_WEIGHT = 0.50

MAX_HARD_WEIGHT = 3.00


# ============================================================
# DECISION THRESHOLD
# ============================================================

DEFAULT_DECISION_THRESHOLD = 0.50


# ============================================================
# CHECKPOINTS
# ============================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
)

BEST_MODEL_NAME = "best_model.pt"

LAST_MODEL_NAME = "last_model.pt"


# ============================================================
# RESUME
# ============================================================

# IMPORTANT:
#
# The existing best_model.pt was produced by the OLD model
# architecture and cannot be loaded into the current model.
#
# Therefore the first run with the current architecture must
# start from scratch.

RESUME_TRAINING = False