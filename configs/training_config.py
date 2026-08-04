"""Training configuration for the Acoustic Drone Recognition System."""

from __future__ import annotations

from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "features"
    / "train_feature_manifest.csv"
)

VALIDATION_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "features"
    / "validation_feature_manifest.csv"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
)

LOG_DIR = (
    PROJECT_ROOT
    / "logs"
)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Training Hyperparameters
# ==========================================================

BATCH_SIZE = 8

NUM_EPOCHS = 50

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

PIN_MEMORY = True

# ==========================================================
# Scheduler
# ==========================================================

SCHEDULER_T_MAX = NUM_EPOCHS

SCHEDULER_MIN_LR = 1e-6

# ==========================================================
# Gradient Clipping
# ==========================================================

GRADIENT_CLIP = 1.0

# ==========================================================
# Checkpoints
# ==========================================================

BEST_MODEL_NAME = "best_model.pt"

LAST_MODEL_NAME = "last_model.pt"

SAVE_EVERY = 1

# ==========================================================
# Randomness
# ==========================================================

RANDOM_SEED = 42

# ==========================================================
# Device
# ==========================================================

import torch

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)