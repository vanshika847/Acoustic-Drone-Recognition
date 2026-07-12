"""
Global configuration for the project.
"""

from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATASET_DIR = PROJECT_ROOT / "datasets" / "raw"

PROCESSED_DATASET_DIR = PROJECT_ROOT / "datasets" / "processed"

METADATA_DIR = PROJECT_ROOT / "datasets" / "metadata"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ==========================================================
# Metadata Files
# ==========================================================

MASTER_METADATA_FILE = METADATA_DIR / "master_metadata.csv"

VALIDATION_REPORT = OUTPUT_DIR / "validation_report.csv"

# ==========================================================
# Audio Configuration
# ==========================================================

TARGET_SAMPLE_RATE = 16000

TARGET_CHANNELS = 1

MIN_DURATION = 1.0

MAX_DURATION = 30.0

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
}