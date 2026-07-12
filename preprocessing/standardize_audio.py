"""
Audio Standardization Engine
"""

from pathlib import Path
import time

import pandas as pd
from tqdm import tqdm

from configs.config import (
    MASTER_METADATA_FILE,
    PROCESSED_DATASET_DIR,
    RAW_DATASET_DIR,
)

from configs.preprocessing_config import (
    TARGET_SAMPLE_RATE,
    NORMALIZE_AUDIO,
    TARGET_PEAK,
    TRIM_SILENCE,
    TOP_DB,
    SAVE_FORMAT,
    SKIP_EXISTING,
)

from utils.logger import setup_logger
from utils.audio_loader import load_audio
from utils.audio_writer import save_audio
from utils.audio_processor import (
    normalize_audio,
    trim_silence,
    remove_dc_offset,
)
from utils.audio_metrics import compute_metrics
from utils.hashing import sha256_file


logger = setup_logger("standardization.log")


def process_audio(row):
    start = time.time()

    input_path = Path(row["path"])

    try:
        relative_path = input_path.relative_to(RAW_DATASET_DIR)
    except ValueError:
        logger.error(f"Cannot determine relative path: {input_path}")
        return {
            "processing_status": "Failed",
            "error": "Invalid path",
            "processing_time": 0,
        }

    output_path = PROCESSED_DATASET_DIR / relative_path
    output_path = output_path.with_suffix(SAVE_FORMAT)

    if SKIP_EXISTING and output_path.exists():
        return {
            "processing_status": "Skipped",
            "processing_time": 0,
        }

    try:

        audio, sr, metadata = load_audio(
            input_path,
            TARGET_SAMPLE_RATE,
        )

        original_metrics = compute_metrics(audio, sr)

        audio = remove_dc_offset(audio)

        if NORMALIZE_AUDIO:
            audio = normalize_audio(
                audio,
                TARGET_PEAK,
            )

        if TRIM_SILENCE:
            audio = trim_silence(
                audio,
                TOP_DB,
            )

        save_audio(
            audio,
            TARGET_SAMPLE_RATE,
            output_path,
        )

        processed_metrics = compute_metrics(
            audio,
            TARGET_SAMPLE_RATE,
        )

        elapsed = round(time.time() - start, 3)

        logger.info(f"Processed {input_path.name}")

        return {
            "processing_status": "Success",
            "processing_time": elapsed,
            "sha256": sha256_file(output_path),

            # Original metadata
            "original_samplerate": metadata["original_samplerate"],
            "original_channels": metadata["original_channels"],
            "format": metadata["format"],
            "subtype": metadata["subtype"],

            # Original metrics
            "original_duration": original_metrics["duration"],
            "original_rms": original_metrics["rms"],
            "original_peak": original_metrics["peak"],

            # Processed metrics
            "processed_duration": processed_metrics["duration"],
            "processed_rms": processed_metrics["rms"],
            "processed_peak": processed_metrics["peak"],
            "dynamic_range": processed_metrics["dynamic_range"],
            "silence_ratio": processed_metrics["silence_ratio"],
        }

    except Exception as e:

        logger.error(f"{input_path}: {e}")

        return {
            "processing_status": "Failed",
            "error": str(e),
            "processing_time": 0,
        }


def main():

    print("=" * 60)
    print("Audio Standardization Engine")
    print("=" * 60)

    df = pd.read_csv(MASTER_METADATA_FILE)

    df = df[df["status"] == "OK"].head(20)

    reports = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
    ):
        reports.append(process_audio(row))

    report = pd.concat(
        [
            df.reset_index(drop=True),
            pd.DataFrame(reports),
        ],
        axis=1,
    )

    output = Path("outputs/standardization_report.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    report.to_csv(output, index=False)

    print("\nColumns:")
    print(report.columns.tolist())

    print("\nProcessing Summary:")
    print(report["processing_status"].value_counts())

    print(f"\nReport saved to: {output}")


if __name__ == "__main__":
    main()