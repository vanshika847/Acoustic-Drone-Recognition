"""
Audio Standardization Engine v2.0

Reads valid audio files from datasets/raw,
standardizes them and saves them into datasets/processed
while preserving the folder structure.
"""

from pathlib import Path
import time

import pandas as pd
from tqdm import tqdm

from configs.config import (
    PROJECT_ROOT,
    RAW_DATASET_DIR,
    PROCESSED_DATASET_DIR,
    MASTER_METADATA_FILE,
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
    remove_dc_offset,
    trim_silence,
)
from utils.audio_metrics import compute_metrics
from utils.hashing import sha256_file


logger = setup_logger("standardization.log")


# --------------------------------------------------
# Path Helpers
# --------------------------------------------------

def resolve_input_path(path_string: str) -> Path:
    """
    Supports both:

    datasets/raw/....

    and

    D:/project/datasets/raw/....
    """

    p = Path(path_string)

    if p.is_absolute():
        return p

    return PROJECT_ROOT / p


def build_output_path(input_path: Path) -> Path:
    """
    Preserve folder hierarchy.
    """

    try:
        relative = input_path.relative_to(RAW_DATASET_DIR)

    except ValueError:

        parts = input_path.parts

        if "datasets" not in parts or "raw" not in parts:
            raise ValueError(
                f"Cannot determine relative path: {input_path}"
            )

        idx = parts.index("raw")

        relative = Path(*parts[idx + 1:])

    output = PROCESSED_DATASET_DIR / relative

    return output.with_suffix(SAVE_FORMAT)


# --------------------------------------------------
# Processing
# --------------------------------------------------

def process_audio(row):

    start = time.time()

    input_path = resolve_input_path(row["path"])

    output_path = build_output_path(input_path)

    if SKIP_EXISTING and output_path.exists():

        return {
            "processing_status": "Skipped",
            "processing_time": 0.0,
        }

    try:

        audio, sr, metadata = load_audio(
            input_path,
            TARGET_SAMPLE_RATE,
        )

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

        metrics = compute_metrics(
            audio,
            TARGET_SAMPLE_RATE,
        )

        elapsed = round(
            time.time() - start,
            3,
        )

        logger.info(f"Processed: {input_path.name}")

        return {
            "processing_status": "Success",
            "processing_time": elapsed,
            "output_path": str(output_path),
            "sha256": sha256_file(output_path),
            **metadata,
            **metrics,
        }

    except Exception as e:

        logger.exception(str(e))

        return {
            "processing_status": "Failed",
            "processing_time": 0,
            "error": str(e),
        }
    # --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 60)
    print("Audio Standardization Engine v2.0")
    print("=" * 60)

    if not MASTER_METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{MASTER_METADATA_FILE}"
        )

    df = pd.read_csv(MASTER_METADATA_FILE)

    print(f"\nLoaded metadata : {len(df)} files")

    if "status" not in df.columns:
        raise ValueError(
            "Column 'status' not found in metadata."
        )

    df = df[df["status"] == "OK"].copy()

    print(f"Valid files     : {len(df)}")

    # -----------------------------
    # TEST MODE
    # -----------------------------
    # Change/remove this after testing
    df = df.head(20)

    print(f"Processing      : {len(df)} files\n")

    reports = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        ncols=100,
    ):
        reports.append(
            process_audio(row)
        )

    report_df = pd.concat(
        [
            df.reset_index(drop=True),
            pd.DataFrame(reports),
        ],
        axis=1,
    )

    output_report = (
        PROJECT_ROOT
        / "outputs"
        / "standardization_report.csv"
    )

    output_report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_df.to_csv(
        output_report,
        index=False,
    )

    print("\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)

    print(
        report_df["processing_status"]
        .value_counts(dropna=False)
    )

    success = (
        report_df["processing_status"]
        == "Success"
    ).sum()

    failed = (
        report_df["processing_status"]
        == "Failed"
    ).sum()

    skipped = (
        report_df["processing_status"]
        == "Skipped"
    ).sum()

    print()

    print(f"Success : {success}")
    print(f"Failed  : {failed}")
    print(f"Skipped : {skipped}")

    print()

    if success > 0:

        avg_time = report_df.loc[
            report_df["processing_status"] == "Success",
            "processing_time",
        ].mean()

        print(
            f"Average processing time : {avg_time:.3f} sec/file"
        )

    print()

    print("Processed audio saved to:")

    print(PROCESSED_DATASET_DIR)

    print()

    print("Report saved to:")

    print(output_report)

    print("=" * 60)


if __name__ == "__main__":
    main()