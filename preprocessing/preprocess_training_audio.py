"""
Convert approved split-manifest recordings into model-ready audio segments.

Purpose
-------
Read only the labelled files already assigned to train, validation, and test
splits; resample them to mono 16 kHz audio, remove DC offset, peak-normalise,
segment them into fixed windows, and write a segment-level manifest.

Raw audio is never modified.

Inputs
------
Split manifests under:
    datasets/processed/manifests/

Referenced files below:
    datasets/raw/

Outputs
-------
PCM WAV segments under:

    datasets/processed/segments/<split>/

One segment-level manifest per split:

    datasets/processed/manifests/train_segments.csv
    datasets/processed/manifests/validation_segments.csv
    datasets/processed/manifests/test_segments.csv

One preprocessing report per split:

    datasets/processed/reports/train_preprocessing.csv
    datasets/processed/reports/validation_preprocessing.csv
    datasets/processed/reports/test_preprocessing.csv

Algorithm
---------
Each source recording is:

1. Loaded as audio.
2. Converted to mono.
3. Resampled to 16 kHz if necessary.
4. Checked for finite values.
5. DC-offset corrected.
6. Peak-normalised.
7. Divided into fixed 4-second windows with 2-second overlap.
8. Final partial windows are zero-padded.
9. Very short recordings (>= minimum_input_seconds) are also retained
   and zero-padded to the full 4-second window.

A deterministic filename based on the source SHA-256 and segment index
makes the operation idempotent.

Usage
-----
From the repository root:

    python -m preprocessing.preprocess_training_audio

With workers:

    python -m preprocessing.preprocess_training_audio --workers 4

Process only training data:

    python -m preprocessing.preprocess_training_audio --splits train

Overwrite existing segments:

    python -m preprocessing.preprocess_training_audio --overwrite
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


# ---------------------------------------------------------------------------
# Project-root support
# ---------------------------------------------------------------------------

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


from configs.config import PROCESSED_DATASET_DIR, RAW_DATASET_DIR
from preprocessing.audio_segmentation import (
    AudioSegmenter,
    SegmentationConfig,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLIT_NAMES = (
    "train",
    "validation",
    "test",
)

SOURCE_MANIFEST_DIRECTORY_NAME = "manifests"

SEGMENT_DIRECTORY_NAME = "segments"

REPORT_DIRECTORY_NAME = "reports"


SEGMENT_MANIFEST_COLUMNS = (
    "split",
    "segment_id",
    "processed_relative_path",
    "source_dataset",
    "source_relative_path",
    "source_sha256",
    "recording_group_id",
    "binary_label",
    "source_category",
    "source_fold",
    "segment_index",
    "start_sample",
    "end_sample",
    "start_seconds",
    "end_seconds",
    "source_duration_seconds",
    "is_padded",
    "sample_rate",
    "segment_samples",
)


REPORT_COLUMNS = (
    "split",
    "source_dataset",
    "source_relative_path",
    "source_sha256",
    "recording_group_id",
    "binary_label",
    "source_duration_seconds",
    "status",
    "segment_count",
    "error",
)


REQUIRED_SOURCE_COLUMNS = {
    "split",
    "dataset",
    "relative_path",
    "file_name",
    "binary_label",
    "source_category",
    "source_fold",
    "recording_group_id",
    "sha256",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AudioPreprocessingError(RuntimeError):
    """Raised when preprocessing configuration or source metadata is unsafe."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AudioPreprocessingConfig:
    """
    Immutable settings for audio loading, normalisation, and segmentation.

    Attributes
    ----------
    sample_rate:
        Target sample rate in Hertz.

    segment_window_seconds:
        Length of every model input segment.

    segment_hop_seconds:
        Distance between consecutive segment starts.

    minimum_input_seconds:
        Absolute minimum duration accepted from a source recording.

        Recordings at or above this duration are retained and padded if
        necessary.

    minimum_final_seconds:
        Minimum duration required for a normal trailing partial segment.

    target_peak:
        Maximum peak amplitude after normalisation.

    remove_dc_offset:
        Whether the waveform mean should be removed.
    """

    sample_rate: int = 16_000

    segment_window_seconds: float = 4.0

    segment_hop_seconds: float = 2.0

    minimum_input_seconds: float = 0.25

    minimum_final_seconds: float = 0.25

    target_peak: float = 0.99

    remove_dc_offset: bool = True

    def __post_init__(self) -> None:
        """Validate preprocessing settings."""

        if self.sample_rate <= 0:
            raise AudioPreprocessingError(
                "sample_rate must be greater than zero."
            )

        if self.segment_window_seconds <= 0.0:
            raise AudioPreprocessingError(
                "segment_window_seconds must be greater than zero."
            )

        if self.segment_hop_seconds <= 0.0:
            raise AudioPreprocessingError(
                "segment_hop_seconds must be greater than zero."
            )

        if self.segment_hop_seconds > self.segment_window_seconds:
            raise AudioPreprocessingError(
                "segment_hop_seconds cannot exceed segment_window_seconds."
            )

        if not 0.0 < self.minimum_input_seconds <= self.segment_window_seconds:
            raise AudioPreprocessingError(
                "minimum_input_seconds must be greater than zero and "
                "no larger than segment_window_seconds."
            )

        if not 0.0 < self.minimum_final_seconds <= self.segment_window_seconds:
            raise AudioPreprocessingError(
                "minimum_final_seconds must be greater than zero and "
                "no larger than segment_window_seconds."
            )

        if not 0.0 < self.target_peak <= 1.0:
            raise AudioPreprocessingError(
                "target_peak must be in the range (0, 1]."
            )

        # Force validation of the segmentation configuration.
        SegmentationConfig(
            sample_rate=self.sample_rate,
            window_seconds=self.segment_window_seconds,
            hop_seconds=self.segment_hop_seconds,
            minimum_final_seconds=self.minimum_final_seconds,
            pad_final_segment=True,
        )

    def segmentation_config(self) -> SegmentationConfig:
        """Return the corresponding segmentation configuration."""

        return SegmentationConfig(
            sample_rate=self.sample_rate,
            window_seconds=self.segment_window_seconds,
            hop_seconds=self.segment_hop_seconds,
            minimum_final_seconds=self.minimum_final_seconds,
            pad_final_segment=True,
        )


# ---------------------------------------------------------------------------
# Request/result data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceProcessingRequest:
    """All information required to process one source recording."""

    row: Mapping[str, str]

    split_name: str

    raw_datasets_directory: Path

    processed_datasets_directory: Path

    config: AudioPreprocessingConfig

    overwrite: bool


@dataclass(frozen=True, slots=True)
class SourceProcessingResult:
    """Segment rows and source-level processing result."""

    segment_rows: tuple[
        dict[str, str | int | float | bool],
        ...
    ]

    report_row: dict[str, str | int | float]


@dataclass(frozen=True, slots=True)
class SplitPreprocessingSummary:
    """Summary for one completed split preprocessing task."""

    split_name: str

    source_manifest_path: Path

    segment_manifest_path: Path

    report_path: Path

    source_recordings: int

    successful_recordings: int

    failed_recordings: int

    written_segments: int


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def _read_source_manifest(
    manifest_path: Path,
) -> list[dict[str, str]]:
    """
    Read and validate one split manifest.

    Parameters
    ----------
    manifest_path:
        Path to train.csv, validation.csv, or test.csv.

    Returns
    -------
    list[dict[str, str]]
        Source manifest rows.
    """

    if not manifest_path.is_file():
        raise AudioPreprocessingError(
            f"Split manifest was not found: '{manifest_path}'."
        )

    with manifest_path.open(
        encoding="utf-8",
        newline="",
    ) as manifest_file:

        reader = csv.DictReader(manifest_file)

        missing_columns = REQUIRED_SOURCE_COLUMNS.difference(
            reader.fieldnames or ()
        )

        if missing_columns:
            missing_text = ", ".join(
                sorted(missing_columns)
            )

            raise AudioPreprocessingError(
                f"Split manifest '{manifest_path}' is missing "
                f"columns: {missing_text}."
            )

        rows = list(reader)

    return rows


# ---------------------------------------------------------------------------
# Source path resolution
# ---------------------------------------------------------------------------


def _resolve_source_path(
    row: Mapping[str, str],
    raw_datasets_directory: Path,
) -> Path:
    """
    Resolve a source audio path safely.

    Prevents relative-path traversal outside the declared dataset directory.
    """

    dataset_name = row["dataset"]

    if not dataset_name:
        raise AudioPreprocessingError(
            "Manifest row contains an empty dataset name."
        )

    if Path(dataset_name).name != dataset_name:
        raise AudioPreprocessingError(
            f"Invalid dataset name in manifest: '{dataset_name}'."
        )

    dataset_directory = (
        raw_datasets_directory / dataset_name
    ).resolve()

    source_path = (
        dataset_directory / row["relative_path"]
    ).resolve()

    if not source_path.is_relative_to(dataset_directory):
        raise AudioPreprocessingError(
            "Source path escapes dataset directory: "
            f"'{row['relative_path']}'."
        )

    if not source_path.is_file():
        raise AudioPreprocessingError(
            f"Source audio file was not found: '{source_path}'."
        )

    return source_path


# ---------------------------------------------------------------------------
# Audio loading and standardisation
# ---------------------------------------------------------------------------


def _load_and_standardise(
    source_path: Path,
    config: AudioPreprocessingConfig,
) -> tuple[np.ndarray, float]:
    """
    Load and standardise one source recording.

    Processing:
        1. Decode audio.
        2. Convert to mono.
        3. Resample to target sample rate.
        4. Validate samples.
        5. Remove DC offset.
        6. Peak-normalise.

    Returns
    -------
    tuple[np.ndarray, float]
        Standardised waveform and its duration in seconds.
    """

    samples, original_sample_rate = sf.read(
        source_path,
        dtype="float32",
        always_2d=True,
    )

    if samples.size == 0:
        raise AudioPreprocessingError(
            f"No audio samples decoded from '{source_path}'."
        )

    if original_sample_rate <= 0:
        raise AudioPreprocessingError(
            f"Invalid source sample rate: {original_sample_rate}."
        )

    # ---------------------------------------------------------------
    # Convert all channels to mono.
    # ---------------------------------------------------------------

    waveform = np.mean(
        samples,
        axis=1,
        dtype=np.float32,
    )

    # ---------------------------------------------------------------
    # Resample.
    # ---------------------------------------------------------------

    if original_sample_rate != config.sample_rate:

        common_divisor = gcd(
            original_sample_rate,
            config.sample_rate,
        )

        up = config.sample_rate // common_divisor
        down = original_sample_rate // common_divisor

        waveform = resample_poly(
            waveform,
            up=up,
            down=down,
        ).astype(
            np.float32,
            copy=False,
        )

    # ---------------------------------------------------------------
    # Validate waveform.
    # ---------------------------------------------------------------

    if waveform.ndim != 1:
        raise AudioPreprocessingError(
            f"Expected mono waveform, got {waveform.ndim}D."
        )

    if waveform.size == 0:
        raise AudioPreprocessingError(
            f"Audio became empty after preprocessing: '{source_path}'."
        )

    if not np.all(np.isfinite(waveform)):
        raise AudioPreprocessingError(
            f"Non-finite audio samples decoded from '{source_path}'."
        )

    # ---------------------------------------------------------------
    # Calculate duration before any padding.
    # ---------------------------------------------------------------

    duration_seconds = (
        waveform.size / config.sample_rate
    )

    # ---------------------------------------------------------------
    # Reject only genuinely unusable recordings.
    #
    # IMPORTANT:
    # We no longer require 1 second.
    # A recording >= 0.25 seconds is retained and padded.
    # ---------------------------------------------------------------

    if duration_seconds < config.minimum_input_seconds:
        raise AudioPreprocessingError(
            f"Audio duration {duration_seconds:.3f}s is below "
            f"minimum accepted duration "
            f"{config.minimum_input_seconds:.3f}s."
        )

    # ---------------------------------------------------------------
    # Remove DC offset.
    # ---------------------------------------------------------------

    if config.remove_dc_offset:

        waveform = (
            waveform
            - np.mean(
                waveform,
                dtype=np.float64,
            )
        )

    # ---------------------------------------------------------------
    # Peak normalisation.
    # ---------------------------------------------------------------

    peak = float(
        np.max(
            np.abs(waveform)
        )
    )

    if peak > 0.0:

        waveform = (
            waveform
            * (
                config.target_peak
                / peak
            )
        )

    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    )

    return waveform, duration_seconds


# ---------------------------------------------------------------------------
# Segment writing
# ---------------------------------------------------------------------------


def _write_segment(
    waveform: np.ndarray,
    sample_rate: int,
    destination: Path,
    overwrite: bool,
) -> None:
    """
    Atomically write one PCM-16 WAV segment.

    Existing files are reused unless overwrite=True.
    """

    if destination.is_file() and not overwrite:
        return

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_suffix(
        ".wav.tmp"
    )

    try:

        sf.write(
            temporary_path,
            waveform,
            sample_rate,
            subtype="PCM_16",
            format="WAV",
        )

        temporary_path.replace(
            destination
        )

    finally:

        if temporary_path.exists():

            try:
                temporary_path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Segment destination
# ---------------------------------------------------------------------------


def _segment_destination(
    request: SourceProcessingRequest,
    segment_index: int,
) -> tuple[Path, Path]:
    """
    Build deterministic segment destination paths.
    """

    row = request.row

    relative_path = (
        Path(SEGMENT_DIRECTORY_NAME)
        / request.split_name
        / f"label_{row['binary_label']}"
        / row["dataset"]
        / f"{row['sha256']}_{segment_index:04d}.wav"
    )

    absolute_path = (
        request.processed_datasets_directory
        / relative_path
    )

    return (
        absolute_path,
        relative_path,
    )


# ---------------------------------------------------------------------------
# Single recording preprocessing
# ---------------------------------------------------------------------------


def _process_source_recording(
    request: SourceProcessingRequest,
) -> SourceProcessingResult:
    """
    Process one source recording.

    A failure affects only that recording and is written to the report.
    """

    row = request.row

    report_base: dict[str, str | int | float] = {
        "split": request.split_name,
        "source_dataset": row["dataset"],
        "source_relative_path": row["relative_path"],
        "source_sha256": row["sha256"],
        "recording_group_id": row["recording_group_id"],
        "binary_label": row["binary_label"],
        "source_duration_seconds": "",
    }

    try:

        # -----------------------------------------------------------
        # Resolve source.
        # -----------------------------------------------------------

        source_path = _resolve_source_path(
            row,
            request.raw_datasets_directory,
        )

        # -----------------------------------------------------------
        # Load and standardise.
        # -----------------------------------------------------------

        waveform, source_duration_seconds = (
            _load_and_standardise(
                source_path,
                request.config,
            )
        )

        # -----------------------------------------------------------
        # Segment.
        # -----------------------------------------------------------

        segmenter = AudioSegmenter(
            request.config.segmentation_config()
        )

        audio_segments = segmenter.segment(
            waveform
        )

        if not audio_segments:

            raise AudioPreprocessingError(
                "No valid audio segments were produced."
            )

        # -----------------------------------------------------------
        # Build segment manifest rows.
        # -----------------------------------------------------------

        segment_rows: list[
            dict[str, str | int | float | bool]
        ] = []

        for audio_segment in audio_segments:

            (
                destination,
                processed_relative_path,
            ) = _segment_destination(
                request,
                audio_segment.index,
            )

            _write_segment(
                waveform=audio_segment.waveform,
                sample_rate=request.config.sample_rate,
                destination=destination,
                overwrite=request.overwrite,
            )

            segment_rows.append(
                {
                    "split": request.split_name,

                    "segment_id": (
                        f"{row['sha256']}:"
                        f"{audio_segment.index}"
                    ),

                    "processed_relative_path":
                        processed_relative_path.as_posix(),

                    "source_dataset":
                        row["dataset"],

                    "source_relative_path":
                        row["relative_path"],

                    "source_sha256":
                        row["sha256"],

                    "recording_group_id":
                        row["recording_group_id"],

                    "binary_label":
                        row["binary_label"],

                    "source_category":
                        row["source_category"],

                    "source_fold":
                        row["source_fold"],

                    "segment_index":
                        audio_segment.index,

                    "start_sample":
                        audio_segment.start_sample,

                    "end_sample":
                        audio_segment.end_sample,

                    "start_seconds":
                        (
                            audio_segment.start_sample
                            / request.config.sample_rate
                        ),

                    "end_seconds":
                        (
                            audio_segment.end_sample
                            / request.config.sample_rate
                        ),

                    "source_duration_seconds":
                        source_duration_seconds,

                    "is_padded":
                        audio_segment.is_padded,

                    "sample_rate":
                        request.config.sample_rate,

                    "segment_samples":
                        audio_segment.waveform.size,
                }
            )

        # -----------------------------------------------------------
        # Success report.
        # -----------------------------------------------------------

        return SourceProcessingResult(
            segment_rows=tuple(segment_rows),

            report_row={
                **report_base,

                "source_duration_seconds":
                    source_duration_seconds,

                "status":
                    "success",

                "segment_count":
                    len(segment_rows),

                "error":
                    "",
            },
        )

    except Exception as error:

        LOGGER.error(
            "Failed preprocessing %s: %s",
            row["relative_path"],
            error,
        )

        return SourceProcessingResult(
            segment_rows=(),

            report_row={
                **report_base,

                "status":
                    "failed",

                "segment_count":
                    0,

                "error":
                    str(error),
            },
        )


# ---------------------------------------------------------------------------
# Atomic CSV writing
# ---------------------------------------------------------------------------


def _write_csv_atomically(
    destination: Path,
    field_names: Sequence[str],
    rows: Iterable[
        Mapping[str, str | int | float | bool]
    ],
) -> None:
    """
    Write a CSV using a temporary file followed by atomic replacement.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{destination.stem}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:

            temporary_path = Path(
                temporary_file.name
            )

            writer = csv.DictWriter(
                temporary_file,
                fieldnames=field_names,
            )

            writer.writeheader()

            writer.writerows(rows)

        temporary_path.replace(
            destination
        )

    finally:

        if (
            temporary_path is not None
            and temporary_path.exists()
        ):

            try:
                temporary_path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Split preprocessing
# ---------------------------------------------------------------------------


def preprocess_split_manifest(
    split_name: str,
    source_manifest_path: Path,
    raw_datasets_directory: Path = RAW_DATASET_DIR,
    processed_datasets_directory: Path = PROCESSED_DATASET_DIR,
    config: AudioPreprocessingConfig = AudioPreprocessingConfig(),
    workers: int = 1,
    overwrite: bool = False,
) -> SplitPreprocessingSummary:
    """
    Preprocess all recordings in one split manifest.
    """

    if split_name not in SPLIT_NAMES:

        raise AudioPreprocessingError(
            f"Unsupported split name '{split_name}'."
        )

    if workers <= 0:

        raise AudioPreprocessingError(
            "workers must be greater than zero."
        )

    source_rows = _read_source_manifest(
        source_manifest_path
    )

    # ---------------------------------------------------------------
    # Verify every row belongs to this split.
    # ---------------------------------------------------------------

    invalid_split_rows = [
        row
        for row in source_rows
        if row["split"] != split_name
    ]

    if invalid_split_rows:

        raise AudioPreprocessingError(
            f"Manifest '{source_manifest_path}' contains "
            f"rows outside split '{split_name}'."
        )

    # ---------------------------------------------------------------
    # Create processing requests.
    # ---------------------------------------------------------------

    requests = [
        SourceProcessingRequest(
            row=row,
            split_name=split_name,
            raw_datasets_directory=raw_datasets_directory,
            processed_datasets_directory=processed_datasets_directory,
            config=config,
            overwrite=overwrite,
        )
        for row in source_rows
    ]

    # ---------------------------------------------------------------
    # Process recordings.
    #
    # executor.map preserves input order, which keeps reports
    # deterministic.
    # ---------------------------------------------------------------

    if workers == 1:

        results = [
            _process_source_recording(request)
            for request in requests
        ]

    else:

        with ThreadPoolExecutor(
            max_workers=workers
        ) as executor:

            results = list(
                executor.map(
                    _process_source_recording,
                    requests,
                )
            )

    # ---------------------------------------------------------------
    # Collect segment rows.
    # ---------------------------------------------------------------

    segment_rows = [
        row
        for result in results
        for row in result.segment_rows
    ]

    # Deterministic ordering.
    segment_rows.sort(
        key=lambda row: str(
            row["segment_id"]
        )
    )

    # ---------------------------------------------------------------
    # Collect reports.
    # ---------------------------------------------------------------

    report_rows = [
        result.report_row
        for result in results
    ]

    # ---------------------------------------------------------------
    # Output paths.
    # ---------------------------------------------------------------

    segment_manifest_path = (
        processed_datasets_directory
        / SOURCE_MANIFEST_DIRECTORY_NAME
        / f"{split_name}_segments.csv"
    )

    report_path = (
        processed_datasets_directory
        / REPORT_DIRECTORY_NAME
        / f"{split_name}_preprocessing.csv"
    )

    # ---------------------------------------------------------------
    # Write outputs atomically.
    # ---------------------------------------------------------------

    _write_csv_atomically(
        segment_manifest_path,
        SEGMENT_MANIFEST_COLUMNS,
        segment_rows,
    )

    _write_csv_atomically(
        report_path,
        REPORT_COLUMNS,
        report_rows,
    )

    # ---------------------------------------------------------------
    # Calculate summary.
    # ---------------------------------------------------------------

    successful_recordings = sum(
        result.report_row["status"]
        == "success"
        for result in results
    )

    failed_recordings = (
        len(source_rows)
        - successful_recordings
    )

    summary = SplitPreprocessingSummary(
        split_name=split_name,

        source_manifest_path=
            source_manifest_path,

        segment_manifest_path=
            segment_manifest_path,

        report_path=
            report_path,

        source_recordings=
            len(source_rows),

        successful_recordings=
            successful_recordings,

        failed_recordings=
            failed_recordings,

        written_segments=
            len(segment_rows),
    )

    LOGGER.info(
        "Preprocessed split '%s': %s",
        split_name,
        summary,
    )

    return summary


# ---------------------------------------------------------------------------
# All-split preprocessing
# ---------------------------------------------------------------------------


def preprocess_all_splits(
    source_manifest_directory: Path =
        PROCESSED_DATASET_DIR
        / SOURCE_MANIFEST_DIRECTORY_NAME,

    raw_datasets_directory: Path =
        RAW_DATASET_DIR,

    processed_datasets_directory: Path =
        PROCESSED_DATASET_DIR,

    config: AudioPreprocessingConfig =
        AudioPreprocessingConfig(),

    workers: int = 1,

    overwrite: bool = False,

    split_names: Sequence[str] =
        SPLIT_NAMES,
) -> tuple[SplitPreprocessingSummary, ...]:
    """
    Preprocess one or more train/validation/test manifests.
    """

    summaries: list[
        SplitPreprocessingSummary
    ] = []

    for split_name in split_names:

        summary = preprocess_split_manifest(
            split_name=split_name,

            source_manifest_path=(
                source_manifest_directory
                / f"{split_name}.csv"
            ),

            raw_datasets_directory=
                raw_datasets_directory,

            processed_datasets_directory=
                processed_datasets_directory,

            config=config,

            workers=workers,

            overwrite=overwrite,
        )

        summaries.append(summary)

    return tuple(summaries)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLIT_NAMES,
        default=SPLIT_NAMES,
        help=(
            "One or more split manifests to preprocess."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of worker threads. "
            "Default: 1."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace already-written segment WAV files."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run manifest-driven training-audio preprocessing."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    arguments = _parse_arguments()

    summaries = preprocess_all_splits(
        workers=arguments.workers,
        overwrite=arguments.overwrite,
        split_names=arguments.splits,
    )

    # ---------------------------------------------------------------
    # Print final summary.
    # ---------------------------------------------------------------

    LOGGER.info(
        "Training audio preprocessing completed."
    )

    for summary in summaries:

        LOGGER.info(
            "%s | sources=%d | successful=%d | "
            "failed=%d | segments=%d",
            summary.split_name,
            summary.source_recordings,
            summary.successful_recordings,
            summary.failed_recordings,
            summary.written_segments,
        )


if __name__ == "__main__":
    main()