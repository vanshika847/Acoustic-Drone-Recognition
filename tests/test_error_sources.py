"""Trace test-set false positives and false negatives back to source recordings.

This script does not modify the model or training data.

It reads:
    outputs/error_analysis/test_predictions.csv
    outputs/error_analysis/false_positives.csv
    outputs/error_analysis/false_negatives.csv

and combines them with:
    outputs/features/test_shard_manifest.csv

The goal is to determine whether model errors are concentrated in
particular recordings, shards, or source groups.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "features"
    / "test_shard_manifest.csv"
)

ERROR_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "error_analysis"
)

PREDICTIONS_FILE = (
    ERROR_DIR
    / "test_predictions.csv"
)

FALSE_POSITIVES_FILE = (
    ERROR_DIR
    / "false_positives.csv"
)

FALSE_NEGATIVES_FILE = (
    ERROR_DIR
    / "false_negatives.csv"
)

OUTPUT_DIR = (
    ERROR_DIR
    / "source_analysis"
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# REQUIRED MANIFEST COLUMNS
# ============================================================

REQUIRED_MANIFEST_COLUMNS = {
    "split",
    "segment_id",
    "binary_label",
    "shard_path",
    "shard_index",
}


# ============================================================
# HELPERS
# ============================================================

def check_file(path: Path) -> None:
    """Ensure a required file exists."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file."""

    check_file(path)

    dataframe = pd.read_csv(
        path,
        keep_default_na=False,
    )

    if dataframe.empty:
        raise ValueError(
            f"CSV is empty:\n{path}"
        )

    return dataframe


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    """
    Find the first matching column from a list of candidates.

    This makes the analysis tolerant of minor column-name
    differences between versions of test_error_analysis.py.
    """

    for candidate in candidates:

        if candidate in dataframe.columns:
            return candidate

    raise ValueError(
        f"Could not find {description} column.\n"
        f"Expected one of: {candidates}\n"
        f"Available columns: {list(dataframe.columns)}"
    )


def extract_recording_id(
    segment_id: str,
) -> str:
    """
    Extract the recording/source identifier from segment_id.

    Expected format:

        recording_hash:segment_number

    Example:

        abcdef123:50

    becomes:

        abcdef123
    """

    value = str(segment_id)

    if ":" in value:
        return value.rsplit(":", 1)[0]

    return value


def extract_segment_number(
    segment_id: str,
) -> str:
    """
    Extract the segment number from segment_id.

    Example:

        abcdef123:50

    becomes:

        50
    """

    value = str(segment_id)

    if ":" in value:
        return value.rsplit(":", 1)[1]

    return ""


def add_derived_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Add recording and segment identifiers."""

    dataframe = dataframe.copy()

    segment_column = find_column(
        dataframe,
        [
            "segment_id",
            "segment",
            "id",
        ],
        "segment ID",
    )

    dataframe["analysis_segment_id"] = (
        dataframe[segment_column]
        .astype(str)
    )

    dataframe["recording_id"] = (
        dataframe["analysis_segment_id"]
        .map(extract_recording_id)
    )

    dataframe["segment_number"] = (
        dataframe["analysis_segment_id"]
        .map(extract_segment_number)
    )

    return dataframe


def print_separator(
    character: str = "-",
    length: int = 70,
) -> None:
    """Print a formatted separator."""

    print(character * length)


# ============================================================
# MANIFEST LOADING
# ============================================================

def load_test_manifest() -> pd.DataFrame:
    """Load and validate the test shard manifest."""

    logger.info(
        "Loading test manifest:\n%s",
        TEST_MANIFEST,
    )

    manifest = load_csv(
        TEST_MANIFEST
    )

    missing = (
        REQUIRED_MANIFEST_COLUMNS
        .difference(manifest.columns)
    )

    if missing:
        raise ValueError(
            "Test manifest is missing required "
            "columns: "
            + ", ".join(sorted(missing))
        )

    manifest["binary_label"] = (
        pd.to_numeric(
            manifest["binary_label"],
            errors="raise",
        )
        .astype(int)
    )

    manifest["shard_index"] = (
        pd.to_numeric(
            manifest["shard_index"],
            errors="raise",
        )
        .astype(int)
    )

    return manifest


# ============================================================
# ERROR FILE LOADING
# ============================================================

def load_error_files() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load predictions, false positives and false negatives."""

    logger.info(
        "Loading prediction files..."
    )

    predictions = load_csv(
        PREDICTIONS_FILE
    )

    false_positives = load_csv(
        FALSE_POSITIVES_FILE
    )

    false_negatives = load_csv(
        FALSE_NEGATIVES_FILE
    )

    predictions = add_derived_columns(
        predictions
    )

    false_positives = add_derived_columns(
        false_positives
    )

    false_negatives = add_derived_columns(
        false_negatives
    )

    return (
        predictions,
        false_positives,
        false_negatives,
    )


# ============================================================
# MANIFEST JOIN
# ============================================================

def attach_manifest_information(
    errors: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach manifest information to an error dataframe.

    The join is performed using segment_id.
    """

    error_segment_column = find_column(
        errors,
        [
            "analysis_segment_id",
        ],
        "error segment ID",
    )

    manifest_segment_column = find_column(
        manifest,
        [
            "segment_id",
        ],
        "manifest segment ID",
    )

    manifest_copy = manifest.copy()

    manifest_copy[
        "analysis_segment_id"
    ] = (
        manifest_copy[
            manifest_segment_column
        ]
        .astype(str)
    )

    manifest_columns = [
        "analysis_segment_id",
        "split",
        "binary_label",
        "shard_path",
        "shard_index",
    ]

    optional_columns = [
        "source",
        "dataset",
        "dataset_name",
        "recording_id",
        "file",
        "file_path",
        "audio_path",
        "original_file",
        "original_path",
    ]

    for column in optional_columns:

        if column in manifest_copy.columns:
            manifest_columns.append(
                column
            )

    manifest_subset = (
        manifest_copy[
            manifest_columns
        ]
        .drop_duplicates(
            subset=["analysis_segment_id"]
        )
    )

    merged = errors.merge(
        manifest_subset,
        on="analysis_segment_id",
        how="left",
        suffixes=(
            "",
            "_manifest",
        ),
    )

    return merged


# ============================================================
# ERROR CONCENTRATION
# ============================================================

def calculate_recording_summary(
    errors: pd.DataFrame,
    error_type: str,
) -> pd.DataFrame:
    """
    Group errors by recording.

    This is one of the most important analyses because multiple
    consecutive segments from the same recording should not be
    interpreted as completely independent failure cases.
    """

    if errors.empty:
        return pd.DataFrame()

    grouped = (
        errors
        .groupby(
            "recording_id",
            dropna=False,
        )
        .agg(
            error_count=(
                "analysis_segment_id",
                "count",
            ),
            first_segment=(
                "segment_number",
                "min",
            ),
            last_segment=(
                "segment_number",
                "max",
            ),
            shard_count=(
                "shard_path",
                "nunique",
            ),
        )
        .reset_index()
    )

    grouped["error_type"] = error_type

    grouped = grouped.sort_values(
        [
            "error_count",
            "recording_id",
        ],
        ascending=[
            False,
            True,
        ],
    )

    return grouped


def calculate_combined_recording_summary(
    false_positives: pd.DataFrame,
    false_negatives: pd.DataFrame,
) -> pd.DataFrame:
    """Create a combined FP/FN recording summary."""

    frames = []

    if not false_positives.empty:

        fp = (
            false_positives
            .groupby(
                "recording_id",
                dropna=False,
            )
            .size()
            .rename("false_positives")
            .reset_index()
        )

        frames.append(fp)

    if not false_negatives.empty:

        fn = (
            false_negatives
            .groupby(
                "recording_id",
                dropna=False,
            )
            .size()
            .rename("false_negatives")
            .reset_index()
        )

        frames.append(fn)

    if not frames:
        return pd.DataFrame()

    combined = frames[0]

    for frame in frames[1:]:

        combined = combined.merge(
            frame,
            on="recording_id",
            how="outer",
        )

    for column in [
        "false_positives",
        "false_negatives",
    ]:

        if column not in combined.columns:
            combined[column] = 0

    combined[
        [
            "false_positives",
            "false_negatives",
        ]
    ] = combined[
        [
            "false_positives",
            "false_negatives",
        ]
    ].fillna(0).astype(int)

    combined["total_errors"] = (
        combined["false_positives"]
        + combined["false_negatives"]
    )

    combined = combined.sort_values(
        [
            "total_errors",
            "false_positives",
            "false_negatives",
        ],
        ascending=False,
    )

    return combined


# ============================================================
# SHARD ANALYSIS
# ============================================================

def calculate_shard_summary(
    false_positives: pd.DataFrame,
    false_negatives: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate errors grouped by shard."""

    frames = []

    if not false_positives.empty:

        fp = (
            false_positives
            .groupby(
                "shard_path",
                dropna=False,
            )
            .size()
            .rename("false_positives")
            .reset_index()
        )

        frames.append(fp)

    if not false_negatives.empty:

        fn = (
            false_negatives
            .groupby(
                "shard_path",
                dropna=False,
            )
            .size()
            .rename("false_negatives")
            .reset_index()
        )

        frames.append(fn)

    if not frames:
        return pd.DataFrame()

    combined = frames[0]

    for frame in frames[1:]:

        combined = combined.merge(
            frame,
            on="shard_path",
            how="outer",
        )

    for column in [
        "false_positives",
        "false_negatives",
    ]:

        if column not in combined.columns:
            combined[column] = 0

    combined[
        [
            "false_positives",
            "false_negatives",
        ]
    ] = combined[
        [
            "false_positives",
            "false_negatives",
        ]
    ].fillna(0).astype(int)

    combined["total_errors"] = (
        combined["false_positives"]
        + combined["false_negatives"]
    )

    combined = combined.sort_values(
        "total_errors",
        ascending=False,
    )

    return combined


# ============================================================
# SOURCE / DATASET ANALYSIS
# ============================================================

def find_source_column(
    dataframe: pd.DataFrame,
) -> str | None:
    """
    Find a likely source/dataset column.

    Returns None if the manifest does not contain one.
    """

    candidates = [
        "source",
        "dataset",
        "dataset_name",
        "source_dataset",
        "collection",
        "corpus",
    ]

    for column in candidates:

        if column in dataframe.columns:
            return column

    return None


def calculate_source_summary(
    false_positives: pd.DataFrame,
    false_negatives: pd.DataFrame,
    source_column: str,
) -> pd.DataFrame:
    """Calculate errors grouped by source/dataset."""

    frames = []

    if not false_positives.empty:

        fp = (
            false_positives
            .groupby(
                source_column,
                dropna=False,
            )
            .size()
            .rename("false_positives")
            .reset_index()
        )

        frames.append(fp)

    if not false_negatives.empty:

        fn = (
            false_negatives
            .groupby(
                source_column,
                dropna=False,
            )
            .size()
            .rename("false_negatives")
            .reset_index()
        )

        frames.append(fn)

    if not frames:
        return pd.DataFrame()

    combined = frames[0]

    for frame in frames[1:]:

        combined = combined.merge(
            frame,
            on=source_column,
            how="outer",
        )

    for column in [
        "false_positives",
        "false_negatives",
    ]:

        if column not in combined.columns:
            combined[column] = 0

    combined[
        [
            "false_positives",
            "false_negatives",
        ]
    ] = combined[
        [
            "false_positives",
            "false_negatives",
        ]
    ].fillna(0).astype(int)

    combined["total_errors"] = (
        combined["false_positives"]
        + combined["false_negatives"]
    )

    return combined.sort_values(
        "total_errors",
        ascending=False,
    )


# ============================================================
# PRINT ERROR RECORDINGS
# ============================================================

def print_top_recordings(
    summary: pd.DataFrame,
    title: str,
    limit: int = 20,
) -> None:
    """Print the recordings responsible for the most errors."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    if summary.empty:

        print("No errors found.")

        return

    display = summary.head(
        limit
    )

    print(
        display.to_string(
            index=False
        )
    )


# ============================================================
# PRINT INDIVIDUAL SEGMENTS
# ============================================================

def print_top_segments(
    dataframe: pd.DataFrame,
    title: str,
    probability_column: str,
    ascending: bool,
    limit: int = 20,
) -> None:
    """Print individual error segments."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    if dataframe.empty:

        print("No samples found.")

        return

    columns = [
        "analysis_segment_id",
        "recording_id",
        "segment_number",
        "shard_path",
    ]

    if probability_column in dataframe.columns:
        columns.append(
            probability_column
        )

    existing_columns = [
        column
        for column in columns
        if column in dataframe.columns
    ]

    display = (
        dataframe
        .sort_values(
            probability_column,
            ascending=ascending,
        )
        .head(limit)
    )

    print(
        display[
            existing_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# CONCENTRATION STATISTICS
# ============================================================

def print_concentration_statistics(
    errors: pd.DataFrame,
    error_name: str,
) -> None:
    """
    Report how concentrated errors are among recordings.

    This helps distinguish:
        75 errors across 75 recordings

    from:
        75 errors concentrated in 3 recordings.
    """

    print()
    print("-" * 70)
    print(
        f"{error_name.upper()} CONCENTRATION"
    )
    print("-" * 70)

    if errors.empty:

        print("No errors.")

        return

    recording_counts = (
        errors
        .groupby(
            "recording_id"
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    total_errors = len(errors)

    unique_recordings = (
        recording_counts.shape[0]
    )

    print(
        f"Total error segments: {total_errors}"
    )

    print(
        f"Unique recordings containing errors: "
        f"{unique_recordings}"
    )

    if unique_recordings > 0:

        top_1 = recording_counts.head(1).sum()
        top_3 = recording_counts.head(3).sum()
        top_5 = recording_counts.head(5).sum()
        top_10 = recording_counts.head(10).sum()

        print(
            f"Top 1 recording:  "
            f"{int(top_1)} errors "
            f"({100 * top_1 / total_errors:.2f}%)"
        )

        print(
            f"Top 3 recordings: "
            f"{int(top_3)} errors "
            f"({100 * top_3 / total_errors:.2f}%)"
        )

        print(
            f"Top 5 recordings: "
            f"{int(top_5)} errors "
            f"({100 * top_5 / total_errors:.2f}%)"
        )

        print(
            f"Top 10 recordings:"
            f" {int(top_10)} errors "
            f"({100 * top_10 / total_errors:.2f}%)"
        )


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_dataframe(
    dataframe: pd.DataFrame,
    filename: str,
) -> Path:
    """Save a dataframe into the source-analysis directory."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        OUTPUT_DIR
        / filename
    )

    dataframe.to_csv(
        path,
        index=False,
    )

    logger.info(
        "Saved: %s",
        path,
    )

    return path


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run complete source-level error analysis."""

    print()
    print("=" * 70)
    print(
        "ACOUSTIC DRONE TEST-SET SOURCE ERROR ANALYSIS"
    )
    print("=" * 70)

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Test manifest: {TEST_MANIFEST}"
    )

    print(
        f"Error analysis: {ERROR_DIR}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CHECKING INPUT FILES")
    print("-" * 70)

    check_file(
        TEST_MANIFEST
    )

    check_file(
        PREDICTIONS_FILE
    )

    check_file(
        FALSE_POSITIVES_FILE
    )

    check_file(
        FALSE_NEGATIVES_FILE
    )

    print(
        "All required files found."
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("LOADING DATA")
    print("-" * 70)

    manifest = load_test_manifest()

    (
        predictions,
        false_positives,
        false_negatives,
    ) = load_error_files()

    print(
        f"Test manifest samples: "
        f"{len(manifest)}"
    )

    print(
        f"Prediction samples: "
        f"{len(predictions)}"
    )

    print(
        f"False positives: "
        f"{len(false_positives)}"
    )

    print(
        f"False negatives: "
        f"{len(false_negatives)}"
    )

    # --------------------------------------------------------
    # Attach manifest information
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("ATTACHING MANIFEST INFORMATION")
    print("-" * 70)

    false_positives = attach_manifest_information(
        false_positives,
        manifest,
    )

    false_negatives = attach_manifest_information(
        false_negatives,
        manifest,
    )

    predictions = attach_manifest_information(
        predictions,
        manifest,
    )

    # --------------------------------------------------------
    # Check unmatched segments
    # --------------------------------------------------------

    fp_unmatched = (
        false_positives[
            "shard_path"
        ]
        .isna()
        .sum()
    )

    fn_unmatched = (
        false_negatives[
            "shard_path"
        ]
        .isna()
        .sum()
    )

    print(
        f"False positives matched to manifest: "
        f"{len(false_positives) - fp_unmatched}/"
        f"{len(false_positives)}"
    )

    print(
        f"False negatives matched to manifest: "
        f"{len(false_negatives) - fn_unmatched}/"
        f"{len(false_negatives)}"
    )

    if fp_unmatched > 0:

        logger.warning(
            "%d false positives could not be "
            "matched to the test manifest.",
            fp_unmatched,
        )

    if fn_unmatched > 0:

        logger.warning(
            "%d false negatives could not be "
            "matched to the test manifest.",
            fn_unmatched,
        )

    # --------------------------------------------------------
    # Recording summaries
    # --------------------------------------------------------

    fp_recordings = calculate_recording_summary(
        false_positives,
        "false_positive",
    )

    fn_recordings = calculate_recording_summary(
        false_negatives,
        "false_negative",
    )

    combined_recordings = (
        calculate_combined_recording_summary(
            false_positives,
            false_negatives,
        )
    )

    # --------------------------------------------------------
    # Print concentration
    # --------------------------------------------------------

    print_concentration_statistics(
        false_positives,
        "False-positive",
    )

    print_concentration_statistics(
        false_negatives,
        "False-negative",
    )

    # --------------------------------------------------------
    # Print top recordings
    # --------------------------------------------------------

    print_top_recordings(
        fp_recordings,
        "TOP RECORDINGS BY FALSE POSITIVES",
        limit=20,
    )

    print_top_recordings(
        fn_recordings,
        "TOP RECORDINGS BY FALSE NEGATIVES",
        limit=20,
    )

    print_top_recordings(
        combined_recordings,
        "TOP RECORDINGS BY TOTAL ERRORS",
        limit=30,
    )

    # --------------------------------------------------------
    # Shard analysis
    # --------------------------------------------------------

    shard_summary = calculate_shard_summary(
        false_positives,
        false_negatives,
    )

    print()
    print("=" * 70)
    print("ERRORS BY SHARD")
    print("=" * 70)

    if shard_summary.empty:

        print(
            "No shard-level errors found."
        )

    else:

        print(
            shard_summary.head(30).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Source / dataset analysis
    # --------------------------------------------------------

    source_column = find_source_column(
        manifest
    )

    if source_column is not None:

        source_summary = (
            calculate_source_summary(
                false_positives,
                false_negatives,
                source_column,
            )
        )

        print()
        print("=" * 70)
        print(
            f"ERRORS BY SOURCE/DATASET "
            f"({source_column})"
        )
        print("=" * 70)

        print(
            source_summary.to_string(
                index=False
            )
        )

        save_dataframe(
            source_summary,
            "errors_by_source.csv",
        )

    else:

        source_summary = pd.DataFrame()

        print()
        print("=" * 70)
        print("SOURCE/DATASET ANALYSIS")
        print("=" * 70)

        print(
            "No source/dataset column was found "
            "in the test manifest."
        )

        print(
            "This is important: the manifest currently "
            "does not directly expose dataset provenance."
        )

    # --------------------------------------------------------
    # Individual error segments
    # --------------------------------------------------------

    probability_column_fp = (
        "drone_probability"
    )

    probability_column_fn = (
        "drone_probability"
    )

    print_top_segments(
        false_positives,
        "HIGHEST-CONFIDENCE FALSE POSITIVES",
        probability_column_fp,
        ascending=False,
        limit=30,
    )

    print_top_segments(
        false_negatives,
        "MOST-CONFIDENT FALSE NEGATIVES",
        probability_column_fn,
        ascending=True,
        limit=30,
    )

    # --------------------------------------------------------
    # Save detailed outputs
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAVING SOURCE-LEVEL ANALYSIS")
    print("=" * 70)

    save_dataframe(
        false_positives,
        "false_positives_with_source.csv",
    )

    save_dataframe(
        false_negatives,
        "false_negatives_with_source.csv",
    )

    save_dataframe(
        fp_recordings,
        "false_positive_recordings.csv",
    )

    save_dataframe(
        fn_recordings,
        "false_negative_recordings.csv",
    )

    save_dataframe(
        combined_recordings,
        "combined_recording_errors.csv",
    )

    save_dataframe(
        shard_summary,
        "errors_by_shard.csv",
    )

    # --------------------------------------------------------
    # Save all predictions with manifest information
    # --------------------------------------------------------

    save_dataframe(
        predictions,
        "all_predictions_with_source.csv",
    )

    # --------------------------------------------------------
    # Final interpretation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SOURCE ERROR ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        "The most important files are:"
    )

    print(
        OUTPUT_DIR
        / "false_positive_recordings.csv"
    )

    print(
        OUTPUT_DIR
        / "false_negative_recordings.csv"
    )

    print(
        OUTPUT_DIR
        / "combined_recording_errors.csv"
    )

    print(
        OUTPUT_DIR
        / "errors_by_shard.csv"
    )

    if source_column is not None:

        print(
            OUTPUT_DIR
            / "errors_by_source.csv"
        )

    print()
    print(
        "Interpretation:"
    )

    print(
        "1. If many errors belong to a few recordings, "
        "we likely have recording-specific/domain-specific "
        "failures."
    )

    print(
        "2. If errors are spread across many recordings, "
        "the problem is more likely generalization."
    )

    print(
        "3. If errors concentrate in one dataset/source, "
        "we should investigate dataset/domain mismatch."
    )

    print(
        "4. If consecutive segments from the same recording "
        "produce many errors, those errors should be treated "
        "as a recording-level failure pattern rather than "
        "75 completely independent failures."
    )

    print()
    print(
        "No model parameters were modified."
    )

    print(
        "No training data was modified."
    )

    print(
        "The test set was used only for analysis."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Source error analysis interrupted."
        )

        raise

    except Exception as error:

        logger.exception(
            "Source error analysis failed."
        )

        raise