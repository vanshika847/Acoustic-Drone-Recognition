"""
Analyze recording concentration, repetition, and cross-split leakage.

This diagnostic script is READ-ONLY.

It does NOT modify:
    - manifests
    - shards
    - feature files
    - original .npy files

It writes diagnostic CSV files under:

    outputs/diagnostics/

Important:
    When manifests do not contain an explicit source/recording/path column,
    recording IDs are inferred from segment_id.

    Expected segment_id pattern:

        <recording_hash>:<segment_index>

    Example:

        0068c3bef61e3ffbc05801c0b8428930827fa8f84260f04f50278be9bf873399:0
        0068c3bef61e3ffbc05801c0b8428930827fa8f84260f04f50278be9bf873399:1
        0068c3bef61e3ffbc05801c0b8428930827fa8f84260f04f50278be9bf873399:2

    These are inferred to belong to recording:

        0068c3bef61e3ffbc05801c0b8428930827fa8f84260f04f50278be9bf873399

    This inference is heuristic and must be inspected before using the
    resulting IDs to modify the dataset split.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import pandas as pd


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "features"
)

DIAGNOSTIC_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
)

SPLITS = (
    "train",
    "validation",
    "test",
)

# Candidate columns that may explicitly identify the original recording.
#
# We intentionally prefer explicit source/path information over inference.
SOURCE_COLUMN_CANDIDATES = (
    "recording_id",
    "recording",
    "recording_name",
    "recording_id",
    "source_id",
    "source",
    "source_file",
    "source_path",
    "audio_path",
    "audio_file",
    "file_path",
    "filepath",
    "file",
    "filename",
    "path",
    "original_path",
    "original_file",
    "input_path",
    "input_file",
)

SEGMENT_COLUMN_CANDIDATES = (
    "segment_id",
    "segment_ids",
    "segment",
    "segment_index",
    "segment_idx",
)

LABEL_COLUMN_CANDIDATES = (
    "binary_label",
    "label",
    "target",
    "targets",
    "class",
    "y",
)


# ============================================================================
# GENERAL HELPERS
# ============================================================================


def find_column(
    columns: list[str],
    candidates: tuple[str, ...],
) -> str | None:
    """
    Find the first matching column using case-insensitive matching.
    """

    normalized = {
        str(column).strip().lower(): column
        for column in columns
    }

    for candidate in candidates:
        match = normalized.get(candidate.lower())

        if match is not None:
            return match

    return None


def format_count(value: int | float) -> str:
    """Format an integer-like count with thousands separators."""

    return f"{int(value):,}"


def ensure_output_directory() -> None:
    """Create the diagnostics directory if necessary."""

    DIAGNOSTIC_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# MANIFEST LOADING
# ============================================================================


def load_manifests() -> pd.DataFrame:
    """
    Load all split manifests.

    Returns
    -------
    pandas.DataFrame
        Combined manifest with a guaranteed 'split' column.
    """

    frames: list[pd.DataFrame] = []

    print("=" * 80)
    print("RECORDING CONCENTRATION / REPETITION ANALYSIS")
    print("=" * 80)

    for split in SPLITS:

        manifest_path = (
            FEATURE_OUTPUT_DIR
            / f"{split}_shard_manifest.csv"
        )

        if not manifest_path.is_file():

            raise FileNotFoundError(
                f"Manifest not found for split '{split}':\n"
                f"{manifest_path}"
            )

        frame = pd.read_csv(
            manifest_path,
            dtype=str,
            keep_default_na=False,
        )

        frame["split"] = split

        frames.append(frame)

        print(
            f"{split:<12} "
            f"{len(frame):,} rows"
        )

    if not frames:

        raise RuntimeError(
            "No manifests were loaded."
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    return combined


# ============================================================================
# COLUMN DETECTION
# ============================================================================


def detect_columns(
    manifest: pd.DataFrame,
) -> tuple[
    str | None,
    str | None,
    str | None,
]:
    """
    Detect source, segment, and label columns.
    """

    columns = list(manifest.columns)

    source_column = find_column(
        columns,
        SOURCE_COLUMN_CANDIDATES,
    )

    segment_column = find_column(
        columns,
        SEGMENT_COLUMN_CANDIDATES,
    )

    label_column = find_column(
        columns,
        LABEL_COLUMN_CANDIDATES,
    )

    print()
    print("Columns:")

    for column in columns:
        print(f"   {column}")

    print()
    print("Detected columns:")
    print(
        f"  source column : {source_column}"
    )
    print(
        f"  segment column: {segment_column}"
    )
    print(
        f"  label column  : {label_column}"
    )

    if segment_column is None:

        raise ValueError(
            "Could not find a segment identifier column."
        )

    if label_column is None:

        raise ValueError(
            "Could not find a label column."
        )

    return (
        source_column,
        segment_column,
        label_column,
    )


# ============================================================================
# RECORDING-ID INFERENCE
# ============================================================================


def infer_recording_id_from_segment(
    segment_id: object,
) -> tuple[str, str]:
    """
    Infer a recording/group identifier from segment_id.

    Expected form:

        <recording_id>:<segment_index>

    Example:

        abcdef123:0
        abcdef123:1
        abcdef123:2

    becomes:

        recording_id = abcdef123

    Returns
    -------
    tuple[str, str]
        inferred recording ID and inference method.
    """

    value = str(segment_id).strip()

    if not value:

        return (
            "",
            "empty_segment_id",
        )

    # The dataset's observed IDs use a final ":<integer>" suffix.
    #
    # Use rsplit rather than split so that only the final suffix is removed.
    #
    # Example:
    #
    #     abc:def:0
    #
    # becomes:
    #
    #     abc:def
    #
    # only when the suffix is actually numeric.
    match = re.match(
        r"^(.*):(\d+)$",
        value,
    )

    if match:

        recording_id = match.group(1).strip()

        if recording_id:

            return (
                recording_id,
                "segment_id_hash_without_index",
            )

    # If the segment ID does not follow the expected pattern, preserve it
    # rather than inventing a grouping rule.
    return (
        value,
        "full_segment_id_fallback",
    )


def build_recording_identifiers(
    manifest: pd.DataFrame,
    source_column: str | None,
    segment_column: str,
) -> tuple[pd.DataFrame, str]:
    """
    Build recording identifiers.

    Explicit source/path columns are preferred.

    If no explicit source column exists, infer recording IDs from segment_id
    by removing the final ':<integer>' segment index.

    Returns
    -------
    tuple[pandas.DataFrame, str]
        Enriched manifest and overall inference description.
    """

    result = manifest.copy()

    if source_column is not None:

        values = (
            result[source_column]
            .astype(str)
            .str.strip()
        )

        invalid = (
            values.eq("")
            | values.str.lower().isin(
                {
                    "nan",
                    "none",
                    "null",
                }
            )
        )

        if invalid.any():

            raise ValueError(
                f"Source column '{source_column}' contains "
                f"{int(invalid.sum())} empty/invalid values."
            )

        result["inferred_recording_id"] = values

        result["inference_method"] = (
            "explicit_source_column"
        )

        return (
            result,
            "explicit_source_column",
        )

    # ------------------------------------------------------------------
    # No explicit recording/source/path column.
    # ------------------------------------------------------------------

    print()
    print(
        "No explicit recording/source/path column exists "
        "in the manifests."
    )

    print(
        "Falling back to inference from segment_id."
    )

    print()
    print("IMPORTANT:")
    print(
        "  The inferred recording IDs are heuristic."
    )

    print(
        "  The final ':<segment_index>' suffix is removed "
        "when it matches the expected pattern."
    )

    inferred: list[str] = []
    methods: list[str] = []

    for value in result[segment_column]:

        recording_id, method = (
            infer_recording_id_from_segment(
                value
            )
        )

        inferred.append(
            recording_id
        )

        methods.append(
            method
        )

    result["inferred_recording_id"] = inferred

    result["inference_method"] = methods

    return (
        result,
        "segment_id_inference",
    )


# ============================================================================
# EXACT SEGMENT DUPLICATES
# ============================================================================


def analyze_exact_segment_duplicates(
    manifest: pd.DataFrame,
    segment_column: str,
) -> pd.DataFrame:
    """
    Find exact duplicate segment IDs across the combined manifests.

    A duplicate is reported when the same segment ID occurs more than once.
    """

    print()
    print("=" * 80)
    print("EXACT SEGMENT-ID DUPLICATES")
    print("=" * 80)

    counts = (
        manifest[segment_column]
        .astype(str)
        .value_counts()
        .rename("occurrences")
        .reset_index()
    )

    counts = counts.rename(
        columns={
            "index": segment_column,
        }
    )

    duplicate_column = counts.columns[0]

    duplicates = counts[
        counts["occurrences"] > 1
    ].copy()

    duplicates = duplicates.sort_values(
        by=[
            "occurrences",
            duplicate_column,
        ],
        ascending=[
            False,
            True,
        ],
    )

    if duplicates.empty:

        print(
            "No duplicated segment IDs found."
        )

    else:

        print(
            f"Found "
            f"{len(duplicates):,} "
            f"duplicated segment IDs."
        )

        print()

        print(
            duplicates.head(20).to_string(
                index=False
            )
        )

    output_path = (
        DIAGNOSTIC_OUTPUT_DIR
        / "duplicate_segment_ids.csv"
    )

    duplicates.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Wrote: {output_path}"
    )

    return duplicates


# ============================================================================
# INFERENCE QUALITY
# ============================================================================


def analyze_recording_inference_quality(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize how recording IDs were inferred.
    """

    print()
    print("=" * 80)
    print("RECORDING-ID INFERENCE QUALITY")
    print("=" * 80)

    rows: list[dict[str, object]] = []

    for split in SPLITS:

        subset = manifest[
            manifest["split"] == split
        ]

        total = len(subset)

        if total == 0:
            continue

        method_counts = (
            subset["inference_method"]
            .value_counts()
        )

        for method, count in method_counts.items():

            rows.append(
                {
                    "split": split,
                    "recording_inference_method": method,
                    "rows": int(count),
                    "split_total": int(total),
                    "fraction": (
                        float(count) / float(total)
                    ),
                }
            )

    result = pd.DataFrame(
        rows,
        columns=[
            "split",
            "recording_inference_method",
            "rows",
            "split_total",
            "fraction",
        ],
    )

    if not result.empty:

        print(
            result.to_string(
                index=False
            )
        )

    output_path = (
        DIAGNOSTIC_OUTPUT_DIR
        / "recording_identifier_quality.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Wrote: {output_path}"
    )

    return result


# ============================================================================
# RECORDING CONCENTRATION
# ============================================================================


def calculate_recording_concentration(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate recording-level sample concentration.

    Each inferred recording receives:

        samples
        unique_segments
        label_0
        label_1
        label_0_fraction
        label_1_fraction
        inference_method
    """

    rows: list[dict[str, object]] = []

    grouped = manifest.groupby(
        [
            "split",
            "inferred_recording_id",
        ],
        sort=True,
        dropna=False,
    )

    for (
        split,
        recording_id,
    ), group in grouped:

        labels = pd.to_numeric(
            group["__analysis_label"],
            errors="coerce",
        )

        if labels.isna().any():

            raise ValueError(
                f"Non-numeric label encountered for "
                f"recording '{recording_id}' "
                f"in split '{split}'."
            )

        label_0 = int(
            (labels == 0).sum()
        )

        label_1 = int(
            (labels == 1).sum()
        )

        samples = int(
            len(group)
        )

        unique_segments = int(
            group["__analysis_segment"]
            .astype(str)
            .nunique()
        )

        if samples == 0:

            label_0_fraction = 0.0
            label_1_fraction = 0.0

        else:

            label_0_fraction = (
                label_0 / samples
            )

            label_1_fraction = (
                label_1 / samples
            )

        methods = (
            group["inference_method"]
            .astype(str)
            .unique()
            .tolist()
        )

        if len(methods) == 1:

            inference_method = methods[0]

        else:

            inference_method = (
                "mixed:" + "|".join(
                    sorted(methods)
                )
            )

        rows.append(
            {
                "split": split,
                "inferred_recording_id": recording_id,
                "samples": samples,
                "unique_segments": unique_segments,
                "label_0": label_0,
                "label_1": label_1,
                "inference_method": inference_method,
                "label_0_fraction": label_0_fraction,
                "label_1_fraction": label_1_fraction,
            }
        )

    result = pd.DataFrame(
        rows,
        columns=[
            "split",
            "inferred_recording_id",
            "samples",
            "unique_segments",
            "label_0",
            "label_1",
            "inference_method",
            "label_0_fraction",
            "label_1_fraction",
        ],
    )

    return result


def print_recording_concentration(
    concentration: pd.DataFrame,
) -> None:
    """Print a readable concentration summary."""

    print()
    print("=" * 80)
    print("RECORDING CONCENTRATION")
    print("=" * 80)

    for split in SPLITS:

        subset = concentration[
            concentration["split"] == split
        ].copy()

        if subset.empty:
            continue

        subset = subset.sort_values(
            by=[
                "samples",
                "inferred_recording_id",
            ],
            ascending=[
                False,
                True,
            ],
        )

        total_samples = int(
            subset["samples"].sum()
        )

        recording_count = int(
            len(subset)
        )

        print()
        print(
            f"{split.upper()}: "
            f"{recording_count:,} inferred recordings"
        )

        print(
            subset.head(20).to_string(
                index=False
            )
        )

        largest = int(
            subset["samples"].max()
        )

        top_5 = int(
            subset.head(5)["samples"].sum()
        )

        top_10 = int(
            subset.head(10)["samples"].sum()
        )

        largest_fraction = (
            largest / total_samples
            if total_samples
            else 0.0
        )

        top_5_fraction = (
            top_5 / total_samples
            if total_samples
            else 0.0
        )

        top_10_fraction = (
            top_10 / total_samples
            if total_samples
            else 0.0
        )

        print()
        print(
            f"  Largest recording: "
            f"{largest:,} samples "
            f"({largest_fraction:.2%})"
        )

        print(
            f"  Top 5 recordings: "
            f"{top_5_fraction:.2%} of split"
        )

        print(
            f"  Top 10 recordings: "
            f"{top_10_fraction:.2%} of split"
        )


# ============================================================================
# RECORDING LABEL CONCENTRATION
# ============================================================================


def calculate_label_concentration(
    concentration: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate label concentration at recording level.

    Correctly uses label_0 + label_1 as total_samples.

    This avoids the bug where one-hot columns were previously treated
    as if they were raw counts.
    """

    result = concentration.copy()

    result["0"] = result["label_0"].astype(int)
    result["1"] = result["label_1"].astype(int)

    result["total_samples"] = (
        result["label_0"].astype(int)
        + result["label_1"].astype(int)
    )

    def majority_label(row: pd.Series) -> int:
        if row["label_0"] >= row["label_1"]:
            return 0

        return 1

    result["majority_label"] = (
        result.apply(
            majority_label,
            axis=1,
        )
    )

    def majority_fraction(row: pd.Series) -> float:
        total = int(
            row["total_samples"]
        )

        if total == 0:
            return 0.0

        return (
            max(
                int(row["label_0"]),
                int(row["label_1"]),
            )
            / total
        )

    result["majority_fraction"] = (
        result.apply(
            majority_fraction,
            axis=1,
        )
    )

    def minority_fraction(row: pd.Series) -> float:
        total = int(
            row["total_samples"]
        )

        if total == 0:
            return 0.0

        return (
            min(
                int(row["label_0"]),
                int(row["label_1"]),
            )
            / total
        )

    result["minority_fraction"] = (
        result.apply(
            minority_fraction,
            axis=1,
        )
    )

    columns = [
        "split",
        "inferred_recording_id",
        "0",
        "1",
        "label_0",
        "label_1",
        "total_samples",
        "majority_label",
        "majority_fraction",
        "minority_fraction",
    ]

    return result[columns]


def print_label_concentration(
    label_concentration: pd.DataFrame,
) -> None:
    """Print label concentration by split."""

    print()
    print("=" * 80)
    print("LABEL CONCENTRATION BY RECORDING")
    print("=" * 80)

    for split in SPLITS:

        subset = label_concentration[
            label_concentration["split"] == split
        ].copy()

        if subset.empty:
            continue

        subset = subset.sort_values(
            by=[
                "total_samples",
                "inferred_recording_id",
            ],
            ascending=[
                False,
                True,
            ],
        )

        print()
        print(
            f"{split.upper()}:"
        )

        print(
            subset.head(20).to_string(
                index=False
            )
        )


# ============================================================================
# CROSS-SPLIT OVERLAP
# ============================================================================


def analyze_cross_split_overlap(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find inferred recording IDs occurring in multiple splits.

    This is the primary diagnostic for potential recording-level leakage.
    """

    print()
    print("=" * 80)
    print("CROSS-SPLIT RECORDING OVERLAP")
    print("=" * 80)

    grouped = (
        manifest.groupby(
            "inferred_recording_id",
            sort=True,
        )["split"]
        .agg(
            lambda values: sorted(
                set(values)
            )
        )
        .reset_index()
    )

    grouped["split_count"] = (
        grouped["split"].map(len)
    )

    overlap = grouped[
        grouped["split_count"] > 1
    ].copy()

    if overlap.empty:

        print(
            "No inferred recording IDs occur "
            "in multiple splits."
        )

    else:

        overlap["splits"] = (
            overlap["split"]
            .map(
                lambda values: ",".join(
                    values
                )
            )
        )

        overlap = overlap[
            [
                "inferred_recording_id",
                "splits",
                "split_count",
            ]
        ]

        print(
            f"WARNING: "
            f"{len(overlap):,} inferred recording IDs "
            f"occur in multiple splits."
        )

        print()

        print(
            overlap.head(100).to_string(
                index=False
            )
        )

    output_path = (
        DIAGNOSTIC_OUTPUT_DIR
        / "cross_split_recording_overlap.csv"
    )

    overlap.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Wrote: {output_path}"
    )

    return overlap


# ============================================================================
# ENRICHED MANIFEST
# ============================================================================


def build_enriched_manifest(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a diagnostic-only enriched manifest.
    """

    result = manifest.copy()

    # Internal analysis columns are removed from the final diagnostic file.
    result = result.drop(
        columns=[
            "__analysis_label",
            "__analysis_segment",
        ],
        errors="ignore",
    )

    return result


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Run all recording concentration diagnostics."""

    ensure_output_directory()

    # ------------------------------------------------------------------
    # Load manifests.
    # ------------------------------------------------------------------

    manifest = load_manifests()

    # ------------------------------------------------------------------
    # Detect columns.
    # ------------------------------------------------------------------

    (
        source_column,
        segment_column,
        label_column,
    ) = detect_columns(
        manifest
    )

    # ------------------------------------------------------------------
    # Build recording IDs.
    # ------------------------------------------------------------------

    (
        manifest,
        inference_description,
    ) = build_recording_identifiers(
        manifest,
        source_column,
        segment_column,
    )

    # ------------------------------------------------------------------
    # Prepare analysis columns.
    #
    # Keeping these separate prevents accidental modification of the
    # original manifest column names.
    # ------------------------------------------------------------------

    manifest["__analysis_segment"] = (
        manifest[segment_column]
        .astype(str)
        .str.strip()
    )

    manifest["__analysis_label"] = (
        manifest[label_column]
        .astype(str)
        .str.strip()
    )

    # ------------------------------------------------------------------
    # Validate labels.
    # ------------------------------------------------------------------

    numeric_labels = pd.to_numeric(
        manifest["__analysis_label"],
        errors="coerce",
    )

    if numeric_labels.isna().any():

        bad_count = int(
            numeric_labels.isna().sum()
        )

        raise ValueError(
            f"Found {bad_count:,} rows with "
            f"non-numeric labels in column "
            f"'{label_column}'."
        )

    invalid_labels = ~numeric_labels.isin(
        [0, 1]
    )

    if invalid_labels.any():

        invalid_values = sorted(
            set(
                numeric_labels[
                    invalid_labels
                ].tolist()
            )
        )

        raise ValueError(
            "Invalid binary labels found: "
            f"{invalid_values}. "
            "Expected only 0 and 1."
        )

    manifest["__analysis_label"] = (
        numeric_labels.astype(int)
    )

    # ------------------------------------------------------------------
    # Validate recording IDs.
    # ------------------------------------------------------------------

    empty_recording_ids = (
        manifest["inferred_recording_id"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    if empty_recording_ids.any():

        raise ValueError(
            f"Found "
            f"{int(empty_recording_ids.sum()):,} "
            f"rows with empty inferred recording IDs."
        )

    # ------------------------------------------------------------------
    # Exact segment duplicates.
    # ------------------------------------------------------------------

    analyze_exact_segment_duplicates(
        manifest,
        segment_column,
    )

    # ------------------------------------------------------------------
    # Inference quality.
    # ------------------------------------------------------------------

    analyze_recording_inference_quality(
        manifest
    )

    # ------------------------------------------------------------------
    # Recording concentration.
    # ------------------------------------------------------------------

    concentration = (
        calculate_recording_concentration(
            manifest
        )
    )

    print_recording_concentration(
        concentration
    )

    concentration_output = (
        DIAGNOSTIC_OUTPUT_DIR
        / "recording_concentration.csv"
    )

    concentration.to_csv(
        concentration_output,
        index=False,
    )

    print()
    print(
        f"Wrote: {concentration_output}"
    )

    # ------------------------------------------------------------------
    # Cross-split overlap.
    # ------------------------------------------------------------------

    overlap = (
        analyze_cross_split_overlap(
            manifest
        )
    )

    # ------------------------------------------------------------------
    # Label concentration.
    # ------------------------------------------------------------------

    label_concentration = (
        calculate_label_concentration(
            concentration
        )
    )

    print_label_concentration(
        label_concentration
    )

    label_output = (
        DIAGNOSTIC_OUTPUT_DIR
        / "recording_label_concentration.csv"
    )

    label_concentration.to_csv(
        label_output,
        index=False,
    )

    print()
    print(
        f"Wrote: {label_output}"
    )

    # ------------------------------------------------------------------
    # Enriched diagnostic manifest.
    # ------------------------------------------------------------------

    enriched = build_enriched_manifest(
        manifest
    )

    enriched_output = (
        DIAGNOSTIC_OUTPUT_DIR
        / "recording_concentration_enriched.csv"
    )

    enriched.to_csv(
        enriched_output,
        index=False,
    )

    print()
    print(
        f"Wrote: {enriched_output}"
    )

    # ------------------------------------------------------------------
    # Final summary.
    # ------------------------------------------------------------------

    total_rows = int(
        len(manifest)
    )

    unique_segments = int(
        manifest[
            segment_column
        ]
        .astype(str)
        .nunique()
    )

    unique_recordings = int(
        manifest[
            "inferred_recording_id"
        ]
        .astype(str)
        .nunique()
    )

    duplicate_segment_count = int(
        (
            manifest[
                segment_column
            ]
            .astype(str)
            .value_counts()
            > 1
        ).sum()
    )

    overlap_count = int(
        len(overlap)
    )

    print()
    print("=" * 80)
    print("FINAL DIAGNOSTIC RESULT")
    print("=" * 80)

    print(
        f"Total rows analyzed: "
        f"{total_rows:,}"
    )

    print(
        f"Unique segment IDs: "
        f"{unique_segments:,}"
    )

    print(
        f"Unique inferred recording IDs: "
        f"{unique_recordings:,}"
    )

    print(
        f"Exact duplicate segment IDs: "
        f"{duplicate_segment_count:,}"
    )

    print(
        f"Cross-split inferred recording overlaps: "
        f"{overlap_count:,}"
    )

    print()
    print(
        "Diagnostic files written under:"
    )

    print(
        f"  {DIAGNOSTIC_OUTPUT_DIR}"
    )

    print()
    print("IMPORTANT:")

    if inference_description == "explicit_source_column":

        print(
            "  Recording IDs came from an explicit "
            "source/recording column."
        )

    else:

        print(
            "  No explicit recording/source/path column "
            "exists in the manifests."
        )

        print(
            "  Recording IDs were inferred from segment_id "
            "by removing the final ':<segment_index>' suffix "
            "when present."
        )

        print()
        print(
            "  This inference is heuristic."
        )

        print(
            "  Inspect the diagnostic CSV files before "
            "using these IDs to modify the dataset split."
        )

    print()
    print(
        "No dataset, manifest, shard, feature, "
        "or original .npy file was modified."
    )

    print()
    print(
        "Next step:"
    )

    print(
        "  Inspect:"
    )

    print(
        "    recording_concentration_enriched.csv"
    )

    print(
        "    recording_identifier_quality.csv"
    )

    print(
        "    recording_concentration.csv"
    )

    print(
        "    recording_label_concentration.csv"
    )

    print(
        "    cross_split_recording_overlap.csv"
    )

    print()
    print(
        "Confirm that the inferred recording IDs "
        "correspond to real source recordings before "
        "changing the training/validation/test split."
    )


if __name__ == "__main__":
    main()