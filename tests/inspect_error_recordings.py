"""
inspect_error_recordings.py

Inspect the recordings responsible for the dominant test-set errors.

This script DOES NOT:
- retrain the model
- run model inference
- modify model parameters
- modify the test set

It DOES:
- load existing error-analysis CSV files
- identify recordings with the most errors
- inspect their manifest metadata
- inspect corresponding NPZ shard contents
- report segment ranges
- report labels and predictions
- report model probabilities
- search the project for possible original/source audio references
- save detailed reports for manual investigation

Run from project root:

    python tests\\inspect_error_recordings.py
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ERROR_ANALYSIS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "error_analysis"
)

SOURCE_ANALYSIS_DIR = (
    ERROR_ANALYSIS_DIR
    / "source_analysis"
)

TEST_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "features"
    / "test_shard_manifest.csv"
)

SHARD_DIR = (
    PROJECT_ROOT
    / "features"
    / "shards"
    / "test"
)

OUTPUT_DIR = (
    SOURCE_ANALYSIS_DIR
    / "recording_inspection"
)

# These are the two recordings responsible for 92% of false positives
TARGET_RECORDINGS = [
    "d323ebf177af404c45bbd02a3d05fafbb202a0df0e6dab45aacd62b4e22f01f8",
    "d33311be5297b532df4c9ff03bb14ace9a2dd1fe72da4c01cdbfa36124d89ff2",
]

# Also inspect the top N other error recordings.
TOP_OTHER_RECORDINGS = 8

# File extensions to search for when trying to recover original audio.
AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
    ".aiff",
    ".au",
}

# Directories that should NOT be recursively searched for audio/source files.
SKIP_SEARCH_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "site-packages",
    "outputs",
    "checkpoints",
}


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("inspect_error_recordings")


# ============================================================================
# HELPERS
# ============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_section(title: str) -> None:
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def find_existing_file(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def normalize_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    normalized = {
        normalize_column_name(c): c
        for c in df.columns
    }

    for candidate in candidates:
        candidate_norm = normalize_column_name(candidate)

        if candidate_norm in normalized:
            return normalized[candidate_norm]

    # Fuzzy fallback
    for col in df.columns:
        col_norm = normalize_column_name(col)

        for candidate in candidates:
            candidate_norm = normalize_column_name(candidate)

            if candidate_norm in col_norm:
                return col

    return None


def safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_segment_id(value: Any) -> tuple[str, int]:
    """
    Expected format:

        recording_hash:segment_number

    Example:

        abc123:50
    """

    text = str(value).strip()

    if ":" not in text:
        return text, -1

    recording_id, segment = text.rsplit(":", 1)

    return recording_id, safe_int(segment)


def compact_path(path: Any) -> str:
    if pd.isna(path):
        return ""

    return str(path)


# ============================================================================
# INPUT VALIDATION
# ============================================================================

def check_inputs() -> None:
    print_header("ACOUSTIC DRONE ERROR RECORDING INSPECTION")

    print(f"Project root:          {PROJECT_ROOT}")
    print(f"Test manifest:         {TEST_MANIFEST}")
    print(f"Error analysis:       {ERROR_ANALYSIS_DIR}")
    print(f"Source analysis:      {SOURCE_ANALYSIS_DIR}")
    print(f"Output directory:     {OUTPUT_DIR}")

    print_section("CHECKING INPUT FILES")

    required = [
        TEST_MANIFEST,
        ERROR_ANALYSIS_DIR / "false_positives.csv",
        ERROR_ANALYSIS_DIR / "false_negatives.csv",
        SOURCE_ANALYSIS_DIR / "false_positives_with_source.csv",
        SOURCE_ANALYSIS_DIR / "false_negatives_with_source.csv",
    ]

    missing = []

    for path in required:
        if path.exists():
            print(f"[OK] {path}")
        else:
            print(f"[MISSING] {path}")
            missing.append(path)

    if missing:
        raise FileNotFoundError(
            "Required files are missing. Run the previous "
            "error-analysis scripts first."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# LOAD ERROR FILES
# ============================================================================

def load_error_files() -> tuple[pd.DataFrame, pd.DataFrame]:
    print_section("LOADING EXISTING ERROR ANALYSIS")

    fp_path = ERROR_ANALYSIS_DIR / "false_positives.csv"
    fn_path = ERROR_ANALYSIS_DIR / "false_negatives.csv"

    fp = pd.read_csv(fp_path)
    fn = pd.read_csv(fn_path)

    print(f"False positives loaded: {len(fp)}")
    print(f"False negatives loaded: {len(fn)}")

    return fp, fn


# ============================================================================
# NORMALIZE ERROR DATA
# ============================================================================

def normalize_error_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    segment_col = find_column(
        df,
        [
            "segment_id",
            "analysis_segment_id",
            "id",
        ],
    )

    recording_col = find_column(
        df,
        [
            "recording_id",
            "source_recording_id",
        ],
    )

    segment_number_col = find_column(
        df,
        [
            "segment_number",
            "segment_idx",
            "segment_index",
        ],
    )

    drone_probability_col = find_column(
        df,
        [
            "drone_probability",
            "probability_drone",
            "drone_prob",
        ],
    )

    shard_col = find_column(
        df,
        [
            "shard_path",
            "shard",
        ],
    )

    if recording_col is None and segment_col is not None:
        parsed = df[segment_col].apply(parse_segment_id)

        df["__recording_id"] = parsed.apply(lambda x: x[0])
        df["__segment_number"] = parsed.apply(lambda x: x[1])

        recording_col = "__recording_id"
        segment_number_col = "__segment_number"

    if recording_col is None:
        raise ValueError(
            "Could not identify recording_id or segment_id."
        )

    if segment_number_col is None:
        df["__segment_number"] = -1
        segment_number_col = "__segment_number"

    df["__recording_id"] = (
        df[recording_col]
        .astype(str)
        .str.strip()
    )

    df["__segment_number"] = pd.to_numeric(
        df[segment_number_col],
        errors="coerce",
    ).fillna(-1).astype(int)

    if drone_probability_col is not None:
        df["__drone_probability"] = pd.to_numeric(
            df[drone_probability_col],
            errors="coerce",
        )
    else:
        df["__drone_probability"] = np.nan

    if shard_col is not None:
        df["__shard_path"] = df[shard_col].astype(str)
    else:
        df["__shard_path"] = ""

    return df


# ============================================================================
# RECORDING ERROR SUMMARY
# ============================================================================

def build_recording_summary(
    fp: pd.DataFrame,
    fn: pd.DataFrame,
) -> pd.DataFrame:

    fp_counts = (
        fp.groupby("__recording_id")
        .size()
        .rename("false_positives")
    )

    fn_counts = (
        fn.groupby("__recording_id")
        .size()
        .rename("false_negatives")
    )

    summary = pd.concat(
        [fp_counts, fn_counts],
        axis=1,
    ).fillna(0)

    summary["false_positives"] = (
        summary["false_positives"].astype(int)
    )

    summary["false_negatives"] = (
        summary["false_negatives"].astype(int)
    )

    summary["total_errors"] = (
        summary["false_positives"]
        + summary["false_negatives"]
    )

    summary = summary.sort_values(
        [
            "total_errors",
            "false_positives",
            "false_negatives",
        ],
        ascending=False,
    )

    summary.index.name = "recording_id"

    return summary.reset_index()


# ============================================================================
# PRINT TARGET RECORDINGS
# ============================================================================

def inspect_target_errors(
    fp: pd.DataFrame,
    fn: pd.DataFrame,
) -> None:

    print_header("DOMINANT FALSE-POSITIVE RECORDINGS")

    for recording_id in TARGET_RECORDINGS:

        rows = fp[
            fp["__recording_id"] == recording_id
        ].copy()

        print()
        print(f"Recording: {recording_id}")
        print(f"False positives: {len(rows)}")

        if rows.empty:
            print("  No false positives found.")
            continue

        rows = rows.sort_values(
            "__segment_number"
        )

        segments = rows[
            "__segment_number"
        ].tolist()

        probabilities = rows[
            "__drone_probability"
        ].dropna()

        print(
            f"Segment range: "
            f"{min(segments)} -> {max(segments)}"
        )

        print(
            f"Highest drone probability: "
            f"{probabilities.max():.6f}"
        )

        print(
            f"Average drone probability: "
            f"{probabilities.mean():.6f}"
        )

        print(
            f"Minimum drone probability: "
            f"{probabilities.min():.6f}"
        )

        print()
        print("Segments:")

        # Print segments in compact groups.
        print_segment_ranges(segments)

        print()
        print("Detailed high-confidence errors:")

        display_cols = [
            "__segment_number",
            "__drone_probability",
            "__shard_path",
        ]

        display_cols = [
            c for c in display_cols
            if c in rows.columns
        ]

        print(
            rows[
                display_cols
            ]
            .sort_values(
                "__drone_probability",
                ascending=False,
            )
            .head(30)
            .to_string(index=False)
        )


def print_segment_ranges(segments: list[int]) -> None:

    if not segments:
        return

    segments = sorted(
        set(
            x for x in segments
            if x >= 0
        )
    )

    if not segments:
        print("  No valid segment numbers.")
        return

    ranges = []

    start = segments[0]
    previous = segments[0]

    for current in segments[1:]:

        if current == previous + 1:
            previous = current
            continue

        if start == previous:
            ranges.append(str(start))
        else:
            ranges.append(
                f"{start}-{previous}"
            )

        start = current
        previous = current

    if start == previous:
        ranges.append(str(start))
    else:
        ranges.append(
            f"{start}-{previous}"
        )

    print("  " + ", ".join(ranges))


# ============================================================================
# MANIFEST INSPECTION
# ============================================================================

def load_manifest() -> pd.DataFrame:

    print_section("LOADING TEST MANIFEST")

    manifest = pd.read_csv(
        TEST_MANIFEST
    )

    print(
        f"Manifest rows: {len(manifest)}"
    )

    print(
        "Manifest columns:"
    )

    for column in manifest.columns:
        print(f"  - {column}")

    return manifest


def normalize_manifest(
    manifest: pd.DataFrame,
) -> pd.DataFrame:

    manifest = manifest.copy()

    segment_col = find_column(
        manifest,
        [
            "segment_id",
            "analysis_segment_id",
            "id",
        ],
    )

    recording_col = find_column(
        manifest,
        [
            "recording_id",
            "source_recording_id",
        ],
    )

    segment_number_col = find_column(
        manifest,
        [
            "segment_number",
            "segment_idx",
            "segment_index",
        ],
    )

    if recording_col is None and segment_col is not None:

        parsed = manifest[
            segment_col
        ].apply(parse_segment_id)

        manifest["__recording_id"] = (
            parsed.apply(lambda x: x[0])
        )

        manifest["__segment_number"] = (
            parsed.apply(lambda x: x[1])
        )

    else:

        if recording_col is not None:
            manifest["__recording_id"] = (
                manifest[recording_col]
                .astype(str)
                .str.strip()
            )

        if segment_number_col is not None:
            manifest["__segment_number"] = pd.to_numeric(
                manifest[segment_number_col],
                errors="coerce",
            ).fillna(-1).astype(int)

    return manifest


def inspect_manifest_matches(
    manifest: pd.DataFrame,
) -> pd.DataFrame:

    print_header("MANIFEST INFORMATION FOR TARGET RECORDINGS")

    target = manifest[
        manifest["__recording_id"].isin(
            TARGET_RECORDINGS
        )
    ].copy()

    print(
        f"Matching manifest rows: {len(target)}"
    )

    if target.empty:
        print(
            "WARNING: Target recordings were not "
            "found directly in the manifest."
        )

        return target

    print()
    print("Target recording counts:")

    print(
        target[
            "__recording_id"
        ]
        .value_counts()
        .to_string()
    )

    return target


# ============================================================================
# SEARCH MANIFEST FOR USEFUL SOURCE INFORMATION
# ============================================================================

def inspect_possible_source_columns(
    manifest: pd.DataFrame,
) -> None:

    print_section("SEARCHING MANIFEST FOR SOURCE/AUDIO METADATA")

    interesting_terms = [
        "source",
        "audio",
        "file",
        "path",
        "recording",
        "dataset",
        "origin",
        "name",
        "raw",
        "wav",
        "mp3",
        "flac",
    ]

    found = []

    for column in manifest.columns:

        name = normalize_column_name(
            column
        )

        if any(
            term in name
            for term in interesting_terms
        ):
            found.append(column)

    if not found:
        print(
            "No obvious source/audio columns found."
        )
        return

    print(
        "Potential source/audio columns:"
    )

    for column in found:
        print(f"  - {column}")

        sample = (
            manifest[column]
            .dropna()
            .astype(str)
            .head(5)
            .tolist()
        )

        for value in sample:
            print(f"      {value}")


# ============================================================================
# NPZ SHARD INSPECTION
# ============================================================================

def inspect_shards(
    fp: pd.DataFrame,
    fn: pd.DataFrame,
) -> None:

    print_header("INSPECTING TEST SHARDS")

    errors = pd.concat(
        [
            fp.assign(
                __error_type="false_positive"
            ),
            fn.assign(
                __error_type="false_negative"
            ),
        ],
        ignore_index=True,
    )

    shard_paths = (
        errors[
            "__shard_path"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not shard_paths:
        print(
            "No shard paths available in error CSVs."
        )
        return

    print(
        f"Unique shards containing errors: "
        f"{len(shard_paths)}"
    )

    for shard_text in shard_paths:

        shard_path = Path(shard_text)

        if not shard_path.exists():

            # Try matching by filename.
            candidate = SHARD_DIR / shard_path.name

            if candidate.exists():
                shard_path = candidate

        print()
        print(f"Shard: {shard_path}")

        if not shard_path.exists():
            print("  [NOT FOUND]")
            continue

        try:
            data = np.load(
                shard_path,
                allow_pickle=True,
            )

            print(
                f"  Keys: {list(data.keys())}"
            )

            for key in data.keys():

                try:
                    value = data[key]

                    print(
                        f"  {key}: "
                        f"shape={getattr(value, 'shape', None)}, "
                        f"dtype={getattr(value, 'dtype', None)}"
                    )

                except Exception as exc:
                    print(
                        f"  {key}: unable to inspect "
                        f"({exc})"
                    )

            data.close()

        except Exception as exc:

            print(
                f"  ERROR loading shard: {exc}"
            )


# ============================================================================
# SEARCH FOR ORIGINAL AUDIO
# ============================================================================

def search_for_audio_files(
    target_recordings: list[str],
) -> list[dict[str, str]]:

    print_header("SEARCHING PROJECT FOR ORIGINAL AUDIO FILES")

    print(
        "This search does not inspect model predictions."
    )

    print(
        "Searching project files for audio filenames "
        "that may correspond to recording IDs..."
    )

    results = []

    # Search likely directories first.
    search_roots = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "datasets",
        PROJECT_ROOT / "audio",
        PROJECT_ROOT / "raw",
        PROJECT_ROOT / "input",
        PROJECT_ROOT / "inputs",
        PROJECT_ROOT / "features",
    ]

    search_roots = [
        root
        for root in search_roots
        if root.exists()
    ]

    # First: exact recording ID filename/path matching.
    for root in search_roots:

        for path in root.rglob("*"):

            if not path.is_file():
                continue

            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue

            path_lower = str(path).lower()

            for recording_id in target_recordings:

                if recording_id.lower() in path_lower:

                    results.append(
                        {
                            "recording_id": recording_id,
                            "audio_path": str(path),
                            "match_type": "recording_id",
                        }
                    )

    # Remove duplicates.
    unique = {}

    for result in results:

        key = (
            result["recording_id"],
            result["audio_path"],
        )

        unique[key] = result

    results = list(unique.values())

    if results:

        print(
            f"Found {len(results)} possible audio matches:"
        )

        for result in results:

            print(
                f"  {result['recording_id']}"
            )

            print(
                f"      {result['audio_path']}"
            )

    else:

        print(
            "No original audio files were found "
            "by direct recording-ID matching."
        )

        print()
        print(
            "This does NOT mean the audio is unavailable."
        )

        print(
            "The recording IDs may be hashes generated "
            "during preprocessing."
        )

    return results


# ============================================================================
# SEARCH PROJECT TEXT FOR RECORDING IDS
# ============================================================================

def search_project_text(
    target_recordings: list[str],
) -> list[dict[str, Any]]:

    print_header("SEARCHING PROJECT FILES FOR RECORDING IDs")

    results = []

    searchable_extensions = {
        ".csv",
        ".json",
        ".jsonl",
        ".txt",
        ".yaml",
        ".yml",
        ".py",
        ".toml",
        ".ini",
        ".cfg",
    }

    for root, dirs, files in __import__(
        "os"
    ).walk(PROJECT_ROOT):

        root_path = Path(root)

        # Avoid expensive/unnecessary directories.
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_SEARCH_DIRS
        ]

        for filename in files:

            path = root_path / filename

            if path.suffix.lower() not in searchable_extensions:
                continue

            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

            except Exception:
                continue

            for recording_id in target_recordings:

                if recording_id not in text:
                    continue

                lines = text.splitlines()

                matching_lines = []

                for line_number, line in enumerate(
                    lines,
                    start=1,
                ):

                    if recording_id in line:

                        matching_lines.append(
                            {
                                "line_number": line_number,
                                "line": line[:1000],
                            }
                        )

                results.append(
                    {
                        "recording_id": recording_id,
                        "file": str(path),
                        "matches": matching_lines,
                    }
                )

    if not results:

        print(
            "No project text files contained the "
            "target recording IDs."
        )

    else:

        print(
            f"Found {len(results)} matching files."
        )

        for result in results:

            print()
            print(
                f"Recording: {result['recording_id']}"
            )

            print(
                f"File: {result['file']}"
            )

            for match in result["matches"][:10]:

                print(
                    f"  line {match['line_number']}: "
                    f"{match['line']}"
                )

    return results


# ============================================================================
# SAVE TARGET ERROR TABLES
# ============================================================================

def save_target_tables(
    fp: pd.DataFrame,
    fn: pd.DataFrame,
) -> None:

    print_section("SAVING TARGET RECORDING TABLES")

    target_fp = fp[
        fp["__recording_id"].isin(
            TARGET_RECORDINGS
        )
    ].copy()

    target_fn = fn[
        fn["__recording_id"].isin(
            TARGET_RECORDINGS
        )
    ].copy()

    fp_path = (
        OUTPUT_DIR
        / "dominant_false_positives.csv"
    )

    fn_path = (
        OUTPUT_DIR
        / "dominant_false_negatives.csv"
    )

    target_fp.to_csv(
        fp_path,
        index=False,
    )

    target_fn.to_csv(
        fn_path,
        index=False,
    )

    print(f"Saved: {fp_path}")
    print(f"Saved: {fn_path}")


# ============================================================================
# SAVE COMPLETE RECORDING SUMMARY
# ============================================================================

def save_recording_summary(
    summary: pd.DataFrame,
) -> None:

    path = (
        OUTPUT_DIR
        / "recording_error_summary.csv"
    )

    summary.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved: {path}"
    )


# ============================================================================
# SAVE JSON REPORT
# ============================================================================

def create_json_report(
    summary: pd.DataFrame,
    fp: pd.DataFrame,
    fn: pd.DataFrame,
    audio_matches: list[dict[str, str]],
    text_matches: list[dict[str, Any]],
) -> None:

    target_summary = summary[
        summary["recording_id"].isin(
            TARGET_RECORDINGS
        )
    ]

    report = {
        "project_root": str(PROJECT_ROOT),
        "test_manifest": str(TEST_MANIFEST),
        "error_analysis_directory": str(
            ERROR_ANALYSIS_DIR
        ),
        "target_recordings": TARGET_RECORDINGS,
        "total_false_positives": int(len(fp)),
        "total_false_negatives": int(len(fn)),
        "target_recording_summary": (
            target_summary
            .to_dict(orient="records")
        ),
        "audio_matches": audio_matches,
        "text_matches": text_matches,
    }

    path = (
        OUTPUT_DIR
        / "recording_inspection_report.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Saved: {path}"
    )


# ============================================================================
# FINAL INTERPRETATION
# ============================================================================

def print_interpretation(
    summary: pd.DataFrame,
) -> None:

    print_header("INTERPRETATION")

    total_fp = int(
        summary["false_positives"].sum()
    )

    target_fp = int(
        summary[
            summary["recording_id"].isin(
                TARGET_RECORDINGS
            )
        ]["false_positives"]
        .sum()
    )

    if total_fp > 0:

        concentration = (
            target_fp / total_fp * 100
        )

    else:
        concentration = 0.0

    print(
        f"False positives in target recordings: "
        f"{target_fp}/{total_fp} "
        f"({concentration:.2f}%)"
    )

    print()

    if concentration >= 80:

        print(
            "STRONG RECORDING-SPECIFIC PATTERN:"
        )

        print(
            "Most false positives are concentrated "
            "in a very small number of recordings."
        )

        print(
            "Do NOT immediately assume that the model "
            "needs another full training run."
        )

        print()

        print(
            "Recommended investigation:"
        )

        print(
            "  1. Locate the original audio."
        )

        print(
            "  2. Listen to the problematic segments."
        )

        print(
            "  3. Inspect their spectrograms."
        )

        print(
            "  4. Verify their labels."
        )

        print(
            "  5. Determine whether the recordings "
            "contain unusual background noise."
        )

        print(
            "  6. Check whether the two recordings "
            "come from the same dataset/source."
        )

    else:

        print(
            "False positives are not highly concentrated."
        )

        print(
            "This is more consistent with a broader "
            "generalization problem."
        )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    check_inputs()

    fp_raw, fn_raw = load_error_files()

    fp = normalize_error_dataframe(
        fp_raw
    )

    fn = normalize_error_dataframe(
        fn_raw
    )

    summary = build_recording_summary(
        fp,
        fn,
    )

    print_header("OVERALL ERROR CONCENTRATION")

    print(
        summary.head(20).to_string(
            index=False
        )
    )

    inspect_target_errors(
        fp,
        fn,
    )

    manifest = load_manifest()

    manifest = normalize_manifest(
        manifest
    )

    inspect_manifest_matches(
        manifest
    )

    inspect_possible_source_columns(
        manifest
    )

    inspect_shards(
        fp,
        fn,
    )

    audio_matches = search_for_audio_files(
        TARGET_RECORDINGS
    )

    text_matches = search_project_text(
        TARGET_RECORDINGS
    )

    save_target_tables(
        fp,
        fn,
    )

    save_recording_summary(
        summary
    )

    create_json_report(
        summary,
        fp,
        fn,
        audio_matches,
        text_matches,
    )

    print_interpretation(
        summary
    )

    print_header(
        "INSPECTION COMPLETE"
    )

    print(
        "No model parameters were modified."
    )

    print(
        "No training data was modified."
    )

    print(
        "No inference was performed."
    )

    print()

    print(
        "Important output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "Key files:"
    )

    print(
        "  recording_error_summary.csv"
    )

    print(
        "  dominant_false_positives.csv"
    )

    print(
        "  dominant_false_negatives.csv"
    )

    print(
        "  recording_inspection_report.json"
    )


if __name__ == "__main__":
    main()