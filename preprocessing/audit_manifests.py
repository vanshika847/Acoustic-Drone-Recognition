"""
Audit final train/validation/test manifests.

Checks:
1. Row counts
2. Binary label distribution
3. Dataset distribution
4. Duplicate SHA-256 files across splits
5. Recording-group leakage across splits
6. Missing required columns
7. Empty manifests

Run from project root:

    python -m preprocessing.audit_manifests
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "manifests"
)

SPLITS = (
    "train",
    "validation",
    "test",
)

REQUIRED_COLUMNS = {
    "dataset",
    "binary_label",
    "sha256",
    "recording_group_id",
}


LOGGER = logging.getLogger(__name__)


class ManifestAuditError(RuntimeError):
    """Raised when the manifest audit cannot be completed safely."""


def load_manifests() -> Dict[str, pd.DataFrame]:
    """Load all final split manifests."""

    manifests: Dict[str, pd.DataFrame] = {}

    for split in SPLITS:
        path = MANIFEST_DIR / f"{split}.csv"

        if not path.is_file():
            raise ManifestAuditError(
                f"Missing manifest: {path}"
            )

        LOGGER.info(
            "Loading %s",
            path,
        )

        dataframe = pd.read_csv(path)

        if dataframe.empty:
            raise ManifestAuditError(
                f"{split}.csv is empty."
            )

        missing_columns = (
            REQUIRED_COLUMNS
            - set(dataframe.columns)
        )

        if missing_columns:
            raise ManifestAuditError(
                f"{split}.csv is missing required "
                f"columns: {sorted(missing_columns)}"
            )

        manifests[split] = dataframe

    return manifests


def audit_row_counts(
    manifests: Dict[str, pd.DataFrame],
) -> None:
    """Print row counts for every split."""

    print()
    print("=" * 70)
    print("ROW COUNTS")
    print("=" * 70)

    total = 0

    for split in SPLITS:
        count = len(manifests[split])
        total += count

        print(
            f"{split:12s}: {count:,}"
        )

    print("-" * 70)
    print(f"{'TOTAL':12s}: {total:,}")


def audit_labels(
    manifests: Dict[str, pd.DataFrame],
) -> None:
    """Print binary label distribution."""

    print()
    print("=" * 70)
    print("BINARY LABEL DISTRIBUTION")
    print("=" * 70)

    for split in SPLITS:
        counts = (
            manifests[split]["binary_label"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        print(
            f"{split:12s}: {counts}"
        )


def audit_datasets(
    manifests: Dict[str, pd.DataFrame],
) -> None:
    """Print dataset distribution per split."""

    print()
    print("=" * 70)
    print("DATASET DISTRIBUTION")
    print("=" * 70)

    for split in SPLITS:
        counts = (
            manifests[split]["dataset"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        print()
        print(f"{split.upper()}")

        for dataset, count in counts.items():
            print(
                f"  {dataset:20s}: {count:,}"
            )


def _normalise_values(
    series: pd.Series,
) -> set[str]:
    """Return non-empty normalized string values."""

    values = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return {
        value
        for value in values
        if value
        and value.lower() != "nan"
    }


def audit_sha256_leakage(
    manifests: Dict[str, pd.DataFrame],
) -> int:
    """Check for identical audio files across splits."""

    print()
    print("=" * 70)
    print("SHA-256 CROSS-SPLIT DUPLICATE CHECK")
    print("=" * 70)

    hashes = {
        split: _normalise_values(
            manifests[split]["sha256"]
        )
        for split in SPLITS
    }

    failures = 0

    comparisons = (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    )

    for left, right in comparisons:
        overlap = (
            hashes[left]
            & hashes[right]
        )

        if overlap:
            failures += len(overlap)

            print(
                f"FAIL: {left} ∩ {right} = "
                f"{len(overlap):,} duplicate SHA-256 values"
            )

        else:
            print(
                f"PASS: {left} ∩ {right} = 0"
            )

    return failures


def audit_recording_group_leakage(
    manifests: Dict[str, pd.DataFrame],
) -> int:
    """Check for recording groups shared across splits."""

    print()
    print("=" * 70)
    print("RECORDING-GROUP CROSS-SPLIT LEAKAGE CHECK")
    print("=" * 70)

    groups = {
        split: _normalise_values(
            manifests[split]["recording_group_id"]
        )
        for split in SPLITS
    }

    failures = 0

    comparisons = (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    )

    for left, right in comparisons:
        overlap = (
            groups[left]
            & groups[right]
        )

        if overlap:
            failures += len(overlap)

            print(
                f"FAIL: {left} ∩ {right} = "
                f"{len(overlap):,} shared groups"
            )

            examples = sorted(overlap)[:10]

            for group in examples:
                print(
                    f"      {group}"
                )

        else:
            print(
                f"PASS: {left} ∩ {right} = 0"
            )

    return failures


def audit_labels_are_valid(
    manifests: Dict[str, pd.DataFrame],
) -> int:
    """Ensure only binary labels 0 and 1 exist."""

    print()
    print("=" * 70)
    print("LABEL VALIDITY")
    print("=" * 70)

    invalid_count = 0

    for split in SPLITS:
        values = set(
            manifests[split]["binary_label"]
            .dropna()
            .tolist()
        )

        invalid = values - {0, 1}

        if invalid:
            invalid_count += len(invalid)

            print(
                f"FAIL: {split} contains "
                f"invalid labels: {sorted(invalid)}"
            )

        else:
            print(
                f"PASS: {split} contains only labels 0 and 1"
            )

    return invalid_count


def audit_dataset_presence(
    manifests: Dict[str, pd.DataFrame],
) -> None:
    """Show which datasets are present in each split."""

    print()
    print("=" * 70)
    print("DATASET PRESENCE")
    print("=" * 70)

    all_datasets = set()

    for dataframe in manifests.values():
        all_datasets.update(
            dataframe["dataset"]
            .dropna()
            .astype(str)
            .str.strip()
        )

    for dataset in sorted(all_datasets):
        presence = []

        for split in SPLITS:
            present = (
                manifests[split]["dataset"]
                .astype(str)
                .eq(dataset)
                .any()
            )

            if present:
                presence.append(split)

        print(
            f"{dataset:20s}: "
            f"{', '.join(presence)}"
        )


def run_audit() -> None:
    """Run the complete manifest audit."""

    print()
    print("=" * 70)
    print("ACOUSTIC DRONE MANIFEST AUDIT")
    print("=" * 70)

    manifests = load_manifests()

    audit_row_counts(
        manifests
    )

    audit_labels(
        manifests
    )

    audit_datasets(
        manifests
    )

    audit_dataset_presence(
        manifests
    )

    sha_failures = audit_sha256_leakage(
        manifests
    )

    group_failures = audit_recording_group_leakage(
        manifests
    )

    label_failures = audit_labels_are_valid(
        manifests
    )

    print()
    print("=" * 70)
    print("FINAL AUDIT RESULT")
    print("=" * 70)

    if (
        sha_failures == 0
        and group_failures == 0
        and label_failures == 0
    ):
        print(
            "PASS: Final manifests are leakage-safe."
        )

        print(
            "PASS: No cross-split SHA-256 duplicates."
        )

        print(
            "PASS: No cross-split recording-group overlap."
        )

        print(
            "PASS: Labels are valid binary values."
        )

    else:
        print(
            "FAIL: Manifest audit found problems."
        )

        if sha_failures:
            print(
                f"- SHA-256 leakage issues: "
                f"{sha_failures:,}"
            )

        if group_failures:
            print(
                f"- Recording-group leakage issues: "
                f"{group_failures:,}"
            )

        if label_failures:
            print(
                f"- Invalid-label issues: "
                f"{label_failures:,}"
            )

        raise ManifestAuditError(
            "Manifest audit failed."
        )


def main() -> None:
    """CLI entry point."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    run_audit()


if __name__ == "__main__":
    main()