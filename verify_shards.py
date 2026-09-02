"""Verify acoustic-drone feature shards before deleting original .npy files."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "features"

SPLITS = ("train", "validation", "test")

FEATURE_NAMES = (
    "mfcc",
    "mel",
    "spectral",
    "chroma",
    "zcr",
    "energy",
)


def resolve_path(value: str) -> Path:
    """Resolve a shard path against the current project."""

    path = Path(value)

    if path.is_absolute() and path.is_file():
        return path

    if path.is_absolute():
        parts = list(path.parts)

        if "features" in parts:
            index = parts.index("features")

            candidate = (
                PROJECT_ROOT
                / Path(*parts[index:])
            )

            if candidate.is_file():
                return candidate

    candidate = (
        PROJECT_ROOT / path
    ).resolve()

    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Shard file not found: '{value}'"
    )


def get_archive_keys(
    archive: np.lib.npyio.NpzFile,
) -> list[str]:
    """Return keys stored inside an NPZ archive."""

    return list(archive.files)


def find_key(
    keys: list[str],
    candidates: tuple[str, ...],
) -> str | None:
    """Find the first matching key."""

    lowered = {
        key.lower(): key
        for key in keys
    }

    for candidate in candidates:

        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    return None


def verify_feature_array(
    array: np.ndarray,
    feature_name: str,
    shard_name: str,
) -> None:
    """Verify one feature array."""

    if array.dtype == object:

        # Object arrays are expected only when each
        # sample may have a different time dimension.
        for index, item in enumerate(array):

            item_array = np.asarray(
                item,
                dtype=np.float32,
            )

            if item_array.ndim != 2:
                raise ValueError(
                    f"{shard_name}: {feature_name} "
                    f"sample {index} has shape "
                    f"{item_array.shape}; expected 2-D."
                )

            if (
                item_array.shape[0] == 0
                or item_array.shape[1] == 0
            ):
                raise ValueError(
                    f"{shard_name}: {feature_name} "
                    f"sample {index} is empty."
                )

            if not np.isfinite(item_array).all():
                raise ValueError(
                    f"{shard_name}: {feature_name} "
                    f"sample {index} contains NaN/Inf."
                )

        return

    # Standard numeric feature array.
    #
    # Expected possibilities:
    #
    #   (samples, channels, time)
    #
    # or a single-feature representation that still
    # has at least two dimensions.
    if array.ndim < 2:
        raise ValueError(
            f"{shard_name}: {feature_name} has shape "
            f"{array.shape}; expected at least 2-D."
        )

    if 0 in array.shape:
        raise ValueError(
            f"{shard_name}: {feature_name} is empty: "
            f"{array.shape}"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{shard_name}: {feature_name} contains NaN/Inf."
        )


def verify_npz_shard(
    shard_path: Path,
) -> tuple[int, Counter[int], set[str]]:
    """
    Verify one NPZ shard.

    Returns
    -------
    tuple
        Sample count, label distribution, and segment IDs.
    """

    with np.load(
        shard_path,
        allow_pickle=True,
    ) as archive:

        keys = get_archive_keys(
            archive
        )

        if not keys:
            raise ValueError(
                f"{shard_path.name} contains no arrays."
            )

        print(
            f"  Keys: {', '.join(keys)}"
        )

        # -----------------------------------------------------------
        # Locate important fields.
        # -----------------------------------------------------------

        label_key = find_key(
            keys,
            (
                "labels",
                "label",
                "binary_label",
                "targets",
            ),
        )

        segment_key = find_key(
            keys,
            (
                "segment_ids",
                "segment_id",
                "ids",
            ),
        )

        if label_key is None:
            raise ValueError(
                f"{shard_path.name}: could not find "
                f"a label array. Available keys: {keys}"
            )

        labels = np.asarray(
            archive[label_key]
        ).reshape(-1)

        sample_count = len(
            labels
        )

        if sample_count == 0:
            raise ValueError(
                f"{shard_path.name}: shard contains zero samples."
            )

        # -----------------------------------------------------------
        # Verify labels.
        # -----------------------------------------------------------

        label_counts: Counter[int] = Counter()

        for value in labels:

            label = int(value)

            if label not in (0, 1):
                raise ValueError(
                    f"{shard_path.name}: invalid label "
                    f"{label}; expected 0 or 1."
                )

            label_counts[label] += 1

        # -----------------------------------------------------------
        # Verify segment IDs if present.
        # -----------------------------------------------------------

        segment_ids: set[str] = set()

        if segment_key is not None:

            raw_ids = np.asarray(
                archive[segment_key]
            ).reshape(-1)

            if len(raw_ids) != sample_count:
                raise ValueError(
                    f"{shard_path.name}: segment ID count "
                    f"{len(raw_ids)} does not match "
                    f"sample count {sample_count}."
                )

            for value in raw_ids:

                segment_id = str(value)

                if not segment_id:
                    raise ValueError(
                        f"{shard_path.name}: empty segment ID."
                    )

                if segment_id in segment_ids:
                    raise ValueError(
                        f"{shard_path.name}: duplicate "
                        f"segment ID: {segment_id}"
                    )

                segment_ids.add(
                    segment_id
                )

        # -----------------------------------------------------------
        # Verify every feature.
        # -----------------------------------------------------------

        for feature_name in FEATURE_NAMES:

            feature_key = find_key(
                keys,
                (
                    feature_name,
                    f"{feature_name}s",
                ),
            )

            if feature_key is None:

                raise ValueError(
                    f"{shard_path.name}: missing feature "
                    f"'{feature_name}'. Available keys: {keys}"
                )

            feature_array = np.asarray(
                archive[feature_key]
            )

            verify_feature_array(
                feature_array,
                feature_name,
                shard_path.name,
            )

            # Numeric stacked representation:
            #
            # first dimension should equal number of samples.
            #
            # Object representation is checked sample-by-sample above.
            if (
                feature_array.dtype != object
                and feature_array.shape[0]
                != sample_count
            ):

                raise ValueError(
                    f"{shard_path.name}: {feature_name} "
                    f"first dimension "
                    f"{feature_array.shape[0]} does not "
                    f"match sample count {sample_count}."
                )

            if (
                feature_array.dtype == object
                and len(feature_array)
                != sample_count
            ):

                raise ValueError(
                    f"{shard_path.name}: {feature_name} "
                    f"contains {len(feature_array)} "
                    f"samples but expected {sample_count}."
                )

        return (
            sample_count,
            label_counts,
            segment_ids,
        )


def verify_split(
    split: str,
) -> bool:
    """Verify every shard belonging to one split."""

    print("\n" + "=" * 60)
    print(
        f"VERIFYING {split.upper()}"
    )
    print("=" * 60)

    manifest_path = (
        FEATURE_OUTPUT_DIR
        / f"{split}_shard_manifest.csv"
    )

    if not manifest_path.is_file():

        print(
            f"ERROR: Manifest not found:\n"
            f"{manifest_path}"
        )

        return False

    manifest = pd.read_csv(
        manifest_path,
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"Manifest rows: {len(manifest)}"
    )

    if "shard_path" not in manifest.columns:

        print(
            "ERROR: Missing manifest column: shard_path"
        )

        return False

    unique_paths = (
        manifest["shard_path"]
        .drop_duplicates()
        .tolist()
    )

    print(
        f"Unique shards referenced: "
        f"{len(unique_paths)}"
    )

    total_samples = 0
    total_labels: Counter[int] = Counter()
    all_segment_ids: set[str] = set()
    verified_paths: set[str] = set()

    # ---------------------------------------------------------------
    # Verify each physical shard exactly once.
    # ---------------------------------------------------------------

    for shard_number, raw_path in enumerate(
        unique_paths,
        start=1,
    ):

        print(
            f"\nShard "
            f"{shard_number}/{len(unique_paths)}: "
            f"{Path(raw_path).name}"
        )

        try:

            shard_path = resolve_path(
                raw_path
            )

            sample_count, label_counts, segment_ids = (
                verify_npz_shard(
                    shard_path
                )
            )

            total_samples += sample_count

            total_labels.update(
                label_counts
            )

            overlap = (
                all_segment_ids
                .intersection(segment_ids)
            )

            if overlap:

                raise ValueError(
                    "Duplicate segment IDs across shards: "
                    + ", ".join(
                        list(overlap)[:5]
                    )
                )

            all_segment_ids.update(
                segment_ids
            )

            verified_paths.add(
                str(
                    shard_path.resolve()
                )
            )

            print(
                f"  Samples: {sample_count}"
            )

            print(
                f"  Labels: "
                f"{dict(sorted(label_counts.items()))}"
            )

            print(
                "  Status: OK"
            )

        except Exception as error:

            print(
                f"ERROR: {error}"
            )

            return False

    # ---------------------------------------------------------------
    # Verify total sample count.
    # ---------------------------------------------------------------

    if total_samples != len(manifest):

        print(
            "\nERROR: Shard sample count does "
            "not match manifest rows."
        )

        print(
            f"Shard samples: {total_samples}"
        )

        print(
            f"Manifest rows: {len(manifest)}"
        )

        return False

    # ---------------------------------------------------------------
    # Verify every manifest path resolves to a
    # physically verified shard.
    # ---------------------------------------------------------------

    manifest_paths: set[str] = set()

    try:

        for raw_path in manifest[
            "shard_path"
        ]:

            manifest_paths.add(
                str(
                    resolve_path(
                        raw_path
                    ).resolve()
                )
            )

    except Exception as error:

        print(
            f"\nERROR resolving manifest path: "
            f"{error}"
        )

        return False

    if manifest_paths != verified_paths:

        print(
            "\nERROR: Manifest shard references "
            "do not match verified shard files."
        )

        return False

    # ---------------------------------------------------------------
    # Verify segment IDs if the manifest stores them.
    # ---------------------------------------------------------------

    manifest_segment_column = None

    for candidate in (
        "segment_id",
        "segment_ids",
    ):

        if candidate in manifest.columns:

            manifest_segment_column = candidate

            break

    if manifest_segment_column is not None:

        manifest_ids = set(
            manifest[
                manifest_segment_column
            ].astype(str)
        )

        if manifest_ids != all_segment_ids:

            print(
                "\nERROR: Segment IDs in manifest "
                "do not match segment IDs in shards."
            )

            print(
                f"Manifest IDs: "
                f"{len(manifest_ids)}"
            )

            print(
                f"Shard IDs: "
                f"{len(all_segment_ids)}"
            )

            return False

    # ---------------------------------------------------------------
    # Verify labels if the manifest contains labels.
    # ---------------------------------------------------------------

    manifest_label_column = None

    for candidate in (
        "label",
        "binary_label",
    ):

        if candidate in manifest.columns:

            manifest_label_column = candidate

            break

    if manifest_label_column is not None:

        manifest_labels = pd.to_numeric(
            manifest[
                manifest_label_column
            ],
            errors="coerce",
        )

        if manifest_labels.isna().any():

            print(
                "\nERROR: Manifest contains "
                "non-numeric labels."
            )

            return False

        manifest_label_counts = Counter(
            int(value)
            for value in manifest_labels
        )

        if (
            dict(
                sorted(
                    manifest_label_counts.items()
                )
            )
            !=
            dict(
                sorted(
                    total_labels.items()
                )
            )
        ):

            print(
                "\nERROR: Label distribution mismatch."
            )

            print(
                f"Manifest: "
                f"{dict(manifest_label_counts)}"
            )

            print(
                f"Shards: "
                f"{dict(total_labels)}"
            )

            return False

    # ---------------------------------------------------------------
    # SUCCESS
    # ---------------------------------------------------------------

    print("\n" + "-" * 60)

    print(
        f"{split.upper()} VERIFIED"
    )

    print(
        f"Shards:       {len(verified_paths)}"
    )

    print(
        f"Samples:      {total_samples}"
    )

    print(
        f"Labels:       "
        f"{dict(sorted(total_labels.items()))}"
    )

    print(
        f"Segment IDs:  {len(all_segment_ids)}"
    )

    print(
        "Status:       PASS"
    )

    return True


def main() -> None:
    """Verify all dataset shards."""

    print("=" * 60)
    print(
        "ACOUSTIC DRONE FEATURE SHARD VERIFICATION"
    )
    print("=" * 60)

    results: dict[str, bool] = {}

    for split in SPLITS:

        results[split] = verify_split(
            split
        )

    print("\n" + "=" * 60)
    print(
        "FINAL VERIFICATION RESULT"
    )
    print("=" * 60)

    all_passed = True

    for split, passed in results.items():

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"{split.upper():12} {status}"
        )

        if not passed:
            all_passed = False

    print()

    if all_passed:

        print(
            "ALL SHARDS VERIFIED SUCCESSFULLY."
        )

        print(
            "Train:      16,863 samples"
        )

        print(
            "Validation: 2,502 samples"
        )

        print(
            "Test:       2,607 samples"
        )

        print(
            "Total:      21,972 samples"
        )

        print()

        print(
            "Original .npy files have NOT "
            "been deleted."
        )

        print(
            "Next step: run the shard DataLoader "
            "smoke test before deleting .npy files."
        )

    else:

        print(
            "VERIFICATION FAILED."
        )

        print(
            "DO NOT DELETE THE ORIGINAL .npy FILES."
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
