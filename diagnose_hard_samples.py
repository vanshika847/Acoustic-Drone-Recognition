"""Forensic diagnostic for extreme per-sample training losses.

READ-ONLY. This script never writes to, deletes, moves, or relabels any
dataset file, feature shard, manifest, or checkpoint. It only reads
existing artifacts and writes a NEW report under outputs/diagnostics/.

Place this file at the project root (next to train.py) and run:

    python diagnose_hard_samples.py
    python diagnose_hard_samples.py --recompute-loss
    python diagnose_hard_samples.py --checkpoint models/checkpoints/best_model.pt
    python diagnose_hard_samples.py --top-k 20

What it does
------------
1. Loads the train shard manifest (the exact CSV train.py trains on) and
   establishes the sample_index -> manifest row mapping EXACTLY the way
   train.py does (no assumptions -- this is verified, not asserted).
2. Loads a checkpoint (best_model.pt by default) and extracts:
     - the stored EMA-smoothed per-sample "hardness" array, if present
     - the stored decision_threshold, if present
3. Optionally (--recompute-loss, on by default if no hardness array is
   found) runs a single no-grad forward pass over the ENTIRE train
   manifest, in manifest order, with NO sampler and NO augmentation, to
   compute an exact, reproducible raw per-sample cross-entropy loss with
   the current model weights.
4. Cross-checks sample_index -> shard_path/shard_index -> actual shard
   content (label match, segment_id match) rather than assuming the
   mapping is correct.
5. Joins segment_id across:
     outputs/features/train_shard_manifest.csv       (what training reads)
     outputs/features/train_feature_manifest.csv      (adds processed wav path)
     datasets/processed/manifests/train_segments.csv  (adds original source
                                                         file, sha256, and
                                                         exact start/end
                                                         seconds)
   to recover the original recording and exact time range, when those
   manifests are still present on disk.
6. Computes basic feature-array health checks per hardest sample: NaN/Inf,
   per-channel near-zero variance (a strong prior for MFCC
   normalisation blow-up, since mfcc.py normalises per channel with
   std + 1e-8), and overall magnitude outliers relative to a
   nominally zero-mean/unit-variance feature space.
7. Flags likely duplicate/near-duplicate samples among the hardest set
   using a cheap rounded-feature hash.
8. Writes a single CSV report with one row per ranked hard sample,
   covering rank, sample_index, segment_id, binary_label, loss (raw
   and/or smoothed hardness), shard_path, shard_index, source dataset,
   source file, sha256, start/end seconds, is_padded, and health flags.

Everything above is descriptive. This script draws NO conclusions about
mislabeling; it surfaces evidence for a human (or a follow-up analysis
pass) to interpret.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Make the project importable when run from the project root.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.training_config import (  # noqa: E402
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DEVICE,
    NUM_WORKERS,
    PIN_MEMORY,
    TRAIN_MANIFEST,
)
from dataset.data_loader import _collate_fixed_features  # noqa: E402
from dataset.feature_dataset import FEATURE_NAMES, FeatureDataset  # noqa: E402
from models.acoustic_drone_model import AcousticDroneModel  # noqa: E402

from torch.utils.data import DataLoader  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics"
FEATURE_MANIFEST_PATH = PROJECT_ROOT / "outputs" / "features" / "train_feature_manifest.csv"
SEGMENT_MANIFEST_PATH = (
    PROJECT_ROOT / "datasets" / "processed" / "manifests" / "train_segments.csv"
)


# ---------------------------------------------------------------------------
# Step 1: load the exact manifest train.py trains on, in the exact
# row order train.py uses. sample_index == row position in this file.
# ---------------------------------------------------------------------------


def load_train_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Train shard manifest not found: {path}")

    manifest = pd.read_csv(path)

    required = {"split", "segment_id", "binary_label", "shard_path", "shard_index"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(
            "Train shard manifest missing required columns: "
            + ", ".join(sorted(missing))
        )

    manifest["binary_label"] = manifest["binary_label"].astype(int)
    manifest["shard_index"] = manifest["shard_index"].astype(int)
    manifest["segment_id"] = manifest["segment_id"].astype(str)

    return manifest.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: load a checkpoint and pull out whatever forensic signal it has.
# ---------------------------------------------------------------------------


def load_checkpoint_raw(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint '{checkpoint_path}' has no 'model_state_dict'."
        )

    return checkpoint


def extract_hardness(
    checkpoint: dict[str, Any],
    expected_len: int,
) -> np.ndarray | None:
    hardness = checkpoint.get("hardness")

    if hardness is None:
        print(
            "[info] Checkpoint does not contain a 'hardness' array. "
            "The smoothed per-sample hardness history is NOT available "
            "from this checkpoint alone."
        )
        return None

    hardness = np.asarray(hardness, dtype=np.float64)

    if hardness.ndim != 1:
        print(
            f"[warn] Checkpoint 'hardness' has unexpected shape "
            f"{hardness.shape}; ignoring it."
        )
        return None

    if len(hardness) != expected_len:
        print(
            f"[warn] Checkpoint 'hardness' length ({len(hardness)}) does not "
            f"match the current train manifest length ({expected_len}). "
            "This means the manifest has changed since this checkpoint was "
            "produced, OR this checkpoint is from a different training run. "
            "Do NOT trust index alignment between this hardness array and "
            "the current manifest -- it is being ignored."
        )
        return None

    return hardness


# ---------------------------------------------------------------------------
# Step 3: fresh, exact, reproducible per-sample loss with the current model.
# No sampler (so no duplicates/omissions), no augmentation, manifest order
# preserved so index i == manifest.iloc[i] with certainty.
# ---------------------------------------------------------------------------


class OrderedIndexedDataset(FeatureDataset):
    """FeatureDataset that yields its own manifest row index, unshuffled."""

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        sample["sample_index"] = int(index)
        return sample


def _collate_with_index(batch: list[dict]) -> dict:
    indices = torch.tensor(
        [int(sample["sample_index"]) for sample in batch],
        dtype=torch.long,
    )
    clean = []
    for sample in batch:
        item = dict(sample)
        item.pop("sample_index", None)
        clean.append(item)
    output = _collate_fixed_features(clean)
    output["sample_index"] = indices
    return output


@torch.no_grad()
def compute_fresh_losses(
    manifest_path: Path,
    model: nn.Module,
) -> np.ndarray:
    """Return an array of raw per-sample CE loss, aligned to manifest rows."""

    dataset = OrderedIndexedDataset(manifest_path, validate_features=True)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        collate_fn=_collate_with_index,
    )

    losses = np.full(len(dataset), np.nan, dtype=np.float64)

    model.eval()

    for batch in loader:
        indices = batch.pop("sample_index").cpu().numpy()
        labels = batch["label"].to(DEVICE)

        features = {name: batch[name].to(DEVICE) for name in FEATURE_NAMES}
        output = model(features)
        logits = output[0] if isinstance(output, tuple) else output

        per_sample_loss = nn.functional.cross_entropy(
            logits,
            labels,
            reduction="none",
        )

        losses[indices] = per_sample_loss.detach().cpu().numpy()

    if np.isnan(losses).any():
        missing = int(np.isnan(losses).sum())
        print(
            f"[warn] {missing} samples were never scored during the fresh "
            "loss pass. This should not happen with drop_last=False and "
            "shuffle=False; investigate the DataLoader/dataset before "
            "trusting the ranking."
        )

    return losses


# ---------------------------------------------------------------------------
# Step 4/9: verify (not assume) that shard_index/shard_path correspond to the
# claimed sample, and pull raw feature-array health stats.
# ---------------------------------------------------------------------------


def resolve_shard_path(value: str, project_root: Path) -> Path:
    raw = Path(value)

    if raw.is_absolute() and raw.is_file():
        return raw

    if raw.is_absolute():
        parts = list(raw.parts)
        if "features" in parts:
            index = parts.index("features")
            candidate = project_root / Path(*parts[index:])
            if candidate.is_file():
                return candidate

    candidate = (project_root / raw).resolve()
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(f"Feature shard not found: '{value}'")


def inspect_sample(
    row: pd.Series,
    project_root: Path,
    shard_cache: dict[Path, Any],
) -> dict[str, Any]:
    """Open the shard read-only and verify + describe one sample."""

    result: dict[str, Any] = {
        "shard_resolved_path": "",
        "shard_label_matches_manifest": None,
        "shard_segment_id_matches_manifest": None,
        "has_nan_or_inf": None,
        "max_abs_feature_value": None,
        "min_channel_std": None,
        "min_channel_std_feature": None,
        "suspected_normalization_blowup": None,
        "feature_hash": "",
        "error": "",
    }

    try:
        shard_path = resolve_shard_path(str(row["shard_path"]), project_root)
        result["shard_resolved_path"] = str(shard_path)

        if shard_path not in shard_cache:
            shard_cache[shard_path] = np.load(shard_path, allow_pickle=False)
        shard = shard_cache[shard_path]

        shard_index = int(row["shard_index"])

        shard_label = int(shard["labels"][shard_index])
        result["shard_label_matches_manifest"] = (
            shard_label == int(row["binary_label"])
        )

        shard_segment_id = str(shard["segment_ids"][shard_index])
        result["shard_segment_id_matches_manifest"] = (
            shard_segment_id == str(row["segment_id"])
        )

        max_abs = 0.0
        min_channel_std = float("inf")
        min_channel_std_feature = ""
        any_nonfinite = False
        hash_parts = []

        for name in FEATURE_NAMES:
            array = np.asarray(shard[name][shard_index], dtype=np.float64)

            if not np.isfinite(array).all():
                any_nonfinite = True

            max_abs = max(max_abs, float(np.max(np.abs(array))))

            # Per-channel std (axis=1 is time). A near-zero std channel
            # is the fingerprint of a near-silent/padded/constant input
            # that a "(x - mean) / (std + 1e-8)" normalization can blow up.
            channel_std = np.std(array, axis=1)
            local_min = float(np.min(channel_std))
            if local_min < min_channel_std:
                min_channel_std = local_min
                min_channel_std_feature = name

            hash_parts.append(np.round(array, 2).tobytes())

        result["has_nan_or_inf"] = any_nonfinite
        result["max_abs_feature_value"] = max_abs
        result["min_channel_std"] = min_channel_std
        result["min_channel_std_feature"] = min_channel_std_feature

        # Heuristic, not a conclusion: ordinary standardized features are
        # usually within roughly +/-6. Values far beyond that, paired with
        # a very small pre-existing channel std, are consistent with a
        # normalization blow-up rather than genuine acoustic difficulty.
        result["suspected_normalization_blowup"] = bool(
            max_abs > 15.0 and min_channel_std < 1e-3
        )

        result["feature_hash"] = hashlib.sha1(b"".join(hash_parts)).hexdigest()[:16]

    except Exception as error:  # noqa: BLE001
        result["error"] = str(error)

    return result


# ---------------------------------------------------------------------------
# Step 5: recover original recording + exact time range via segment_id joins.
# ---------------------------------------------------------------------------


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        print(f"[info] Optional manifest not found, skipping join: {path}")
        return None
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def build_lineage_lookup() -> pd.DataFrame | None:
    feature_manifest = load_optional_csv(FEATURE_MANIFEST_PATH)
    segment_manifest = load_optional_csv(SEGMENT_MANIFEST_PATH)

    if feature_manifest is None and segment_manifest is None:
        return None

    lineage = None
    if feature_manifest is not None:
        keep_cols = [
            c
            for c in (
                "segment_id",
                "processed_relative_path",
                "segment_file_name",
            )
            if c in feature_manifest.columns
        ]
        lineage = feature_manifest[keep_cols].drop_duplicates("segment_id")

    if segment_manifest is not None:
        keep_cols = [
            c
            for c in (
                "segment_id",
                "source_dataset",
                "source_relative_path",
                "source_sha256",
                "recording_group_id",
                "start_seconds",
                "end_seconds",
                "source_duration_seconds",
                "is_padded",
            )
            if c in segment_manifest.columns
        ]
        segment_subset = segment_manifest[keep_cols].drop_duplicates("segment_id")

        lineage = (
            segment_subset
            if lineage is None
            else lineage.merge(segment_subset, on="segment_id", how="outer")
        )

    return lineage


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "best_model.pt",
        help="Checkpoint to load for fresh-loss recomputation and hardness extraction.",
    )
    parser.add_argument(
        "--recompute-loss",
        action="store_true",
        help="Force a fresh no-grad forward pass over the whole train set, "
        "even if the checkpoint already contains a hardness array.",
    )
    parser.add_argument(
        "--no-recompute-loss",
        action="store_true",
        help="Skip the fresh forward pass even if no hardness array is found "
        "(rank by checkpoint hardness only).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of hardest samples to report, per ranking and per class.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("HARD-SAMPLE FORENSIC DIAGNOSTIC (read-only)")
    print("=" * 78)

    manifest = load_train_manifest(Path(TRAIN_MANIFEST))
    n = len(manifest)
    print(f"Train manifest: {TRAIN_MANIFEST}  ({n} rows)")

    checkpoint = load_checkpoint_raw(args.checkpoint)
    print(f"Checkpoint: {args.checkpoint}")

    hardness = extract_hardness(checkpoint, expected_len=n)

    do_recompute = args.recompute_loss or (hardness is None and not args.no_recompute_loss)

    fresh_losses: np.ndarray | None = None
    if do_recompute:
        print("Running fresh no-grad forward pass over the full train manifest...")
        model = AcousticDroneModel().to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        fresh_losses = compute_fresh_losses(Path(TRAIN_MANIFEST), model)
    else:
        print("Skipping fresh forward pass (using checkpoint hardness only).")

    # Choose the ranking signal: prefer fresh raw loss (exact, reproducible,
    # tied to the actual checkpoint weights) but always carry hardness too
    # when available, so both can be reported side by side.
    if fresh_losses is not None:
        rank_signal = fresh_losses
        rank_signal_name = "fresh_raw_loss"
    elif hardness is not None:
        rank_signal = hardness
        rank_signal_name = "checkpoint_hardness"
    else:
        raise RuntimeError(
            "No ranking signal available: checkpoint had no 'hardness' array "
            "and fresh-loss recomputation was disabled. Re-run without "
            "--no-recompute-loss."
        )

    print(f"Ranking by: {rank_signal_name}")

    labels = manifest["binary_label"].to_numpy()

    lineage = build_lineage_lookup()

    shard_cache: dict[Path, Any] = {}

    def build_rows(order: np.ndarray, group_label: str) -> list[dict[str, Any]]:
        rows = []
        for rank, idx in enumerate(order, start=1):
            row = manifest.iloc[idx]
            health = inspect_sample(row, PROJECT_ROOT, shard_cache)

            record: dict[str, Any] = {
                "group": group_label,
                "rank": rank,
                "sample_index": int(idx),
                "segment_id": row["segment_id"],
                "binary_label": int(row["binary_label"]),
                "fresh_raw_loss": (
                    float(fresh_losses[idx]) if fresh_losses is not None else ""
                ),
                "checkpoint_hardness": (
                    float(hardness[idx]) if hardness is not None else ""
                ),
                "shard_path": row["shard_path"],
                "shard_index": int(row["shard_index"]),
                **health,
            }

            if lineage is not None:
                match = lineage[lineage["segment_id"] == row["segment_id"]]
                if not match.empty:
                    for col in match.columns:
                        if col == "segment_id":
                            continue
                        record[col] = match.iloc[0][col]

            rows.append(record)
        return rows

    overall_order = np.argsort(-rank_signal)[: args.top_k]
    drone_mask = labels == 1
    background_mask = labels == 0

    drone_order = np.argsort(-np.where(drone_mask, rank_signal, -np.inf))[: args.top_k]
    background_order = np.argsort(-np.where(background_mask, rank_signal, -np.inf))[
        : args.top_k
    ]

    all_rows = (
        build_rows(overall_order, "overall_top_k")
        + build_rows(drone_order, "hardest_drone")
        + build_rows(background_order, "hardest_background")
    )

    report = pd.DataFrame(all_rows)

    # Flag duplicate/near-duplicate feature content within the reported set.
    if "feature_hash" in report.columns:
        report["possible_duplicate_in_report"] = report["feature_hash"].duplicated(
            keep=False
        ) & (report["feature_hash"] != "")

    report_path = OUTPUT_DIR / "hard_sample_forensics.csv"
    report.to_csv(report_path, index=False)

    for shard in shard_cache.values():
        try:
            shard.close()
        except Exception:  # noqa: BLE001
            pass

    print()
    print(f"Wrote {len(report)} rows to: {report_path}")
    print(
        "This file was written UNDER outputs/diagnostics/. "
        "No dataset, shard, manifest, or checkpoint file was modified."
    )


if __name__ == "__main__":
    main()
