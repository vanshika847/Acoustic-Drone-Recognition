"""Diagnostic evaluation for the acoustic drone detector.

This evaluator is intentionally more detailed than the training-time metrics.

It:
    - evaluates validation or test data using the checkpoint's threshold;
    - reports confusion-matrix metrics;
    - searches for the threshold that maximises drone F1;
    - reports useful threshold/recall trade-offs;
    - identifies false-negative drone samples;
    - identifies borderline drone samples;
    - saves per-sample predictions to CSV;
    - saves false negatives separately for investigation.

IMPORTANT:
    This project uses shard-based feature manifests.

    Therefore evaluation MUST use:
        train_shard_manifest.csv
        validation_shard_manifest.csv
        test_shard_manifest.csv

    and NOT:
        *_feature_manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from configs.training_config import (
    BATCH_SIZE,
    DEVICE,
    NUM_WORKERS,
    PIN_MEMORY,
)

from dataset.data_loader import _collate_fixed_features
from dataset.feature_dataset import FeatureDataset
from models.acoustic_drone_model import AcousticDroneModel


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# SHARD MANIFESTS
# ============================================================

VALIDATION_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "features"
    / "validation_shard_manifest.csv"
)

TEST_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "features"
    / "test_shard_manifest.csv"
)


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = (
    "mfcc",
    "mel",
    "spectral",
    "chroma",
    "zcr",
    "energy",
)


# ============================================================
# MANIFEST VALIDATION
# ============================================================

REQUIRED_SHARD_COLUMNS = {
    "split",
    "segment_id",
    "binary_label",
    "shard_path",
    "shard_index",
}


def validate_shard_manifest(
    manifest: Path,
) -> None:
    """Validate that the supplied manifest is a shard manifest."""

    if not manifest.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest}"
        )

    try:
        dataframe = pd.read_csv(manifest, nrows=5)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read manifest: {manifest}\n"
            f"Original error: {exc}"
        ) from exc

    columns = set(dataframe.columns)

    missing = REQUIRED_SHARD_COLUMNS - columns

    if missing:
        raise ValueError(
            "The evaluation manifest is not a valid shard manifest.\n"
            f"Manifest: {manifest}\n"
            f"Missing columns: {', '.join(sorted(missing))}\n\n"
            "Expected shard manifest columns:\n"
            "  split\n"
            "  segment_id\n"
            "  binary_label\n"
            "  shard_path\n"
            "  shard_index\n\n"
            "Do NOT pass *_feature_manifest.csv here.\n"
            "Use *_shard_manifest.csv."
        )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_detector(
    checkpoint: str | Path,
    device: torch.device,
) -> tuple[AcousticDroneModel, float]:
    """Load detector weights and the stored decision threshold."""

    checkpoint = Path(checkpoint).resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    payload = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in payload:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    model = AcousticDroneModel().to(device)

    try:
        model.load_state_dict(
            payload["model_state_dict"]
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint weights are incompatible with the current "
            "AcousticDroneModel architecture.\n"
            f"Checkpoint: {checkpoint}\n"
            "This usually means the checkpoint was created with an "
            "older model architecture."
        ) from exc

    model.eval()

    threshold = float(
        payload.get(
            "decision_threshold",
            0.50,
        )
    )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"Invalid decision threshold in checkpoint: {threshold}"
        )

    return model, threshold


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calculate binary detector metrics at a probability threshold."""

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    if len(probabilities) != len(labels):
        raise ValueError(
            "Probabilities and labels must contain the same number "
            "of samples."
        )

    predictions = probabilities >= threshold

    tp = int(
        np.sum(
            (predictions == 1)
            & (labels == 1)
        )
    )

    fp = int(
        np.sum(
            (predictions == 1)
            & (labels == 0)
        )
    )

    tn = int(
        np.sum(
            (predictions == 0)
            & (labels == 0)
        )
    )

    fn = int(
        np.sum(
            (predictions == 0)
            & (labels == 1)
        )
    )

    total = len(labels)

    correct = tp + tn

    accuracy = (
        100.0 * correct / total
        if total
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy_percent": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


# ============================================================
# BEST THRESHOLD
# ============================================================

def find_best_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Find the threshold that maximises drone F1."""

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.linspace(
        0.05,
        0.95,
        181,
    ):
        metrics = calculate_metrics(
            probabilities,
            labels,
            float(threshold),
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)

    return best_threshold, max(0.0, best_f1)


# ============================================================
# PREDICTION COLLECTION
# ============================================================

def collect_predictions(
    model: AcousticDroneModel,
    dataset: FeatureDataset,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict],
    float,
]:
    """Run inference and collect per-sample diagnostic information."""

    all_probabilities: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    records: list[dict] = []

    criterion = nn.CrossEntropyLoss(
        reduction="none"
    )

    total_loss = 0.0
    total_samples = 0

    # DataLoader uses shuffle=False, therefore samples correspond
    # sequentially to dataset indices.
    dataset_cursor = 0

    with torch.no_grad():
        for batch in loader:
            features = {
                name: batch[name].to(
                    device,
                    non_blocking=True,
                )
                for name in FEATURE_NAMES
            }

            labels = batch["label"].to(device)

            logits, _ = model(features)

            losses = criterion(
                logits,
                labels,
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            probability_np = (
                probabilities
                .cpu()
                .numpy()
            )

            labels_np = (
                labels
                .cpu()
                .numpy()
            )

            losses_np = (
                losses
                .cpu()
                .numpy()
            )

            batch_size = len(labels_np)

            for offset in range(batch_size):
                dataset_index = (
                    dataset_cursor + offset
                )

                sample = dataset[
                    dataset_index
                ]

                record = {
                    "manifest_index": int(
                        dataset_index
                    ),
                    "segment_id": str(
                        sample.get(
                            "segment_id",
                            dataset_index,
                        )
                    ),
                    "true_label": int(
                        labels_np[offset]
                    ),
                    "drone_probability": float(
                        probability_np[offset]
                    ),
                    "background_probability": float(
                        1.0 - probability_np[offset]
                    ),
                    "sample_loss": float(
                        losses_np[offset]
                    ),
                }

                # Include useful shard metadata when exposed
                # by FeatureDataset.
                for key in (
                    "shard_path",
                    "shard_index",
                    "split",
                    "source",
                    "file_path",
                    "audio_path",
                ):
                    if key in sample:
                        value = sample[key]

                        if isinstance(
                            value,
                            torch.Tensor,
                        ):
                            if value.numel() == 1:
                                value = value.item()
                            else:
                                value = value.tolist()

                        record[key] = value

                records.append(record)

            all_probabilities.append(
                probability_np
            )

            all_labels.append(
                labels_np
            )

            total_loss += float(
                losses.sum().item()
            )

            total_samples += batch_size
            dataset_cursor += batch_size

    if not all_probabilities:
        raise ValueError(
            "Evaluation loader produced zero batches."
        )

    probabilities = np.concatenate(
        all_probabilities
    )

    labels = np.concatenate(
        all_labels
    )

    mean_loss = (
        total_loss / total_samples
        if total_samples
        else 0.0
    )

    return (
        probabilities,
        labels,
        records,
        mean_loss,
    )


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    title: str,
    metrics: dict[str, float],
) -> None:
    """Print metrics in a readable format."""

    print("")
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"Threshold   : {metrics['threshold']:.4f}"
    )

    print(
        f"Accuracy    : {metrics['accuracy_percent']:.2f}%"
    )

    print(
        f"Precision   : {metrics['precision']:.4f}"
    )

    print(
        f"Recall      : {metrics['recall']:.4f}"
    )

    print(
        f"Specificity : {metrics['specificity']:.4f}"
    )

    print(
        f"Drone F1    : {metrics['f1']:.4f}"
    )

    print(
        "Confusion   : "
        f"TP={int(metrics['tp'])}, "
        f"FP={int(metrics['fp'])}, "
        f"TN={int(metrics['tn'])}, "
        f"FN={int(metrics['fn'])}"
    )


# ============================================================
# THRESHOLD TABLE
# ============================================================

def print_threshold_table(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Show how recall/F1 change at useful operating thresholds."""

    print("")
    print("=" * 70)
    print("THRESHOLD TRADE-OFF")
    print("=" * 70)

    print(
        f"{'Threshold':>10} "
        f"{'Precision':>12} "
        f"{'Recall':>12} "
        f"{'F1':>12} "
        f"{'FP':>8} "
        f"{'FN':>8}"
    )

    print("-" * 70)

    for threshold in (
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ):
        metrics = calculate_metrics(
            probabilities,
            labels,
            threshold,
        )

        print(
            f"{threshold:10.2f} "
            f"{metrics['precision']:12.4f} "
            f"{metrics['recall']:12.4f} "
            f"{metrics['f1']:12.4f} "
            f"{int(metrics['fp']):8d} "
            f"{int(metrics['fn']):8d}"
        )


# ============================================================
# SAVE DIAGNOSTIC FILES
# ============================================================

def save_diagnostic_files(
    records: list[dict],
    probabilities: np.ndarray,
    labels: np.ndarray,
    checkpoint_threshold: float,
    checkpoint: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Save predictions, false negatives and borderline drones."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if len(records) != len(probabilities):
        raise ValueError(
            "Number of records does not match number of predictions."
        )

    if len(records) != len(labels):
        raise ValueError(
            "Number of records does not match number of labels."
        )

    dataframe = pd.DataFrame(records)

    dataframe["true_label"] = labels.astype(int)

    dataframe["checkpoint_threshold"] = float(
        checkpoint_threshold
    )

    dataframe[
        "predicted_label_at_checkpoint_threshold"
    ] = (
        probabilities >= checkpoint_threshold
    ).astype(int)

    dataframe["threshold_distance"] = (
        probabilities - checkpoint_threshold
    )

    dataframe["absolute_threshold_distance"] = (
        np.abs(
            probabilities
            - checkpoint_threshold
        )
    )

    dataframe["is_false_negative"] = (
        (dataframe["true_label"] == 1)
        & (
            dataframe[
                "predicted_label_at_checkpoint_threshold"
            ]
            == 0
        )
    )

    dataframe["is_false_positive"] = (
        (dataframe["true_label"] == 0)
        & (
            dataframe[
                "predicted_label_at_checkpoint_threshold"
            ]
            == 1
        )
    )

    # Sort difficult examples first.
    dataframe = dataframe.sort_values(
        by="sample_loss",
        ascending=False,
    )

    checkpoint_stem = checkpoint.stem

    all_predictions_path = (
        output_dir
        / f"{checkpoint_stem}_predictions.csv"
    )

    false_negative_path = (
        output_dir
        / f"{checkpoint_stem}_false_negatives.csv"
    )

    borderline_path = (
        output_dir
        / f"{checkpoint_stem}_borderline_drones.csv"
    )

    dataframe.to_csv(
        all_predictions_path,
        index=False,
    )

    false_negatives = dataframe[
        dataframe["is_false_negative"]
    ].copy()

    false_negatives = false_negatives.sort_values(
        by="drone_probability",
        ascending=True,
    )

    false_negatives.to_csv(
        false_negative_path,
        index=False,
    )

    # Actual drones whose probability is within 0.20 of
    # the checkpoint decision threshold.
    borderline_drones = dataframe[
        (dataframe["true_label"] == 1)
        & (
            dataframe[
                "absolute_threshold_distance"
            ]
            <= 0.20
        )
    ].copy()

    borderline_drones = borderline_drones.sort_values(
        by="drone_probability",
        ascending=True,
    )

    borderline_drones.to_csv(
        borderline_path,
        index=False,
    )

    return (
        all_predictions_path,
        false_negative_path,
        borderline_path,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    manifest: str | Path,
    checkpoint: str | Path,
    *,
    device: torch.device = DEVICE,
    output_dir: str | Path = "evaluation_results",
) -> dict[str, float]:
    """Evaluate a detector and save detailed diagnostics."""

    manifest = Path(manifest).resolve()
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir).resolve()

    # --------------------------------------------------------
    # Validate shard manifest before constructing dataset.
    # --------------------------------------------------------

    validate_shard_manifest(
        manifest
    )

    # --------------------------------------------------------
    # Load dataset.
    # --------------------------------------------------------

    dataset = FeatureDataset(
        manifest,
        validate_features=True,
    )

    if len(dataset) == 0:
        raise ValueError(
            f"Evaluation dataset is empty: {manifest}"
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        collate_fn=_collate_fixed_features,
    )

    # --------------------------------------------------------
    # Load checkpoint.
    # --------------------------------------------------------

    model, checkpoint_threshold = load_detector(
        checkpoint,
        device,
    )

    # --------------------------------------------------------
    # Run inference.
    # --------------------------------------------------------

    (
        probabilities,
        labels,
        records,
        mean_loss,
    ) = collect_predictions(
        model,
        dataset,
        loader,
        device,
    )

    # --------------------------------------------------------
    # Check sample counts.
    # --------------------------------------------------------

    if len(probabilities) != len(labels):
        raise RuntimeError(
            "Prediction/label count mismatch."
        )

    if len(probabilities) != len(dataset):
        raise RuntimeError(
            "Model produced a different number of predictions "
            "than dataset samples."
        )

    # --------------------------------------------------------
    # Checkpoint threshold metrics.
    # --------------------------------------------------------

    checkpoint_metrics = calculate_metrics(
        probabilities,
        labels,
        checkpoint_threshold,
    )

    # --------------------------------------------------------
    # Best F1 threshold.
    #
    # This is diagnostic only.
    # --------------------------------------------------------

    best_threshold, best_f1 = find_best_threshold(
        probabilities,
        labels,
    )

    best_threshold_metrics = calculate_metrics(
        probabilities,
        labels,
        best_threshold,
    )

    # --------------------------------------------------------
    # Save diagnostic files.
    # --------------------------------------------------------

    (
        all_predictions_path,
        false_negative_path,
        borderline_path,
    ) = save_diagnostic_files(
        records=records,
        probabilities=probabilities,
        labels=labels,
        checkpoint_threshold=checkpoint_threshold,
        checkpoint=checkpoint,
        output_dir=output_dir,
    )

    # --------------------------------------------------------
    # Reload diagnostic dataframe for console analysis.
    # --------------------------------------------------------

    dataframe = pd.read_csv(
        all_predictions_path
    )

    false_negatives = dataframe[
        dataframe["is_false_negative"]
    ].copy()

    false_positive_count = int(
        checkpoint_metrics["fp"]
    )

    false_negative_count = len(
        false_negatives
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("")
    print("=" * 70)
    print("ACOUSTIC DRONE DETECTOR — DIAGNOSTIC EVALUATION")
    print("=" * 70)

    print(
        f"Manifest    : {manifest}"
    )

    print(
        f"Checkpoint  : {checkpoint}"
    )

    print(
        f"Samples     : {len(labels)}"
    )

    print(
        f"Mean loss   : {mean_loss:.6f}"
    )

    print(
        f"Device      : {device}"
    )

    print_metrics(
        "CHECKPOINT THRESHOLD",
        checkpoint_metrics,
    )

    print_metrics(
        "BEST F1 THRESHOLD ON THIS DATASET",
        best_threshold_metrics,
    )

    print(
        f"\nBest threshold F1: {best_f1:.4f}"
    )

    print_threshold_table(
        probabilities,
        labels,
    )

    # ========================================================
    # ERROR ANALYSIS
    # ========================================================

    print("")
    print("=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    print(
        f"False negatives (missed drones): "
        f"{false_negative_count}"
    )

    print(
        f"False positives (background called drone): "
        f"{false_positive_count}"
    )

    if false_negative_count:
        print("")
        print("HARDEST MISSED DRONES")
        print("-" * 70)

        columns = [
            "manifest_index",
            "segment_id",
            "drone_probability",
            "sample_loss",
            "shard_path",
            "shard_index",
        ]

        available_columns = [
            column
            for column in columns
            if column in false_negatives.columns
        ]

        print(
            false_negatives[
                available_columns
            ]
            .head(20)
            .to_string(index=False)
        )

    else:
        print("")
        print(
            "No false-negative drones at the checkpoint threshold."
        )

    # ========================================================
    # DIAGNOSTIC FILES
    # ========================================================

    print("")
    print("=" * 70)
    print("DIAGNOSTIC FILES")
    print("=" * 70)

    print(
        f"All predictions   : {all_predictions_path}"
    )

    print(
        f"False negatives   : {false_negative_path}"
    )

    print(
        f"Borderline drones : {borderline_path}"
    )

    print("")
    print(
        "IMPORTANT: The best-F1 threshold shown here is diagnostic."
    )

    print(
        "Do not change the production threshold based only on "
        "the test set."
    )

    print(
        "For a final model, threshold selection should be done "
        "on validation."
    )

    return {
        "loss": float(mean_loss),
        **checkpoint_metrics,
        "best_threshold": float(best_threshold),
        "best_threshold_f1": float(best_f1),
        "false_negative_count": float(
            false_negative_count
        ),
        "false_positive_count": float(
            false_positive_count
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run detailed acoustic drone detector evaluation "
            "using shard-based feature manifests."
        )
    )

    parser.add_argument(
        "--split",
        choices=(
            "validation",
            "test",
        ),
        default="validation",
        help=(
            "Dataset split to evaluate. "
            "Default: validation."
        ),
    )

    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Optional explicit shard manifest path. "
            "Overrides --split."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default="models/checkpoints/best_model.pt",
        help="Detector checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help=(
            "Directory where diagnostic CSV files are saved."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Select manifest.
    # --------------------------------------------------------

    if args.manifest is not None:
        manifest = Path(
            args.manifest
        )
    elif args.split == "validation":
        manifest = VALIDATION_MANIFEST
    elif args.split == "test":
        manifest = TEST_MANIFEST
    else:
        raise ValueError(
            f"Unsupported split: {args.split}"
        )

    # --------------------------------------------------------
    # Print selected configuration.
    # --------------------------------------------------------

    print("")
    print("=" * 70)
    print("EVALUATION CONFIGURATION")
    print("=" * 70)

    print(
        f"Split       : {args.split}"
    )

    print(
        f"Manifest    : {manifest.resolve()}"
    )

    print(
        f"Checkpoint  : {Path(args.checkpoint).resolve()}"
    )

    print(
        f"Device      : {DEVICE}"
    )

    # --------------------------------------------------------
    # Evaluate.
    # --------------------------------------------------------

    evaluate(
        manifest,
        args.checkpoint,
        device=DEVICE,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()