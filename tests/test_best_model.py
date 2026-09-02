"""Evaluate the best saved Acoustic Drone Recognition model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Project imports
# ============================================================

from configs.training_config import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)

from dataset.data_loader import create_dataloader

from models.acoustic_drone_model import AcousticDroneModel

from utils.checkpoint import load_checkpoint


# ============================================================
# Paths
# ============================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "best_model.pt"
)

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
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    correct: int,
    total: int,
) -> tuple[float, float, float, float]:
    """Calculate accuracy, precision, recall and F1."""

    accuracy = (
        100.0 * correct / total
        if total > 0
        else 0.0
    )

    precision = (
        true_positive
        / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate model and return complete classification metrics."""

    model.eval()

    running_loss = 0.0

    correct = 0
    total = 0

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():

        for batch_index, batch in enumerate(dataloader):

            # ------------------------------------------------
            # Move tensors to device
            # ------------------------------------------------

            features = {
                name: batch[name].to(device)
                for name in (
                    "mfcc",
                    "mel",
                    "spectral",
                    "chroma",
                    "zcr",
                    "energy",
                )
            }

            labels = batch["label"].to(device)

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            output = model(features)

            # ------------------------------------------------
            # Support models returning:
            #
            #   logits
            #
            # or
            #
            #   (logits, attention)
            # ------------------------------------------------

            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            loss = criterion(
                logits,
                labels,
            )

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            predictions = logits.argmax(
                dim=1
            )

            # ------------------------------------------------
            # Overall accuracy
            # ------------------------------------------------

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            # ------------------------------------------------
            # Confusion matrix components
            #
            # Class 0 = Background
            # Class 1 = Drone
            # ------------------------------------------------

            true_positive += (
                (
                    (predictions == 1)
                    & (labels == 1)
                )
                .sum()
                .item()
            )

            true_negative += (
                (
                    (predictions == 0)
                    & (labels == 0)
                )
                .sum()
                .item()
            )

            false_positive += (
                (
                    (predictions == 1)
                    & (labels == 0)
                )
                .sum()
                .item()
            )

            false_negative += (
                (
                    (predictions == 0)
                    & (labels == 1)
                )
                .sum()
                .item()
            )

            running_loss += loss.item()

            if (
                batch_index + 1
            ) % 50 == 0:

                print(
                    f"Evaluated "
                    f"{batch_index + 1}/"
                    f"{len(dataloader)} batches..."
                )

    average_loss = (
        running_loss / len(dataloader)
        if len(dataloader) > 0
        else 0.0
    )

    accuracy, precision, recall, f1 = (
        calculate_metrics(
            true_positive,
            false_positive,
            false_negative,
            correct,
            total,
        )
    )

    background_recall = (
        true_negative
        / (true_negative + false_positive)
        if (true_negative + false_positive) > 0
        else 0.0
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "background_recall": background_recall,
        "total": total,
    }


# ============================================================
# Print results
# ============================================================

def print_results(
    name: str,
    results: dict,
) -> None:
    """Print evaluation results."""

    print()
    print("=" * 70)
    print(f"{name.upper()} RESULTS")
    print("=" * 70)

    print(
        f"Loss:              "
        f"{results['loss']:.6f}"
    )

    print(
        f"Accuracy:          "
        f"{results['accuracy']:.2f}%"
    )

    print(
        f"Drone Precision:   "
        f"{results['precision']:.4f}"
    )

    print(
        f"Drone Recall:      "
        f"{results['recall']:.4f}"
    )

    print(
        f"Drone F1:          "
        f"{results['f1']:.4f}"
    )

    print(
        f"Background Recall: "
        f"{results['background_recall']:.4f}"
    )

    print()
    print("Confusion Matrix")
    print("----------------")
    print(
        "                 Predicted"
    )
    print(
        "              Background  Drone"
    )

    print(
        f"Actual "
        f"Background   "
        f"{results['true_negative']:10d}  "
        f"{results['false_positive']:5d}"
    )

    print(
        f"Actual "
        f"Drone        "
        f"{results['false_negative']:10d}  "
        f"{results['true_positive']:5d}"
    )

    print()
    print(
        f"False positives: "
        f"{results['false_positive']}"
    )

    print(
        f"False negatives: "
        f"{results['false_negative']}"
    )

    print(
        f"Total samples:   "
        f"{results['total']}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Run validation and test evaluation."""

    print("=" * 70)
    print(
        "ACOUSTIC DRONE BEST MODEL EVALUATION"
    )
    print("=" * 70)

    print(
        f"Device:     {DEVICE}"
    )

    print(
        f"Checkpoint: {CHECKPOINT}"
    )

    print(
        f"Validation: {VALIDATION_MANIFEST}"
    )

    print(
        f"Test:       {TEST_MANIFEST}"
    )

    # --------------------------------------------------------
    # Verify files
    # --------------------------------------------------------

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Best checkpoint not found:\n"
            f"{CHECKPOINT}"
        )

    if not VALIDATION_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Validation manifest not found:\n"
            f"{VALIDATION_MANIFEST}"
        )

    if not TEST_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Test manifest not found:\n"
            f"{TEST_MANIFEST}"
        )

    # --------------------------------------------------------
    # Validation DataLoader
    # --------------------------------------------------------

    print()
    print(
        "Creating validation DataLoader..."
    )

    validation_loader = create_dataloader(
        manifest_path=VALIDATION_MANIFEST,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        validate_features=True,
    )

    print(
        f"Validation samples: "
        f"{len(validation_loader.dataset)}"
    )

    print(
        f"Validation batches: "
        f"{len(validation_loader)}"
    )

    # --------------------------------------------------------
    # Test DataLoader
    # --------------------------------------------------------

    print()
    print(
        "Creating test DataLoader..."
    )

    test_loader = create_dataloader(
        manifest_path=TEST_MANIFEST,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        validate_features=True,
    )

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    print(
        f"Test batches: "
        f"{len(test_loader)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print(
        "Building AcousticDroneModel..."
    )

    model = AcousticDroneModel().to(
        DEVICE
    )

    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters())}"
    )

    # --------------------------------------------------------
    # Optimizer and scheduler
    #
    # They are required because the project's checkpoint
    # stores complete training state.
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50,
        eta_min=1e-6,
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print()
    print(
        "Loading best checkpoint..."
    )

    epoch, best_accuracy = load_checkpoint(
        CHECKPOINT,
        model,
        optimizer,
        scheduler,
    )

    print(
        f"Checkpoint epoch: "
        f"{epoch + 1}"
    )

    print(
        f"Saved best validation accuracy: "
        f"{best_accuracy:.2f}%"
    )

    # --------------------------------------------------------
    # Criterion
    #
    # For evaluation we use ordinary CrossEntropyLoss.
    # The class weighting is a training strategy and does
    # not change the classification predictions.
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Validation evaluation
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "EVALUATING VALIDATION SET"
    )

    print(
        "=" * 70
    )

    validation_results = evaluate(
        model,
        validation_loader,
        criterion,
        DEVICE,
    )

    print_results(
        "Validation",
        validation_results,
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "EVALUATING TEST SET"
    )

    print(
        "=" * 70
    )

    test_results = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE,
    )

    print_results(
        "Test",
        test_results,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "EVALUATION COMPLETE"
    )
    print("=" * 70)

    print(
        f"Validation F1: "
        f"{validation_results['f1']:.4f}"
    )

    print(
        f"Test F1:       "
        f"{test_results['f1']:.4f}"
    )

    print(
        f"Validation accuracy: "
        f"{validation_results['accuracy']:.2f}%"
    )

    print(
        f"Test accuracy:       "
        f"{test_results['accuracy']:.2f}%"
    )

    print()

    print(
        "Next step: use these results to decide "
        "whether additional training is worthwhile."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()