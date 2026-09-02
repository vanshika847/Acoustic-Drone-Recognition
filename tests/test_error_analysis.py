"""
Test-set error analysis for the Acoustic Drone Recognition System.

This script evaluates the frozen best checkpoint on the held-out test set
and identifies:

    - False positives:
        Actual background -> predicted drone

    - False negatives:
        Actual drone -> predicted background

Outputs are written to:

    outputs/error_analysis/

Files:

    test_predictions.csv
    false_positives.csv
    false_negatives.csv

The test set is used ONLY for evaluation/error analysis.

No model parameters are changed.
No optimizer is created.
No training is performed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Project imports
# ============================================================

from configs.training_config import (  # noqa: E402
    BATCH_SIZE,
    DEVICE,
    NUM_WORKERS,
)

from dataset.data_loader import create_dataloader  # noqa: E402

from models.network import AcousticDroneNet  # noqa: E402


# ============================================================
# Paths
# ============================================================

TEST_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "features"
    / "test_shard_manifest.csv"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "best_model.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "error_analysis"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "test_predictions.csv"
)

FALSE_POSITIVES_FILE = (
    OUTPUT_DIR
    / "false_positives.csv"
)

FALSE_NEGATIVES_FILE = (
    OUTPUT_DIR
    / "false_negatives.csv"
)


# ============================================================
# Configuration
# ============================================================

device = DEVICE


# ============================================================
# Helper functions
# ============================================================

def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> AcousticDroneNet:
    """
    Build the model and load the frozen best checkpoint.
    """

    print()
    print("=" * 70)
    print("BUILDING MODEL")
    print("=" * 70)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device:     {device}")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Best checkpoint not found: {checkpoint_path}"
        )

    model = AcousticDroneNet().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch", "unknown"),
    )

    print(
        "Saved best validation accuracy:",
        f"{checkpoint.get('best_accuracy', 0.0):.2f}%",
    )

    print(
        "Model parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
    )

    return model


def extract_model_outputs(
    outputs,
) -> torch.Tensor:
    """
    Extract logits from the model output.

    The current AcousticDroneModel returns:

        logits, attention

    This helper also supports a model that returns logits directly.
    """

    if isinstance(outputs, tuple):
        return outputs[0]

    if isinstance(outputs, list):
        return outputs[0]

    if isinstance(outputs, dict):
        if "logits" not in outputs:
            raise ValueError(
                "Model returned a dictionary without "
                "'logits'."
            )

        return outputs["logits"]

    return outputs


def calculate_metrics(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
) -> dict[str, float]:
    """
    Calculate binary classification metrics.
    """

    total = (
        true_positive
        + false_positive
        + false_negative
        + true_negative
    )

    accuracy = (
        100.0
        * (
            true_positive
            + true_negative
        )
        / total
        if total > 0
        else 0.0
    )

    precision = (
        true_positive
        / (
            true_positive
            + false_positive
        )
        if (
            true_positive
            + false_positive
        ) > 0
        else 0.0
    )

    recall = (
        true_positive
        / (
            true_positive
            + false_negative
        )
        if (
            true_positive
            + false_negative
        ) > 0
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        ) > 0
        else 0.0
    )

    false_positive_rate = (
        false_positive
        / (
            false_positive
            + true_negative
        )
        if (
            false_positive
            + true_negative
        ) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
    }


# ============================================================
# Main analysis
# ============================================================

def run_error_analysis() -> None:
    """
    Run complete test-set error analysis.
    """

    print()
    print("=" * 70)
    print("ACOUSTIC DRONE TEST-SET ERROR ANALYSIS")
    print("=" * 70)

    print(f"Device:     {device}")
    print(f"Test:       {TEST_MANIFEST}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Output:     {OUTPUT_DIR}")

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not TEST_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Test manifest not found:\n{TEST_MANIFEST}"
        )

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Best checkpoint not found:\n{CHECKPOINT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Create test DataLoader
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CREATING TEST DATALOADER")
    print("-" * 70)

    test_loader = create_dataloader(
        manifest_path=TEST_MANIFEST,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        validate_features=True,
    )

    print(
        f"Test samples: {len(test_loader.dataset)}"
    )

    print(
        f"Test batches: {len(test_loader)}"
    )

    # --------------------------------------------------------
    # Load frozen model
    # --------------------------------------------------------

    model = load_model(
        CHECKPOINT,
        device,
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING TEST-SET INFERENCE")
    print("=" * 70)

    all_rows: list[dict] = []

    total_loss = 0.0
    total_samples = 0

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    criterion = torch.nn.CrossEntropyLoss(
        reduction="sum"
    )

    with torch.no_grad():

        progress_bar = tqdm(
            test_loader,
            desc="Evaluating test set",
        )

        for batch_index, batch in enumerate(
            progress_bar,
            start=1,
        ):

            # ------------------------------------------------
            # Move tensor features to device
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

            outputs = model(features)

            logits = extract_model_outputs(
                outputs
            )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                logits,
                labels,
            )

            total_loss += loss.item()

            # ------------------------------------------------
            # Probabilities
            # ------------------------------------------------

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            drone_probabilities = (
                probabilities[:, 1]
            )

            background_probabilities = (
                probabilities[:, 0]
            )

            predictions = logits.argmax(
                dim=1
            )

            # ------------------------------------------------
            # Batch statistics
            # ------------------------------------------------

            batch_size = labels.size(0)

            total_samples += batch_size

            true_positive += (
                (
                    (predictions == 1)
                    & (labels == 1)
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

            true_negative += (
                (
                    (predictions == 0)
                    & (labels == 0)
                )
                .sum()
                .item()
            )

            # ------------------------------------------------
            # Store sample-level predictions
            # ------------------------------------------------

            segment_ids = batch[
                "segment_id"
            ]

            labels_cpu = labels.cpu().tolist()

            predictions_cpu = (
                predictions.cpu().tolist()
            )

            drone_probabilities_cpu = (
                drone_probabilities.cpu().tolist()
            )

            background_probabilities_cpu = (
                background_probabilities.cpu().tolist()
            )

            probabilities_cpu = (
                probabilities.cpu().tolist()
            )

            for index in range(batch_size):

                true_label = int(
                    labels_cpu[index]
                )

                predicted_label = int(
                    predictions_cpu[index]
                )

                drone_probability = float(
                    drone_probabilities_cpu[index]
                )

                background_probability = float(
                    background_probabilities_cpu[index]
                )

                confidence = max(
                    drone_probability,
                    background_probability,
                )

                if (
                    true_label == 0
                    and predicted_label == 1
                ):
                    error_type = (
                        "false_positive"
                    )

                elif (
                    true_label == 1
                    and predicted_label == 0
                ):
                    error_type = (
                        "false_negative"
                    )

                elif (
                    true_label == 1
                    and predicted_label == 1
                ):
                    error_type = (
                        "true_positive"
                    )

                else:
                    error_type = (
                        "true_negative"
                    )

                all_rows.append(
                    {
                        "segment_id": segment_ids[index],
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "background_probability": (
                            background_probability
                        ),
                        "drone_probability": (
                            drone_probability
                        ),
                        "confidence": confidence,
                        "error_type": error_type,
                        "correct": (
                            true_label
                            == predicted_label
                        ),
                    }
                )

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    predictions_df = pd.DataFrame(
        all_rows
    )

    if predictions_df.empty:
        raise RuntimeError(
            "No predictions were generated."
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
    )

    average_loss = (
        total_loss / total_samples
        if total_samples > 0
        else 0.0
    )

    false_positives_df = (
        predictions_df[
            predictions_df["error_type"]
            == "false_positive"
        ]
        .copy()
    )

    false_negatives_df = (
        predictions_df[
            predictions_df["error_type"]
            == "false_negative"
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Sort errors by confidence
    # --------------------------------------------------------

    # False positives with highest drone confidence first.
    false_positives_df = (
        false_positives_df.sort_values(
            by="drone_probability",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # False negatives with lowest drone confidence first.
    #
    # A false negative with a drone probability close to 0
    # is a very confident miss.
    false_negatives_df = (
        false_negatives_df.sort_values(
            by="drone_probability",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save complete prediction file
    # --------------------------------------------------------

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Save false positives
    # --------------------------------------------------------

    false_positives_df.to_csv(
        FALSE_POSITIVES_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Save false negatives
    # --------------------------------------------------------

    false_negatives_df.to_csv(
        FALSE_NEGATIVES_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST-SET ERROR ANALYSIS RESULTS")
    print("=" * 70)

    print(
        f"Total samples:       {total_samples}"
    )

    print(
        f"Average loss:        {average_loss:.6f}"
    )

    print(
        f"Accuracy:             "
        f"{metrics['accuracy']:.2f}%"
    )

    print(
        f"Drone precision:      "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Drone recall:         "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"Drone F1:             "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Background recall:    "
        f"{true_negative / (true_negative + false_positive):.4f}"
        if (
            true_negative
            + false_positive
        ) > 0
        else "Background recall:    0.0000"
    )

    print(
        f"False-positive rate:   "
        f"{metrics['false_positive_rate']:.4f}"
    )

    print()
    print("-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)

    print()
    print(
        "                 Predicted"
    )

    print(
        "              Background  Drone"
    )

    print(
        f"Actual Background"
        f"    {true_negative:8d}"
        f" {false_positive:8d}"
    )

    print(
        f"Actual Drone"
        f"          {false_negative:8d}"
        f" {true_positive:8d}"
    )

    print()
    print("-" * 70)
    print("ERROR COUNTS")
    print("-" * 70)

    print(
        f"False positives:      "
        f"{len(false_positives_df)}"
    )

    print(
        f"False negatives:      "
        f"{len(false_negatives_df)}"
    )

    print(
        f"True positives:       "
        f"{true_positive}"
    )

    print(
        f"True negatives:       "
        f"{true_negative}"
    )

    # --------------------------------------------------------
    # False-positive analysis
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FALSE POSITIVES")
    print("=" * 70)

    print(
        "Actual background, predicted drone."
    )

    print()

    if false_positives_df.empty:

        print(
            "No false positives found."
        )

    else:

        print(
            f"Found {len(false_positives_df)} "
            "false positives."
        )

        print()
        print(
            "Top false positives by drone confidence:"
        )

        display_columns = [
            "segment_id",
            "drone_probability",
            "background_probability",
            "confidence",
        ]

        print(
            false_positives_df[
                display_columns
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # False-negative analysis
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FALSE NEGATIVES")
    print("=" * 70)

    print(
        "Actual drone, predicted background."
    )

    print()

    if false_negatives_df.empty:

        print(
            "No false negatives found."
        )

    else:

        print(
            f"Found {len(false_negatives_df)} "
            "false negatives."
        )

        print()
        print(
            "Most confident drone misses:"
        )

        display_columns = [
            "segment_id",
            "drone_probability",
            "background_probability",
            "confidence",
        ]

        print(
            false_negatives_df[
                display_columns
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Confidence analysis
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ERROR CONFIDENCE ANALYSIS")
    print("=" * 70)

    if not false_positives_df.empty:

        fp_mean = (
            false_positives_df[
                "drone_probability"
            ].mean()
        )

        fp_min = (
            false_positives_df[
                "drone_probability"
            ].min()
        )

        fp_max = (
            false_positives_df[
                "drone_probability"
            ].max()
        )

        print()
        print(
            "False-positive drone probability:"
        )

        print(
            f"  Mean: {fp_mean:.4f}"
        )

        print(
            f"  Min:  {fp_min:.4f}"
        )

        print(
            f"  Max:  {fp_max:.4f}"
        )

    if not false_negatives_df.empty:

        fn_mean = (
            false_negatives_df[
                "drone_probability"
            ].mean()
        )

        fn_min = (
            false_negatives_df[
                "drone_probability"
            ].min()
        )

        fn_max = (
            false_negatives_df[
                "drone_probability"
            ].max()
        )

        print()
        print(
            "False-negative drone probability:"
        )

        print(
            f"  Mean: {fn_mean:.4f}"
        )

        print(
            f"  Min:  {fn_min:.4f}"
        )

        print(
            f"  Max:  {fn_max:.4f}"
        )

    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print()
    print(
        f"All predictions:"
        f"\n  {PREDICTIONS_FILE}"
    )

    print()
    print(
        f"False positives:"
        f"\n  {FALSE_POSITIVES_FILE}"
    )

    print()
    print(
        f"False negatives:"
        f"\n  {FALSE_NEGATIVES_FILE}"
    )

    print()
    print("=" * 70)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        "The test set was used only for inference."
    )

    print(
        "No model parameters were modified."
    )

    print()
    print(
        "Next step: inspect the false-positive and "
        "false-negative CSV files to identify "
        "systematic failure patterns."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    try:

        run_error_analysis()

    except KeyboardInterrupt:

        print()
        print(
            "Error analysis interrupted by user."
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("ERROR ANALYSIS FAILED")
        print("=" * 70)

        print()
        print(
            f"{type(error).__name__}: {error}"
        )

        raise