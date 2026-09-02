"""Single-segment and batch inference for the acoustic drone detector."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from configs.training_config import (
    DEFAULT_DECISION_THRESHOLD,
    DEVICE,
)
from dataset.data_loader import _collate_fixed_features
from dataset.feature_dataset import (
    FEATURE_NAMES,
    FeatureDataset,
)
from models.acoustic_drone_model import AcousticDroneModel


def load_model(
    checkpoint_path: str | Path,
    device: torch.device = DEVICE,
) -> tuple[AcousticDroneModel, float]:
    """Load the trained detector and its saved decision threshold."""

    checkpoint_path = Path(
        checkpoint_path
    ).resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

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

    model = AcousticDroneModel().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    threshold = float(
        checkpoint.get(
            "decision_threshold",
            DEFAULT_DECISION_THRESHOLD,
        )
    )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "Invalid decision threshold in checkpoint: "
            f"{threshold}"
        )

    return model, threshold


def resolve_threshold(
    saved_threshold: float,
    override_threshold: float | None,
) -> float:
    """Resolve the threshold used for inference."""

    if override_threshold is None:
        return saved_threshold

    threshold = float(
        override_threshold
    )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "Inference threshold must be between "
            "0 and 1."
        )

    return threshold


@torch.no_grad()
def predict_batch(
    model: AcousticDroneModel,
    batch: dict[str, Any],
    *,
    threshold: float,
    device: torch.device = DEVICE,
) -> dict[str, Any]:
    """Predict a collated batch."""

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    features = {
        name: batch[name].to(
            device,
            non_blocking=True,
        )
        for name in FEATURE_NAMES
    }

    logits, attention = model(features)

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[:, 1]

    background_probability = (
        1.0 - probabilities
    )

    is_drone = (
        probabilities >= threshold
    )

    # Positive values mean the sample is more strongly
    # classified as drone relative to the decision threshold.
    threshold_margin = (
        probabilities - threshold
    )

    # Confidence of the selected class.
    confidence = torch.maximum(
        probabilities,
        background_probability,
    )

    return {
        "drone_probability": (
            probabilities.cpu()
        ),
        "background_probability": (
            background_probability.cpu()
        ),
        "is_drone": (
            is_drone.cpu()
        ),
        "confidence": (
            confidence.cpu()
        ),
        "threshold_margin": (
            threshold_margin.cpu()
        ),
        "attention_weights": (
            attention.cpu()
        ),
        "logits": logits.cpu(),
    }


def predict_manifest_index(
    manifest_path: str | Path,
    sample_index: int,
    checkpoint_path: str | Path,
    *,
    threshold_override: float | None = None,
    device: torch.device = DEVICE,
) -> dict[str, Any]:
    """Run inference for one row of a feature manifest."""

    dataset = FeatureDataset(
        manifest_path,
        validate_features=True,
    )

    if sample_index < 0:
        raise IndexError(
            f"sample_index {sample_index} "
            "cannot be negative."
        )

    if sample_index >= len(dataset):
        raise IndexError(
            f"sample_index {sample_index} "
            f"is outside dataset bounds "
            f"[0, {len(dataset) - 1}]."
        )

    sample = dataset[
        sample_index
    ]

    batch = _collate_fixed_features(
        [sample]
    )

    model, saved_threshold = load_model(
        checkpoint_path,
        device,
    )

    threshold = resolve_threshold(
        saved_threshold,
        threshold_override,
    )

    result = predict_batch(
        model,
        batch,
        threshold=threshold,
        device=device,
    )

    drone_probability = float(
        result[
            "drone_probability"
        ][0].item()
    )

    background_probability = float(
        result[
            "background_probability"
        ][0].item()
    )

    confidence = float(
        result[
            "confidence"
        ][0].item()
    )

    threshold_margin = float(
        result[
            "threshold_margin"
        ][0].item()
    )

    is_drone = bool(
        result[
            "is_drone"
        ][0].item()
    )

    return {
        "segment_id": sample[
            "segment_id"
        ],
        "manifest_index": int(
            sample_index
        ),
        "true_label": int(
            sample["label"].item()
        ),
        "drone_probability": (
            drone_probability
        ),
        "background_probability": (
            background_probability
        ),
        "confidence": confidence,
        "threshold": threshold,
        "saved_threshold": (
            saved_threshold
        ),
        "threshold_margin": (
            threshold_margin
        ),
        "is_drone": is_drone,
        "prediction": (
            "DRONE"
            if is_drone
            else "BACKGROUND"
        ),
        "attention_weights": (
            result[
                "attention_weights"
            ][0].tolist()
        ),
    }


def print_prediction(
    result: dict[str, Any],
) -> None:
    """Print one prediction in a diagnostic-friendly format."""

    print("=" * 70)
    print("ACOUSTIC DRONE DETECTOR")
    print("=" * 70)

    print(
        f"Segment ID          : "
        f"{result['segment_id']}"
    )

    print(
        f"Manifest index      : "
        f"{result['manifest_index']}"
    )

    print(
        f"True label          : "
        f"{result['true_label']}"
    )

    print(
        f"Drone probability   : "
        f"{result['drone_probability']:.6f}"
    )

    print(
        f"Background prob.    : "
        f"{result['background_probability']:.6f}"
    )

    print(
        f"Confidence          : "
        f"{result['confidence']:.6f}"
    )

    print(
        f"Saved threshold     : "
        f"{result['saved_threshold']:.6f}"
    )

    print(
        f"Used threshold      : "
        f"{result['threshold']:.6f}"
    )

    print(
        f"Threshold margin    : "
        f"{result['threshold_margin']:+.6f}"
    )

    print(
        f"Prediction          : "
        f"{result['prediction']}"
    )

    print("")
    print("Feature attention:")
    print("-" * 40)

    for name, weight in zip(
        FEATURE_NAMES,
        result["attention_weights"],
    ):
        print(
            f"{name:12s}: {weight:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run acoustic drone detector inference "
            "on one manifest sample."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help="Feature shard manifest.",
    )

    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Manifest row index to classify.",
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pt",
        help="Trained detector checkpoint.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional threshold override. "
            "By default the threshold saved in the "
            "checkpoint is used."
        ),
    )

    args = parser.parse_args()

    result = predict_manifest_index(
        args.manifest,
        args.index,
        args.checkpoint,
        threshold_override=args.threshold,
        device=DEVICE,
    )

    print_prediction(
        result
    )


if __name__ == "__main__":
    main()