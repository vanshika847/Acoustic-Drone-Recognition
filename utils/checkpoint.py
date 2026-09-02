"""Checkpoint utilities for acoustic drone detector training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


CHECKPOINT_VERSION = 2


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None,
    scheduler: LRScheduler | None,
    epoch: int,
    best_f1: float,
    *,
    decision_threshold: float = 0.50,
    hardness: Any = None,
    best_epoch: int | None = None,
    validation_metrics: dict[str, float] | None = None,
    training_metrics: dict[str, float] | None = None,
) -> None:
    """
    Save a complete detector checkpoint.

    The checkpoint contains:
        - model parameters
        - optimizer state
        - scheduler state
        - current epoch
        - best validation F1
        - decision threshold
        - hard-example hardness values
        - best epoch
        - latest validation metrics
        - latest training metrics

    Older checkpoints remain loadable.
    """

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not 0.0 < decision_threshold < 1.0:
        raise ValueError(
            "decision_threshold must be between 0 and 1."
        )

    payload: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "best_epoch": (
            int(best_epoch)
            if best_epoch is not None
            else int(epoch)
        ),
        "best_f1": float(best_f1),

        # Kept for compatibility with older evaluation code.
        "best_accuracy": float(best_f1),

        "decision_threshold": float(
            decision_threshold
        ),

        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = (
            optimizer.state_dict()
        )

    if scheduler is not None:
        payload["scheduler_state_dict"] = (
            scheduler.state_dict()
        )

    if hardness is not None:
        payload["hardness"] = hardness

    if validation_metrics is not None:
        payload["validation_metrics"] = {
            str(key): float(value)
            for key, value in validation_metrics.items()
        }

    if training_metrics is not None:
        payload["training_metrics"] = {
            str(key): float(value)
            for key, value in training_metrics.items()
        }

    torch.save(
        payload,
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> tuple[int, float]:
    """
    Load a checkpoint.

    Returns:
        (epoch, best_f1)
    """

    checkpoint = torch.load(
        Path(path),
        map_location="cpu",
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

    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if (
        scheduler is not None
        and "scheduler_state_dict" in checkpoint
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    epoch = int(
        checkpoint.get(
            "epoch",
            -1,
        )
    )

    best_f1 = float(
        checkpoint.get(
            "best_f1",
            checkpoint.get(
                "best_accuracy",
                0.0,
            ),
        )
    )

    return epoch, best_f1


def load_checkpoint_metadata(
    path: str | Path,
) -> dict[str, Any]:
    """
    Load checkpoint metadata without modifying a model.

    This function intentionally returns the additional training
    information when it exists, while remaining compatible with
    older checkpoints.
    """

    checkpoint = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=False,
    )

    return {
        "checkpoint_version": int(
            checkpoint.get(
                "checkpoint_version",
                1,
            )
        ),

        "epoch": int(
            checkpoint.get(
                "epoch",
                -1,
            )
        ),

        "best_epoch": int(
            checkpoint.get(
                "best_epoch",
                checkpoint.get(
                    "epoch",
                    -1,
                ),
            )
        ),

        "best_f1": float(
            checkpoint.get(
                "best_f1",
                checkpoint.get(
                    "best_accuracy",
                    0.0,
                ),
            )
        ),

        "decision_threshold": float(
            checkpoint.get(
                "decision_threshold",
                0.50,
            )
        ),

        "hardness": checkpoint.get(
            "hardness"
        ),

        "validation_metrics": checkpoint.get(
            "validation_metrics",
            {},
        ),

        "training_metrics": checkpoint.get(
            "training_metrics",
            {},
        ),
    }


__all__ = [
    "CHECKPOINT_VERSION",
    "save_checkpoint",
    "load_checkpoint",
    "load_checkpoint_metadata",
]