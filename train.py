"""Model training pipeline for the Acoustic Drone Recognition System."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from configs.training_config import (
    BATCH_SIZE,
    BEST_MODEL_NAME,
    CHECKPOINT_DIR,
    DEVICE,
    GRADIENT_CLIP,
    LAST_MODEL_NAME,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    PIN_MEMORY,
    RANDOM_SEED,
    SCHEDULER_MIN_LR,
    SCHEDULER_T_MAX,
    TRAIN_MANIFEST,
    VALIDATION_MANIFEST,
    WEIGHT_DECAY,
)

from dataset.data_loader import create_dataloader

from models.network import AcousticDroneNet

from utils.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)

# -------------------------------------------------------
# Logging
# -------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------
# Reproducibility
# -------------------------------------------------------

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# -------------------------------------------------------
# Device
# -------------------------------------------------------

device = DEVICE

logger.info("Training on device: %s", device)

def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
) -> tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model:
            AcousticDroneNet.

        dataloader:
            Training DataLoader.

        criterion:
            Loss function.

        optimizer:
            AdamW optimizer.

        device:
            CPU or CUDA.

    Returns:
        Average loss and accuracy.
    """

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False,
    )

    for batch in progress_bar:

        # ------------------------------------------
        # Move tensors to device
        # ------------------------------------------

        for key, value in batch.items():

            if isinstance(value, torch.Tensor):

                batch[key] = value.to(device)

        labels = batch["label"]

        # ------------------------------------------
        # Forward
        # ------------------------------------------

        optimizer.zero_grad()

        logits, _ = model(batch)

        loss = criterion(
            logits,
            labels,
        )

        # ------------------------------------------
        # Backward
        # ------------------------------------------

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP,
        )

        optimizer.step()

        # ------------------------------------------
        # Statistics
        # ------------------------------------------

        running_loss += loss.item()

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100 * correct / total:.2f}%"
        )

    average_loss = running_loss / len(dataloader)

    accuracy = 100.0 * correct / total

    return average_loss, accuracy
def validate_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model on the validation dataset.

    Args:
        model:
            AcousticDroneNet.

        dataloader:
            Validation DataLoader.

        criterion:
            Loss function.

        device:
            CPU or CUDA.

    Returns:
        Average validation loss and accuracy.
    """

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        leave=False,
    )

    with torch.no_grad():

        for batch in progress_bar:

            # ------------------------------------------
            # Move tensors to device
            # ------------------------------------------

            for key, value in batch.items():

                if isinstance(value, torch.Tensor):

                    batch[key] = value.to(device)

            labels = batch["label"]

            # ------------------------------------------
            # Forward
            # ------------------------------------------

            logits, _ = model(batch)

            loss = criterion(
                logits,
                labels,
            )

            # ------------------------------------------
            # Statistics
            # ------------------------------------------

            running_loss += loss.item()

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100 * correct / total:.2f}%"
            )

    average_loss = running_loss / len(dataloader)

    accuracy = 100.0 * correct / total

    return average_loss, accuracy
def train() -> None:
    """
    Train the Acoustic Drone Recognition Network.
    """

    # -------------------------------------------------------
    # Data
    # -------------------------------------------------------

    logger.info("Creating dataloaders...")

    train_loader = create_dataloader(
        manifest_path=TRAIN_MANIFEST,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    validation_loader = create_dataloader(
        manifest_path=VALIDATION_MANIFEST,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # -------------------------------------------------------
    # Model
    # -------------------------------------------------------

    logger.info("Building AcousticDroneNet...")

    model = AcousticDroneNet().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=SCHEDULER_T_MAX,
        eta_min=SCHEDULER_MIN_LR,
    )

    # -------------------------------------------------------
    # Resume Training
    # -------------------------------------------------------

    start_epoch = 0

    best_accuracy = 0.0

    last_checkpoint = (
        CHECKPOINT_DIR
        / LAST_MODEL_NAME
    )

    if last_checkpoint.exists():

        logger.info(
            "Loading checkpoint: %s",
            last_checkpoint,
        )

        start_epoch, best_accuracy = (
            load_checkpoint(
                last_checkpoint,
                model,
                optimizer,
                scheduler,
            )
        )

        start_epoch += 1

        logger.info(
            "Resuming from epoch %d",
            start_epoch,
        )

    # -------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------

    logger.info("Starting training...")

    for epoch in range(
        start_epoch,
        NUM_EPOCHS,
    ):

        logger.info(
            "Epoch %d / %d",
            epoch + 1,
            NUM_EPOCHS,
        )

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        validation_loss, validation_accuracy = (
            validate_one_epoch(
                model,
                validation_loader,
                criterion,
                device,
            )
        )

        scheduler.step()

        # ---------------------------------------------------
        # Save Best Model
        # ---------------------------------------------------

        if validation_accuracy > best_accuracy:

            best_accuracy = validation_accuracy

            save_checkpoint(
                CHECKPOINT_DIR / BEST_MODEL_NAME,
                model,
                optimizer,
                scheduler,
                epoch,
                best_accuracy,
            )

            logger.info(
                "New best model saved "
                "(%.2f%%)",
                best_accuracy,
            )

        # ---------------------------------------------------
        # Save Last Model
        # ---------------------------------------------------

        save_checkpoint(
            CHECKPOINT_DIR / LAST_MODEL_NAME,
            model,
            optimizer,
            scheduler,
            epoch,
            best_accuracy,
        )

        # ---------------------------------------------------
        # Epoch Summary
        # ---------------------------------------------------

        logger.info(
            (
                "Train Loss: %.4f | "
                "Train Acc: %.2f%% | "
                "Val Loss: %.4f | "
                "Val Acc: %.2f%%"
            ),
            train_loss,
            train_accuracy,
            validation_loss,
            validation_accuracy,
        )

    logger.info("Training completed.")


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user."
        )
    except Exception as error:
        logger.exception(error)
        raise