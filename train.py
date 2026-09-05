"""Balanced + hard-example-mining training for the acoustic drone detector.

Important changes from the previous trainer:
1. Balanced sampling is retained, but class-weighted CE is removed. Using both
   was double-compensating the minority class.
2. Hardness is based on ordinary per-sample CE, not class-weighted CE.
3. Repeated samples in an epoch are averaged before updating hardness.
4. Validation is untouched and uses the natural validation distribution.
5. The decision threshold that maximises validation drone F1 is stored with
   the best checkpoint.
6. The best checkpoint is selected by drone F1, not accuracy.
7. The new model uses GroupNorm/LayerNorm, so small batches are safer.
"""

from __future__ import annotations

import copy
import csv
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from configs.training_config import (
    AUGMENT_PROBABILITY,
    BALANCED_SAMPLING_RATIO,
    BATCH_SIZE,
    BEST_MODEL_NAME,
    CHECKPOINT_DIR,
    DEFAULT_DECISION_THRESHOLD,
    DEVICE,
    EPOCH_SAMPLE_MULTIPLIER,
    EMA_DECAY,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_ACCUMULATION_STEPS,
    FEATURE_DROPOUT_PROBABILITY,
    GRADIENT_CLIP,
    HARD_MINING_STRENGTH,
    HARD_MINING_WARMUP_EPOCHS,
    HARDNESS_MOMENTUM,
    LABEL_SMOOTHING,
    LAST_MODEL_NAME,
    LEARNING_RATE,
    MAX_HARD_WEIGHT,
    MIN_HARD_WEIGHT,
    NUM_EPOCHS,
    NUM_WORKERS,
    PIN_MEMORY,
    RANDOM_SEED,
    RESUME_TRAINING,
    SAMPLE_WITH_REPLACEMENT,
    SCHEDULER_MIN_LR,
    SCHEDULER_T_MAX,
    TRAINING_HISTORY_PATH,
    WARMUP_EPOCHS,
    TIME_MASK_MAX_FRAMES,
    TRAIN_MANIFEST,
    VALIDATION_MANIFEST,
    TEST_MANIFEST,
    WEIGHT_DECAY,
)

from dataset.data_loader import (
    _collate_fixed_features,
)
from dataset.feature_dataset import (
    FEATURE_NAMES,
    FeatureDataset,
)
from models.acoustic_drone_model import AcousticDroneModel
from utils.split_integrity import assert_training_manifests_are_safe

from utils.checkpoint import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(RANDOM_SEED)


def move_batch_to_device(
    batch: dict,
    target_device: torch.device,
) -> dict:
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(
                target_device,
                non_blocking=True,
            )
    return batch


def calculate_metrics(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    correct: int,
    total: int,
) -> tuple[float, float, float, float]:
    accuracy = (
        100.0 * correct / total
        if total
        else 0.0
    )

    precision = (
        true_positive
        / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return accuracy, precision, recall, f1


class IndexedFeatureDataset(FeatureDataset):
    """FeatureDataset that exposes the manifest row index."""

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        sample["sample_index"] = int(index)
        return sample


def indexed_collate(
    batch: list[dict],
) -> dict:
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    indices = torch.tensor(
        [
            int(sample["sample_index"])
            for sample in batch
        ],
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


class ModelEMA:
    """Exponential moving average of model weights for stable validation."""

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1).")
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.model.state_dict()
        model_state = model.state_dict()
        for name, ema_value in ema_state.items():
            value = model_state[name].detach()
            if not torch.is_floating_point(ema_value):
                ema_value.copy_(value)
            else:
                ema_value.mul_(self.decay).add_(value, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }


class HardExampleSampler(Sampler[int]):
    """Blend class-balanced sampling with per-sample hard-example weights."""

    def __init__(
        self,
        labels: np.ndarray,
        hardness: np.ndarray,
        *,
        num_samples: int,
        balanced_ratio: float,
        hard_strength: float,
        min_hard_weight: float,
        max_hard_weight: float,
        replacement: bool,
    ) -> None:
        labels = np.asarray(labels, dtype=np.int64)
        hardness = np.asarray(hardness, dtype=np.float64)

        if labels.ndim != 1 or hardness.ndim != 1:
            raise ValueError("labels and hardness must be 1-D.")
        if len(labels) != len(hardness):
            raise ValueError("labels and hardness must have equal length.")
        if len(labels) == 0:
            raise ValueError("Cannot sample an empty dataset.")
        if not 0.0 <= balanced_ratio <= 1.0:
            raise ValueError("balanced_ratio must be in [0, 1].")
        if hard_strength < 0.0:
            raise ValueError("hard_strength must be >= 0.")
        if min_hard_weight <= 0.0:
            raise ValueError("min_hard_weight must be > 0.")
        if max_hard_weight < min_hard_weight:
            raise ValueError("max_hard_weight must be >= min_hard_weight.")

        counts = np.bincount(
            labels,
            minlength=2,
        )

        if counts[0] == 0 or counts[1] == 0:
            raise ValueError(
                "Training data must contain both background and drone samples."
            )

        self.labels = labels
        self.hardness = hardness
        self.num_samples = int(num_samples)
        self.balanced_ratio = float(balanced_ratio)
        self.hard_strength = float(hard_strength)
        self.min_hard_weight = float(min_hard_weight)
        self.max_hard_weight = float(max_hard_weight)
        self.replacement = bool(replacement)

    def _balanced_weights(self) -> np.ndarray:
        counts = np.bincount(
            self.labels,
            minlength=2,
        ).astype(np.float64)

        weights = np.zeros(
            len(self.labels),
            dtype=np.float64,
        )
        weights[self.labels == 0] = 1.0 / counts[0]
        weights[self.labels == 1] = 1.0 / counts[1]

        return weights / weights.sum()

    def _hard_weights(self) -> np.ndarray:
        hardness = np.nan_to_num(
            self.hardness,
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )
        hardness = np.maximum(
            hardness,
            1e-8,
        )

        median = float(np.median(hardness))
        if median <= 1e-8:
            median = 1.0

        weights = hardness / median
        weights = np.clip(
            weights,
            self.min_hard_weight,
            self.max_hard_weight,
        )

        weights = np.power(
            weights,
            self.hard_strength,
        )

        return weights / weights.sum()

    def probabilities(self) -> np.ndarray:
        balanced = self._balanced_weights()
        hard = self._hard_weights()

        probabilities = (
            self.balanced_ratio * balanced
            + (1.0 - self.balanced_ratio) * hard
        )

        return probabilities / probabilities.sum()

    def __iter__(self):
        probabilities = self.probabilities()

        indices = np.random.choice(
            len(self.labels),
            size=self.num_samples,
            replace=self.replacement,
            p=probabilities,
        )

        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


def build_training_loader(
    manifest_path: Path,
    hardness: np.ndarray,
) -> DataLoader:
    dataset = IndexedFeatureDataset(
        manifest_path,
        validate_features=True,
    )

    manifest = dataset.dataframe
    labels = manifest["binary_label"].astype(int).to_numpy()

    num_samples = max(
        1,
        int(
            len(dataset)
            * EPOCH_SAMPLE_MULTIPLIER
        ),
    )

    sampler = HardExampleSampler(
        labels,
        hardness,
        num_samples=num_samples,
        balanced_ratio=BALANCED_SAMPLING_RATIO,
        hard_strength=HARD_MINING_STRENGTH,
        min_hard_weight=MIN_HARD_WEIGHT,
        max_hard_weight=MAX_HARD_WEIGHT,
        replacement=SAMPLE_WITH_REPLACEMENT,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        collate_fn=indexed_collate,
    )


def build_validation_loader(
    manifest_path: Path,
) -> DataLoader:
    dataset = FeatureDataset(
        manifest_path,
        validate_features=True,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        collate_fn=_collate_fixed_features,
    )


def augment_features(
    batch: dict,
    *,
    probability: float,
) -> dict:
    """Apply conservative feature-space masking to training batches only."""

    if probability <= 0.0:
        return batch

    if torch.rand(1).item() >= probability:
        return batch

    time_length = batch[FEATURE_NAMES[0]].shape[-1]
    max_mask = min(
        TIME_MASK_MAX_FRAMES,
        max(1, time_length // 8),
    )

    if max_mask > 0:
        mask_length = int(
            torch.randint(
                1,
                max_mask + 1,
                (1,),
            ).item()
        )

        start_max = max(
            1,
            time_length - mask_length + 1,
        )
        start = int(
            torch.randint(
                0,
                start_max,
                (1,),
            ).item()
        )

        end = start + mask_length

        for name in FEATURE_NAMES:
            batch[name] = batch[name].clone()
            batch[name][..., start:end] = 0.0

    if (
        FEATURE_DROPOUT_PROBABILITY > 0.0
        and torch.rand(1).item()
        < FEATURE_DROPOUT_PROBABILITY
    ):
        name_index = int(
            torch.randint(
                0,
                len(FEATURE_NAMES),
                (1,),
            ).item()
        )
        name = FEATURE_NAMES[name_index]
        batch[name] = batch[name].clone()
        batch[name].zero_()

    return batch


def forward_model(
    model: nn.Module,
    batch: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = {
        name: batch[name]
        for name in FEATURE_NAMES
    }

    output = model(features)

    if not isinstance(output, tuple) or len(output) < 1:
        raise TypeError(
            "Model must return (logits, attention_weights)."
        )

    logits = output[0]
    attention = (
        output[1]
        if len(output) > 1
        else None
    )

    if not isinstance(logits, torch.Tensor):
        raise TypeError("Model logits must be a tensor.")

    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(
            "Expected logits shape (batch, 2), "
            f"got {tuple(logits.shape)}."
        )

    return logits, attention


def update_hardness(
    old_hardness: np.ndarray,
    observed_losses: np.ndarray,
    observed_counts: np.ndarray,
    *,
    momentum: float,
) -> np.ndarray:
    if old_hardness.shape != observed_losses.shape:
        raise ValueError("Hardness arrays must have equal shape.")

    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must be in [0, 1).")

    updated = old_hardness.copy()
    seen = observed_counts > 0

    if np.any(seen):
        mean_losses = np.zeros_like(observed_losses)
        mean_losses[seen] = (
            observed_losses[seen]
            / observed_counts[seen]
        )

        updated[seen] = (
            momentum * old_hardness[seen]
            + (1.0 - momentum) * mean_losses[seen]
        )

    return updated


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    target_device: torch.device,
    epoch_loss_sums: np.ndarray,
    epoch_loss_counts: np.ndarray,
) -> tuple[float, float, float, float, float]:
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    tp = fp = fn = 0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for batch_index, batch in enumerate(progress):
        sample_indices = batch["sample_index"].clone()

        batch = move_batch_to_device(
            batch,
            target_device,
        )

        batch = augment_features(
            batch,
            probability=AUGMENT_PROBABILITY,
        )

        labels = batch["label"]

        if batch_index % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.zero_grad(set_to_none=True)

        logits, _ = forward_model(
            model,
            batch,
        )

        per_sample_loss = nn.functional.cross_entropy(
            logits,
            labels,
            reduction="none",
        )

        loss = per_sample_loss.mean()
        (loss / GRADIENT_ACCUMULATION_STEPS).backward()

        is_update_step = (
            (batch_index + 1) % GRADIENT_ACCUMULATION_STEPS == 0
            or (batch_index + 1) == len(loader)
        )
        if is_update_step:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP,
            )
            optimizer.step()

        loss_values = (
            per_sample_loss
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        indices = (
            sample_indices
            .cpu()
            .numpy()
        )

        np.add.at(
            epoch_loss_sums,
            indices,
            loss_values,
        )
        np.add.at(
            epoch_loss_counts,
            indices,
            1,
        )

        predictions = logits.argmax(dim=1)

        correct += int(
            (predictions == labels).sum().item()
        )
        total += int(labels.numel())

        tp += int(
            (
                (predictions == 1)
                & (labels == 1)
            ).sum().item()
        )
        fp += int(
            (
                (predictions == 1)
                & (labels == 0)
            ).sum().item()
        )
        fn += int(
            (
                (predictions == 0)
                & (labels == 1)
            ).sum().item()
        )

        running_loss += float(loss.item())

        _, precision, recall, f1 = calculate_metrics(
            tp,
            fp,
            fn,
            correct,
            total,
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            precision=f"{precision:.3f}",
            recall=f"{recall:.3f}",
            f1=f"{f1:.3f}",
        )

    if len(loader) == 0:
        raise ValueError("Training loader has zero batches.")

    accuracy, precision, recall, f1 = calculate_metrics(
        tp,
        fp,
        fn,
        correct,
        total,
    )

    return (
        running_loss / len(loader),
        accuracy,
        precision,
        recall,
        f1,
    )


def find_best_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Find the validation threshold with the best drone F1."""

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    if len(probabilities) == 0:
        return DEFAULT_DECISION_THRESHOLD, 0.0

    best_threshold = DEFAULT_DECISION_THRESHOLD
    best_f1 = -1.0
    best_recall = -1.0
    best_distance = float("inf")

    # Fine but finite grid. Among essentially equivalent F1 values, prefer
    # the threshold closest to 0.50 so the exported operating point is not
    # driven by tiny validation-set probability fluctuations.
    for threshold in np.linspace(
        0.05,
        0.95,
        181,
    ):
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
        fn = int(
            np.sum(
                (predictions == 0)
                & (labels == 1)
            )
        )

        _, _, current_recall, f1 = calculate_metrics(
            tp,
            fp,
            fn,
            int(np.sum(predictions == labels)),
            len(labels),
        )

        distance = abs(float(threshold) - 0.50)
        near_best = f1 >= best_f1 - 0.001

        if (
            f1 > best_f1 + 0.001
            or (near_best and current_recall > best_recall + 1e-12)
            or (
                near_best
                and np.isclose(current_recall, best_recall, atol=1e-12)
                and distance < best_distance
            )
        ):
            best_f1 = max(best_f1, f1)
            best_recall = max(best_recall, current_recall)
            best_distance = distance
            best_threshold = float(threshold)

    return best_threshold, max(0.0, best_f1)


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    target_device: torch.device,
) -> tuple[float, float, float, float, float, float]:
    model.eval()

    running_loss = 0.0
    all_probabilities: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        progress = tqdm(
            loader,
            desc="Validation",
            leave=False,
        )

        for batch in progress:
            batch = move_batch_to_device(
                batch,
                target_device,
            )

            labels = batch["label"]

            logits, _ = forward_model(
                model,
                batch,
            )

            loss = criterion(
                logits,
                labels,
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            running_loss += float(loss.item())

            all_probabilities.append(
                probabilities.cpu().numpy()
            )
            all_labels.append(
                labels.cpu().numpy()
            )

    if len(loader) == 0:
        raise ValueError("Validation loader has zero batches.")

    probabilities = np.concatenate(
        all_probabilities
    )
    labels = np.concatenate(
        all_labels
    )

    threshold, _ = find_best_threshold(
        probabilities,
        labels,
    )

    predictions = (
        probabilities >= threshold
    )

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
    fn = int(
        np.sum(
            (predictions == 0)
            & (labels == 1)
        )
    )
    correct = int(
        np.sum(
            predictions
            == labels
        )
    )

    accuracy, precision, recall, f1 = calculate_metrics(
        tp,
        fp,
        fn,
        correct,
        len(labels),
    )

    return (
        running_loss / len(loader),
        accuracy,
        precision,
        recall,
        f1,
        threshold,
    )


def log_hardness_statistics(
    hardness: np.ndarray,
    manifest: pd.DataFrame,
) -> None:
    labels = (
        manifest["binary_label"]
        .astype(int)
        .to_numpy()
    )

    for class_id, name in (
        (0, "background"),
        (1, "drone"),
    ):
        values = hardness[
            labels == class_id
        ]

        if len(values) == 0:
            continue

        logger.info(
            (
                "Hardness | %s | mean=%.4f | "
                "median=%.4f | p90=%.4f | max=%.4f"
            ),
            name,
            float(np.mean(values)),
            float(np.median(values)),
            float(np.percentile(values, 90)),
            float(np.max(values)),
        )


def load_manifest(
    path: Path,
) -> pd.DataFrame:
    path = Path(path).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}"
        )

    manifest = pd.read_csv(path)

    required = {
        "split",
        "segment_id",
        "binary_label",
        "shard_path",
        "shard_index",
    }

    missing = required.difference(
        manifest.columns
    )
    if missing:
        raise ValueError(
            "Manifest missing columns: "
            + ", ".join(sorted(missing))
        )

    manifest["binary_label"] = (
        pd.to_numeric(
            manifest["binary_label"],
            errors="raise",
        ).astype(int)
    )

    if not manifest["binary_label"].isin([0, 1]).all():
        raise ValueError(
            "Manifest labels must be 0 or 1."
        )

    counts = (
        manifest["binary_label"]
        .value_counts()
        .sort_index()
    )

    if 0 not in counts or 1 not in counts:
        raise ValueError(
            "Manifest must contain both classes."
        )

    logger.info(
        "Manifest: %s | samples=%d | background=%d | drone=%d",
        path,
        len(manifest),
        int(counts[0]),
        int(counts[1]),
    )

    return manifest


def train() -> None:
    logger.info("=" * 78)
    logger.info("ACOUSTIC DRONE DETECTOR TRAINING")
    logger.info("=" * 78)
    logger.info("Device: %s", DEVICE)
    logger.info("Effective batch size: %d", BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    logger.info("Learning rate: %.6f", LEARNING_RATE)
    logger.info("Weight decay: %.6f", WEIGHT_DECAY)
    logger.info("Label smoothing: %.3f", LABEL_SMOOTHING)
    logger.info("Hard mining: strength=%.2f, balanced_ratio=%.2f, warmup=%d", HARD_MINING_STRENGTH, BALANCED_SAMPLING_RATIO, HARD_MINING_WARMUP_EPOCHS)
    logger.info("Augmentation probability: %.2f", AUGMENT_PROBABILITY)

    train_manifest = load_manifest(
        Path(TRAIN_MANIFEST)
    )
    validation_manifest = load_manifest(
        Path(VALIDATION_MANIFEST)
    )

    # FAIL CLOSED: verify all three shard manifests before any training step.
    # Test is checked for isolation only; it is never used for model selection.
    assert_training_manifests_are_safe(
        Path(TRAIN_MANIFEST),
        Path(VALIDATION_MANIFEST),
        Path(TEST_MANIFEST),
    )
    logger.info("PASS: train/validation/test recording, source-SHA, and segment identities are disjoint.")

    num_samples = len(train_manifest)

    # Equal initial hardness means the warmup behaves as balanced sampling.
    hardness = np.ones(
        num_samples,
        dtype=np.float64,
    )

    model = AcousticDroneModel().to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )
    logger.info(
        "Model parameters: %d",
        parameter_count,
    )

    ema = ModelEMA(model, decay=EMA_DECAY)
    logger.info("EMA validation model: enabled (decay=%.3f)", EMA_DECAY)

    # Do not use class weights: the sampler already controls class exposure.
    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    def lr_lambda(epoch_index: int) -> float:
        if WARMUP_EPOCHS > 0 and epoch_index < WARMUP_EPOCHS:
            return max(1.0e-3, (epoch_index + 1) / WARMUP_EPOCHS)
        progress = (epoch_index - WARMUP_EPOCHS) / max(1, SCHEDULER_T_MAX - WARMUP_EPOCHS)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        floor = SCHEDULER_MIN_LR / LEARNING_RATE
        return floor + (1.0 - floor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )

    start_epoch = 0
    best_f1 = 0.0
    best_threshold = DEFAULT_DECISION_THRESHOLD

    last_checkpoint = (
        CHECKPOINT_DIR
        / LAST_MODEL_NAME
    )

    if RESUME_TRAINING and last_checkpoint.exists():
        loaded_epoch, loaded_best = load_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            scheduler,
        )

        metadata = load_checkpoint_metadata(
            last_checkpoint
        )

        start_epoch = loaded_epoch + 1
        best_f1 = loaded_best
        best_threshold = metadata[
            "decision_threshold"
        ]

        saved_hardness = metadata.get(
            "hardness"
        )

        if (
            saved_hardness is not None
            and len(saved_hardness) == num_samples
        ):
            hardness = np.asarray(
                saved_hardness,
                dtype=np.float64,
            )

        logger.info(
            "Resumed from epoch %d | best F1=%.4f | threshold=%.3f",
            start_epoch + 1,
            best_f1,
            best_threshold,
        )

    elif RESUME_TRAINING:
        logger.warning(
            "Resume requested but checkpoint does not exist. "
            "Starting fresh."
        )

    validation_loader = build_validation_loader(
        Path(VALIDATION_MANIFEST)
    )

    history_rows: list[dict[str, float | int | str]] = []
    epochs_without_improvement = 0

    for epoch in range(
        start_epoch,
        NUM_EPOCHS,
    ):
        logger.info("")
        logger.info("=" * 78)
        logger.info(
            "Epoch %d / %d",
            epoch + 1,
            NUM_EPOCHS,
        )
        logger.info(
            "Learning rate: %.8f",
            optimizer.param_groups[0]["lr"],
        )

        hard_mining_active = (
            epoch
            >= HARD_MINING_WARMUP_EPOCHS
        )

        logger.info(
            "Hard-example mining: %s",
            "ENABLED" if hard_mining_active else "WARMUP",
        )

        sampler_hardness = (
            hardness
            if hard_mining_active
            else np.ones_like(hardness)
        )

        train_loader = build_training_loader(
            Path(TRAIN_MANIFEST),
            sampler_hardness,
        )

        epoch_loss_sums = np.zeros(
            num_samples,
            dtype=np.float64,
        )
        epoch_loss_counts = np.zeros(
            num_samples,
            dtype=np.int64,
        )

        train_loss, train_acc, train_precision, train_recall, train_f1 = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                DEVICE,
                epoch_loss_sums,
                epoch_loss_counts,
            )
        )

        ema.update(model)

        hardness = update_hardness(
            hardness,
            epoch_loss_sums,
            epoch_loss_counts,
            momentum=HARDNESS_MOMENTUM,
        )

        if hard_mining_active:
            log_hardness_statistics(
                hardness,
                train_manifest,
            )

        (
            validation_loss,
            validation_acc,
            validation_precision,
            validation_recall,
            validation_f1,
            validation_threshold,
        ) = validate_one_epoch(
            ema.model,
            validation_loader,
            criterion,
            DEVICE,
        )

        scheduler.step()

        logger.info(
            (
                "Train | loss=%.4f | acc=%.2f%% | "
                "precision=%.4f | recall=%.4f | f1=%.4f"
            ),
            train_loss,
            train_acc,
            train_precision,
            train_recall,
            train_f1,
        )

        logger.info(
            (
                "Validation | loss=%.4f | acc=%.2f%% | "
                "precision=%.4f | recall=%.4f | f1=%.4f | threshold=%.3f"
            ),
            validation_loss,
            validation_acc,
            validation_precision,
            validation_recall,
            validation_f1,
            validation_threshold,
        )

        improved = (
            validation_f1 > best_f1 + EARLY_STOPPING_MIN_DELTA
            or (
                abs(validation_f1 - best_f1) <= EARLY_STOPPING_MIN_DELTA
                and validation_loss < (
                    history_rows[-1]["validation_loss"]
                    if history_rows
                    else float("inf")
                )
            )
        )

        history_rows.append(
            {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_f1": train_f1,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_acc,
                "validation_precision": validation_precision,
                "validation_recall": validation_recall,
                "validation_f1": validation_f1,
                "validation_threshold": validation_threshold,
            }
        )

        if improved:
            best_f1 = max(best_f1, validation_f1)
            best_threshold = validation_threshold
            epochs_without_improvement = 0

            # The EMA model is the exported inference model because it is
            # evaluated on untouched validation data and is usually more
            # stable than the final instantaneous weights.
            save_checkpoint(
                CHECKPOINT_DIR / BEST_MODEL_NAME,
                ema.model,
                optimizer,
                scheduler,
                epoch,
                best_f1,
                decision_threshold=best_threshold,
                hardness=hardness,
            )

            logger.info(
                "NEW BEST | EMA drone F1=%.4f | threshold=%.3f",
                best_f1,
                best_threshold,
            )
        else:
            epochs_without_improvement += 1
            logger.info(
                "No validation improvement for %d/%d epoch(s).",
                epochs_without_improvement,
                EARLY_STOPPING_PATIENCE,
            )

        save_checkpoint(
            CHECKPOINT_DIR / LAST_MODEL_NAME,
            model,
            optimizer,
            scheduler,
            epoch,
            best_f1,
            decision_threshold=best_threshold,
            hardness=hardness,
        )

        TRAINING_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRAINING_HISTORY_PATH.open("w", encoding="utf-8", newline="") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=list(history_rows[0]))
            writer.writeheader()
            writer.writerows(history_rows)

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            logger.info(
                "Early stopping triggered after %d epochs without improvement.",
                epochs_without_improvement,
            )
            break

    logger.info("")
    logger.info("=" * 78)
    logger.info("TRAINING COMPLETED")
    logger.info("Best validation drone F1: %.4f", best_f1)
    logger.info("Decision threshold: %.3f", best_threshold)
    logger.info(
        "Best checkpoint: %s",
        CHECKPOINT_DIR / BEST_MODEL_NAME,
    )
    logger.info("=" * 78)


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception:
        logger.exception("Training failed.")
        raise
