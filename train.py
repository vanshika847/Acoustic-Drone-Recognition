"""Robust training for the binary acoustic drone detector.

Training strategy
-----------------
1. Class-balanced sampling handles the class imbalance.
2. Moderate hard-example mining focuses on genuinely difficult samples.
3. Extreme individual losses are clipped before becoming hardness values.
4. Repeated samples in an epoch are averaged before hardness is updated.
5. Validation is never balanced, augmented, or hard-mined.
6. The decision threshold is selected using validation Drone F1.
7. The best checkpoint is selected by validation Drone F1.
8. Early stopping prevents unnecessary over-training after validation
   performance has plateaued.
9. Validation confusion counts are logged so missed drones are visible.
10. The training run starts cleanly when RESUME_TRAINING=False.
"""

from __future__ import annotations

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
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    EPOCH_SAMPLE_MULTIPLIER,
    FEATURE_DROPOUT_PROBABILITY,
    GRADIENT_CLIP,
    HARD_MINING_STRENGTH,
    HARD_MINING_WARMUP_EPOCHS,
    HARDNESS_MOMENTUM,
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
    TIME_MASK_MAX_FRAMES,
    TRAIN_MANIFEST,
    VALIDATION_MANIFEST,
    WEIGHT_DECAY,
)

from dataset.data_loader import _collate_fixed_features
from dataset.feature_dataset import FEATURE_NAMES, FeatureDataset
from models.acoustic_drone_model import AcousticDroneModel
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


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

def move_batch_to_device(
    batch: dict,
    target_device: torch.device,
) -> dict:
    """Move tensor values in a batch to the selected device."""

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(
                target_device,
                non_blocking=True,
            )

    return batch


# ============================================================
# METRICS
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

    return (
        accuracy,
        precision,
        recall,
        f1,
    )


# ============================================================
# INDEXED DATASET
# ============================================================

class IndexedFeatureDataset(FeatureDataset):
    """FeatureDataset that exposes the manifest row index."""

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        sample = super().__getitem__(index)
        sample["sample_index"] = int(index)
        return sample


def indexed_collate(
    batch: list[dict],
) -> dict:
    """Collate samples while preserving their manifest indices."""

    if not batch:
        raise ValueError(
            "Cannot collate an empty batch."
        )

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


# ============================================================
# HARD-EXAMPLE SAMPLER
# ============================================================

class HardExampleSampler(Sampler[int]):
    """
    Combine class-balanced sampling with moderate hard-example sampling.

    The sampler deliberately prevents extreme hardness values from
    dominating the probability distribution.
    """

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

        labels = np.asarray(
            labels,
            dtype=np.int64,
        )

        hardness = np.asarray(
            hardness,
            dtype=np.float64,
        )

        if labels.ndim != 1:
            raise ValueError(
                "labels must be 1-D."
            )

        if hardness.ndim != 1:
            raise ValueError(
                "hardness must be 1-D."
            )

        if len(labels) != len(hardness):
            raise ValueError(
                "labels and hardness must have equal length."
            )

        if len(labels) == 0:
            raise ValueError(
                "Cannot sample an empty dataset."
            )

        if not 0.0 <= balanced_ratio <= 1.0:
            raise ValueError(
                "balanced_ratio must be in [0, 1]."
            )

        if hard_strength < 0.0:
            raise ValueError(
                "hard_strength must be >= 0."
            )

        if min_hard_weight <= 0.0:
            raise ValueError(
                "min_hard_weight must be > 0."
            )

        if max_hard_weight < min_hard_weight:
            raise ValueError(
                "max_hard_weight must be >= min_hard_weight."
            )

        counts = np.bincount(
            labels,
            minlength=2,
        )

        if counts[0] == 0 or counts[1] == 0:
            raise ValueError(
                "Training data must contain both "
                "background and drone samples."
            )

        self.labels = labels
        self.hardness = hardness
        self.num_samples = int(num_samples)
        self.balanced_ratio = float(
            balanced_ratio
        )
        self.hard_strength = float(
            hard_strength
        )
        self.min_hard_weight = float(
            min_hard_weight
        )
        self.max_hard_weight = float(
            max_hard_weight
        )
        self.replacement = bool(
            replacement
        )

    def _balanced_weights(self) -> np.ndarray:
        """Create inverse-frequency class-balanced weights."""

        counts = np.bincount(
            self.labels,
            minlength=2,
        ).astype(np.float64)

        weights = np.zeros(
            len(self.labels),
            dtype=np.float64,
        )

        weights[self.labels == 0] = (
            1.0 / counts[0]
        )

        weights[self.labels == 1] = (
            1.0 / counts[1]
        )

        return weights / weights.sum()

    def _hard_weights(self) -> np.ndarray:
        """
        Convert hardness values into bounded sampling weights.

        Important:
        Extremely large losses cannot directly create extremely large
        sampling probabilities.
        """

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

        median = float(
            np.median(hardness)
        )

        if median <= 1e-8:
            median = 1.0

        normalized = (
            hardness / median
        )

        normalized = np.clip(
            normalized,
            self.min_hard_weight,
            self.max_hard_weight,
        )

        weights = np.power(
            normalized,
            self.hard_strength,
        )

        weights = np.nan_to_num(
            weights,
            nan=1.0,
            posinf=self.max_hard_weight,
            neginf=self.min_hard_weight,
        )

        weights = np.maximum(
            weights,
            1e-12,
        )

        return weights / weights.sum()

    def probabilities(self) -> np.ndarray:
        """Combine balanced and hard-example probabilities."""

        balanced = self._balanced_weights()
        hard = self._hard_weights()

        probabilities = (
            self.balanced_ratio * balanced
            + (
                1.0 - self.balanced_ratio
            ) * hard
        )

        probabilities = np.maximum(
            probabilities,
            1e-12,
        )

        return (
            probabilities
            / probabilities.sum()
        )

    def __iter__(self):
        probabilities = self.probabilities()

        indices = np.random.choice(
            len(self.labels),
            size=self.num_samples,
            replace=self.replacement,
            p=probabilities,
        )

        return iter(
            indices.tolist()
        )

    def __len__(self) -> int:
        return self.num_samples


# ============================================================
# LOADERS
# ============================================================

def build_training_loader(
    manifest_path: Path,
    hardness: np.ndarray,
) -> DataLoader:
    """Build the balanced + hard-example training loader."""

    dataset = IndexedFeatureDataset(
        manifest_path,
        validate_features=True,
    )

    manifest = dataset.dataframe

    labels = (
        manifest["binary_label"]
        .astype(int)
        .to_numpy()
    )

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
    """
    Build the natural validation loader.

    No balancing.
    No augmentation.
    No hard mining.
    """

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


# ============================================================
# AUGMENTATION
# ============================================================

def augment_features(
    batch: dict,
    *,
    probability: float,
) -> dict:
    """
    Apply conservative feature-space augmentation.

    Augmentation is applied only to training batches.
    """

    if probability <= 0.0:
        return batch

    if torch.rand(1).item() >= probability:
        return batch

    time_length = batch[
        FEATURE_NAMES[0]
    ].shape[-1]

    max_mask = min(
        TIME_MASK_MAX_FRAMES,
        max(
            1,
            time_length // 8,
        ),
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
            time_length
            - mask_length
            + 1,
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
        feature_index = int(
            torch.randint(
                0,
                len(FEATURE_NAMES),
                (1,),
            ).item()
        )

        feature_name = FEATURE_NAMES[
            feature_index
        ]

        batch[feature_name] = (
            batch[feature_name].clone()
        )

        batch[feature_name].zero_()

    return batch


# ============================================================
# MODEL FORWARD
# ============================================================

def forward_model(
    model: nn.Module,
    batch: dict,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
]:
    """Run the model and validate its output."""

    features = {
        name: batch[name]
        for name in FEATURE_NAMES
    }

    output = model(features)

    if (
        not isinstance(output, tuple)
        or len(output) < 1
    ):
        raise TypeError(
            "Model must return "
            "(logits, attention_weights)."
        )

    logits = output[0]

    attention = (
        output[1]
        if len(output) > 1
        else None
    )

    if not isinstance(
        logits,
        torch.Tensor,
    ):
        raise TypeError(
            "Model logits must be a tensor."
        )

    if (
        logits.ndim != 2
        or logits.shape[1] != 2
    ):
        raise ValueError(
            "Expected logits shape "
            "(batch, 2), got "
            f"{tuple(logits.shape)}."
        )

    return (
        logits,
        attention,
    )


# ============================================================
# HARDNESS UPDATE
# ============================================================

def update_hardness(
    old_hardness: np.ndarray,
    observed_loss_sums: np.ndarray,
    observed_counts: np.ndarray,
    *,
    momentum: float,
) -> np.ndarray:
    """
    Update per-sample hardness using the mean observed loss.

    Repeated samples are averaged before updating hardness.
    Extremely large losses are clipped before entering the EMA.
    """

    if (
        old_hardness.shape
        != observed_loss_sums.shape
    ):
        raise ValueError(
            "Hardness arrays must have equal shape."
        )

    if not 0.0 <= momentum < 1.0:
        raise ValueError(
            "momentum must be in [0, 1)."
        )

    updated = old_hardness.copy()

    seen = observed_counts > 0

    if not np.any(seen):
        return updated

    mean_losses = np.zeros_like(
        observed_loss_sums
    )

    mean_losses[seen] = (
        observed_loss_sums[seen]
        / observed_counts[seen]
    )

    # --------------------------------------------------------
    # Extreme-loss protection.
    #
    # A single corrupted / pathological sample should not
    # become the dominant hard-mining target.
    #
    # The upper bound is derived from the observed epoch loss
    # distribution rather than from a fixed arbitrary value.
    # --------------------------------------------------------

    valid_losses = mean_losses[seen]

    if len(valid_losses) > 0:
        median = float(
            np.median(valid_losses)
        )

        q75 = float(
            np.percentile(
                valid_losses,
                75,
            )
        )

        q90 = float(
            np.percentile(
                valid_losses,
                90,
            )
        )

        spread = max(
            q90 - q75,
            q75 - median,
            1e-6,
        )

        upper_limit = (
            q90 + 4.0 * spread
        )

        upper_limit = max(
            upper_limit,
            median * 2.0,
            1.0,
        )

        mean_losses[seen] = np.clip(
            mean_losses[seen],
            1e-6,
            upper_limit,
        )

    updated[seen] = (
        momentum * old_hardness[seen]
        + (
            1.0 - momentum
        ) * mean_losses[seen]
    )

    updated = np.nan_to_num(
        updated,
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )

    updated = np.maximum(
        updated,
        1e-6,
    )

    return updated


# ============================================================
# HARDNESS STATISTICS
# ============================================================

def log_hardness_statistics(
    hardness: np.ndarray,
    manifest: pd.DataFrame,
) -> None:
    """Log hardness statistics separately for each class."""

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
                "Hardness | %s | "
                "mean=%.4f | median=%.4f | "
                "p90=%.4f | max=%.4f"
            ),
            name,
            float(np.mean(values)),
            float(np.median(values)),
            float(
                np.percentile(
                    values,
                    90,
                )
            ),
            float(np.max(values)),
        )


def log_hardest_samples(
    hardness: np.ndarray,
    manifest: pd.DataFrame,
    *,
    top_k: int = 10,
) -> None:
    """Log the hardest training samples for investigation."""

    if len(hardness) == 0:
        return

    top_k = min(
        top_k,
        len(hardness),
    )

    indices = np.argsort(
        hardness
    )[-top_k:][::-1]

    logger.info(
        "Hardest training samples:"
    )

    for rank, index in enumerate(
        indices,
        start=1,
    ):
        row = manifest.iloc[
            int(index)
        ]

        segment_id = row.get(
            "segment_id",
            "unknown",
        )

        label = row.get(
            "binary_label",
            "unknown",
        )

        shard_path = row.get(
            "shard_path",
            "unknown",
        )

        shard_index = row.get(
            "shard_index",
            "unknown",
        )

        logger.info(
            (
                "  #%d | index=%d | label=%s | "
                "hardness=%.5f | segment=%s | "
                "shard=%s | shard_index=%s"
            ),
            rank,
            int(index),
            label,
            float(hardness[index]),
            segment_id,
            shard_path,
            shard_index,
        )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    target_device: torch.device,
    optimizer: torch.optim.Optimizer,
    epoch_loss_sums: np.ndarray,
    epoch_loss_counts: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
]:
    """Train one epoch using ordinary unweighted CE."""

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    tp = 0
    fp = 0
    fn = 0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for batch in progress:

        sample_indices = (
            batch["sample_index"]
            .clone()
        )

        batch = move_batch_to_device(
            batch,
            target_device,
        )

        batch = augment_features(
            batch,
            probability=AUGMENT_PROBABILITY,
        )

        labels = batch["label"]

        optimizer.zero_grad(
            set_to_none=True
        )

        logits, _ = forward_model(
            model,
            batch,
        )

        # IMPORTANT:
        # Ordinary CE.
        # Class balancing is handled by the sampler.
        per_sample_loss = (
            nn.functional.cross_entropy(
                logits,
                labels,
                reduction="none",
            )
        )

        loss = per_sample_loss.mean()

        loss.backward()

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

        predictions = (
            logits.argmax(dim=1)
        )

        correct += int(
            (
                predictions == labels
            ).sum().item()
        )

        total += int(
            labels.numel()
        )

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

        running_loss += float(
            loss.item()
        )

        _, precision, recall, f1 = (
            calculate_metrics(
                tp,
                fp,
                fn,
                correct,
                total,
            )
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            precision=f"{precision:.3f}",
            recall=f"{recall:.3f}",
            f1=f"{f1:.3f}",
        )

    if len(loader) == 0:
        raise ValueError(
            "Training loader has zero batches."
        )

    accuracy, precision, recall, f1 = (
        calculate_metrics(
            tp,
            fp,
            fn,
            correct,
            total,
        )
    )

    return (
        running_loss / len(loader),
        accuracy,
        precision,
        recall,
        f1,
    )


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """
    Find the threshold producing the best Drone F1.

    If multiple thresholds have essentially the same F1,
    prefer the threshold with better recall. This is useful
    because the current model's main weakness is missed drones.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    if len(probabilities) == 0:
        return (
            DEFAULT_DECISION_THRESHOLD,
            0.0,
        )

    best_threshold = (
        DEFAULT_DECISION_THRESHOLD
    )

    best_f1 = -1.0
    best_recall = -1.0
    best_precision = -1.0

    for threshold in np.linspace(
        0.05,
        0.95,
        181,
    ):
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
                predictions == labels
            )
        )

        _, precision, recall, f1 = (
            calculate_metrics(
                tp,
                fp,
                fn,
                correct,
                len(labels),
            )
        )

        # Primary objective: F1.
        #
        # Tie-breaker: recall.
        #
        # Second tie-breaker: precision.
        if (
            f1 > best_f1
            or (
                np.isclose(
                    f1,
                    best_f1,
                    atol=1e-12,
                )
                and recall > best_recall
            )
            or (
                np.isclose(
                    f1,
                    best_f1,
                    atol=1e-12,
                )
                and np.isclose(
                    recall,
                    best_recall,
                    atol=1e-12,
                )
                and precision > best_precision
            )
        ):
            best_f1 = f1
            best_recall = recall
            best_precision = precision
            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        max(0.0, best_f1),
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    target_device: torch.device,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    int,
    int,
    int,
    int,
]:
    """
    Validate on the natural validation distribution.

    Returns:
        loss
        accuracy
        precision
        recall
        f1
        threshold
        true positives
        false positives
        false negatives
        true negatives
    """

    model.eval()

    running_loss = 0.0

    all_probabilities: list[
        np.ndarray
    ] = []

    all_labels: list[
        np.ndarray
    ] = []

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

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1,
                )[:, 1]
            )

            running_loss += float(
                loss.item()
            )

            all_probabilities.append(
                probabilities
                .cpu()
                .numpy()
            )

            all_labels.append(
                labels
                .cpu()
                .numpy()
            )

    if len(loader) == 0:
        raise ValueError(
            "Validation loader has zero batches."
        )

    probabilities = np.concatenate(
        all_probabilities
    )

    labels = np.concatenate(
        all_labels
    )

    threshold, _ = (
        find_best_threshold(
            probabilities,
            labels,
        )
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

    tn = int(
        np.sum(
            (predictions == 0)
            & (labels == 0)
        )
    )

    correct = (
        tp + tn
    )

    accuracy, precision, recall, f1 = (
        calculate_metrics(
            tp,
            fp,
            fn,
            correct,
            len(labels),
        )
    )

    return (
        running_loss / len(loader),
        accuracy,
        precision,
        recall,
        f1,
        threshold,
        tp,
        fp,
        fn,
        tn,
    )


# ============================================================
# MANIFEST
# ============================================================

def load_manifest(
    path: Path,
) -> pd.DataFrame:
    """Load and validate a manifest."""

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
            + ", ".join(
                sorted(missing)
            )
        )

    manifest["binary_label"] = (
        pd.to_numeric(
            manifest["binary_label"],
            errors="raise",
        ).astype(int)
    )

    if not manifest[
        "binary_label"
    ].isin([0, 1]).all():
        raise ValueError(
            "Manifest labels must be 0 or 1."
        )

    counts = (
        manifest["binary_label"]
        .value_counts()
        .sort_index()
    )

    if (
        0 not in counts
        or 1 not in counts
    ):
        raise ValueError(
            "Manifest must contain both classes."
        )

    logger.info(
        (
            "Manifest: %s | samples=%d | "
            "background=%d | drone=%d"
        ),
        path,
        len(manifest),
        int(counts[0]),
        int(counts[1]),
    )

    return manifest


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train() -> None:
    """Run the complete training experiment."""

    logger.info("=" * 78)
    logger.info(
        "ROBUST ACOUSTIC DRONE DETECTOR TRAINING"
    )
    logger.info("=" * 78)

    logger.info(
        "Device: %s",
        DEVICE,
    )

    logger.info(
        "Random seed: %d",
        RANDOM_SEED,
    )

    logger.info(
        "Max epochs: %d",
        NUM_EPOCHS,
    )

    logger.info(
        "Early stopping patience: %d",
        EARLY_STOPPING_PATIENCE,
    )

    logger.info(
        "Balanced sampling ratio: %.2f",
        BALANCED_SAMPLING_RATIO,
    )

    logger.info(
        "Hard mining strength: %.2f",
        HARD_MINING_STRENGTH,
    )

    logger.info(
        "Hard mining warmup: %d epochs",
        HARD_MINING_WARMUP_EPOCHS,
    )

    # --------------------------------------------------------
    # Load manifests.
    # --------------------------------------------------------

    train_manifest = load_manifest(
        Path(TRAIN_MANIFEST)
    )

    validation_manifest = load_manifest(
        Path(VALIDATION_MANIFEST)
    )

    num_samples = len(
        train_manifest
    )

    # --------------------------------------------------------
    # Initial hardness.
    #
    # All samples start equally hard.
    # Therefore the first warmup epochs behave like
    # class-balanced sampling.
    # --------------------------------------------------------

    hardness = np.ones(
        num_samples,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Model.
    # --------------------------------------------------------

    model = AcousticDroneModel().to(
        DEVICE
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    logger.info(
        "Model parameters: %d",
        parameter_count,
    )

    # --------------------------------------------------------
    # Ordinary CE.
    #
    # Do NOT add class weights here because the sampler already
    # compensates for class imbalance.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Training state.
    # --------------------------------------------------------

    start_epoch = 0

    best_f1 = 0.0

    best_threshold = (
        DEFAULT_DECISION_THRESHOLD
    )

    epochs_without_improvement = 0

    best_epoch = -1

    # --------------------------------------------------------
    # Checkpoint paths.
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    last_checkpoint = (
        CHECKPOINT_DIR
        / LAST_MODEL_NAME
    )

    best_checkpoint = (
        CHECKPOINT_DIR
        / BEST_MODEL_NAME
    )

    # --------------------------------------------------------
    # Resume support.
    #
    # For THIS new experiment keep RESUME_TRAINING=False.
    # --------------------------------------------------------

    if (
        RESUME_TRAINING
        and last_checkpoint.exists()
    ):

        (
            loaded_epoch,
            loaded_best,
        ) = load_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            scheduler,
        )

        metadata = (
            load_checkpoint_metadata(
                last_checkpoint
            )
        )

        start_epoch = (
            loaded_epoch + 1
        )

        best_f1 = float(
            loaded_best
        )

        best_threshold = float(
            metadata.get(
                "decision_threshold",
                DEFAULT_DECISION_THRESHOLD,
            )
        )

        saved_hardness = metadata.get(
            "hardness"
        )

        if (
            saved_hardness is not None
            and len(saved_hardness)
            == num_samples
        ):
            hardness = np.asarray(
                saved_hardness,
                dtype=np.float64,
            )

        logger.info(
            (
                "Resumed from epoch %d | "
                "best F1=%.4f | threshold=%.3f"
            ),
            start_epoch + 1,
            best_f1,
            best_threshold,
        )

    elif RESUME_TRAINING:

        logger.warning(
            (
                "Resume requested but checkpoint "
                "does not exist. Starting fresh."
            )
        )

    # --------------------------------------------------------
    # Training loop.
    # --------------------------------------------------------

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
            "Learning rate: %.10f",
            optimizer.param_groups[0]["lr"],
        )

        # ----------------------------------------------------
        # Hard mining activation.
        # ----------------------------------------------------

        hard_mining_active = (
            epoch
            >= HARD_MINING_WARMUP_EPOCHS
        )

        logger.info(
            "Hard-example mining: %s",
            (
                "ENABLED"
                if hard_mining_active
                else "WARMUP"
            ),
        )

        if hard_mining_active:
            sampler_hardness = hardness
        else:
            sampler_hardness = (
                np.ones_like(hardness)
            )

        # ----------------------------------------------------
        # Loaders.
        # ----------------------------------------------------

        train_loader = (
            build_training_loader(
                Path(TRAIN_MANIFEST),
                sampler_hardness,
            )
        )

        validation_loader = (
            build_validation_loader(
                Path(VALIDATION_MANIFEST)
            )
        )

        logger.info(
            "Training samples this epoch: %d",
            len(train_loader.sampler),
        )

        logger.info(
            "Training batches: %d",
            len(train_loader),
        )

        logger.info(
            "Validation batches: %d",
            len(validation_loader),
        )

        # ----------------------------------------------------
        # Per-sample loss accumulation.
        #
        # Repeated samples are averaged before hardness update.
        # ----------------------------------------------------

        epoch_loss_sums = np.zeros(
            num_samples,
            dtype=np.float64,
        )

        epoch_loss_counts = np.zeros(
            num_samples,
            dtype=np.int64,
        )

        # ----------------------------------------------------
        # Train.
        # ----------------------------------------------------

        (
            train_loss,
            train_acc,
            train_precision,
            train_recall,
            train_f1,
        ) = train_one_epoch(
            model,
            train_loader,
            DEVICE,
            optimizer,
            epoch_loss_sums,
            epoch_loss_counts,
        )

        # ----------------------------------------------------
        # Update hardness.
        # ----------------------------------------------------

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

            log_hardest_samples(
                hardness,
                train_manifest,
                top_k=10,
            )

        # ----------------------------------------------------
        # Validation.
        # ----------------------------------------------------

        (
            validation_loss,
            validation_acc,
            validation_precision,
            validation_recall,
            validation_f1,
            validation_threshold,
            validation_tp,
            validation_fp,
            validation_fn,
            validation_tn,
        ) = validate_one_epoch(
            model,
            validation_loader,
            criterion,
            DEVICE,
        )

        # ----------------------------------------------------
        # Scheduler.
        # ----------------------------------------------------

        scheduler.step()

        # ----------------------------------------------------
        # Training log.
        # ----------------------------------------------------

        logger.info(
            (
                "Train | loss=%.4f | "
                "acc=%.2f%% | precision=%.4f | "
                "recall=%.4f | f1=%.4f"
            ),
            train_loss,
            train_acc,
            train_precision,
            train_recall,
            train_f1,
        )

        # ----------------------------------------------------
        # Validation log.
        # ----------------------------------------------------

        logger.info(
            (
                "Validation | loss=%.4f | "
                "acc=%.2f%% | precision=%.4f | "
                "recall=%.4f | f1=%.4f | "
                "threshold=%.3f"
            ),
            validation_loss,
            validation_acc,
            validation_precision,
            validation_recall,
            validation_f1,
            validation_threshold,
        )

        # ----------------------------------------------------
        # Explicit confusion matrix.
        # ----------------------------------------------------

        logger.info(
            (
                "Validation confusion | "
                "TN=%d | FP=%d | FN=%d | TP=%d"
            ),
            validation_tn,
            validation_fp,
            validation_fn,
            validation_tp,
        )

        logger.info(
            "Validation missed drones (FN): %d",
            validation_fn,
        )

        logger.info(
            "Validation false alarms (FP): %d",
            validation_fp,
        )

        # ----------------------------------------------------
        # Best model logic.
        #
        # Primary metric = Drone F1.
        # A tiny improvement below min_delta does not count
        # as a meaningful improvement.
        # ----------------------------------------------------

        meaningful_improvement = (
            validation_f1
            > best_f1
            + EARLY_STOPPING_MIN_DELTA
        )

        if meaningful_improvement:

            previous_best = best_f1

            best_f1 = float(
                validation_f1
            )

            best_threshold = float(
                validation_threshold
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                scheduler,
                epoch,
                best_f1,
                decision_threshold=(
                    best_threshold
                ),
                hardness=hardness,
            )

            logger.info(
                (
                    "NEW BEST | "
                    "Drone F1 %.4f -> %.4f | "
                    "threshold=%.3f | "
                    "epoch=%d"
                ),
                previous_best,
                best_f1,
                best_threshold,
                epoch + 1,
            )

        else:

            epochs_without_improvement += 1

            logger.info(
                (
                    "No meaningful validation "
                    "F1 improvement | "
                    "patience=%d/%d"
                ),
                epochs_without_improvement,
                EARLY_STOPPING_PATIENCE,
            )

        # ----------------------------------------------------
        # Last checkpoint.
        #
        # This is always the latest state and is useful if
        # training is interrupted.
        # ----------------------------------------------------

        save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            scheduler,
            epoch,
            best_f1,
            decision_threshold=(
                best_threshold
            ),
            hardness=hardness,
        )

        # ----------------------------------------------------
        # Early stopping.
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            logger.info("")
            logger.info(
                "=" * 78
            )

            logger.info(
                (
                    "EARLY STOPPING | "
                    "Validation Drone F1 has not "
                    "meaningfully improved for %d epochs."
                ),
                EARLY_STOPPING_PATIENCE,
            )

            logger.info(
                "Best epoch: %d",
                best_epoch + 1,
            )

            logger.info(
                "Best validation Drone F1: %.4f",
                best_f1,
            )

            logger.info(
                "Best decision threshold: %.3f",
                best_threshold,
            )

            logger.info(
                "=" * 78
            )

            break

    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------

    logger.info("")
    logger.info("=" * 78)
    logger.info(
        "TRAINING COMPLETED"
    )
    logger.info("=" * 78)

    logger.info(
        "Best validation Drone F1: %.4f",
        best_f1,
    )

    logger.info(
        "Best decision threshold: %.3f",
        best_threshold,
    )

    if best_epoch >= 0:
        logger.info(
            "Best epoch: %d",
            best_epoch + 1,
        )

    logger.info(
        "Best checkpoint: %s",
        best_checkpoint,
    )

    logger.info(
        "Last checkpoint: %s",
        last_checkpoint,
    )

    logger.info("=" * 78)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        train()

    except KeyboardInterrupt:
        logger.info(
            "Training interrupted by user."
        )

    except Exception:
        logger.exception(
            "Training failed."
        )
        raise