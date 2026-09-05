"""Segment-level and recording-level metrics for drone detection."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def confusion_counts(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> tuple[int, int, int, int]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = probabilities >= threshold
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    return tp, fp, tn, fn


def binary_detection_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    tp, fp, tn, fn = confusion_counts(probabilities, labels, threshold)
    total = len(labels)
    accuracy = 100.0 * (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
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


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)

    sorted_scores = scores[order]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average_rank
        start = end

    sum_pos_ranks = float(ranks[labels == 1].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(labels == 1))
    if n_pos == 0 or len(labels) == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    return float(np.trapz(precision, recall))


def find_best_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    default_threshold: float = 0.50,
    f1_tolerance: float = 0.001,
) -> tuple[float, float]:
    """Select a validation threshold by drone F1, preferring higher recall."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(probabilities) == 0:
        return default_threshold, 0.0

    best_threshold = default_threshold
    best_f1 = -1.0
    best_recall = -1.0
    best_distance = float("inf")

    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = binary_detection_metrics(probabilities, labels, float(threshold))
        f1 = metrics["f1"]
        recall = metrics["recall"]
        distance = abs(float(threshold) - 0.50)
        near_best = f1 >= best_f1 - f1_tolerance
        improved = (
            f1 > best_f1 + f1_tolerance
            or (near_best and recall > best_recall + 1e-12)
            or (
                near_best
                and abs(recall - best_recall) <= 1e-12
                and distance < best_distance
            )
        )
        if improved:
            best_f1 = f1
            best_recall = recall
            best_distance = distance
            best_threshold = float(threshold)

    return best_threshold, max(0.0, best_f1)


def aggregate_recording_predictions(
    recording_ids: Iterable[str],
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    method: str = "max",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collapse segment predictions to one score/label per recording.

    ``method='max'`` uses the strongest drone probability in the recording,
    which is the recall-oriented detector aggregation. ``method='mean'``
    averages segment probabilities.
    """

    frame = pd.DataFrame(
        {
            "recording_id": list(recording_ids),
            "probability": np.asarray(probabilities, dtype=np.float64),
            "label": np.asarray(labels, dtype=np.int64),
        }
    )
    if method == "max":
        scores = frame.groupby("recording_id", sort=False)["probability"].max()
    elif method == "mean":
        scores = frame.groupby("recording_id", sort=False)["probability"].mean()
    else:
        raise ValueError("Recording aggregation method must be 'max' or 'mean'.")

    recording_labels = frame.groupby("recording_id", sort=False)["label"].max()
    mixed = frame.groupby("recording_id")["label"].nunique()
    if int((mixed > 1).sum()) != 0:
        raise ValueError("A recording contains mixed binary labels.")

    recording_names = list(scores.index.astype(str))
    return scores.to_numpy(), recording_labels.to_numpy(), recording_names
