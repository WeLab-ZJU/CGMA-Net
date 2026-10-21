from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= start) & (y_prob < end if end < 1.0 else y_prob <= end)
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += abs(avg_acc - avg_conf) * (mask.sum() / len(y_true))
    return float(ece)


def binary_classification_metrics(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true_np = np.asarray(list(y_true)).astype(int)
    y_prob_np = np.asarray(list(y_prob)).astype(float)
    y_pred_np = (y_prob_np >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1]).ravel()
    sensitivity = recall_score(y_true_np, y_pred_np, zero_division=0)
    specificity = tn / max(tn + fp, 1)

    metrics = {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true_np, y_pred_np, zero_division=0)),
        "f1": float(f1_score(y_true_np, y_pred_np, zero_division=0)),
        "auc": float(roc_auc_score(y_true_np, y_prob_np)) if len(np.unique(y_true_np)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y_true_np, y_prob_np)),
        "brier": float(brier_score_loss(y_true_np, y_prob_np)),
        "ece": float(expected_calibration_error(y_true_np, y_prob_np)),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }
    return metrics


def subgroup_metrics(
    y_true: List[int],
    y_prob: List[float],
    groups: List[str],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for group_name in sorted(set(groups)):
        indices = [idx for idx, group in enumerate(groups) if group == group_name]
        if not indices:
            continue
        group_true = [y_true[idx] for idx in indices]
        group_prob = [y_prob[idx] for idx in indices]
        out[group_name] = binary_classification_metrics(group_true, group_prob, threshold=threshold)
        out[group_name]["count"] = float(len(indices))
    return out
