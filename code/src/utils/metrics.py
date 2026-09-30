"""Evaluation metrics — Top-1, Tail Recall, Macro F1, ECE, Cross-Domain Gap."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np


def top1_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def per_class_recall(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    recall = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            recall[c] = (preds[mask] == c).mean()
        else:
            recall[c] = np.nan
    return recall


def per_class_precision(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    prec = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        mask = preds == c
        if mask.sum() > 0:
            prec[c] = (labels[mask] == c).mean()
        else:
            prec[c] = np.nan
    return prec


def tail_recall(preds: np.ndarray, labels: np.ndarray, tail_classes: Sequence[int]) -> float:
    if len(tail_classes) == 0:
        return float("nan")
    rec = per_class_recall(preds, labels, num_classes=max(preds.max(), labels.max(), max(tail_classes)) + 1)
    vals = rec[list(tail_classes)]
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def macro_f1(preds: np.ndarray, labels: np.ndarray, num_classes: int,
             classes: Sequence[int] | None = None) -> float:
    """对 ``classes``（默认：标签中出现过的类别）取平均的 F1。

    与 sklearn ``f1_score(average="macro", labels=classes, zero_division=0)`` 一致：
    某类在标签中存在但从未被预测时，精确率无定义，F1 记 0。
    此前这类情形得到 NaN 并被丢出平均，等于把模型完全放弃的类别从分母里拿掉，
    会系统性抬高偏向头部类的方法的 Macro-F1。
    """
    if classes is None:
        classes = np.unique(labels)
    f1 = []
    for c in classes:
        tp = float(((preds == c) & (labels == c)).sum())
        denom = float((preds == c).sum() + (labels == c).sum())
        f1.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1)) if f1 else float("nan")


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """ECE with equal-width confidence bins."""
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() > 0:
            acc = accuracies[in_bin].mean()
            conf = confidences[in_bin].mean()
            ece += (in_bin.sum() / n) * abs(acc - conf)
    return float(ece)


def cross_domain_gap(source_acc: float, target_acc: float) -> float:
    return float(source_acc - target_acc) * 100.0


def evaluate_all(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
    tail_classes: Sequence[int],
) -> Dict[str, float]:
    return {
        "top1": top1_accuracy(preds, labels) * 100.0,
        "tail_recall": tail_recall(preds, labels, tail_classes) * 100.0,
        "macro_f1": macro_f1(preds, labels, num_classes) * 100.0,
        "ece": expected_calibration_error(probs, labels),
    }
