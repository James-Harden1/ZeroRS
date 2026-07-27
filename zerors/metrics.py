"""Evaluation metrics used by the original evaluator."""

from __future__ import annotations

import numpy as np


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> tuple[float, int, int]:
    prediction, target = prediction.astype(bool), target.astype(bool)
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return intersection / (union + 1e-6), intersection, union


def mask_box(mask: np.ndarray) -> list[int] | None:
    y, x = np.where(mask)
    return None if len(x) == 0 else [int(x.min()), int(y.min()), int(x.max()), int(y.max())]


def box_iou(prediction: list[int] | None, target: list[int]) -> tuple[float, int, int]:
    if prediction is None:
        prediction = [0, 0, 0, 0]
    x1, y1 = max(prediction[0], target[0]), max(prediction[1], target[1])
    x2, y2 = min(prediction[2], target[2]), min(prediction[3], target[3])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    prediction_area = (prediction[2] - prediction[0] + 1) * (prediction[3] - prediction[1] + 1)
    target_area = (target[2] - target[0] + 1) * (target[3] - target[1] + 1)
    union = prediction_area + target_area - intersection
    return intersection / (union + 1e-6), intersection, union


def summarize(ious: list[float], intersections: list[int], unions: list[int]) -> dict[str, float]:
    if not ious:
        return {"Pr@0.3": 0.0, "Pr@0.5": 0.0, "Pr@0.7": 0.0, "mIoU": 0.0, "oIoU": 0.0}
    values = np.asarray(ious)
    return {
        "Pr@0.3": float(np.mean(values > 0.3)),
        "Pr@0.5": float(np.mean(values > 0.5)),
        "Pr@0.7": float(np.mean(values > 0.7)),
        "mIoU": float(values.mean()),
        "oIoU": float(sum(intersections) / (sum(unions) + 1e-6)),
    }
