"""VLM-attention verification and the two original fallback paths."""

from __future__ import annotations

from collections import deque
from typing import Protocol, Sequence

import cv2
import numpy as np

from .attention import normalize_attention
from .types import MaskScore, SelectionResult, TriggerPoint


class SamPredictorLike(Protocol):
    def set_image(self, image: np.ndarray) -> None: ...
    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output: bool = True): ...


def _contours(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def point_score(mask: np.ndarray, points: Sequence[TriggerPoint], proximity: float = 40.0) -> float:
    """Compute S_point: supported triggers contribute 1.0 or 1.5."""
    if not np.any(mask):
        return 0.0
    contours = _contours(mask)
    score = 0.0
    for point in points:
        hit = 0 <= point.y < mask.shape[0] and 0 <= point.x < mask.shape[1] and mask[point.y, point.x]
        if not hit:
            hit = any(cv2.pointPolygonTest(contour, (point.x, point.y), True) >= -proximity for contour in contours)
        if hit:
            score += 1.5 if point.value > 0.9 else 1.0
    return score


def score_mask(mask: np.ndarray, normalized_attention: np.ndarray, points: Sequence[TriggerPoint]) -> MaskScore:
    """Return the lexicographic pair (S_point, S_attn) from the original code."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return MaskScore(0.0, 0.0)
    return MaskScore(point_score(mask, points), float(normalized_attention[mask].mean()))


def select_best_mask(candidates: Sequence[np.ndarray], attention: np.ndarray, points: Sequence[TriggerPoint]) -> tuple[int, MaskScore] | None:
    if not candidates:
        return None
    normalized = normalize_attention(attention)
    best_index = 0
    best_score = MaskScore(-1.0, -1.0)
    for index, candidate in enumerate(candidates):
        current = score_mask(np.asarray(candidate, dtype=bool), normalized, points)
        if (current.point_score, current.mean_attention) > (best_score.point_score, best_score.mean_attention):
            best_index, best_score = index, current
    return best_index, best_score


def minimum_trigger_distance(mask: np.ndarray, points: Sequence[TriggerPoint]) -> float:
    if not np.any(mask) or not points:
        return 9999.0
    contours = _contours(mask)
    minimum = 9999.0
    for point in points:
        if 0 <= point.y < mask.shape[0] and 0 <= point.x < mask.shape[1] and mask[point.y, point.x]:
            return 0.0
        for contour in contours:
            signed_distance = cv2.pointPolygonTest(contour, (point.x, point.y), True)
            distance = abs(signed_distance) if signed_distance < 0 else 0.0
            minimum = min(minimum, distance)
    return minimum


def region_grow(attention: np.ndarray, threshold: float = 0.3, seed_values: int = 7) -> np.ndarray:
    """Use all pixels at the seven largest unique values as 8-connected BFS seeds."""
    prediction = normalize_attention(attention)
    height, width = prediction.shape
    visited = np.zeros_like(prediction, dtype=bool)
    region = np.zeros_like(prediction, dtype=np.uint8)
    top_values = np.unique(prediction)[::-1][:seed_values]
    queue: deque[tuple[int, int]] = deque()
    for value in top_values:
        for y, x in np.argwhere(prediction == value):
            queue.append((int(y), int(x)))
            visited[y, x] = True
            region[y, x] = 1
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx] and prediction[ny, nx] > threshold:
                queue.append((ny, nx))
                visited[ny, nx] = True
                region[ny, nx] = 1
    return region.astype(bool)


def box_from_region(region: np.ndarray) -> np.ndarray | None:
    if not np.any(region):
        return None
    ys, xs = np.where(region)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()])


def fallback_peak_points(attention: np.ndarray, radius: int = 20, stride: int = 4) -> list[list[int]]:
    _, maximum, _, maximum_location = cv2.minMaxLoc(attention.astype(np.float32))
    center_x, center_y = maximum_location
    height, width = attention.shape
    points: list[list[int]] = []
    for y in range(max(0, center_y - radius), min(height, center_y + radius), stride):
        for x in range(max(0, center_x - radius), min(width, center_x + radius), stride):
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2 and attention[y, x] > maximum / 2.0:
                points.append([x, y])
    if [center_x, center_y] not in points:
        points.append([center_x, center_y])
    return points


def _best_sam_mask(sam: SamPredictorLike, *, box=None, points=None) -> np.ndarray:
    if box is not None:
        masks, scores, _ = sam.predict(box=np.asarray(box), multimask_output=True)
    else:
        prompts = np.asarray(points)
        masks, scores, _ = sam.predict(
            point_coords=prompts,
            point_labels=np.ones(len(prompts), dtype=np.int32),
            multimask_output=True,
        )
    return np.asarray(masks[int(np.argmax(scores))], dtype=bool)


def select_or_fallback(image: np.ndarray, attention: np.ndarray, candidates: Sequence[np.ndarray], points: Sequence[TriggerPoint], sam: SamPredictorLike) -> SelectionResult:
    """Apply the exact candidate-first, then two fallback conditions of evaluation.py."""
    if not candidates:
        coarse = region_grow(attention)
        box = box_from_region(coarse)
        if box is None:
            return SelectionResult(np.zeros(image.shape[:2], dtype=bool), "Fallback_Failed", None)
        sam.set_image(image)
        return SelectionResult(_best_sam_mask(sam, box=box), "SAM_RegionGrow_Box", None)

    selected = select_best_mask(candidates, attention, points)
    assert selected is not None
    index, score = selected
    mask = np.asarray(candidates[index], dtype=bool)
    if minimum_trigger_distance(mask, points) > 100.0:
        sam.set_image(image)
        return SelectionResult(_best_sam_mask(sam, points=fallback_peak_points(attention)), "SAM_Heatmap_Fallback", score)
    return SelectionResult(mask, "candidate", score)
