from __future__ import annotations

import cv2
import numpy as np

from .attention import normalize_heatmap
from .config import SelectionConfig
from .types import MaskCandidate


def score_candidate(candidate: MaskCandidate, heatmap: np.ndarray, points: np.ndarray, proximity_px: float) -> float:
    """Rank independently generated masks using VLM attention only at verification."""
    mask = np.asarray(candidate.mask, dtype=bool)
    if not mask.any():
        return float("-inf")
    heat = normalize_heatmap(heatmap)
    support = float(heat[mask].mean())
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    covered = 0.0
    for x, y in points.astype(int):
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            if mask[y, x] or any(cv2.pointPolygonTest(contour, (int(x), int(y)), True) >= -proximity_px for contour in contours):
                covered += float(heat[y, x])
    return support + covered + 0.05 * candidate.confidence


def select_candidate(candidates: list[MaskCandidate], heatmap: np.ndarray, points: np.ndarray, config: SelectionConfig) -> MaskCandidate | None:
    if not candidates:
        return None
    ranked = [(score_candidate(candidate, heatmap, points, config.point_proximity_px), candidate) for candidate in candidates]
    return max(ranked, key=lambda item: item[0])[1]
