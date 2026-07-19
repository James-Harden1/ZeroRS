from __future__ import annotations

from collections import deque
from typing import Iterable

import cv2
import numpy as np

from .config import PromptRefinementConfig
from .types import PromptRegion


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Convert an arbitrary finite attention map to the [0, 1] range."""
    array = np.asarray(heatmap, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D heatmap, got shape {array.shape}")
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(array.min()), float(array.max())
    if high - low < 1e-8:
        return np.zeros_like(array)
    return (array - low) / (high - low)


def prepare_heatmap(heatmap: np.ndarray, image_shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Resize VLM attention to image space, normalize, then remove local noise."""
    height, width = image_shape
    resized = cv2.resize(normalize_heatmap(heatmap), (width, height), interpolation=cv2.INTER_CUBIC)
    kernel = max(3, int(round(sigma * 4 + 1)) | 1)
    return normalize_heatmap(cv2.GaussianBlur(resized, (kernel, kernel), sigmaX=sigma))


def _box_from_points(points: np.ndarray, image_shape: tuple[int, int], padding: int) -> np.ndarray:
    height, width = image_shape
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    return np.array(
        [max(0, x1 - padding), max(0, y1 - padding), min(width - 1, x2 + padding), min(height - 1, y2 + padding)],
        dtype=np.float32,
    )


def salient_regions(heatmap: np.ndarray, config: PromptRefinementConfig) -> list[PromptRegion]:
    """Turn deep-layer VLM attention into dense point prompts and enclosing boxes.

    Each connected saliency component creates one point-box prompt pair.  Points
    are sampled around its centroid only where attention remains above half of
    the image-wide maximum, keeping SAM prompts local and semantically grounded.
    """
    smooth = normalize_heatmap(heatmap)
    binary = (smooth >= config.saliency_threshold).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    maximum = float(smooth.max())
    regions: list[PromptRegion] = []

    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < config.min_component_area:
            continue
        cx, cy = np.rint(centroids[label]).astype(int)
        yy, xx = np.indices(smooth.shape)
        local = ((xx - cx) ** 2 + (yy - cy) ** 2 <= config.point_radius**2) & (smooth >= maximum * 0.5)
        coordinates = np.column_stack(np.where(local)[::-1])
        if coordinates.size == 0:
            coordinates = np.array([[cx, cy]], dtype=np.int32)
        points = coordinates[:: max(1, config.point_stride)].astype(np.float32)
        if not np.any(np.all(points == (cx, cy), axis=1)):
            points = np.vstack([points, np.array([[cx, cy]], dtype=np.float32)])
        regions.append(
            PromptRegion(
                centroid=(cx, cy),
                points=points,
                box=_box_from_points(points, smooth.shape, config.box_padding),
                saliency=float(smooth[cy, cx]),
            )
        )
    return sorted(regions, key=lambda region: region.saliency, reverse=True)[: config.max_regions]


def region_grow(heatmap: np.ndarray, seeds: Iterable[tuple[int, int]], threshold: float) -> np.ndarray:
    """Grow an 8-connected mask from valid seeds, used by the first fallback."""
    image = normalize_heatmap(heatmap)
    height, width = image.shape
    result = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x, y in seeds:
        if 0 <= x < width and 0 <= y < height and image[y, x] >= threshold:
            result[y, x] = True
            queue.append((x, y))
    while queue:
        x, y = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and not result[ny, nx] and image[ny, nx] >= threshold:
                    result[ny, nx] = True
                    queue.append((nx, ny))
    return result
