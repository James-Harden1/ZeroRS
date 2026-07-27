"""Candidate extraction for the original ODISE branch."""

from __future__ import annotations

import cv2
import numpy as np


def candidates_from_panoptic(output: dict, image_shape: tuple[int, int]) -> list[np.ndarray]:
    """Union category-0 regions and split the union into 8-connected masks."""
    if "panoptic_seg" not in output:
        return []
    panoptic, segments = output["panoptic_seg"]
    labels = panoptic.detach().cpu().numpy() if hasattr(panoptic, "detach") else np.asarray(panoptic)
    height, width = image_shape
    full_mask = np.zeros((height, width), dtype=np.uint8)
    for segment in segments:
        if segment["category_id"] != 0:
            continue
        mask = (labels == segment["id"]).astype(np.uint8)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        full_mask = cv2.bitwise_or(full_mask, mask)
    if not np.any(full_mask):
        return []
    count, labels, _, _ = cv2.connectedComponentsWithStats(full_mask, connectivity=8)
    return [(labels == label).astype(bool) for label in range(1, count)]


def clean_query(query: str) -> str:
    return query.replace("-", " ").replace("_", " ").strip()
