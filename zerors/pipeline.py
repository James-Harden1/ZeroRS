from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import ZeroRSConfig
from .point_box_refiner import PointBoxPromptRefiner
from .selection import select_candidate
from .types import MaskCandidate


class AttentionProvider(Protocol):
    def __call__(self, image: np.ndarray, query: str) -> np.ndarray: ...


class DiffusionProvider(Protocol):
    def __call__(self, image: np.ndarray, query: str) -> list[MaskCandidate]: ...


class ZeroRSPipeline:
    """Late-fusion ZeroRS pipeline retaining independent VLM-SAM and DM branches."""

    def __init__(self, attention_provider: AttentionProvider, diffusion_provider: DiffusionProvider, refiner: PointBoxPromptRefiner, config: ZeroRSConfig) -> None:
        self.attention_provider = attention_provider
        self.diffusion_provider = diffusion_provider
        self.refiner = refiner
        self.config = config

    def predict(self, image: np.ndarray, query: str) -> MaskCandidate | None:
        attention = self.attention_provider(image, query)
        heatmap, regions = self.refiner.build_prompts(attention, image.shape[:2])
        point_box_candidates = self.refiner.refine(image, attention)
        diffusion_candidates = self.diffusion_provider(image, query)
        candidates = [*diffusion_candidates, *point_box_candidates]
        points = np.array([region.centroid for region in regions], dtype=np.float32)
        best = select_candidate(candidates, heatmap, points, self.config.selection)
        if best is not None and best.confidence >= self.config.selection.confidence_threshold:
            return best
        return self.refiner.fallback_from_region_grow(image, attention) or best
