from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from .attention import prepare_heatmap, region_grow, salient_regions
from .config import PromptRefinementConfig
from .types import MaskCandidate, PromptRegion


class SamPredictorLike(Protocol):
    def set_image(self, image: np.ndarray) -> None: ...

    def predict(self, *, point_coords=None, point_labels=None, box=None, multimask_output: bool = True): ...


class PointBoxPromptRefiner:
    """Refine VLM attention into point-box SAM predictions without retraining.

    The module implements the paper's point-box prompt refinement: weighted VLM
    attention -> saliency components -> local point diffusion + enclosing box ->
    SAM candidate masks.  It deliberately keeps prompts local instead of using a
    box spanning every active component.
    """

    def __init__(self, predictor: SamPredictorLike, config: PromptRefinementConfig = PromptRefinementConfig()) -> None:
        self.predictor = predictor
        self.config = config

    def build_prompts(self, attention: np.ndarray, image_shape: tuple[int, int]) -> tuple[np.ndarray, list[PromptRegion]]:
        heatmap = prepare_heatmap(attention, image_shape, self.config.gaussian_sigma)
        return heatmap, salient_regions(heatmap, self.config)

    def refine(self, image: np.ndarray, attention: np.ndarray) -> list[MaskCandidate]:
        heatmap, regions = self.build_prompts(attention, image.shape[:2])
        self.predictor.set_image(image)
        candidates: list[MaskCandidate] = []
        for index, region in enumerate(regions):
            masks, scores, _ = self.predictor.predict(
                point_coords=region.points,
                point_labels=np.ones(len(region.points), dtype=np.int32),
                box=region.box[None, :],
                multimask_output=True,
            )
            best = int(np.argmax(scores))
            candidates.append(
                MaskCandidate(
                    mask=np.asarray(masks[best], dtype=bool),
                    source="point_box_sam",
                    confidence=float(scores[best]),
                    metadata={"prompt_index": index, "box": region.box.tolist(), "centroid": region.centroid, "saliency": region.saliency},
                )
            )
        return self._nms(candidates)

    def fallback_from_region_grow(self, image: np.ndarray, attention: np.ndarray) -> MaskCandidate | None:
        """First fallback: region growing -> box prompt; then point-only fallback."""
        heatmap, regions = self.build_prompts(attention, image.shape[:2])
        self.predictor.set_image(image)
        seeds = [region.centroid for region in regions]
        coarse = region_grow(heatmap, seeds, self.config.saliency_threshold)
        if coarse.any():
            ys, xs = np.where(coarse)
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
            masks, scores, _ = self.predictor.predict(box=box[None, :], multimask_output=True)
            best = int(np.argmax(scores))
            return MaskCandidate(np.asarray(masks[best], dtype=bool), "fallback_region_box", float(scores[best]), {"box": box.tolist()})
        if not regions:
            return None
        region = regions[0]
        masks, scores, _ = self.predictor.predict(
            point_coords=region.points,
            point_labels=np.ones(len(region.points), dtype=np.int32),
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        return MaskCandidate(np.asarray(masks[best], dtype=bool), "fallback_points", float(scores[best]))

    def _nms(self, candidates: list[MaskCandidate]) -> list[MaskCandidate]:
        kept: list[MaskCandidate] = []
        for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
            duplicate = False
            for prior in kept:
                union = np.logical_or(candidate.mask, prior.mask).sum()
                overlap = np.logical_and(candidate.mask, prior.mask).sum() / max(1, union)
                if overlap >= self.config.nms_iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return kept
