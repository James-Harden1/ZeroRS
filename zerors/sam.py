"""Minimal Segment Anything adapter shared by stage 2 and stage 3."""

from __future__ import annotations

import numpy as np


class SamAdapter:
    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        from segment_anything import SamPredictor, sam_model_registry

        self.predictor = SamPredictor(sam_model_registry["vit_h"](checkpoint=checkpoint_path).to(device))

    def set_image(self, image: np.ndarray) -> None:
        self.predictor.set_image(image)

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output: bool = True):
        return self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output,
        )
