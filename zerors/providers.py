from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from .attention import normalize_heatmap
from .types import MaskCandidate


class QwenAttentionProvider:
    """Extract a language-conditioned image attention map from Qwen2.5-VL.

    Imports are intentionally delayed so the package can be inspected and tested
    without loading Qwen or its CUDA dependencies.
    """

    def __init__(self, model, processor, layers: Sequence[int], weights: Sequence[float], max_new_tokens: int = 10) -> None:
        self.model = model
        self.processor = processor
        self.layers = tuple(layers)
        self.weights = tuple(weights)
        self.max_new_tokens = max_new_tokens

    def __call__(self, image: np.ndarray, query: str) -> np.ndarray:
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        prompt = "Return the target bounding box as [x1, y1, x2, y2]."
        pil_image = Image.fromarray(image)
        messages = [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": f"Locate: {query}. {prompt}"}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device)
        ids = inputs.input_ids[0].tolist()
        start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        start, end = ids.index(start_id) + 1, ids.index(end_id)
        grid = self.processor.image_processor(images=image_inputs)["image_grid_thw"][0].cpu().numpy()
        grid_h, grid_w = int(grid[1] / 2), int(grid[2] / 2)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, output_attentions=True, return_dict_in_generate=True)
        if not generated.attentions:
            raise RuntimeError("Qwen did not return generation attentions; use eager attention implementation.")
        fused = np.zeros((grid_h, grid_w), dtype=np.float32)
        for layer, weight in zip(self.layers, self.weights):
            steps = [step[layer][0, :, -1, start:end].mean(dim=0) for step in generated.attentions]
            vector = torch.stack(steps).mean(dim=0).float().cpu().numpy()
            if vector.size != grid_h * grid_w:
                side = int(math.sqrt(vector.size))
                vector = vector[: side * side]
                layer_map = vector.reshape(side, side)
                layer_map = cv2.resize(layer_map, (grid_w, grid_h), interpolation=cv2.INTER_CUBIC)
            else:
                layer_map = vector.reshape(grid_h, grid_w)
            fused += float(weight) * normalize_heatmap(layer_map)
        return normalize_heatmap(fused)


class OdiseCandidateProvider:
    """Adapt an existing ``OdiseRefiner.predict_crop`` instance to ZeroRS."""

    def __init__(self, odise_refiner) -> None:
        self.odise_refiner = odise_refiner

    def __call__(self, image: np.ndarray, query: str) -> list[MaskCandidate]:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        output = self.odise_refiner.predict_crop(bgr, [query.replace("-", " ").replace("_", " ").strip()])
        if "panoptic_seg" not in output:
            return []
        panoptic, segments = output["panoptic_seg"]
        labels = panoptic.detach().cpu().numpy() if hasattr(panoptic, "detach") else np.asarray(panoptic)
        proposals: list[MaskCandidate] = []
        for segment in segments:
            if segment.get("category_id") != 0:
                continue
            mask = labels == segment["id"]
            if mask.any():
                proposals.append(MaskCandidate(mask=mask, source="odise", confidence=float(segment.get("score", 0.0))))
        return proposals
