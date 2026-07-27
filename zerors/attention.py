"""Qwen attention extraction and trigger-point construction."""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch

from .types import TriggerPoint


def normalize_attention(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)


def extract_attention_map(model, processor, image_path: str, query: str) -> np.ndarray | None:
    """Reproduce the attention extraction used in the original evaluator."""
    from qwen_vl_utils import process_vision_info

    prompt = "The output format should be like [x1, y1, x2, y2] without any other text."
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": f"Locate it according to the following description. {query} {prompt}"},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device)

    token_ids = inputs["input_ids"][0].tolist()
    try:
        vision_start = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        start = token_ids.index(vision_start) + 1
        end = token_ids.index(vision_end)
    except ValueError:
        return None

    image_inputs_aux = processor.image_processor(images=image_inputs)
    grid = image_inputs_aux["image_grid_thw"][0].cpu().numpy()
    grid_height, grid_width = int(grid[1] / 2), int(grid[2] / 2)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=10, output_attentions=True, return_dict_in_generate=True
        )
    if generated.attentions is None:
        return None

    weights = {16: 0.1, 17: 0.1, 18: 0.3, 19: 0.5}
    fused: np.ndarray | None = None
    for layer in (16, 17, 18, 19):
        token_attention = [
            step[layer][0, :, -1, start:end].mean(dim=0)
            for step in generated.attentions
        ]
        vector = torch.stack(token_attention).mean(dim=0)
        try:
            layer_map = vector.reshape(grid_height, grid_width).float().cpu().numpy()
        except RuntimeError:
            side = int(math.sqrt(vector.shape[0]))
            layer_map = vector[: side * side].reshape(side, side).float().cpu().numpy()
        layer_map = normalize_attention(layer_map)
        fused = layer_map * weights[layer] if fused is None else fused + layer_map * weights[layer]
    return fused


def extract_trigger_points(heatmap: np.ndarray, threshold: float = 0.3) -> tuple[list[TriggerPoint], np.ndarray]:
    """Blur, threshold, and reduce valid connected components to centroids."""
    normalized = normalize_attention(heatmap)
    blurred = cv2.GaussianBlur((normalized * 255).astype(np.uint8), (7, 7), 0)
    _, binary = cv2.threshold(blurred, int(255 * threshold), 255, cv2.THRESH_BINARY)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    points: list[TriggerPoint] = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < 5:
            continue
        x, y = int(centroids[label][0]), int(centroids[label][1])
        points.append(TriggerPoint(x=x, y=y, value=float(blurred[y, x]) / 255.0))
    return points, binary


def resize_attention(heatmap: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    return cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
