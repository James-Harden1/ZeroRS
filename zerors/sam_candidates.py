"""Qwen box parsing and the original one-step SAM box refinement."""

from __future__ import annotations

import json
import re
from typing import Protocol

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms


class SamPredictorLike(Protocol):
    def set_image(self, image: np.ndarray) -> None: ...
    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output: bool = True): ...


BOX_PROMPT = (
    "Locate the target object described by the user: '{query}'.\n"
    "Strictly follow spatial constraints in the description. And try to locate the target base on the instruction of query.\n"
    "Ignore other objects of the same category that do not match the position. But use their relative relations with the target.\n"
    "If it is difficult to locate the target, try to think of the features of the target, use them to locate the most likely one.\n"
    "Output the bounding box that the target query ask for might exist in, in JSON format: [x1, y1, x2, y2].\n"
    "If there are multiple parts, output a list of boxes.\n"
    "Example Output: [[100, 200, 300, 400]]"
)


def parse_qwen_boxes(response: str, image_size: tuple[int, int]) -> list[list[int]]:
    """Keep the original JSON/regex parser and coordinate conversion rules."""
    width, height = image_size
    boxes: list[list[float]] = []
    try:
        cleaned = response.replace("```json", "").replace("```", "").strip()
        matched = re.search(r"(\[\[.*?\]\]|\[.*?\])", cleaned, re.DOTALL)
        if matched:
            parsed = json.loads(matched.group(1))
            if isinstance(parsed, list):
                if parsed and isinstance(parsed[0], dict) and "bbox_2d" in parsed[0]:
                    boxes.extend(item["bbox_2d"] for item in parsed)
                elif parsed and isinstance(parsed[0], (int, float)):
                    boxes.append(parsed)
                else:
                    boxes = parsed
    except Exception:
        pass
    if not boxes:
        boxes = [[int(value) for value in match] for match in re.findall(r"\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]", response)]

    valid: list[list[int]] = []
    for box in boxes:
        if len(box) != 4:
            continue
        if all(0 <= value <= 1 for value in box):
            x1, y1, x2, y2 = int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height)
        elif all(0 <= value <= 1000 for value in box) and not all(value > 1 for value in box):
            x1, y1, x2, y2 = int(box[0] / 1000 * width), int(box[1] / 1000 * height), int(box[2] / 1000 * width), int(box[3] / 1000 * height)
        else:
            x1, y1, x2, y2 = [int(value) for value in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 > 2 and y2 - y1 > 2:
            valid.append([x1, y1, x2, y2])
    return valid


def refine_box_once(sam: SamPredictorLike, input_box: list[int] | np.ndarray, padding: int = 10) -> tuple[np.ndarray, float]:
    """Initial box-prompt prediction followed by exactly one mask-derived box prompt."""
    masks, scores, _ = sam.predict(box=np.asarray(input_box)[None, :], multimask_output=True)
    index = int(np.argmax(scores))
    best_mask, best_score = masks[index].astype(np.uint8), float(scores[index])
    ys, xs = np.where(best_mask > 0)
    if len(ys) == 0:
        return best_mask, best_score
    height, width = best_mask.shape
    x1, y1 = max(0, xs.min() - padding), max(0, ys.min() - padding)
    x2, y2 = min(width, xs.max() + padding), min(height, ys.max() + padding)
    masks, scores, _ = sam.predict(box=np.asarray([x1, y1, x2, y2])[None, :], multimask_output=True)
    index = int(np.argmax(scores))
    return masks[index].astype(np.uint8), float(scores[index])


def generate_sam_candidates(sam: SamPredictorLike, image: np.ndarray, boxes: list[list[int]]) -> list[np.ndarray]:
    sam.set_image(image)
    proposals: list[dict] = []
    for box in boxes:
        try:
            mask, score = refine_box_once(sam, box)
        except Exception:
            continue
        if mask.sum() < 20:
            continue
        ys, xs = np.where(mask > 0)
        proposals.append({"bbox": [xs.min(), ys.min(), xs.max(), ys.max()], "mask": mask.astype(bool), "score": score})
    if not proposals:
        return []
    boxes_tensor = torch.tensor([proposal["bbox"] for proposal in proposals], dtype=torch.float32)
    scores_tensor = torch.tensor([proposal["score"] for proposal in proposals], dtype=torch.float32)
    kept = nms(boxes_tensor, scores_tensor, iou_threshold=0.6)
    return [proposals[index]["mask"] for index in kept]


def qwen_prompt(query: str) -> str:
    return BOX_PROMPT.format(query=query)
