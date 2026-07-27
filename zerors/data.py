"""Dataset and intermediate-file helpers shared by the three stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


def load_samples(dataset_json: Path, split: str, limit: int) -> list[dict]:
    with dataset_json.open("r", encoding="utf-8") as handle:
        return json.load(handle)[split][:limit]


def sample_image_path(image_dir: Path, image_id: int) -> Path:
    return image_dir / f"{image_id:05d}.jpg"


def candidate_path(candidate_dir: Path, index: int, image_id: int, source: str) -> Path:
    return candidate_dir / f"{index}_{image_id:05d}_{source}.npy"


def read_rgb(image_path: Path) -> np.ndarray:
    return np.asarray(Image.open(image_path).convert("RGB"))


def read_bgr(image_path: Path) -> np.ndarray:
    return cv2.cvtColor(read_rgb(image_path), cv2.COLOR_RGB2BGR)


def save_masks(path: Path, masks: Iterable[np.ndarray]) -> None:
    items = [np.asarray(mask, dtype=bool) for mask in masks]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.stack(items, axis=0) if items else np.array([]))


def load_masks(path: Path, image_shape: tuple[int, int]) -> list[np.ndarray]:
    if not path.exists():
        return []
    stored = np.load(path)
    if stored.size == 0:
        return []
    height, width = image_shape
    masks: list[np.ndarray] = []
    for mask in stored:
        binary = np.asarray(mask, dtype=bool)
        if binary.shape != (height, width):
            binary = cv2.resize(binary.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        masks.append(binary)
    return masks


def ground_truth_path(mask_dir: Path, index: int) -> Path | None:
    for name in (f"{index}.png", f"{index:05d}.png"):
        path = mask_dir / name
        if path.exists():
            return path
    return None
