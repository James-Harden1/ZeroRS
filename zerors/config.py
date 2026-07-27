"""Shared runtime settings for the original ZeroRS pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    dataset_json: Path
    image_dir: Path
    candidate_dir: Path
    output_dir: Path
    qwen_model: Path
    sam_checkpoint: Path
    mask_dir: Path | None = None
    odise_config: Path | None = None
    odise_weights: Path | None = None


@dataclass(frozen=True)
class PipelineSettings:
    split: str = "test"
    sample_limit: int = 6000
    attention_layers: tuple[int, ...] = (16, 17, 18, 19)
    attention_weights: tuple[float, ...] = (0.1, 0.1, 0.3, 0.5)
    trigger_threshold: float = 0.3
    trigger_min_area: int = 5
    point_proximity: float = 40.0
    distance_fallback: float = 100.0
    region_threshold: float = 0.3
    region_seed_values: int = 7
    fallback_radius: int = 20
    fallback_stride: int = 4
    sam_box_padding: int = 10
    sam_nms_iou: float = 0.6
    min_mask_area: int = 20

    def __post_init__(self) -> None:
        if len(self.attention_layers) != len(self.attention_weights):
            raise ValueError("attention_layers and attention_weights must have equal lengths")
