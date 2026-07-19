from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class PromptRefinementConfig:
    """Hyperparameters for the Point-Box Prompt Refinement module."""

    saliency_threshold: float = 0.30
    gaussian_sigma: float = 7.0
    min_component_area: int = 5
    point_radius: int = 20
    point_stride: int = 4
    max_regions: int = 7
    box_padding: int = 2
    nms_iou_threshold: float = 0.60


@dataclass(frozen=True)
class SelectionConfig:
    """Candidate verification and fallback thresholds."""

    point_proximity_px: float = 40.0
    distance_fallback_px: float = 100.0
    confidence_threshold: float = 0.70


@dataclass(frozen=True)
class ZeroRSConfig:
    """Runtime paths and algorithm settings for a ZeroRS experiment."""

    dataset_json: Path
    image_dir: Path
    output_dir: Path
    qwen_model: Path | None = None
    sam_checkpoint: Path | None = None
    odise_config: Path | None = None
    odise_weights: Path | None = None
    split: str = "test"
    prompt: PromptRefinementConfig = PromptRefinementConfig()
    selection: SelectionConfig = SelectionConfig()
    attention_layers: Tuple[int, ...] = (16, 17, 18, 19)
    attention_weights: Tuple[float, ...] = (0.1, 0.1, 0.3, 0.5)

    def __post_init__(self) -> None:
        if len(self.attention_layers) != len(self.attention_weights):
            raise ValueError("attention_layers and attention_weights must have the same length")
        if abs(sum(self.attention_weights) - 1.0) > 1e-6:
            raise ValueError("attention_weights must sum to 1")
