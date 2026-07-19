from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MaskCandidate:
    """A binary segmentation proposal plus provenance and confidence."""

    mask: np.ndarray
    source: str
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptRegion:
    """Salient-region prompts derived from a VLM attention heatmap."""

    centroid: tuple[int, int]
    points: np.ndarray
    box: np.ndarray
    saliency: float
