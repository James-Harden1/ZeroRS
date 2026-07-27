"""Small data records used by candidate verification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TriggerPoint:
    x: int
    y: int
    value: float


@dataclass(frozen=True)
class MaskScore:
    point_score: float
    mean_attention: float


@dataclass(frozen=True)
class SelectionResult:
    mask: object
    source: str
    score: MaskScore | None
