"""
AbstractMusic core types.

These dataclasses define the minimal contracts used between the MusicManager
and backend implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class GeneratedAsset:
    """A generated binary asset (e.g., WAV audio)."""

    data: bytes
    mime_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioGenerationRequest:
    """Text-to-audio request contract."""

    prompt: str
    negative_prompt: Optional[str] = None
    duration_s: Optional[float] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    # Backend-specific passthrough.
    extra: Dict[str, Any] = field(default_factory=dict)

