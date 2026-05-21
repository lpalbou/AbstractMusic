"""
AbstractMusic core types.

These dataclasses define the minimal contracts used between the MusicManager
and backend implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence


@dataclass(frozen=True)
class ProviderModelInfo:
    """A model entry returned by a provider catalog or packaged registry."""

    id: str
    object: Optional[str] = None
    created: Optional[int] = None
    owned_by: Optional[str] = None
    capabilities: Sequence[str] = field(default_factory=tuple)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MusicBackendCapabilities:
    """Backend-level capability constraints."""

    supported_tasks: Optional[Sequence[str]] = None
    output_formats: Optional[Sequence[str]] = None
    supports_lyrics: Optional[bool] = None
    supports_negative_prompt: Optional[bool] = None
    supports_guidance_scale: Optional[bool] = None
    supports_reference_audio: Optional[bool] = None
    supports_video: Optional[bool] = None
    max_duration_s: Optional[float] = None
    sample_rates_hz: Optional[Sequence[int]] = None
    model_id: Optional[str] = None
    license: Optional[str] = None
    commercial_allowed: Optional[bool] = None
    official_8bit_available: Optional[bool] = None
    preferred_precision: Optional[str] = None


@dataclass(frozen=True)
class GeneratedAsset:
    """A generated binary asset (e.g., WAV audio)."""

    data: bytes
    mime_type: str
    media_type: str = "audio"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MusicSectionPlan:
    """Provider-neutral description of one musical section."""

    name: str
    duration_ms: Optional[int] = None
    positive_styles: Sequence[str] = field(default_factory=tuple)
    negative_styles: Sequence[str] = field(default_factory=tuple)
    lines: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "positive_styles": list(self.positive_styles),
            "negative_styles": list(self.negative_styles),
            "lines": list(self.lines),
        }


@dataclass(frozen=True)
class MusicCompositionPlan:
    """Provider-neutral structured music plan.

    Backends translate this into their native request shape. This keeps the
    planning intelligence separate from provider-specific transport details.
    """

    positive_styles: Sequence[str] = field(default_factory=tuple)
    negative_styles: Sequence[str] = field(default_factory=tuple)
    sections: Sequence[MusicSectionPlan] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "positive_styles": list(self.positive_styles),
            "negative_styles": list(self.negative_styles),
            "sections": [
                section.to_dict() if hasattr(section, "to_dict") else dict(section)  # type: ignore[arg-type]
                for section in self.sections
            ],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AudioGenerationRequest:
    """Text-to-audio request contract."""

    prompt: str
    lyrics: Optional[str] = None
    vocal_language: Optional[str] = None
    negative_prompt: Optional[str] = None
    duration_s: Optional[float] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    format: str = "wav"
    sample_rate: Optional[int] = None
    composition_plan: Optional[Any] = None
    # Backend-specific passthrough.
    extra: Dict[str, Any] = field(default_factory=dict)
