"""
AbstractMusic backend interface.

Backends implement the actual inference/generation logic.
"""

from __future__ import annotations

from typing import Protocol

from ..types import AudioGenerationRequest, GeneratedAsset


class MusicBackend(Protocol):
    """Protocol for music/audio generation backends."""

    backend_id: str

    def generate_audio(self, request: AudioGenerationRequest) -> GeneratedAsset: ...

