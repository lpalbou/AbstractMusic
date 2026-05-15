"""
AbstractMusic backend interface.

Backends implement the actual inference/generation logic.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence

from ..types import AudioGenerationRequest, GeneratedAsset, MusicBackendCapabilities, ProviderModelInfo


class MusicBackend(Protocol):
    """Protocol for music/audio generation backends."""

    backend_id: str

    def get_capabilities(self) -> Optional[MusicBackendCapabilities]:
        """Return backend-level capability constraints, when known."""
        return None

    def list_provider_models(self, *, task: Optional[str] = None) -> Sequence[ProviderModelInfo]:
        """Return provider-advertised model entries, when available."""
        _ = task
        return ()

    def preload(self) -> None:
        """Best-effort: load model weights into memory."""
        return None

    def unload(self) -> None:
        """Best-effort: release model weights from memory."""
        return None

    def generate_audio(self, request: AudioGenerationRequest) -> GeneratedAsset: ...
