"""
AbstractMusic high-level manager.

This mirrors AbstractVision's `VisionManager`: it is intentionally thin and
delegates execution to a configured backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from .artifacts import MediaStore
from .errors import BackendNotConfiguredError
from .types import AudioGenerationRequest, GeneratedAsset
from .backends.base_backend import MusicBackend


@dataclass
class MusicManager:
    """High-level orchestrator for music/audio generation tasks."""

    backend: Optional[MusicBackend] = None
    store: Optional[MediaStore] = None

    def _require_backend(self) -> MusicBackend:
        if self.backend is None:
            raise BackendNotConfiguredError(
                "No music backend configured. Provide a backend to MusicManager(backend=...) before calling."
            )
        return self.backend

    def _maybe_store(self, asset: GeneratedAsset, *, tags: Optional[Dict[str, str]] = None) -> Union[GeneratedAsset, Dict[str, Any]]:
        if self.store is None:
            return asset
        return self.store.store_bytes(
            asset.data,
            content_type=asset.mime_type,
            filename="music.wav" if asset.mime_type == "audio/wav" else None,
            metadata=asset.metadata,
            tags=tags,
        )

    def generate_audio(self, prompt: str, **kwargs: Any) -> Union[GeneratedAsset, Dict[str, Any]]:
        backend = self._require_backend()
        req = AudioGenerationRequest(
            prompt=str(prompt or ""),
            negative_prompt=kwargs.pop("negative_prompt", None),
            duration_s=kwargs.pop("duration_s", None),
            num_inference_steps=kwargs.pop("num_inference_steps", None),
            guidance_scale=kwargs.pop("guidance_scale", None),
            seed=kwargs.pop("seed", None),
            extra=dict(kwargs),
        )
        asset = backend.generate_audio(req)
        return self._maybe_store(
            asset, tags={"kind": "generated_media", "modality": "audio", "task": "text_to_music"}
        )

    def t2m(self, prompt: str, **kwargs: Any) -> bytes:
        """Convenience: generate WAV bytes directly (library-mode)."""
        out = self.generate_audio(prompt, **kwargs)
        if isinstance(out, dict):
            # In library-mode, store is usually None; if configured, return stored content is up to caller.
            raise TypeError("MusicManager.t2m returned an artifact ref dict; use generate_audio(...) to handle stores.")
        return bytes(out.data)

