"""
AbstractMusic high-level manager.

This mirrors AbstractVision's `VisionManager`: it is intentionally thin and
delegates execution to a configured backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from .artifacts import MediaStore
from .errors import BackendNotConfiguredError, CapabilityNotSupportedError
from .model_capabilities import MusicModelCapabilitiesRegistry
from .types import AudioGenerationRequest, GeneratedAsset
from .backends.base_backend import MusicBackend


@dataclass
class MusicManager:
    """High-level orchestrator for music/audio generation tasks."""

    backend: Optional[MusicBackend] = None
    store: Optional[MediaStore] = None
    model_id: Optional[str] = None
    registry: Optional[MusicModelCapabilitiesRegistry] = None

    def __post_init__(self) -> None:
        if self.model_id and self.registry is None:
            self.registry = MusicModelCapabilitiesRegistry()

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

    def _require_model_support(self, task: str) -> None:
        if not self.model_id:
            return
        reg = self.registry or MusicModelCapabilitiesRegistry()
        self.registry = reg
        reg.require_support(str(self.model_id), str(task))

    def _require_backend_support(self, backend: MusicBackend, request: AudioGenerationRequest, task: str) -> None:
        try:
            caps = backend.get_capabilities()
        except Exception:
            caps = None
        if caps is None:
            return
        if caps.supported_tasks is not None and str(task) not in {str(t) for t in caps.supported_tasks}:
            raise CapabilityNotSupportedError(f"Backend does not support task {task!r}.")
        if request.lyrics and caps.supports_lyrics is False:
            raise CapabilityNotSupportedError("Backend does not support lyrics.")
        if request.negative_prompt and caps.supports_negative_prompt is False:
            raise CapabilityNotSupportedError("Backend does not support negative_prompt.")
        if request.guidance_scale is not None and caps.supports_guidance_scale is False:
            raise CapabilityNotSupportedError("Backend does not support guidance_scale.")
        if request.format and caps.output_formats is not None:
            allowed = {str(fmt).lower() for fmt in caps.output_formats}
            if str(request.format).lower() not in allowed:
                raise CapabilityNotSupportedError(
                    f"Backend does not support format={request.format!r}; supported formats: {sorted(allowed)}"
                )
        if request.duration_s is not None and caps.max_duration_s is not None:
            if float(request.duration_s) > float(caps.max_duration_s):
                raise CapabilityNotSupportedError(
                    f"Backend max duration is {float(caps.max_duration_s):g}s; requested {float(request.duration_s):g}s."
                )

    def generate_audio(self, prompt: str, **kwargs: Any) -> Union[GeneratedAsset, Dict[str, Any]]:
        backend = self._require_backend()
        task = "text_to_music"
        req = AudioGenerationRequest(
            prompt=str(prompt or ""),
            lyrics=kwargs.pop("lyrics", None),
            vocal_language=kwargs.pop("vocal_language", None),
            negative_prompt=kwargs.pop("negative_prompt", None),
            duration_s=kwargs.pop("duration_s", None),
            num_inference_steps=kwargs.pop("num_inference_steps", None),
            guidance_scale=kwargs.pop("guidance_scale", None),
            seed=kwargs.pop("seed", None),
            format=str(kwargs.pop("format", "wav") or "wav"),
            sample_rate=kwargs.pop("sample_rate", None),
            extra=dict(kwargs),
        )
        self._require_model_support(task)
        self._require_backend_support(backend, req, task)
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
