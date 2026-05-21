"""
AbstractMusic high-level manager.

This mirrors AbstractVision's `VisionManager`: it is intentionally thin and
delegates execution to a configured backend.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Union

from .artifacts import MediaStore
from .errors import BackendNotConfiguredError, CapabilityNotSupportedError
from .model_capabilities import MusicModelCapabilitiesRegistry
from .prompt_planner import MusicPlanningRequest, create_music_prompt_plan, compile_music_prompt_plan
from .types import AudioGenerationRequest, GeneratedAsset
from .backends.base_backend import MusicBackend


@dataclass
class MusicManager:
    """High-level orchestrator for music/audio generation tasks."""

    backend: Optional[MusicBackend] = None
    store: Optional[MediaStore] = None
    model_id: Optional[str] = None
    registry: Optional[MusicModelCapabilitiesRegistry] = None
    text_planner: Optional[Any] = None
    text_planner_mode: str = "auto"

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

    def plan_text(
        self,
        request: MusicPlanningRequest,
        *,
        text_planner: Optional[Any] = None,
        mode: Optional[str] = None,
    ):
        """Create a validated music text plan through the configured planner."""

        return create_music_prompt_plan(
            request,
            provider=self.text_planner if text_planner is None else text_planner,
            mode=str(mode or self.text_planner_mode or "deterministic"),
        )

    def generate_audio(self, prompt: str, **kwargs: Any) -> Union[GeneratedAsset, Dict[str, Any]]:
        backend = self._require_backend()
        task = "text_to_music"
        request_metadata = kwargs.pop("metadata", None)
        extra_request_metadata = kwargs.pop("request_metadata", None)
        if request_metadata is None:
            request_metadata = extra_request_metadata
        elif isinstance(request_metadata, dict) and isinstance(extra_request_metadata, dict):
            request_metadata = {**request_metadata, **extra_request_metadata}

        planning_value = kwargs.pop("planning", kwargs.pop("plan_text", False))
        planning_enabled = bool(planning_value) and str(planning_value).strip().lower() not in {
            "0",
            "false",
            "no",
            "none",
            "off",
        }
        text_planner = kwargs.pop("text_planner", None)
        text_planner_mode = str(kwargs.pop("text_planner_mode", self.text_planner_mode) or self.text_planner_mode)
        instrumental = bool(kwargs.pop("instrumental", False))
        enhance_prompt = bool(kwargs.pop("enhance_prompt", False))
        structure_prompt = bool(kwargs.pop("structure_prompt", True))
        auto_lyrics = bool(kwargs.pop("auto_lyrics", False))

        lyrics = kwargs.pop("lyrics", None)
        vocal_language = kwargs.pop("vocal_language", None)
        negative_prompt = kwargs.pop("negative_prompt", None)
        duration_s = kwargs.pop("duration_s", None)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        guidance_scale = kwargs.pop("guidance_scale", None)
        seed = kwargs.pop("seed", None)
        output_format = str(kwargs.pop("format", "wav") or "wav")
        sample_rate = kwargs.pop("sample_rate", None)
        composition_plan = kwargs.pop("composition_plan", None)

        if planning_enabled:
            try:
                caps = backend.get_capabilities()
            except Exception:
                caps = None
            plan_request = MusicPlanningRequest(
                prompt=str(prompt or ""),
                lyrics=lyrics,
                vocal_language=vocal_language,
                duration_s=duration_s,
                bpm=kwargs.get("bpm"),
                keyscale=kwargs.get("keyscale"),
                timesignature=kwargs.get("timesignature"),
                instrumental=instrumental,
                enhance_prompt=enhance_prompt,
                structure_prompt=structure_prompt,
                auto_lyrics=auto_lyrics,
                backend=str(getattr(backend, "backend_id", "")),
                model_id=getattr(caps, "model_id", None) if caps is not None else self.model_id,
            )
            plan = self.plan_text(plan_request, text_planner=text_planner, mode=text_planner_mode)
            native_lyrics_supported = not (caps is not None and caps.supports_lyrics is False)
            compiled = compile_music_prompt_plan(
                plan,
                backend=str(getattr(backend, "backend_id", "")),
                native_lyrics_supported=native_lyrics_supported,
            )
            prompt = compiled.prompt
            lyrics = compiled.lyrics
            if composition_plan is None:
                composition_plan = compiled.composition_plan
            vocal_language = compiled.metadata.get("vocal_language")
            for key in ("bpm", "keyscale", "timesignature"):
                value = compiled.metadata.get(key)
                if value is not None:
                    kwargs[key] = value
            if isinstance(request_metadata, dict):
                request_metadata = {**compiled.metadata, **request_metadata}
            else:
                request_metadata = dict(compiled.metadata)

        req = AudioGenerationRequest(
            prompt=str(prompt or ""),
            lyrics=lyrics,
            vocal_language=vocal_language,
            negative_prompt=negative_prompt,
            duration_s=duration_s,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            format=output_format,
            sample_rate=sample_rate,
            composition_plan=composition_plan,
            extra=dict(kwargs),
        )
        self._require_model_support(task)
        self._require_backend_support(backend, req, task)
        asset = backend.generate_audio(req)
        if isinstance(request_metadata, dict) and request_metadata:
            asset = replace(asset, metadata={**dict(asset.metadata or {}), **dict(request_metadata)})
        return self._maybe_store(
            asset, tags={"kind": "generated_media", "modality": "audio", "task": "text_to_music"}
        )

    def t2m(self, prompt: str, **kwargs: Any) -> bytes:
        """Convenience: generate audio bytes directly (library-mode)."""
        out = self.generate_audio(prompt, **kwargs)
        if isinstance(out, dict):
            # In library-mode, store is usually None; if configured, return stored content is up to caller.
            raise TypeError("MusicManager.t2m returned an artifact ref dict; use generate_audio(...) to handle stores.")
        return bytes(out.data)
