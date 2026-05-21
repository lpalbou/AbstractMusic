"""
MusicGen backend adapter.

This backend uses the Hugging Face Transformers MusicGen implementation
directly. It is intentionally separate from the generic Diffusers backend
because MusicGen is not a Diffusers pipeline.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from ..errors import OptionalDependencyMissingError
from ..huggingface import require_hf_repo_id
from ..types import AudioGenerationRequest, GeneratedAsset, MusicBackendCapabilities, ProviderModelInfo
from .diffusers_audio import _encode_wav_bytes, _resolve_device, _resolve_dtype


def _lazy_import_torch():
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): torch. "
            "Install via: pip install 'abstractmusic[musicgen]'"
        ) from e
    return torch


def _lazy_import_transformers():
    try:
        from transformers import AutoProcessor, MusicgenForConditionalGeneration  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): transformers with MusicGen support. "
            "Install via: pip install 'abstractmusic[musicgen]'"
        ) from e
    return AutoProcessor, MusicgenForConditionalGeneration


def _move_inputs_to_device(inputs: Any, device: str) -> Any:
    to_fn = getattr(inputs, "to", None)
    if callable(to_fn):
        try:
            return to_fn(device)
        except Exception:
            pass
    if isinstance(inputs, dict):
        out: Dict[str, Any] = {}
        for key, value in inputs.items():
            value_to = getattr(value, "to", None)
            out[key] = value_to(device) if callable(value_to) else value
        return out
    return inputs


def _first_audio(audio_values: Any) -> Any:
    if hasattr(audio_values, "detach"):
        ndim = int(getattr(audio_values, "ndim", 0) or 0)
        if ndim >= 3:
            return audio_values[0]
        if ndim == 2:
            # MusicGen commonly returns (batch, samples) for mono.
            return audio_values[0]
        return audio_values
    try:
        return audio_values[0]
    except Exception:
        return audio_values


@dataclass(frozen=True)
class MusicGenBackendConfig:
    model_id: str = "facebook/musicgen-small"
    device: str = "auto"
    torch_dtype: str = "auto"
    duration_s: float = 10.0
    token_rate_hz: int = 50
    do_sample: bool = True
    guidance_scale: Optional[float] = 3.0
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", require_hf_repo_id(self.model_id, field_name="model_id"))


class MusicGenBackend:
    """Local text-to-music backend powered by Transformers MusicGen."""

    backend_id = "abstractmusic:musicgen"

    def __init__(self, *, config: Optional[MusicGenBackendConfig] = None) -> None:
        self._config = config or MusicGenBackendConfig()
        self._processor = None
        self._model = None
        self._device: Optional[str] = None

    def get_capabilities(self) -> MusicBackendCapabilities:
        return MusicBackendCapabilities(
            supported_tasks=("text_to_music",),
            output_formats=("wav",),
            supports_lyrics=False,
            supports_negative_prompt=False,
            supports_guidance_scale=True,
            supports_reference_audio=False,
            supports_video=False,
            max_duration_s=30.0,
            sample_rates_hz=(32000,),
            model_id=str(self._config.model_id),
            license="CC-BY-NC-4.0",
            commercial_allowed=False,
            official_8bit_available=False,
            preferred_precision="official Transformers weights; no official 8-bit artifact reviewed",
        )

    def list_provider_models(self, *, task: Optional[str] = None) -> Sequence[ProviderModelInfo]:
        if task is not None and task != "text_to_music":
            return ()
        return (
            ProviderModelInfo(
                id=str(self._config.model_id),
                object="model",
                owned_by="facebook",
                capabilities=("text_to_music",),
                raw={"license": "CC-BY-NC-4.0", "backend": self.backend_id},
            ),
        )

    def preload(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        torch = _lazy_import_torch()
        AutoProcessor, MusicgenForConditionalGeneration = _lazy_import_transformers()

        device = _resolve_device(torch, self._config.device)
        dtype = _resolve_dtype(torch, self._config.torch_dtype, device=device)

        processor = AutoProcessor.from_pretrained(str(self._config.model_id))
        model = MusicgenForConditionalGeneration.from_pretrained(str(self._config.model_id), torch_dtype=dtype)
        to_fn = getattr(model, "to", None)
        if callable(to_fn):
            model = to_fn(device)

        eval_fn = getattr(model, "eval", None)
        if callable(eval_fn):
            eval_fn()

        self._processor = processor
        self._model = model
        self._device = device

    def unload(self) -> None:
        """Best-effort: release MusicGen weights and allocator caches."""
        self._processor = None
        self._model = None
        self._device = None
        try:
            torch = _lazy_import_torch()
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "empty_cache"):
                try:
                    mps.empty_cache()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass

    def _sample_rate(self) -> int:
        model = self._model
        try:
            audio_encoder = getattr(getattr(model, "config", None), "audio_encoder", None)
            sr = getattr(audio_encoder, "sampling_rate", None)
            if isinstance(sr, int) and sr > 0:
                return int(sr)
        except Exception:
            pass
        return 32000

    def generate_audio(self, request: AudioGenerationRequest) -> GeneratedAsset:
        self.preload()
        assert self._processor is not None
        assert self._model is not None

        torch = _lazy_import_torch()
        prompt = str(request.prompt or "")
        duration_s = float(request.duration_s if request.duration_s is not None else self._config.duration_s)
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if duration_s > 30:
            raise ValueError("MusicGen small backend currently caps duration_s at 30 seconds")

        inputs = self._processor(text=[prompt], padding=True, return_tensors="pt")
        device = str(self._device or "cpu")
        inputs = _move_inputs_to_device(inputs, device)

        seed = request.seed
        if isinstance(seed, int) and seed >= 0:
            try:
                torch.manual_seed(int(seed))
                if device == "cuda":
                    torch.cuda.manual_seed_all(int(seed))
            except Exception:
                pass

        extra = request.extra or {}
        token_rate = int(extra.get("token_rate_hz") or self._config.token_rate_hz or 50)
        max_new_tokens = int(extra.get("max_new_tokens") or round(float(duration_s) * float(token_rate)))

        guidance_scale = (
            float(request.guidance_scale)
            if request.guidance_scale is not None
            else (float(self._config.guidance_scale) if self._config.guidance_scale is not None else None)
        )

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": max(1, max_new_tokens),
            "do_sample": bool(extra.get("do_sample", self._config.do_sample)),
        }
        if guidance_scale is not None:
            generate_kwargs["guidance_scale"] = float(guidance_scale)
        temperature = extra.get("temperature", self._config.temperature)
        if temperature is not None:
            generate_kwargs["temperature"] = float(temperature)
        top_k = extra.get("top_k", self._config.top_k)
        if top_k is not None:
            generate_kwargs["top_k"] = int(top_k)
        top_p = extra.get("top_p", self._config.top_p)
        if top_p is not None:
            generate_kwargs["top_p"] = float(top_p)

        with torch.no_grad():
            audio_values = self._model.generate(**inputs, **generate_kwargs)

        sample_rate = self._sample_rate()
        wav_bytes = _encode_wav_bytes(_first_audio(audio_values), sample_rate=sample_rate)
        metadata: Dict[str, Any] = {
            "backend": self.backend_id,
            "model_id": str(self._config.model_id),
            "device": device,
            "duration_s": float(duration_s),
            "sampling_rate": int(sample_rate),
            "max_new_tokens": int(max_new_tokens),
            "guidance_scale": guidance_scale,
        }
        if seed is not None:
            metadata["seed"] = int(seed)
        return GeneratedAsset(data=bytes(wav_bytes), mime_type="audio/wav", metadata=metadata)
