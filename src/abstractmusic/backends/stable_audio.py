"""
Stable Audio Open backend adapter.

This backend targets Stability AI's gated Stable Audio Open Small checkpoint.
AbstractMusic vendors the minimal `stable-audio-tools==0.0.19` model code under
`abstractmusic.vendor.stable_audio_open_min` so users do not need to install
the upstream package (which pulls heavy UI/training dependencies). The model
is gated on Hugging Face, so loading may require `huggingface-cli login` and
license acceptance.
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Sequence

from ..errors import AbstractMusicError, OptionalDependencyMissingError
from ..huggingface import require_hf_repo_id
from ..types import AudioGenerationRequest, GeneratedAsset, MusicBackendCapabilities, ProviderModelInfo
from .diffusers_audio import _encode_wav_bytes, _resolve_device


def _lazy_import_torch():
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): torch. "
            "Install via: pip install 'abstractmusic[stable-audio]'"
        ) from e
    return torch


def _lazy_import_stable_audio_open_min():
    try:
        from ..vendor import stable_audio_open_min  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): abstractmusic stable-audio vendor runtime. "
            "Reinstall AbstractMusic or file a bug report."
        ) from e
    return stable_audio_open_min


def _time_shift(dist_shift: Any, t: Any, seq_len: int) -> Any:
    if dist_shift is None:
        return t
    time_shift = getattr(dist_shift, "time_shift", None)
    if callable(time_shift):
        return time_shift(t, seq_len)
    return t


def _sample_rf_pingpong(torch: Any, model_fn: Any, noise: Any, *, steps: int, dist_shift: Any, **extra_args: Any) -> Any:
    t = torch.linspace(1, 0, int(steps) + 1, device=noise.device, dtype=noise.dtype)
    t = _time_shift(dist_shift, t, int(noise.shape[-1]))
    ts = noise.new_ones([noise.shape[0]])
    x = noise
    for i in range(len(t) - 1):
        denoised = x - t[i] * model_fn(x, t[i] * ts, **extra_args)
        t_next = t[i + 1]
        x = (1 - t_next) * denoised + t_next * torch.randn_like(x)
    return x


def _sample_rf_euler(torch: Any, model_fn: Any, noise: Any, *, steps: int, dist_shift: Any, **extra_args: Any) -> Any:
    t = torch.linspace(1, 0, int(steps) + 1, device=noise.device, dtype=noise.dtype)
    t = _time_shift(dist_shift, t, int(noise.shape[-1]))
    x = noise
    for t_curr, t_prev in zip(t[:-1], t[1:]):
        t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)
        dt = t_prev - t_curr
        v = model_fn(x, t_curr_tensor, **extra_args)
        x = x + dt * v
    return x


def _generate_diffusion_cond_minimal(
    torch: Any,
    model: Any,
    *,
    steps: int,
    cfg_scale: float,
    conditioning: Any,
    sample_size: int,
    sampler_type: str,
    seed: int,
    device: str,
    ) -> Any:
    """Minimal Stable Audio RF generation path.

    AbstractMusic keeps this rectified-flow loop local so the `stable-audio`
    backend does not depend on upstream sampling stacks (k-diffusion, UI, and
    training tooling). Stable Audio Open Small uses the rectified-flow path,
    which is small enough to keep here.
    """

    pretransform = getattr(model, "pretransform", None)
    latent_sample_size = int(sample_size)
    if pretransform is not None:
        latent_sample_size = latent_sample_size // int(getattr(pretransform, "downsampling_ratio", 1) or 1)

    seed = int(seed) if int(seed) >= 0 else int(torch.randint(0, 2**31 - 1, (1,)).item())
    torch.manual_seed(seed)
    if str(device) == "cuda":
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    noise = torch.randn([1, int(getattr(model, "io_channels")), latent_sample_size], device=device)
    conditioning_tensors = model.conditioner(conditioning, device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {
        key: value.type(model_dtype) if value is not None and hasattr(value, "type") else value
        for key, value in conditioning_inputs.items()
    }

    diff_objective = str(getattr(model, "diffusion_objective", ""))
    if diff_objective not in {"rectified_flow", "rf_denoiser"}:
        raise ValueError(
            "StableAudioBackend's minimal dependency path only supports rectified-flow Stable Audio models. "
            f"Model diffusion_objective={diff_objective!r}."
        )

    sampler = str(sampler_type or "pingpong").strip().lower()
    sampler_kwargs = {
        **conditioning_inputs,
        "dist_shift": getattr(model, "dist_shift", None),
        "cfg_scale": float(cfg_scale),
        "batch_cfg": True,
        "rescale_cfg": True,
    }
    if sampler == "pingpong":
        sampled = _sample_rf_pingpong(torch, model.model, noise, steps=int(steps), **sampler_kwargs)
    elif sampler == "euler":
        sampled = _sample_rf_euler(torch, model.model, noise, steps=int(steps), **sampler_kwargs)
    else:
        raise ValueError("StableAudioBackend supports sampler_type='pingpong' or 'euler' in minimal mode.")

    if pretransform is not None:
        sampled = sampled.to(next(pretransform.parameters()).dtype)
        sampled = pretransform.decode(sampled)

    try:
        if str(device) == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass
    return sampled


@dataclass(frozen=True)
class StableAudioBackendConfig:
    model_id: str = "stabilityai/stable-audio-open-small"
    device: str = "auto"
    duration_s: float = 11.0
    num_inference_steps: int = 8
    guidance_scale: float = 1.0
    sampler_type: str = "pingpong"
    auto_retry_cpu_on_mps_error: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", require_hf_repo_id(self.model_id, field_name="model_id"))


class StableAudioBackend:
    """Local text-to-audio backend powered by stable-audio-tools."""

    backend_id = "abstractmusic:stable-audio"

    def __init__(self, *, config: Optional[StableAudioBackendConfig] = None) -> None:
        self._config = config or StableAudioBackendConfig()
        self._model = None
        self._model_config: Optional[Dict[str, Any]] = None
        self._device: Optional[str] = None

    def get_capabilities(self) -> MusicBackendCapabilities:
        return MusicBackendCapabilities(
            supported_tasks=("text_to_audio", "text_to_music"),
            output_formats=("wav",),
            supports_lyrics=False,
            supports_negative_prompt=False,
            supports_guidance_scale=True,
            supports_reference_audio=False,
            supports_video=False,
            max_duration_s=11.0,
            sample_rates_hz=(44100,),
            model_id=str(self._config.model_id),
            license="Stability AI Community License",
            commercial_allowed=False,
            official_8bit_available=False,
            preferred_precision="official stable-audio-tools checkpoint; no official 8-bit artifact reviewed",
        )

    def list_provider_models(self, *, task: Optional[str] = None) -> Sequence[ProviderModelInfo]:
        if task is not None and task not in {"text_to_audio", "text_to_music"}:
            return ()
        return (
            ProviderModelInfo(
                id=str(self._config.model_id),
                object="model",
                owned_by="stabilityai",
                capabilities=("text_to_audio", "text_to_music"),
                raw={"license": "Stability AI Community License", "backend": self.backend_id},
            ),
        )

    def preload(self) -> None:
        if self._model is not None and self._model_config is not None:
            return

        torch = _lazy_import_torch()
        stable_audio_open = _lazy_import_stable_audio_open_min()

        device = _resolve_device(torch, self._config.device)
        try:
            model, model_config = stable_audio_open.get_pretrained_model(str(self._config.model_id))
        except Exception as exc:
            msg = str(exc)
            name = type(exc).__name__
            lowered = msg.lower()
            if "gated" in lowered or "forbidden" in lowered or "403" in lowered or name in {"GatedRepoError"}:
                raise AbstractMusicError(
                    "Cannot download Stable Audio Open Small weights from Hugging Face (gated). "
                    "Accept the model terms on Hugging Face and set `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`, "
                    "or run `huggingface-cli login`."
                ) from exc
            raise
        to_fn = getattr(model, "to", None)
        if callable(to_fn):
            model = to_fn(device)
        eval_fn = getattr(model, "eval", None)
        if callable(eval_fn):
            eval_fn()

        self._model = model
        self._model_config = dict(model_config or {})
        self._device = device

    def unload(self) -> None:
        """Best-effort: release Stable Audio Open model weights and allocator caches."""
        self._model = None
        self._model_config = None
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
        cfg = self._model_config or {}
        sr = cfg.get("sample_rate")
        try:
            if int(sr) > 0:
                return int(sr)
        except Exception:
            pass
        return 44100

    def _sample_size(self) -> int:
        cfg = self._model_config or {}
        sample_size = cfg.get("sample_size")
        try:
            if int(sample_size) > 0:
                return int(sample_size)
        except Exception:
            pass
        return int(self._sample_rate() * 11.0)

    def _generate_with_current_device(
        self,
        *,
        prompt: str,
        duration_s: float,
        steps: int,
        cfg_scale: float,
        sampler_type: str,
        seed: Optional[int],
    ) -> Any:
        torch = _lazy_import_torch()
        assert self._model is not None

        device = str(self._device or "cpu")
        if isinstance(seed, int) and seed >= 0:
            try:
                torch.manual_seed(int(seed))
                if device == "cuda":
                    torch.cuda.manual_seed_all(int(seed))
            except Exception:
                pass

        conditioning = [{"prompt": prompt, "seconds_total": float(duration_s)}]
        with torch.no_grad():
            return _generate_diffusion_cond_minimal(
                torch,
                self._model,
                steps=int(steps),
                cfg_scale=float(cfg_scale),
                conditioning=conditioning,
                sample_size=int(self._sample_size()),
                sampler_type=str(sampler_type),
                seed=int(seed) if seed is not None else -1,
                device=device,
            )

    def generate_audio(self, request: AudioGenerationRequest) -> GeneratedAsset:
        self.preload()
        assert self._model is not None

        prompt = str(request.prompt or "")
        duration_s = float(request.duration_s if request.duration_s is not None else self._config.duration_s)
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if duration_s > 11:
            raise ValueError("Stable Audio Open Small supports up to 11 seconds")

        steps = int(request.num_inference_steps if request.num_inference_steps is not None else self._config.num_inference_steps)
        cfg_scale = float(request.guidance_scale if request.guidance_scale is not None else self._config.guidance_scale)
        extra = request.extra or {}
        sampler_type = str(extra.get("sampler_type", self._config.sampler_type) or self._config.sampler_type)
        seed = request.seed

        try:
            audio = self._generate_with_current_device(
                prompt=prompt,
                duration_s=duration_s,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler_type=sampler_type,
                seed=seed,
            )
        except RuntimeError as e:
            msg = str(e)
            is_mps = str(self._device or "").lower() == "mps"
            if is_mps and self._config.auto_retry_cpu_on_mps_error:
                print(
                    "WARNING #FALLBACK : stable-audio-tools failed on Apple MPS; retrying on CPU. "
                    f"Original error: {msg}",
                    file=sys.stderr,
                )
                self._model = None
                self._model_config = None
                self._device = None
                self._config = replace(self._config, device="cpu")
                self.preload()
                audio = self._generate_with_current_device(
                    prompt=prompt,
                    duration_s=duration_s,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    sampler_type=sampler_type,
                    seed=seed,
                )
            else:
                raise

        if hasattr(audio, "detach"):
            if int(getattr(audio, "ndim", 0) or 0) >= 3:
                audio = audio[0]
        elif isinstance(audio, (list, tuple)) and audio:
            audio = audio[0]

        sample_rate = self._sample_rate()
        wav_bytes = _encode_wav_bytes(audio, sample_rate=sample_rate)

        metadata: Dict[str, Any] = {
            "backend": self.backend_id,
            "model_id": str(self._config.model_id),
            "device": str(self._device or "cpu"),
            "duration_s": float(duration_s),
            "sampling_rate": int(sample_rate),
            "num_inference_steps": int(steps),
            "guidance_scale": float(cfg_scale),
            "sampler_type": sampler_type,
        }
        if seed is not None:
            metadata["seed"] = int(seed)
        return GeneratedAsset(data=bytes(wav_bytes), mime_type="audio/wav", metadata=metadata)
