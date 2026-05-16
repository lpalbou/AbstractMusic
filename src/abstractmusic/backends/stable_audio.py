"""
Stable Audio Open backend adapter.

This backend wraps the official `stable-audio-tools` inference path for
Stability AI's Stable Audio Open Small model. The model is gated on Hugging
Face, so loading may require `huggingface-cli login` and license acceptance.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Sequence

from ..errors import OptionalDependencyMissingError
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


def _lazy_import_stable_audio_tools():
    try:
        from stable_audio_tools import get_pretrained_model  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): stable-audio-tools. "
            "Install via: pip install 'abstractmusic[stable-audio]' && "
            "pip install --no-deps 'stable-audio-tools==0.0.19'"
        ) from e
    return get_pretrained_model


class _DistributionShift:
    def __init__(
        self,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        max_length: int = 4096,
        min_length: int = 256,
        use_sine: bool = False,
    ) -> None:
        self.base_shift = float(base_shift)
        self.max_shift = float(max_shift)
        self.max_length = int(max_length)
        self.min_length = int(min_length)
        self.use_sine = bool(use_sine)

    def time_shift(self, t: Any, seq_len: int) -> Any:
        import math

        sigma = 1.0
        denom = max(float(self.max_length - self.min_length), 1.0)
        mu = -(self.base_shift + (self.max_shift - self.base_shift) * (float(seq_len) - self.min_length) / denom)
        t_out = 1 - math.exp(mu) / (math.exp(mu) + (1 / (1 - t) - 1) ** sigma)
        if self.use_sine:
            t_out = (t_out * (math.pi / 2)).sin() if hasattr(t_out, "sin") else t_out
        return t_out


def _install_stable_audio_minimal_shims() -> None:
    """Install tiny inference shims before stable-audio-tools imports model code.

    The published stable-audio-tools package imports k-diffusion, UI, and
    training-adjacent dependency chains even for the RF pingpong path used by
    Stable Audio Open Small. These shims let model construction import while
    this backend owns the minimal generation loop.
    """

    inference_name = "stable_audio_tools.inference"
    if inference_name not in sys.modules:
        try:
            __import__(inference_name)
        except Exception:
            inference_mod = types.ModuleType(inference_name)
            inference_mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules[inference_name] = inference_mod

    sampling_name = "stable_audio_tools.inference.sampling"
    if sampling_name not in sys.modules:
        sampling_mod = types.ModuleType(sampling_name)
        sampling_mod.DistributionShift = _DistributionShift

        def _unsupported_sample(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("Stable Audio v-diffusion sampling is not available in AbstractMusic's minimal backend.")

        sampling_mod.sample = _unsupported_sample
        sys.modules[sampling_name] = sampling_mod

    generation_name = "stable_audio_tools.inference.generation"
    if generation_name not in sys.modules:
        generation_mod = types.ModuleType(generation_name)

        def _generate(model: Any, **kwargs: Any) -> Any:
            torch = _lazy_import_torch()
            return _generate_diffusion_cond_minimal(
                torch,
                model,
                steps=int(kwargs.get("steps", 8)),
                cfg_scale=float(kwargs.get("cfg_scale", 1.0)),
                conditioning=kwargs.get("conditioning"),
                sample_size=int(kwargs.get("sample_size", getattr(model, "sample_size", 44100 * 11))),
                sampler_type=str(kwargs.get("sampler_type", "pingpong")),
                seed=int(kwargs.get("seed", -1)),
                device=str(kwargs.get("device", "cpu")),
            )

        generation_mod.generate_diffusion_cond = _generate
        sys.modules[generation_name] = generation_mod


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

    This intentionally avoids importing `stable_audio_tools.inference.generation`
    because that module imports all of k-diffusion and pulls UI/training-oriented
    dependencies. Stable Audio Open Small uses the rectified-flow path, which is
    small enough to keep here.
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
        get_pretrained_model = _lazy_import_stable_audio_tools()
        _install_stable_audio_minimal_shims()

        device = _resolve_device(torch, self._config.device)
        model, model_config = get_pretrained_model(str(self._config.model_id))
        to_fn = getattr(model, "to", None)
        if callable(to_fn):
            model = to_fn(device)
        eval_fn = getattr(model, "eval", None)
        if callable(eval_fn):
            eval_fn()

        self._model = model
        self._model_config = dict(model_config or {})
        self._device = device

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
