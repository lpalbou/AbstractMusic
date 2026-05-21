"""
Stable Audio 3 backend adapter.

This backend uses AbstractMusic-owned runtime code under
`abstractmusic.vendor.stable_audio3_min` plus Hugging Face model files. It must
not import or wrap the upstream `stable_audio_3` package.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..errors import OptionalDependencyMissingError
from ..huggingface import require_hf_repo_id
from ..types import AudioGenerationRequest, GeneratedAsset, MusicBackendCapabilities, ProviderModelInfo
from .diffusers_audio import _encode_wav_bytes, _resolve_device


MODEL_ID_SMALL_MUSIC = "stabilityai/stable-audio-3-small-music"
MODEL_ID_SMALL_SFX = "stabilityai/stable-audio-3-small-sfx"
MODEL_ID_MEDIUM = "stabilityai/stable-audio-3-medium"

_MODEL_INFO: Dict[str, Dict[str, Any]] = {
    MODEL_ID_SMALL_MUSIC: {
        "name": "small-music",
        "display": "Stable Audio 3 Small Music",
        "max_duration_s": 120.0,
        "sample_rate_hz": 44100,
        "hardware": "CPU",
    },
    MODEL_ID_SMALL_SFX: {
        "name": "small-sfx",
        "display": "Stable Audio 3 Small SFX",
        "max_duration_s": 120.0,
        "sample_rate_hz": 44100,
        "hardware": "CPU",
    },
    MODEL_ID_MEDIUM: {
        "name": "medium",
        "display": "Stable Audio 3 Medium",
        "max_duration_s": 380.0,
        "sample_rate_hz": 44100,
        "hardware": "CUDA",
    },
}

_MODEL_ALIASES = {
    "small": MODEL_ID_SMALL_MUSIC,
    "small-music": MODEL_ID_SMALL_MUSIC,
    "small-sfx": MODEL_ID_SMALL_SFX,
    "sfx": MODEL_ID_SMALL_SFX,
    "stable-audio-3-small": MODEL_ID_SMALL_MUSIC,
    "stable-audio-3-small-music": MODEL_ID_SMALL_MUSIC,
    MODEL_ID_SMALL_MUSIC.lower(): MODEL_ID_SMALL_MUSIC,
    "stable-audio-3-small-sfx": MODEL_ID_SMALL_SFX,
    MODEL_ID_SMALL_SFX.lower(): MODEL_ID_SMALL_SFX,
    "medium": MODEL_ID_MEDIUM,
    "stable-audio-3-medium": MODEL_ID_MEDIUM,
    MODEL_ID_MEDIUM.lower(): MODEL_ID_MEDIUM,
}


def _lazy_import_torch():
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): torch. "
            "Install via: pip install 'abstractmusic[stable-audio-3]'"
        ) from e
    return torch


def _lazy_import_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): huggingface_hub. "
            "Install via: pip install 'abstractmusic[stable-audio-3]'"
        ) from e
    return hf_hub_download


def _lazy_import_safetensors():
    try:
        from safetensors.torch import load_file  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): safetensors. "
            "Install via: pip install 'abstractmusic[stable-audio-3]'"
        ) from e
    return load_file


def _canonical_model_id(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return MODEL_ID_SMALL_MUSIC
    aliased = _MODEL_ALIASES.get(raw.lower())
    if aliased:
        return aliased
    repo_id = require_hf_repo_id(raw, field_name="model_id")
    if repo_id not in _MODEL_INFO:
        allowed = ", ".join(sorted(_MODEL_INFO))
        raise ValueError(f"StableAudio3Backend supports model_id values: {allowed}. Got {repo_id!r}.")
    return repo_id


def _hf_token() -> Optional[str]:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return None


def _prepare_hf_env() -> None:
    token = _hf_token()
    if token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = token


def _resolve_model_half(dtype: str, *, device: str) -> bool:
    d = str(dtype or "").strip().lower() or "auto"
    dev = str(device or "").strip().lower() or "cpu"
    return dev == "cuda" and d in {"auto", "float16", "fp16", "half"}


def _is_finite_audio(torch: Any, audio: Any) -> bool:
    try:
        if hasattr(audio, "detach"):
            return bool(torch.isfinite(audio.detach()).all().item())
    except Exception:
        return False
    try:
        import numpy as np  # type: ignore

        return bool(np.isfinite(np.asarray(audio)).all())
    except Exception:
        return False


def _first_audio(audio: Any) -> Any:
    if hasattr(audio, "detach"):
        ndim = int(getattr(audio, "ndim", 0) or 0)
        if ndim >= 3:
            return audio[0]
        return audio
    if isinstance(audio, (list, tuple)) and audio:
        return audio[0]
    return audio


def _align_sample_size(model_config: Dict[str, Any], target_samples: int) -> int:
    model_block = dict(model_config.get("model") or {})
    pretransform = dict(model_block.get("pretransform") or {})
    pre_cfg = dict(pretransform.get("config") or {})
    ratio = int(pre_cfg.get("downsampling_ratio") or 1)
    if ratio > 1:
        target_samples = ((int(target_samples) + ratio - 1) // ratio) * ratio
        encoder_cfg = dict(dict(pre_cfg.get("encoder") or {}).get("config") or {})
        strides = encoder_cfg.get("strides") or [1]
        try:
            stride = int(strides[0])
        except Exception:
            stride = 1
        chunk_size = int(encoder_cfg.get("chunk_size") or 32)
        latent_align = max(1, chunk_size // max(1, stride))
        align = max(1, ratio * latent_align)
        target_samples = ((int(target_samples) + align - 1) // align) * align
    sample_size = int(model_config.get("sample_size") or target_samples)
    return min(int(target_samples), sample_size)


def _copy_state_dict(model: Any, state_dict: Dict[str, Any]) -> None:
    model_state = model.state_dict()
    remapped: Dict[str, Any] = {}
    for key, value in state_dict.items():
        out_key = key
        if out_key not in model_state:
            parts = out_key.split(".")
            for i in range(1, len(parts)):
                candidate = ".".join(parts[:i]) + "." + ".".join(parts[i + 1 :])
                if candidate in model_state:
                    out_key = candidate
                    break
        if out_key in model_state and tuple(value.shape) == tuple(model_state[out_key].shape):
            remapped[out_key] = value
    model.load_state_dict(remapped, strict=False)


def _normalize_config_for_selected_repo(model_config: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    data = json.loads(json.dumps(model_config))
    conditioning = data.get("model", {}).get("conditioning", {})
    for item in conditioning.get("configs", []):
        if item.get("type") == "t5gemma":
            cfg = item.setdefault("config", {})
            cfg["repo_id"] = model_id
            cfg.setdefault("subfolder", "t5gemma-b-b-ul2")
    return data


@dataclass(frozen=True)
class StableAudio3BackendConfig:
    model_id: str = MODEL_ID_SMALL_MUSIC
    device: str = "auto"
    torch_dtype: str = "auto"
    duration_s: float = 30.0
    num_inference_steps: int = 16
    guidance_scale: float = 1.0
    sampler_type: str = "pingpong"
    duration_padding_sec: float = 6.0
    apg_scale: float = 1.0
    chunked_decode: Optional[bool] = True
    auto_retry_cpu_on_mps_error: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _canonical_model_id(self.model_id))


class _StableAudio3Runtime:
    def __init__(self, config: StableAudio3BackendConfig) -> None:
        self.config = config
        self.model = None
        self.model_config: Dict[str, Any] = {}
        self.device: str = "cpu"
        self.model_half = False

    def load(self) -> None:
        if self.model is not None:
            return

        _prepare_hf_env()
        torch = _lazy_import_torch()
        hf_hub_download = _lazy_import_huggingface_hub()
        load_file = _lazy_import_safetensors()
        from abstractmusic.vendor.stable_audio3_min.factory import create_diffusion_cond_from_config

        token = _hf_token()
        model_id = str(self.config.model_id)
        config_path = Path(hf_hub_download(repo_id=model_id, filename="model_config.json", token=token))
        ckpt_path = Path(hf_hub_download(repo_id=model_id, filename="model.safetensors", token=token))
        model_config = _normalize_config_for_selected_repo(json.loads(config_path.read_text()), model_id)

        device = _resolve_device(torch, self.config.device)
        model_half = _resolve_model_half(self.config.torch_dtype, device=device)

        model = create_diffusion_cond_from_config(model_config)
        state = load_file(str(ckpt_path))
        _copy_state_dict(model, state)
        model.to(device).eval().requires_grad_(False)
        if model_half:
            model.to(torch.float16)

        self.model = model
        self.model_config = model_config
        self.device = str(device)
        self.model_half = bool(model_half)

    def unload(self) -> None:
        """Best-effort: release the loaded model and allocator caches."""
        self.model = None
        self.model_config = {}
        self.device = "cpu"
        self.model_half = False
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
        if self.model is not None:
            try:
                sr = int(getattr(self.model, "sample_rate"))
                if sr > 0:
                    return sr
            except Exception:
                pass
        return int(_MODEL_INFO[str(self.config.model_id)]["sample_rate_hz"])

    def generate(
        self,
        *,
        prompt: str,
        duration_s: float,
        steps: int,
        cfg_scale: float,
        seed: Optional[int],
        sampler_type: str,
        duration_padding_sec: float,
        apg_scale: float,
        chunked_decode: Optional[bool],
    ) -> Any:
        self.load()
        assert self.model is not None

        torch = _lazy_import_torch()
        from abstractmusic.vendor.stable_audio3_min.inference.sampling import sample_diffusion

        device = self.device
        sample_rate = self._sample_rate()
        target_samples = int((float(duration_s) + float(duration_padding_sec)) * float(sample_rate))
        audio_sample_size = _align_sample_size(self.model_config, target_samples)
        pretransform = getattr(self.model, "pretransform", None)
        latent_sample_size = int(audio_sample_size)
        if pretransform is not None:
            latent_sample_size = latent_sample_size // int(getattr(pretransform, "downsampling_ratio", 1) or 1)

        seed_value = int(seed) if seed is not None else -1
        if seed_value < 0:
            seed_value = int(torch.randint(0, 99999, (1,)).item())
        torch.manual_seed(seed_value)
        if str(device) == "cuda":
            torch.cuda.manual_seed_all(seed_value)

        noise = torch.randn([1, int(getattr(self.model, "io_channels")), latent_sample_size], device=device)
        conditioning = [{"prompt": prompt, "seconds_total": float(duration_s)}]
        conditioning_tensors = self.model.conditioner(conditioning, device)

        mask = torch.zeros((1, 1, latent_sample_size), device=device)
        inpaint_input = torch.zeros((1, int(getattr(self.model, "io_channels")), latent_sample_size), device=device)
        conditioning_tensors["inpaint_mask"] = [mask]
        conditioning_tensors["inpaint_masked_input"] = [inpaint_input]
        conditioning_inputs = self.model.get_conditioning_inputs(conditioning_tensors)

        model_dtype = next(self.model.model.parameters()).dtype
        noise = noise.type(model_dtype)
        conditioning_inputs = {
            key: value.type(model_dtype) if value is not None and hasattr(value, "type") else value
            for key, value in conditioning_inputs.items()
        }

        result = sample_diffusion(
            model=self.model.model,
            noise=noise,
            cond_inputs=conditioning_inputs,
            diffusion_objective=str(getattr(self.model, "diffusion_objective")),
            steps=int(steps),
            cfg_scale=float(cfg_scale),
            conditioning=conditioning,
            sample_rate=sample_rate,
            pretransform=pretransform,
            mask_padding_attention=bool(getattr(self.model, "mask_padding_attention", False)),
            use_effective_length_for_schedule=bool(getattr(self.model, "use_effective_length_for_schedule", False)),
            headroom_seconds=float(duration_padding_sec),
            dist_shift=getattr(self.model, "sampling_dist_shift", None),
            sampler_type=str(sampler_type or "pingpong"),
            batch_cfg=True,
            rescale_cfg=True,
            apg_scale=float(apg_scale),
            chunked_decode=chunked_decode,
            disable_tqdm=True,
        )
        result = result.to(torch.float32).clamp(-1, 1)
        max_length = int(float(duration_s) * float(sample_rate))
        return result[:, :, :max_length]


class StableAudio3Backend:
    """Local text-to-music backend powered by an AbstractMusic-owned SA3 runtime."""

    backend_id = "abstractmusic:stable-audio-3"

    def __init__(self, *, config: Optional[StableAudio3BackendConfig] = None) -> None:
        self._config = config or StableAudio3BackendConfig()
        self._runtime = _StableAudio3Runtime(self._config)

    def get_capabilities(self) -> MusicBackendCapabilities:
        info = _MODEL_INFO[str(self._config.model_id)]
        return MusicBackendCapabilities(
            supported_tasks=("text_to_music", "text_to_audio"),
            output_formats=("wav",),
            supports_lyrics=False,
            supports_negative_prompt=False,
            supports_guidance_scale=True,
            supports_reference_audio=False,
            supports_video=False,
            max_duration_s=float(info["max_duration_s"]),
            sample_rates_hz=(int(info["sample_rate_hz"]),),
            model_id=str(self._config.model_id),
            license="Stability AI Community License plus text-encoder terms",
            commercial_allowed=False,
            official_8bit_available=False,
            preferred_precision="package-owned runtime; float32 on CPU/MPS, float16 only on CUDA",
        )

    def list_provider_models(self, *, task: Optional[str] = None) -> Sequence[ProviderModelInfo]:
        if task is not None and task not in {"text_to_music", "text_to_audio"}:
            return ()
        return tuple(
            ProviderModelInfo(
                id=model_id,
                object="model",
                owned_by="stabilityai",
                capabilities=("text_to_music", "text_to_audio"),
                raw={"backend": self.backend_id, "license": "Stability AI Community License", **info},
            )
            for model_id, info in _MODEL_INFO.items()
        )

    def preload(self) -> None:
        self._runtime.load()

    def unload(self) -> None:
        self._runtime.unload()

    def generate_audio(self, request: AudioGenerationRequest) -> GeneratedAsset:
        if request.lyrics:
            raise ValueError("Stable Audio 3 backend does not support lyrics.")
        if request.negative_prompt:
            raise ValueError("Stable Audio 3 backend does not support negative_prompt yet.")

        duration_s = float(request.duration_s if request.duration_s is not None else self._config.duration_s)
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        max_duration_s = float(_MODEL_INFO[str(self._config.model_id)]["max_duration_s"])
        if duration_s > max_duration_s:
            raise ValueError(f"Stable Audio 3 model supports up to {max_duration_s:g} seconds")

        steps = int(request.num_inference_steps if request.num_inference_steps is not None else self._config.num_inference_steps)
        cfg_scale = float(request.guidance_scale if request.guidance_scale is not None else self._config.guidance_scale)
        extra = dict(request.extra or {})
        sampler_type = str(extra.get("sampler_type", self._config.sampler_type) or self._config.sampler_type)
        duration_padding_sec = float(extra.get("duration_padding_sec", self._config.duration_padding_sec))
        apg_scale = float(extra.get("apg_scale", self._config.apg_scale))
        chunked_decode = extra.get("chunked_decode", self._config.chunked_decode)
        if isinstance(chunked_decode, str):
            chunked_decode = chunked_decode.strip().lower() in {"1", "on", "true", "yes"}

        try:
            audio = self._runtime.generate(
                prompt=str(request.prompt or ""),
                duration_s=duration_s,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=request.seed,
                sampler_type=sampler_type,
                duration_padding_sec=duration_padding_sec,
                apg_scale=apg_scale,
                chunked_decode=chunked_decode,
            )
        except RuntimeError as e:
            is_mps = str(getattr(self._runtime, "device", "")).lower() == "mps"
            if is_mps and self._config.auto_retry_cpu_on_mps_error:
                print(
                    "WARNING #FALLBACK : Stable Audio 3 internal runtime failed on Apple MPS; "
                    f"retrying on CPU float32. Original error: {e}",
                    file=sys.stderr,
                )
                self._config = replace(self._config, device="cpu", torch_dtype="float32")
                self._runtime = _StableAudio3Runtime(self._config)
                audio = self._runtime.generate(
                    prompt=str(request.prompt or ""),
                    duration_s=duration_s,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    seed=request.seed,
                    sampler_type=sampler_type,
                    duration_padding_sec=duration_padding_sec,
                    apg_scale=apg_scale,
                    chunked_decode=chunked_decode,
                )
            else:
                raise

        torch = _lazy_import_torch()
        if not _is_finite_audio(torch, audio):
            raise RuntimeError("Stable Audio 3 internal runtime produced non-finite audio.")

        sample_rate = int(self._runtime._sample_rate())
        wav_bytes = _encode_wav_bytes(_first_audio(audio), sample_rate=sample_rate)
        metadata: Dict[str, Any] = {
            "backend": self.backend_id,
            "model_id": str(self._config.model_id),
            "device": str(getattr(self._runtime, "device", "cpu")),
            "model_half": bool(getattr(self._runtime, "model_half", False)),
            "duration_s": float(duration_s),
            "sampling_rate": int(sample_rate),
            "num_inference_steps": int(steps),
            "guidance_scale": float(cfg_scale),
            "sampler_type": sampler_type,
            "duration_padding_sec": float(duration_padding_sec),
            "apg_scale": float(apg_scale),
            "chunked_decode": chunked_decode,
            "runtime": "abstractmusic.vendor.stable_audio3_min",
        }
        if request.seed is not None:
            metadata["seed"] = int(request.seed)
        return GeneratedAsset(data=bytes(wav_bytes), mime_type="audio/wav", metadata=metadata)
