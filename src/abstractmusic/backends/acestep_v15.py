"""
ACE-Step v1.5 local (in-process) backend for AbstractMusic.

This backend integrates the Hugging Face checkpoint repo `ACE-Step/Ace-Step1.5`
directly, without relying on any external ACE-Step server/daemon.

Design notes:
- We load ACE-Step components from the HF cache on first use.
- ACE-Step's custom Transformers model code is **vendored** into
  `abstractmusic.vendor.acestep_v15_turbo`, so we do **not** use
  `trust_remote_code`.
- Output is encoded as WAV bytes using the stdlib `wave` module.
- On Apple Silicon (MPS), ACE-Step upstream advises disabling bf16; we follow
  their approach by defaulting to float16 on MPS (bf16 disabled), float32 on CPU,
  and bf16 on CUDA/XPU.
- Any device fallback is explicit and tagged with #FALLBACK.
"""

from __future__ import annotations

import inspect
import io
import math
import os
import random
import re
import subprocess
import sys
import wave
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional, Tuple

from ..audio_analysis import (
    inspect_harmonic_diversity_bytes,
    inspect_music_signal_bytes,
    inspect_spectrotemporal_modulation_bytes,
)
from ..errors import OptionalDependencyMissingError
from ..huggingface import require_hf_repo_id
from ..types import AudioGenerationRequest, GeneratedAsset, MusicBackendCapabilities

if os.environ.get("DIFFUSERS_SLOW_IMPORT", "").strip().upper() in {"1", "ON", "YES", "TRUE"}:
    os.environ["DIFFUSERS_SLOW_IMPORT"] = "0"


def _lazy_import_torch():
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): torch. Install via: pip install 'torch'"
        ) from e
    return torch


def _lazy_import_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): numpy. Install via: pip install 'numpy'"
        ) from e
    return np


def _lazy_import_transformers():
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): transformers. Install via: pip install 'transformers'"
        ) from e
    return AutoModel, AutoTokenizer


def _lazy_import_transformers_lm():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): transformers. Install via: pip install 'transformers'"
        ) from e
    return AutoModelForCausalLM, AutoTokenizer


def _patch_transformers_rope_validation() -> None:
    """Avoid deprecated rope_config_validation warnings by using validate_rope directly."""

    try:
        from transformers import modeling_rope_utils as rope_utils  # type: ignore
    except Exception:
        return
    original = getattr(rope_utils, "rope_config_validation", None)
    if original is None:
        return
    if getattr(original, "_abstractmusic_patched", False):
        return

    def _rope_config_validation(config: Any, ignore_keys: Any = None) -> None:
        if hasattr(config, "standardize_rope_params"):
            try:
                config.standardize_rope_params()
            except Exception:
                pass
        if hasattr(config, "validate_rope"):
            try:
                config.validate_rope(ignore_keys=ignore_keys)
                return
            except Exception:
                print(
                    "WARNING #FALLBACK : validate_rope failed; skipping RoPE validation "
                    "to avoid deprecated rope_config_validation warning.",
                    file=sys.stderr,
                )
                return
        # No validate_rope available; skip deprecated validation.
        print(
            "WARNING #FALLBACK : RoPE validation unavailable; skipping deprecated "
            "rope_config_validation.",
            file=sys.stderr,
        )

    _rope_config_validation._abstractmusic_patched = True  # type: ignore[attr-defined]
    rope_utils.rope_config_validation = _rope_config_validation  # type: ignore[assignment]


def _patch_transformers_torch_dtype_property() -> None:
    """Map deprecated torch_dtype property to dtype without warnings."""

    try:
        from transformers.configuration_utils import PretrainedConfig  # type: ignore
    except Exception:
        return
    if getattr(PretrainedConfig, "_abstractmusic_torch_dtype_patched", False):
        return

    def _get(self):
        return getattr(self, "dtype", None)

    def _set(self, value):
        setattr(self, "dtype", value)

    PretrainedConfig.torch_dtype = property(_get, _set)  # type: ignore[assignment]
    PretrainedConfig._abstractmusic_torch_dtype_patched = True  # type: ignore[attr-defined]


def _patch_oobleck_weight_norm(torch_mod: Any) -> None:
    """Use the modern weight_norm API with key conversion to avoid warnings."""

    try:
        from torch.nn.utils.parametrizations import weight_norm as new_weight_norm  # type: ignore
    except Exception:
        return
    try:
        _disable_diffusers_slow_import()
        import diffusers.models.autoencoders.autoencoder_oobleck as oobleck_mod  # type: ignore
    except Exception:
        return

    if getattr(oobleck_mod, "_abstractmusic_weight_norm_patched", False):
        return

    oobleck_mod.weight_norm = new_weight_norm  # type: ignore[assignment]
    AutoencoderOobleck = getattr(oobleck_mod, "AutoencoderOobleck", None)
    if AutoencoderOobleck is None:
        return

    original_fix = getattr(AutoencoderOobleck, "_fix_state_dict_keys_on_load", None)

    def _fix_state_dict_keys_on_load(self, state_dict: Any) -> None:
        if callable(original_fix):
            original_fix(self, state_dict)
        if not isinstance(state_dict, dict):
            return
        to_add: Dict[str, Any] = {}
        to_remove = []
        for key in list(state_dict.keys()):
            if not isinstance(key, str) or not key.endswith(".weight_g"):
                continue
            prefix = key[: -len("weight_g")]
            v_key = f"{prefix}weight_v"
            if v_key not in state_dict:
                continue
            to_add[f"{prefix}parametrizations.weight.original0"] = state_dict[key]
            to_add[f"{prefix}parametrizations.weight.original1"] = state_dict[v_key]
            to_remove.extend([key, v_key])
        for key in to_remove:
            state_dict.pop(key, None)
        state_dict.update(to_add)

    AutoencoderOobleck._fix_state_dict_keys_on_load = _fix_state_dict_keys_on_load  # type: ignore[assignment]
    oobleck_mod._abstractmusic_weight_norm_patched = True  # type: ignore[attr-defined]


def _lazy_import_acestep_v15_turbo_model():
    """Import the vendored ACE-Step v1.5 turbo Transformers model class.

    We vendor this code because Transformers `trust_remote_code` only searches
    the repo root for custom modules, but ACE-Step stores its modules under the
    checkpoint subfolder (`acestep-v15-turbo/`).
    """

    try:
        from ..vendor.acestep_v15_turbo.modeling_acestep_v15_turbo import (  # type: ignore
            AceStepConditionGenerationModel,
        )
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Failed to import vendored ACE-Step v1.5 turbo model code. "
            "Ensure `abstractmusic` is installed correctly (vendored module missing), and that "
            "optional runtime deps are present (einops)."
        ) from e
    return AceStepConditionGenerationModel


def _lazy_import_diffusers_oobleck():
    try:
        _disable_diffusers_slow_import()
        from diffusers.models.autoencoders.autoencoder_oobleck import (  # type: ignore
            AutoencoderOobleck,
        )
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): diffusers. Install via: pip install 'diffusers'"
        ) from e
    return AutoencoderOobleck


def _lazy_import_hf_hub_download():
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): huggingface_hub. "
            "Install via: pip install 'huggingface_hub'"
        ) from e
    return hf_hub_download


def _disable_diffusers_slow_import() -> None:
    """Ensure Diffusers does not eagerly import unrelated transformer modules."""

    os.environ["DIFFUSERS_SLOW_IMPORT"] = "0"
    try:
        import diffusers.utils.import_utils as diffusers_import_utils  # type: ignore

        diffusers_import_utils.DIFFUSERS_SLOW_IMPORT = False  # type: ignore[attr-defined]
    except Exception:
        pass


def _patch_hf_hub_download() -> None:
    """Strip deprecated hf_hub_download args injected by upstream callers."""

    def _wrap(fn: Any):
        if getattr(fn, "_abstractmusic_patched", False):
            return fn

        def _hf_hub_download(*args: Any, **kwargs: Any):
            if "local_dir_use_symlinks" in kwargs:
                kwargs.pop("local_dir_use_symlinks", None)
            return fn(*args, **kwargs)

        _hf_hub_download._abstractmusic_patched = True  # type: ignore[attr-defined]
        return _hf_hub_download

    try:
        import huggingface_hub  # type: ignore
    except Exception:
        return
    try:
        import huggingface_hub.file_download as hf_file_download  # type: ignore
    except Exception:
        hf_file_download = None
    try:
        import diffusers.configuration_utils as diffusers_config_utils  # type: ignore
    except Exception:
        diffusers_config_utils = None

    if hasattr(huggingface_hub, "hf_hub_download"):
        huggingface_hub.hf_hub_download = _wrap(huggingface_hub.hf_hub_download)  # type: ignore[assignment]
    if hf_file_download is not None and hasattr(hf_file_download, "hf_hub_download"):
        hf_file_download.hf_hub_download = _wrap(hf_file_download.hf_hub_download)  # type: ignore[assignment]
    if diffusers_config_utils is not None and hasattr(diffusers_config_utils, "hf_hub_download"):
        diffusers_config_utils.hf_hub_download = _wrap(diffusers_config_utils.hf_hub_download)  # type: ignore[assignment]


def _from_pretrained_transformers(cls: Any, *, dtype: Any, **kwargs: Any) -> Any:
    """Call Transformers from_pretrained using `dtype` to avoid warnings."""

    return cls.from_pretrained(dtype=dtype, **kwargs)


def _from_pretrained_diffusers(cls: Any, *, dtype: Any, **kwargs: Any) -> Any:
    """Call Diffusers from_pretrained using torch_dtype."""

    return cls.from_pretrained(torch_dtype=dtype, **kwargs)


def _postprocess_caption_yaml_value(text: str) -> str:
    """Flatten YAML-style multi-line caption output into one readable line."""
    if not text:
        return text
    parts = [line.strip() for line in str(text).splitlines() if line.strip()]
    return " ".join(parts)


def _get_system_memory_gb() -> Optional[float]:
    """Best-effort system memory in GiB."""

    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
            total_bytes = int(out.strip())
            if total_bytes > 0:
                return float(total_bytes) / float(1024**3)
        except Exception:
            pass

    try:
        import psutil  # type: ignore

        total_bytes = int(psutil.virtual_memory().total)
        if total_bytes > 0:
            return float(total_bytes) / float(1024**3)
    except Exception:
        pass

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_bytes = int(pages) * int(page_size)
    except Exception:
        return None
    if total_bytes <= 0:
        return None
    return float(total_bytes) / float(1024**3)


def _normalize_mps_env_ratio(ratio: Any, *, label: str) -> Optional[float]:
    try:
        r = float(ratio)
    except Exception:
        print(
            f"WARNING #FALLBACK : Invalid {label} (non-numeric). Ignoring value={ratio!r}.",
            file=sys.stderr,
        )
        return None
    if r <= 0:
        print(
            f"WARNING #FALLBACK : Invalid {label} (<=0). Ignoring value={r!r}.",
            file=sys.stderr,
        )
        return None
    if r > 1.0:
        print(
            f"WARNING #FALLBACK : {label}={r:.4f} is out of range; clamping to 1.0.",
            file=sys.stderr,
        )
        return 1.0
    return r


def _normalize_mps_fraction(ratio: Any, *, label: str) -> Optional[float]:
    try:
        r = float(ratio)
    except Exception:
        print(
            f"WARNING #FALLBACK : Invalid {label} (non-numeric). Ignoring value={ratio!r}.",
            file=sys.stderr,
        )
        return None
    if r <= 0:
        print(
            f"WARNING #FALLBACK : Invalid {label} (<=0). Ignoring value={r!r}.",
            file=sys.stderr,
        )
        return None
    if r > 2.0:
        print(
            f"WARNING #FALLBACK : {label}={r:.4f} exceeds the allowed 2.0 max; clamping to 2.0.",
            file=sys.stderr,
        )
        return 2.0
    return r


def _maybe_configure_mps_high_watermark(
    *, mps_max_memory_gb: Optional[float], mps_high_watermark_ratio: Optional[float]
) -> None:
    """Configure the MPS high-watermark ratio if requested."""

    env_high_raw = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
    env_low_raw = os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO")

    env_high = None
    env_high_invalid = False
    if env_high_raw:
        env_high = _normalize_mps_env_ratio(env_high_raw, label="PYTORCH_MPS_HIGH_WATERMARK_RATIO")
        env_high_invalid = env_high is None
    env_low = None
    env_low_invalid = False
    if env_low_raw:
        env_low = _normalize_mps_env_ratio(env_low_raw, label="PYTORCH_MPS_LOW_WATERMARK_RATIO")
        env_low_invalid = env_low is None

    ratio = None
    user_specified = mps_high_watermark_ratio is not None or mps_max_memory_gb is not None
    if mps_high_watermark_ratio is not None:
        ratio = _normalize_mps_fraction(mps_high_watermark_ratio, label="mps_high_watermark_ratio")
    if ratio is None and mps_max_memory_gb is not None:
        try:
            max_gb = float(mps_max_memory_gb)
        except Exception:
            max_gb = None
        if max_gb is not None and max_gb > 0:
            if max_gb > 16.0:
                print(
                    "WARNING #FALLBACK : mps_max_memory_gb exceeds the allowed 16 GiB cap; "
                    "clamping to 16.0.",
                    file=sys.stderr,
                )
                max_gb = 16.0
            base_gb = None
            torch_mod = sys.modules.get("torch")
            if torch_mod is not None:
                try:
                    base_bytes = int(torch_mod.mps.recommended_max_memory())
                    if base_bytes > 0:
                        base_gb = float(base_bytes) / float(1024**3)
                except Exception:
                    base_gb = None
            if base_gb is None or base_gb <= 0:
                base_gb = _get_system_memory_gb()
            if base_gb is None or base_gb <= 0:
                print(
                    "WARNING #FALLBACK : Unable to resolve MPS/system memory; "
                    "using a conservative 0.66 MPS memory fraction.",
                    file=sys.stderr,
                )
                ratio = 0.66
            else:
                ratio = max_gb / float(base_gb)

    if ratio is not None:
        ratio = _normalize_mps_fraction(ratio, label="mps_high_watermark_ratio")

    if ratio is None and not user_specified and env_high is not None:
        ratio = env_high
    if ratio is None and not user_specified and env_high_invalid:
        print(
            "WARNING #FALLBACK : Invalid MPS high-watermark env var detected; "
            "resetting to a safe default of 0.80.",
            file=sys.stderr,
        )
        ratio = 0.80

    if ratio is None:
        return

    env_ratio = _normalize_mps_env_ratio(min(ratio, 1.0), label="PYTORCH_MPS_HIGH_WATERMARK_RATIO")
    if env_ratio is None:
        return

    if env_high is None or abs(env_high - env_ratio) > 1e-6 or user_specified:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = f"{env_ratio:.4f}"

    if env_low is None or env_low_invalid or env_low >= env_ratio:
        low_ratio = max(0.05, min(env_ratio * 0.9, env_ratio - 0.02))
        if low_ratio <= 0:
            low_ratio = max(0.01, env_ratio * 0.8)
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = f"{low_ratio:.4f}"

    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        try:
            torch_mod.mps.set_per_process_memory_fraction(float(ratio))
        except Exception:
            pass


def _resolve_device(torch_mod: Any, device: str) -> str:
    d = str(device or "").strip().lower() or "auto"
    if d != "auto":
        return d
    try:
        if bool(getattr(torch_mod.cuda, "is_available", lambda: False)()):
            return "cuda"
    except Exception:
        pass
    try:
        mps = getattr(torch_mod.backends, "mps", None)
        if mps is not None and bool(getattr(mps, "is_available", lambda: False)()):
            return "mps"
    except Exception:
        pass
    # Intel XPU (optional)
    try:
        xpu = getattr(torch_mod, "xpu", None)
        if xpu is not None and bool(getattr(xpu, "is_available", lambda: False)()):
            return "xpu"
    except Exception:
        pass
    return "cpu"


def _dtype_from_str(torch_mod: Any, dtype: str) -> Any:
    d = str(dtype or "").strip().lower()
    if d in {"float16", "fp16"}:
        return torch_mod.float16
    if d in {"bfloat16", "bf16"}:
        return torch_mod.bfloat16
    return torch_mod.float32


def _default_model_dtype(torch_mod: Any, device: str) -> Any:
    # Mirrors upstream ACE-Step guidance:
    # - CUDA/XPU: bf16
    # - MPS (macOS): avoid bf16 → fp16
    # - CPU: fp32
    if str(device) in {"cuda", "xpu"}:
        return torch_mod.bfloat16
    if str(device) == "mps":
        return torch_mod.float16
    return torch_mod.float32


def _default_vae_dtype(torch_mod: Any, device: str) -> Any:
    # VAE decode can be memory-heavy; fp16 helps on accelerators (CUDA/XPU/MPS).
    if str(device) in {"cuda", "xpu", "mps"}:
        return torch_mod.float16
    return torch_mod.float32


def _resolve_text_encoder_runtime(torch_mod: Any, *, model_device: str, model_dtype: Any) -> Tuple[str, Any]:
    """Resolve text-encoder device/dtype.

    On MPS, Qwen3 embedding can hit MPSGraph mixed-dtype kernel issues in some
    builds. Running the text encoder on CPU float32 is more stable; we then cast
    resulting hidden states to the DiT model dtype/device.
    """

    if str(model_device) == "mps":
        return "cpu", torch_mod.float32
    return model_device, model_dtype


def _encode_wav_bytes(audio: Any, *, sample_rate: int) -> bytes:
    """Encode waveform into 16-bit PCM WAV bytes.

    Accepts:
    - torch.Tensor [C, T] or [T] or [1, C, T]
    - numpy arrays with the same shapes
    """

    np = _lazy_import_numpy()
    torch = _lazy_import_torch()

    if isinstance(audio, torch.Tensor):
        x = audio.detach().to("cpu")
        x = x.float().numpy()
    else:
        x = np.asarray(audio, dtype=np.float32)

    # Normalize to [T, C]
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim == 2:
        # Could be [C, T] or [T, C]. Heuristic: if first dim is 1/2 assume channels-first.
        if x.shape[0] in {1, 2} and x.shape[1] > x.shape[0]:
            x = x.T
    else:
        raise ValueError(f"Unsupported audio array shape: {tuple(x.shape)}")

    # Clip and convert to 16-bit PCM.
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2", copy=False)
    n_channels = int(pcm.shape[1])

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes(order="C"))
    return buf.getvalue()


def _decode_wav_bytes(wav_bytes: bytes) -> Tuple[Any, int]:
    """Decode 16-bit PCM WAV bytes into a float32 torch tensor [C, T]."""
    np = _lazy_import_numpy()
    torch = _lazy_import_torch()

    with wave.open(io.BytesIO(bytes(wav_bytes)), "rb") as wf:
        sample_rate = int(wf.getframerate())
        n_channels = int(wf.getnchannels())
        sample_width = int(wf.getsampwidth())
        n_frames = int(wf.getnframes())
        raw = wf.readframes(n_frames)

    if sample_width != 2:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    pcm = np.frombuffer(raw, dtype="<i2")
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels)
    else:
        pcm = pcm.reshape(-1, 1)
    audio = torch.from_numpy(pcm.astype(np.float32, copy=False).T / 32767.0)
    return audio, sample_rate


def _peak_normalize(wav: Any, *, target_db: float = -1.0) -> Any:
    """Peak-normalize an audio tensor/array to a target peak dBFS (default -1 dB)."""
    torch = _lazy_import_torch()
    np = _lazy_import_numpy()

    target_peak = float(10 ** (float(target_db) / 20.0))

    if isinstance(wav, torch.Tensor):
        x = wav
        peak = float(x.abs().max().item()) if x.numel() else 0.0
        if peak <= 0:
            return x
        return x * (target_peak / peak)

    x = np.asarray(wav, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0:
        return x
    return x * (target_peak / peak)


def _remove_dc_offset(wav: Any) -> Any:
    """Remove per-channel DC offset from waveform tensor/array."""
    torch = _lazy_import_torch()
    np = _lazy_import_numpy()

    if isinstance(wav, torch.Tensor):
        x = wav
        if x.numel() == 0:
            return x
        if x.ndim == 3:
            # Prefer [B, C, T] convention used by decode() in this backend.
            if int(x.shape[1]) <= 8 and int(x.shape[2]) > int(x.shape[1]):
                return x - x.mean(dim=-1, keepdim=True)
            # Fallback for [B, T, C].
            return x - x.mean(dim=-2, keepdim=True)
        if x.ndim == 2:
            # [C, T] or [T, C]
            if int(x.shape[0]) <= 8 and int(x.shape[1]) > int(x.shape[0]):
                return x - x.mean(dim=-1, keepdim=True)
            return x - x.mean(dim=0, keepdim=True)
        if x.ndim == 1:
            return x - x.mean()
        return x

    x = np.asarray(wav, dtype=np.float32)
    if x.size == 0:
        return x
    if x.ndim == 3:
        if int(x.shape[1]) <= 8 and int(x.shape[2]) > int(x.shape[1]):
            return x - x.mean(axis=-1, keepdims=True)
        return x - x.mean(axis=-2, keepdims=True)
    if x.ndim == 2:
        if int(x.shape[0]) <= 8 and int(x.shape[1]) > int(x.shape[0]):
            return x - x.mean(axis=-1, keepdims=True)
        return x - x.mean(axis=0, keepdims=True)
    if x.ndim == 1:
        return x - x.mean()
    return x


def _ensure_finite_audio(wav: Any) -> Any:
    """Replace NaN/Inf values in waveform before PCM encoding."""
    torch = _lazy_import_torch()
    np = _lazy_import_numpy()

    if isinstance(wav, torch.Tensor):
        try:
            has_non_finite = not bool(torch.isfinite(wav).all().item())
        except Exception:
            has_non_finite = False
        if has_non_finite:
            print(
                "WARNING #FALLBACK : Non-finite values detected in decoded waveform; replacing NaN/Inf before WAV encoding.",
                file=sys.stderr,
            )
            wav = torch.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0)
        return wav

    x = np.asarray(wav, dtype=np.float32)
    if not np.isfinite(x).all():
        print(
            "WARNING #FALLBACK : Non-finite values detected in decoded waveform; replacing NaN/Inf before WAV encoding.",
            file=sys.stderr,
        )
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    return x


def _sft_gen_prompt(instruction: str, caption: str, metas: str) -> str:
    # Upstream ACE-Step prompt format (see ACE-Step v1.5 repo/model card).
    return f"# Instruction\n{instruction}\n\n# Caption\n{caption}\n\n# Metas\n{metas}<|endoftext|>\n"


def _format_lyrics(lyrics: str, language: str) -> str:
    return f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"


_DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
_DEFAULT_COVER_DIT_INSTRUCTION = "Generate audio semantic tokens based on the given conditions:"


def _default_meta_string(duration_s: float) -> str:
    # Mirrors upstream _create_default_meta.
    ds = max(1, int(round(float(duration_s))))
    return (
        "- bpm: N/A\n"
        "- timesignature: N/A\n"
        "- keyscale: N/A\n"
        f"- duration: {ds} seconds\n"
    )


def _meta_string(
    *,
    duration_s: float,
    bpm: Optional[int],
    keyscale: str,
    timesignature: str,
) -> str:
    ds = max(1, int(round(float(duration_s))))
    bpm_text = "N/A" if bpm is None else str(int(bpm))
    keyscale_text = str(keyscale or "").strip() or "N/A"
    timesignature_text = str(timesignature or "").strip() or "N/A"
    return (
        f"- bpm: {bpm_text}\n"
        f"- timesignature: {timesignature_text}\n"
        f"- keyscale: {keyscale_text}\n"
        f"- duration: {ds} seconds\n"
    )


# Pin to a known-good commit for deterministic weights/config.
# Users may override via AceStepV15BackendConfig(revision=...) / --revision.
_DEFAULT_ACESTEP_V15_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"


@dataclass(frozen=True)
class AceStepV15BackendConfig:
    """Configuration for ACE-Step v1.5 local inference."""

    repo_id: str = "ACE-Step/Ace-Step1.5"
    revision: Optional[str] = _DEFAULT_ACESTEP_V15_REVISION
    local_files_only: bool = True

    device: str = "auto"
    torch_dtype: str = "auto"  # auto|float32|float16|bfloat16
    vae_torch_dtype: str = "auto"  # auto|float32|float16|bfloat16
    mps_max_memory_gb: Optional[float] = 16.0
    mps_high_watermark_ratio: Optional[float] = None

    # Model subfolders inside the repo.
    dit_subfolder: str = "acestep-v15-turbo"
    vae_subfolder: str = "vae"
    text_encoder_subfolder: str = "Qwen3-Embedding-0.6B"

    # Generation defaults.
    default_duration_s: float = 10.0
    frames_per_second: int = 25
    refer_timbre_frames: int = 750
    fix_nfe: int = 8  # Turbo schedule length (ACE-Step v1.5 turbo)
    shift: float = 3.0
    infer_method: str = "ode"
    dcw_enabled: bool = True
    dcw_mode: str = "double"
    dcw_scaler: float = 0.05
    dcw_high_scaler: float = 0.02
    enable_normalization: bool = True
    normalization_db: float = -1.0
    # Match upstream text2music conditioning defaults.
    # The package-owned DiT path collapses toward static tones when the
    # text-to-music context is a repeated silence latent. A seeded random
    # source context keeps the non-cover path musically varied while remaining
    # deterministic for a given request seed.
    use_random_src_latents: bool = True
    use_sft_prompt: bool = True
    chunk_mask_mode: str = "auto"  # auto|zeros|ones
    reference_mode: str = "silence"  # zeros|silence

    # VAE decode tiling (helps on MPS/unified memory).
    mps_decode_chunk_frames: int = 32
    mps_decode_overlap_frames: int = 8
    cpu_decode_chunk_frames: int = 256
    cpu_decode_overlap_frames: int = 64

    # When running on MPS, retry on CPU if an op is unsupported.
    auto_retry_cpu_on_mps_error: bool = True
    mps_force_cpu_decode_min_duration_s: float = 30.0
    mps_segment_generation_min_duration_s: float = 30.0
    mps_segment_duration_s: float = 10.0
    mps_segment_crossfade_s: float = 0.0

    # Experimental internal 5Hz LM planner. Keep it opt-in because feeding its
    # coarse audio-code hints as cover conditioning can imprint 5Hz artifacts.
    use_audio_code_planner: bool = False
    planner_min_duration_s: float = 10.0
    lm_model_subfolder: str = "acestep-5Hz-lm-1.7B"
    lm_fallback_repo_id: str = "ACE-Step/acestep-5Hz-lm-0.6B"
    lm_device: str = "auto"  # auto|cuda|xpu|mps|cpu
    lm_torch_dtype: str = "auto"  # auto|float32|float16|bfloat16
    lm_temperature: float = 0.85
    lm_cfg_scale: float = 2.0
    lm_top_k: int = 0
    lm_top_p: float = 0.9
    lm_negative_prompt: str = "NO USER INPUT"
    planner_cover_strength: float = 0.50
    allow_direct_text_fallback: bool = True
    quality_retry_enabled: bool = True
    quality_retry_max_attempts: int = 3
    quality_retry_fallback_seeds: tuple[int, ...] = (123, 124, 321)

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_id", require_hf_repo_id(self.repo_id, field_name="repo_id"))
        object.__setattr__(
            self,
            "lm_fallback_repo_id",
            require_hf_repo_id(self.lm_fallback_repo_id, field_name="lm_fallback_repo_id"),
        )


class AceStepV15Backend:
    """ACE-Step 1.5 backend (local in-process)."""

    backend_id = "abstractmusic:acestep-v15"

    def __init__(self, *, config: AceStepV15BackendConfig) -> None:
        self._config = config
        self._loaded = False

        self._device: Optional[str] = None
        self._text_device: Optional[str] = None
        self._dtype: Any = None
        self._vae_dtype: Any = None

        self._model: Any = None
        self._vae: Any = None
        self._text_tokenizer: Any = None
        self._text_encoder: Any = None
        self._silence_latent: Any = None
        self._lm_tokenizer: Any = None
        self._lm_model: Any = None
        self._lm_device: Optional[str] = None
        self._lm_dtype: Any = None
        self._lm_model_label: Optional[str] = None
        self._lm_audio_token_ids: tuple[int, ...] = ()
        self._lm_think_end_token_id: Optional[int] = None

    def get_capabilities(self) -> MusicBackendCapabilities:
        return MusicBackendCapabilities(
            supported_tasks=("text_to_music",),
            output_formats=("wav",),
            supports_lyrics=True,
            supports_negative_prompt=False,
            supports_guidance_scale=False,
            supports_reference_audio=False,
            supports_video=False,
            max_duration_s=600.0,
            sample_rates_hz=(48000,),
            model_id=str(self._config.repo_id),
            license="MIT",
            commercial_allowed=True,
            official_8bit_available=False,
            preferred_precision="bf16 on CUDA/XPU, fp16 on MPS, fp32 on CPU when no official 8-bit artifact is available",
        )

    def _resolve_dtypes(self, torch_mod: Any, device: str) -> Tuple[Any, Any]:
        if str(self._config.torch_dtype).strip().lower() == "auto":
            dtype = _default_model_dtype(torch_mod, device)
        else:
            dtype = _dtype_from_str(torch_mod, str(self._config.torch_dtype))

        if str(self._config.vae_torch_dtype).strip().lower() == "auto":
            vae_dtype = _default_vae_dtype(torch_mod, device)
        else:
            vae_dtype = _dtype_from_str(torch_mod, str(self._config.vae_torch_dtype))

        # CPU float16 is not reliable.
        if device == "cpu" and dtype == torch_mod.float16:
            print(
                "WARNING #FALLBACK : ACE-Step model torch_dtype=float16 on CPU is not reliably supported. "
                "Using float32 instead.",
                file=sys.stderr,
            )
            dtype = torch_mod.float32
        if device == "cpu" and vae_dtype == torch_mod.float16:
            print(
                "WARNING #FALLBACK : ACE-Step VAE torch_dtype=float16 on CPU is not reliably supported. "
                "Using float32 instead.",
                file=sys.stderr,
            )
            vae_dtype = torch_mod.float32

        # Upstream ACE-Step guidance: macOS/MPS should avoid bf16.
        if device == "mps" and dtype == torch_mod.bfloat16:
            print(
                "WARNING #FALLBACK : Apple Silicon MPS does not support bfloat16 reliably for ACE-Step. "
                "Using float16 instead. (Upstream guidance: disable bf16 on macOS.)",
                file=sys.stderr,
            )
            dtype = torch_mod.float16
        if device == "mps" and vae_dtype == torch_mod.bfloat16:
            vae_dtype = torch_mod.float16

        return dtype, vae_dtype

    def _ensure_loaded(self) -> None:
        if (
            self._loaded
            and self._model is not None
            and self._vae is not None
            and self._text_tokenizer is not None
            and self._text_encoder is not None
            and self._silence_latent is not None
        ):
            return

        device_pref = str(self._config.device or "auto").strip().lower() or "auto"
        if device_pref in {"auto", "mps"}:
            _maybe_configure_mps_high_watermark(
                mps_max_memory_gb=self._config.mps_max_memory_gb,
                mps_high_watermark_ratio=self._config.mps_high_watermark_ratio,
            )

        torch = _lazy_import_torch()
        AceStepConditionGenerationModel = _lazy_import_acestep_v15_turbo_model()
        AutoModel, AutoTokenizer = _lazy_import_transformers()
        _patch_transformers_rope_validation()
        _patch_transformers_torch_dtype_property()
        AutoencoderOobleck = _lazy_import_diffusers_oobleck()
        _patch_hf_hub_download()

        device = _resolve_device(torch, self._config.device)
        dtype, vae_dtype = self._resolve_dtypes(torch, device)
        text_device, text_dtype = _resolve_text_encoder_runtime(torch, model_device=device, model_dtype=dtype)
        if device == "mps" and text_device != device:
            print(
                "WARNING #FALLBACK : Running ACE-Step text encoder on CPU float32 for MPS compatibility; "
                "conditioning hidden states are cast back to the model dtype/device.",
                file=sys.stderr,
            )

        repo_id = str(self._config.repo_id)
        revision = self._config.revision
        local_files_only = bool(self._config.local_files_only)

        # Load core DiT model (ACE-Step v1.5 turbo).
        try:
            self._model = _from_pretrained_transformers(
                AceStepConditionGenerationModel,
                dtype=dtype,
                pretrained_model_name_or_path=repo_id,
                revision=revision,
                local_files_only=local_files_only,
                subfolder=str(self._config.dit_subfolder),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ACE-Step DiT model from {repo_id!r}: {e}") from e
        try:
            self._model.to(device)
            self._model.eval()
        except Exception:
            pass

        # Load text encoder (Qwen3 embedding) + tokenizer.
        try:
            self._text_tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                revision=revision,
                subfolder=str(self._config.text_encoder_subfolder),
                trust_remote_code=False,
                local_files_only=local_files_only,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ACE-Step text tokenizer from {repo_id!r}: {e}") from e

        try:
            self._text_encoder = _from_pretrained_transformers(
                AutoModel,
                dtype=text_dtype,
                pretrained_model_name_or_path=repo_id,
                revision=revision,
                subfolder=str(self._config.text_encoder_subfolder),
                trust_remote_code=False,
                local_files_only=local_files_only,
            )
            self._text_encoder.to(text_device)
            self._text_encoder.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load ACE-Step text encoder from {repo_id!r}: {e}") from e

        # Load VAE (Diffusers AutoencoderOobleck).
        try:
            _patch_oobleck_weight_norm(torch)
            self._vae = _from_pretrained_diffusers(
                AutoencoderOobleck,
                dtype=vae_dtype,
                pretrained_model_name_or_path=repo_id,
                revision=revision,
                subfolder=str(self._config.vae_subfolder),
                low_cpu_mem_usage=False,
                local_files_only=local_files_only,
            )
            self._vae.to(device)
            self._vae.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load ACE-Step VAE from {repo_id!r}: {e}") from e

        # Load and transpose silence latent (stored as [1, C, T] in repo).
        try:
            hf_hub_download = _lazy_import_hf_hub_download()
            sl_path = hf_hub_download(
                repo_id=repo_id,
                revision=revision,
                filename=f"{self._config.dit_subfolder}/silence_latent.pt",
                local_files_only=local_files_only,
            )
            sl = torch.load(sl_path, map_location="cpu")  # tensor
            if not isinstance(sl, torch.Tensor):
                raise TypeError(f"silence_latent.pt expected torch.Tensor, got {type(sl)!r}")
            # Upstream transpose: [1, C, T] -> [1, T, C]
            sl = sl.transpose(1, 2).contiguous()
            self._silence_latent = sl.to(device=device, dtype=dtype)
        except Exception as e:
            raise RuntimeError(f"Failed to load silence_latent.pt from {repo_id!r}: {e}") from e

        self._device = device
        self._text_device = text_device
        self._dtype = dtype
        self._vae_dtype = vae_dtype
        self._loaded = True

    def _release_lm_runtime(self) -> None:
        self._lm_model = None
        self._lm_tokenizer = None
        self._lm_device = None
        self._lm_dtype = None
        self._lm_audio_token_ids = ()
        self._lm_think_end_token_id = None
        try:
            import gc

            gc.collect()
        except Exception:
            pass

    def _seed_sampling_rng(self, seed: int) -> None:
        torch = _lazy_import_torch()
        seed_value = int(seed)
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and hasattr(torch, "mps"):
            try:
                torch.mps.manual_seed(seed_value)
            except Exception:
                pass

    def _tokenize_text_input(self, text: str, *, max_length: int) -> Tuple[Any, Any]:
        """Tokenize text using upstream ACE-Step truncation limits."""
        tok = self._text_tokenizer
        if tok is None:
            raise RuntimeError("Text tokenizer not loaded")

        kwargs = {
            "return_tensors": "pt",
            "padding": "longest",
            "truncation": True,
            "max_length": int(max_length),
        }
        try:
            out = tok(text, **kwargs)
        except TypeError:
            kwargs.pop("max_length", None)
            out = tok(text, **kwargs)
        input_ids = getattr(out, "input_ids", None)
        attention_mask = getattr(out, "attention_mask", None)
        if input_ids is None:
            raise RuntimeError("Tokenizer did not return input_ids")
        if attention_mask is None:
            # Some tokenizers may omit attention_mask for single sequences; generate one.
            torch = _lazy_import_torch()
            attention_mask = torch.ones_like(input_ids)

        return input_ids, attention_mask.bool()

    def _get_silence_latent_slice(self, length: int, *, device: Optional[str] = None, dtype: Optional[Any] = None) -> Any:
        silence = self._silence_latent
        if silence is None:
            raise RuntimeError("silence_latent not loaded")
        target_len = max(1, int(length))
        available = int(silence.shape[1])
        if target_len <= available:
            out = silence[:, :target_len, :]
        else:
            repeats = (target_len + available - 1) // available
            out = silence.repeat(1, repeats, 1)[:, :target_len, :]
        target_device = silence.device if device is None else device
        target_dtype = silence.dtype if dtype is None else dtype
        return out.to(device=target_device, dtype=target_dtype)

    def _encode_text_hidden(self, input_ids: Any, attention_mask: Any) -> Any:
        torch = _lazy_import_torch()
        enc = self._text_encoder
        if enc is None:
            raise RuntimeError("Text encoder not loaded")

        # Best-effort signature-based kwargs.
        kwargs: Dict[str, Any] = {"input_ids": input_ids}
        try:
            import inspect

            sig = inspect.signature(enc.forward)  # type: ignore[attr-defined]
            params = set(sig.parameters.keys())
            if "attention_mask" in params:
                kwargs["attention_mask"] = attention_mask
            if "lyric_attention_mask" in params:
                kwargs["lyric_attention_mask"] = None
        except Exception:
            # Fallback: try attention_mask (common).
            kwargs["attention_mask"] = attention_mask

        with torch.inference_mode():
            out = enc(**kwargs)
        hs = getattr(out, "last_hidden_state", None)
        if hs is None and isinstance(out, (tuple, list)) and out:
            hs = out[0]
        if hs is None:
            hs = out
        return hs

    def _embed_lyrics(self, lyric_input_ids: Any) -> Any:
        torch = _lazy_import_torch()
        enc = self._text_encoder
        if enc is None:
            raise RuntimeError("Text encoder not loaded")
        emb = None
        try:
            emb = enc.get_input_embeddings()
        except Exception:
            emb = getattr(enc, "embed_tokens", None)
        if emb is None or not callable(emb):
            raise RuntimeError("Text encoder does not expose an embedding layer")
        with torch.inference_mode():
            return emb(lyric_input_ids)

    def _decode_latents(self, latents_bt64: Any) -> Any:
        """Decode latents [B, T, 64] into waveform tensor [B, 2, samples]."""
        torch = _lazy_import_torch()
        if self._vae is None:
            raise RuntimeError("VAE not loaded")
        device = str(self._device or "cpu")

        pred_latents_for_decode = latents_bt64.transpose(1, 2).contiguous()  # [B, 64, T]
        pred_latents_for_decode = pred_latents_for_decode.to(dtype=getattr(self._vae, "dtype", torch.float32))

        def _vae_decode(chunk: Any) -> Any:
            out = self._vae.decode(chunk)  # diffusers returns DecoderOutput(sample=...)
            sample = getattr(out, "sample", None)
            if sample is None:
                sample = getattr(out, "samples", None)
            if sample is None and isinstance(out, dict):
                sample = out.get("sample") or out.get("samples")
            if sample is None:
                raise TypeError("VAE decode returned an unexpected type (missing .sample)")
            return sample

        def _move_vae_to_cpu_float32() -> None:
            if self._vae is None:
                raise RuntimeError("VAE not loaded")
            try:
                self._vae.to(device="cpu", dtype=torch.float32)
            except TypeError:
                self._vae.to("cpu")
                try:
                    self._vae.float()
                except Exception:
                    pass
            self._vae.eval()

        def _release_runtime_for_cpu_decode() -> None:
            if str(device) != "mps":
                return
            try:
                self._model = None
            except Exception:
                pass
            try:
                self._text_encoder = None
                self._text_tokenizer = None
                self._silence_latent = None
            except Exception:
                pass
            self._release_lm_runtime()
            self._loaded = False
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
            try:
                import gc

                gc.collect()
            except Exception:
                pass

        def _switch_to_cpu_decode(*, reason: str, announce: bool) -> None:
            nonlocal device, pred_latents_for_decode, wav_accum, chunk_frames, overlap_frames, stride
            if announce:
                print(reason, file=sys.stderr)
            _release_runtime_for_cpu_decode()
            _move_vae_to_cpu_float32()
            pred_latents_for_decode = pred_latents_for_decode.detach().to("cpu", dtype=torch.float32)
            device = "cpu"
            chunk_frames = int(self._config.cpu_decode_chunk_frames)
            overlap_frames = int(self._config.cpu_decode_overlap_frames)
            stride = max(1, int(chunk_frames - overlap_frames))
            if wav_accum is not None:
                try:
                    wav_accum = wav_accum.to("cpu", dtype=torch.float32)
                except Exception:
                    wav_accum = wav_accum.to("cpu")

        # For MPS: use small chunks to avoid kernel limits.
        if device == "mps":
            chunk_frames = int(self._config.mps_decode_chunk_frames)
            overlap_frames = int(self._config.mps_decode_overlap_frames)
        elif device == "cpu":
            chunk_frames = int(self._config.cpu_decode_chunk_frames)
            overlap_frames = int(self._config.cpu_decode_overlap_frames)
        else:
            chunk_frames = 0
            overlap_frames = 0

        if chunk_frames <= 0 or pred_latents_for_decode.shape[-1] <= chunk_frames:
            return _vae_decode(pred_latents_for_decode)

        stride = max(1, int(chunk_frames - overlap_frames))
        total_frames = int(pred_latents_for_decode.shape[-1])
        wav_accum = None
        wav_len = 0
        ratio = None
        overlap_samples = 0

        force_cpu_duration = float(self._config.mps_force_cpu_decode_min_duration_s)
        if (
            device == "mps"
            and force_cpu_duration > 0.0
            and total_frames >= int(round(force_cpu_duration * float(self._config.frames_per_second)))
        ):
            _switch_to_cpu_decode(
                reason=(
                    "WARNING #FALLBACK : Long ACE-Step VAE decode on MPS is memory-unstable; "
                    "switching decode to CPU float32."
                ),
                announce=True,
            )

        for start in range(0, total_frames, stride):
            end = min(total_frames, start + chunk_frames)
            chunk = pred_latents_for_decode[..., start:end]
            # Once we hit an MPS failure and choose CPU decode, keep decoding on CPU
            # for the remainder of the waveform to avoid device thrash.
            if device == "cpu":
                chunk = chunk.to("cpu", dtype=torch.float32)
            try:
                wav_chunk = _vae_decode(chunk)  # [B, 2, S]
            except (NotImplementedError, RuntimeError) as e:
                if device == "mps" and bool(self._config.auto_retry_cpu_on_mps_error):
                    _switch_to_cpu_decode(
                        reason=(
                            "WARNING #FALLBACK : MPS VAE decode failed; retrying decode on CPU float32. "
                            f"(error={type(e).__name__}: {e})"
                        ),
                        announce=True,
                    )
                    chunk = pred_latents_for_decode[..., start:end]
                    wav_chunk = _vae_decode(chunk)
                else:
                    raise

            if ratio is None:
                ratio = float(wav_chunk.shape[-1]) / float(max(1, int(end - start)))
                overlap_samples = int(round(float(overlap_frames) * ratio))

            if wav_accum is None:
                wav_accum = wav_chunk
                wav_len = int(wav_chunk.shape[-1])
                continue

            if overlap_samples <= 0:
                wav_accum = torch.cat([wav_accum, wav_chunk], dim=-1)
                wav_len = int(wav_accum.shape[-1])
                continue

            # Crossfade overlap.
            if wav_chunk.device != wav_accum.device:
                if device == "cpu":
                    wav_accum = wav_accum.to("cpu")
                    wav_chunk = wav_chunk.to("cpu")
                else:
                    wav_chunk = wav_chunk.to(wav_accum.device)
            o = min(overlap_samples, int(wav_chunk.shape[-1]), int(wav_accum.shape[-1]))
            if o <= 0:
                wav_accum = torch.cat([wav_accum, wav_chunk], dim=-1)
                wav_len = int(wav_accum.shape[-1])
                continue

            fade_out = torch.linspace(1.0, 0.0, o, device=wav_accum.device, dtype=wav_accum.dtype)
            fade_in = 1.0 - fade_out
            a_tail = wav_accum[..., -o:] * fade_out
            b_head = wav_chunk[..., :o] * fade_in
            mixed = a_tail + b_head
            wav_accum = torch.cat([wav_accum[..., :-o], mixed, wav_chunk[..., o:]], dim=-1)
            wav_len = int(wav_accum.shape[-1])

            if end >= total_frames:
                break

        if wav_accum is None:
            raise RuntimeError("VAE tiled decode produced no output")
        return wav_accum

    def _planner_enabled_for_duration(self, duration_s: float) -> bool:
        if not bool(self._config.use_audio_code_planner):
            return False
        return float(duration_s) >= float(self._config.planner_min_duration_s)

    def _resolve_lm_runtime(self, torch_mod: Any) -> Tuple[str, Any]:
        device_pref = str(self._config.lm_device or "auto").strip().lower() or "auto"
        if device_pref == "auto":
            if str(self._config.device or "auto").strip().lower() in {"cuda", "xpu"}:
                device = _resolve_device(torch_mod, str(self._config.device))
            else:
                device = "cpu"
        else:
            device = device_pref

        if device == "auto":
            device = "cpu"

        dtype_pref = str(self._config.lm_torch_dtype or "auto").strip().lower() or "auto"
        if dtype_pref == "auto":
            if device in {"cuda", "xpu"}:
                dtype = torch_mod.bfloat16
            elif device == "mps":
                dtype = torch_mod.float16
            else:
                dtype = torch_mod.float32
        else:
            dtype = _dtype_from_str(torch_mod, dtype_pref)

        if device == "cpu" and dtype == torch_mod.float16:
            dtype = torch_mod.float32
        if device == "mps" and dtype == torch_mod.bfloat16:
            dtype = torch_mod.float16
        return device, dtype

    def _lm_candidate_sources(self, lm_device: str) -> tuple[tuple[str, Optional[str], str], ...]:
        primary = (
            str(self._config.repo_id),
            str(self._config.lm_model_subfolder or "").strip() or None,
            str(self._config.lm_model_subfolder or "").strip() or "acestep-5Hz-lm-1.7B",
        )
        fallback_repo = str(self._config.lm_fallback_repo_id or "").strip()
        candidates = [primary]
        if fallback_repo:
            candidates.append((fallback_repo, None, fallback_repo.rsplit("/", 1)[-1]))
        return tuple(candidates)

    def _ensure_lm_loaded(self) -> None:
        if self._lm_model is not None and self._lm_tokenizer is not None:
            return

        torch = _lazy_import_torch()
        AutoModelForCausalLM, AutoTokenizer = _lazy_import_transformers_lm()
        lm_device, lm_dtype = self._resolve_lm_runtime(torch)

        last_error: Optional[Exception] = None
        local_files_only = bool(self._config.local_files_only)
        for repo_id, subfolder, label in self._lm_candidate_sources(lm_device):
            try:
                tok_kwargs = {
                    "pretrained_model_name_or_path": repo_id,
                    "revision": self._config.revision if repo_id == str(self._config.repo_id) else None,
                    "trust_remote_code": False,
                    "local_files_only": local_files_only,
                }
                if subfolder:
                    tok_kwargs["subfolder"] = subfolder
                tok = AutoTokenizer.from_pretrained(**tok_kwargs)

                model_kwargs = {
                    "pretrained_model_name_or_path": repo_id,
                    "revision": self._config.revision if repo_id == str(self._config.repo_id) else None,
                    "local_files_only": local_files_only,
                }
                if subfolder:
                    model_kwargs["subfolder"] = subfolder
                model = _from_pretrained_transformers(AutoModelForCausalLM, dtype=lm_dtype, **model_kwargs)
                model.to(lm_device)
                model.eval()

                if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
                    tok.pad_token = tok.eos_token

                audio_tokens = [
                    token_id
                    for token, token_id in tok.get_added_vocab().items()
                    if isinstance(token, str) and token.startswith("<|audio_code_")
                ]
                if not audio_tokens:
                    raise RuntimeError("LM tokenizer did not expose any <|audio_code_...|> tokens.")
                audio_token_ids = tuple(sorted(audio_tokens))

                think_end_ids = tok.encode("</think>", add_special_tokens=False)
                if not think_end_ids:
                    raise RuntimeError("LM tokenizer could not encode </think>.")

                self._lm_tokenizer = tok
                self._lm_model = model
                self._lm_device = lm_device
                self._lm_dtype = lm_dtype
                self._lm_model_label = label
                self._lm_audio_token_ids = audio_token_ids
                self._lm_think_end_token_id = int(think_end_ids[-1])
                return
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"Failed to load internal ACE-Step 5Hz LM planner: {last_error}") from last_error

    def _build_lm_cot_prompt(self, caption: str, lyrics: str) -> str:
        if self._lm_tokenizer is None:
            raise RuntimeError("LM tokenizer not loaded")
        return self._lm_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n"},
                {"role": "user", "content": f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _build_lm_codes_prompt(self, caption: str, lyrics: str, cot_text: str) -> str:
        if self._lm_tokenizer is None:
            raise RuntimeError("LM tokenizer not loaded")
        formatted = self._lm_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n"},
                {"role": "user", "content": f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return formatted + cot_text + "\n\n"

    def _build_lm_codes_unconditional_prompt(self, negative_prompt: str) -> str:
        if self._lm_tokenizer is None:
            raise RuntimeError("LM tokenizer not loaded")
        user_prompt = str(negative_prompt or "").strip() or "NO USER INPUT"
        formatted = self._lm_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n"},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return formatted + "<think>\n\n</think>\n\n"

    def _apply_lm_top_k_filter(self, logits: Any, top_k: int) -> Any:
        torch = _lazy_import_torch()
        if int(top_k) <= 0:
            return logits
        k = min(int(top_k), int(logits.shape[-1]))
        threshold = torch.topk(logits, k)[0][..., -1, None]
        return logits.masked_fill(logits < threshold, float("-inf"))

    def _apply_lm_top_p_filter(self, logits: Any, top_p: float) -> Any:
        torch = _lazy_import_torch()
        if not (0.0 < float(top_p) < 1.0):
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits.float(), dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > float(top_p)
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        return logits.masked_fill(indices_to_remove, float("-inf"))

    def _sample_lm_token(self, logits: Any, temperature: float) -> Any:
        torch = _lazy_import_torch()
        if float(temperature) <= 0.0:
            return torch.argmax(logits, dim=-1)
        scaled = logits.float() / float(temperature)
        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _generate_lm_tokens(
        self,
        *,
        prompt_text: str,
        max_new_tokens: int,
        eos_token_id: int,
        prefix_allowed_tokens_fn: Any = None,
    ) -> str:
        torch = _lazy_import_torch()
        self._ensure_lm_loaded()
        assert self._lm_tokenizer is not None
        assert self._lm_model is not None
        assert self._lm_device is not None

        prompt = self._lm_tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=False)
        input_ids = prompt.input_ids.to(self._lm_device)
        attention_mask = prompt.attention_mask.to(self._lm_device)

        with torch.inference_mode():
            outputs = self._lm_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=float(self._config.lm_temperature),
                top_k=int(self._config.lm_top_k),
                top_p=float(self._config.lm_top_p),
                max_new_tokens=int(max_new_tokens),
                eos_token_id=int(eos_token_id),
                pad_token_id=int(self._lm_tokenizer.eos_token_id),
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                use_cache=True,
            )

        generated_ids = outputs[0, input_ids.shape[-1] :]
        return self._lm_tokenizer.decode(generated_ids, skip_special_tokens=False)

    def _generate_lm_audio_codes_cfg(
        self,
        *,
        caption: str,
        lyrics: str,
        cot_text: str,
        target_code_count: int,
    ) -> Any:
        torch = _lazy_import_torch()
        self._ensure_lm_loaded()
        assert self._lm_tokenizer is not None
        assert self._lm_model is not None
        assert self._lm_device is not None
        if not self._lm_audio_token_ids:
            raise RuntimeError("LM tokenizer did not expose any <|audio_code_...|> tokens.")

        conditional_prompt = self._build_lm_codes_prompt(caption, lyrics, cot_text)
        unconditional_prompt = self._build_lm_codes_unconditional_prompt(str(self._config.lm_negative_prompt))

        original_padding_side = getattr(self._lm_tokenizer, "padding_side", "right")
        self._lm_tokenizer.padding_side = "left"
        try:
            batch = self._lm_tokenizer(
                [conditional_prompt, unconditional_prompt],
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
        finally:
            self._lm_tokenizer.padding_side = original_padding_side

        input_ids = batch.input_ids.to(self._lm_device)
        attention_mask = batch.attention_mask.to(self._lm_device)
        prompt_len = int(input_ids.shape[-1])
        eos_token_id = int(self._lm_tokenizer.eos_token_id)
        pad_token_id = eos_token_id
        valid_audio_indices = torch.tensor(self._lm_audio_token_ids, device=self._lm_device, dtype=torch.long)

        generated_ids = input_ids.clone()
        attn_mask = attention_mask.clone()
        past_key_values = None
        model_kwargs: Dict[str, Any] = {"attention_mask": attn_mask}
        max_new_tokens = int(target_code_count) + 10
        generated_code_count = 0

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                if past_key_values is None:
                    outputs = self._lm_model(
                        input_ids=generated_ids,
                        attention_mask=attn_mask,
                        use_cache=True,
                    )
                else:
                    outputs = self._lm_model(
                        input_ids=generated_ids[:, -1:],
                        attention_mask=attn_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                logits = outputs.logits[:, -1, :]
                cond_logits = logits[0:1].float()
                uncond_logits = logits[1:2].float()
                cfg_logits = torch.full_like(cond_logits, float("-inf"))

                if generated_code_count < int(target_code_count):
                    cfg_valid = uncond_logits[:, valid_audio_indices] + float(self._config.lm_cfg_scale) * (
                        cond_logits[:, valid_audio_indices] - uncond_logits[:, valid_audio_indices]
                    )
                    cfg_logits[:, valid_audio_indices] = cfg_valid
                else:
                    cfg_logits[:, eos_token_id] = 0.0

                cfg_logits = torch.nan_to_num(cfg_logits, nan=float("-inf"))
                cfg_logits = self._apply_lm_top_k_filter(cfg_logits, int(self._config.lm_top_k))
                cfg_logits = self._apply_lm_top_p_filter(cfg_logits, float(self._config.lm_top_p))

                next_token = self._sample_lm_token(cfg_logits, float(self._config.lm_temperature))
                token_id = int(next_token[0].item())

                next_token_pair = next_token.repeat(2).unsqueeze(-1)
                generated_ids = torch.cat([generated_ids, next_token_pair], dim=1)
                attn_mask = torch.cat(
                    [attn_mask, torch.ones((2, 1), device=self._lm_device, dtype=attn_mask.dtype)],
                    dim=1,
                )
                model_kwargs["attention_mask"] = attn_mask
                if hasattr(outputs, "past_key_values"):
                    past_key_values = outputs.past_key_values

                if token_id == eos_token_id:
                    break
                generated_code_count += 1

        generated_token_ids = generated_ids[0, prompt_len:]
        output_text = self._lm_tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        audio_token_texts = re.findall(r"<\|audio_code_(\d+)\|>", output_text)
        if len(audio_token_texts) < int(target_code_count):
            raise RuntimeError(
                f"Internal ACE-Step LM planner generated only {len(audio_token_texts)} audio codes "
                f"for target_count={int(target_code_count)}."
            )
        code_ids = [int(x) for x in audio_token_texts[: int(target_code_count)]]
        return torch.tensor(code_ids, dtype=torch.long).unsqueeze(0).unsqueeze(-1)

    def _parse_lm_output(self, output_text: str) -> Tuple[Dict[str, Any], str]:
        metadata: Dict[str, Any] = {}
        audio_codes = "".join(re.findall(r"<\|audio_code_\d+\|>", str(output_text or "")))

        reasoning_text = None
        match = re.search(r"<think>(.*?)</think>", str(output_text or ""), re.DOTALL)
        if match:
            reasoning_text = str(match.group(1)).strip()
        elif audio_codes:
            reasoning_text = str(output_text).split("<|audio_code_")[0].strip()
        else:
            reasoning_text = str(output_text or "").strip()

        if reasoning_text:
            current_key = None
            current_value_lines = []

            def _save() -> None:
                nonlocal current_key, current_value_lines
                if not current_key or not current_value_lines:
                    current_key = None
                    current_value_lines = []
                    return
                value = "\n".join(current_value_lines).strip()
                if current_key == "caption":
                    metadata["caption"] = _postprocess_caption_yaml_value(value)
                elif current_key == "bpm":
                    try:
                        metadata["bpm"] = int(value)
                    except Exception:
                        metadata["bpm"] = value
                elif current_key == "duration":
                    try:
                        metadata["duration"] = int(value)
                    except Exception:
                        metadata["duration"] = value
                elif current_key == "language":
                    metadata["language"] = value
                elif current_key == "keyscale":
                    metadata["keyscale"] = value
                elif current_key == "timesignature":
                    metadata["timesignature"] = value
                current_key = None
                current_value_lines = []

            for line in reasoning_text.splitlines():
                if line.strip().startswith("<"):
                    continue
                if line and not line[0].isspace() and ":" in line:
                    _save()
                    key, value = line.split(":", 1)
                    current_key = str(key).strip().lower()
                    if value.strip():
                        current_value_lines.append(value.strip())
                elif current_key and (line.startswith(" ") or line.startswith("\t")):
                    current_value_lines.append(line.strip())
            _save()

        return metadata, audio_codes

    def _format_metadata_as_cot(self, metadata: Dict[str, Any]) -> str:
        ordered_keys = ("bpm", "caption", "duration", "keyscale", "language", "timesignature")
        lines = []
        for key in ordered_keys:
            value = metadata.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            if key == "timesignature" and text.endswith("/4"):
                text = text.split("/")[0]
            lines.append(f"{key}: {text}")
        body = "\n".join(lines)
        return f"<think>\n{body}\n</think>"

    def _generate_planner_metadata(
        self,
        *,
        caption: str,
        lyrics: str,
        duration_s: float,
        vocal_language: str,
        bpm: Optional[int],
        keyscale: str,
        timesignature: str,
    ) -> Dict[str, Any]:
        self._ensure_lm_loaded()
        assert self._lm_think_end_token_id is not None
        prompt = self._build_lm_cot_prompt(caption, lyrics)
        output_text = self._generate_lm_tokens(
            prompt_text=prompt,
            max_new_tokens=max(256, int(round(float(duration_s) * 5.0)) + 500),
            eos_token_id=int(self._lm_think_end_token_id),
        )
        metadata, _ = self._parse_lm_output(output_text)
        if not metadata.get("caption"):
            metadata["caption"] = caption
        if not metadata.get("duration"):
            metadata["duration"] = int(round(float(duration_s)))
        if lyrics.strip() and not metadata.get("language"):
            metadata["language"] = vocal_language
        if bpm is not None and metadata.get("bpm") is None:
            metadata["bpm"] = int(bpm)
        if keyscale and metadata.get("keyscale") is None:
            metadata["keyscale"] = keyscale
        if timesignature and metadata.get("timesignature") is None:
            metadata["timesignature"] = timesignature
        return metadata

    def _generate_audio_code_ids(
        self,
        *,
        caption: str,
        lyrics: str,
        cot_text: str,
        target_code_count: int,
    ) -> Any:
        cfg_scale = float(self._config.lm_cfg_scale)
        if cfg_scale > 1.0:
            return self._generate_lm_audio_codes_cfg(
                caption=caption,
                lyrics=lyrics,
                cot_text=cot_text,
                target_code_count=target_code_count,
            )

        torch = _lazy_import_torch()
        self._ensure_lm_loaded()
        assert self._lm_tokenizer is not None
        assert self._lm_audio_token_ids

        prompt_text = self._build_lm_codes_prompt(caption, lyrics, cot_text)
        prompt = self._lm_tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=False)
        prompt_len = int(prompt.input_ids.shape[-1])
        eos_token_id = int(self._lm_tokenizer.eos_token_id)
        allowed_audio_ids = self._lm_audio_token_ids

        def _prefix_allowed_tokens_fn(_batch_id: int, input_ids: Any) -> Any:
            generated_count = max(0, int(input_ids.shape[-1]) - prompt_len)
            if generated_count < int(target_code_count):
                return allowed_audio_ids
            return (eos_token_id,)

        output_text = self._generate_lm_tokens(
            prompt_text=prompt_text,
            max_new_tokens=int(target_code_count) + 10,
            eos_token_id=eos_token_id,
            prefix_allowed_tokens_fn=_prefix_allowed_tokens_fn,
        )
        audio_token_texts = re.findall(r"<\|audio_code_(\d+)\|>", output_text)
        if len(audio_token_texts) < int(target_code_count):
            raise RuntimeError(
                f"Internal ACE-Step LM planner generated only {len(audio_token_texts)} audio codes "
                f"for target_count={int(target_code_count)}."
            )
        code_ids = [int(x) for x in audio_token_texts[: int(target_code_count)]]
        return torch.tensor(code_ids, dtype=torch.long).unsqueeze(0).unsqueeze(-1)

    def _maybe_plan_audio_codes(
        self,
        *,
        prompt: str,
        lyrics: str,
        duration_s: float,
        vocal_language: str,
        bpm: Optional[int],
        keyscale: str,
        timesignature: str,
        seed: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        if not self._planner_enabled_for_duration(duration_s):
            return None
        if seed is not None:
            self._seed_sampling_rng(int(seed))

        planner_metadata = self._generate_planner_metadata(
            caption=prompt or "music",
            lyrics=lyrics,
            duration_s=duration_s,
            vocal_language=vocal_language,
            bpm=bpm,
            keyscale=keyscale,
            timesignature=timesignature,
        )
        cot_text = self._format_metadata_as_cot(planner_metadata)
        target_code_count = max(1, int(round(float(duration_s) * 5.0)))
        audio_codes = self._generate_audio_code_ids(
            caption=str(planner_metadata.get("caption") or prompt or "music"),
            lyrics=lyrics,
            cot_text=cot_text,
            target_code_count=target_code_count,
        )
        return {
            "caption": str(planner_metadata.get("caption") or prompt or "music"),
            "vocal_language": str(planner_metadata.get("language") or vocal_language or "unknown"),
            "bpm": planner_metadata.get("bpm"),
            "keyscale": str(planner_metadata.get("keyscale") or keyscale or ""),
            "timesignature": str(planner_metadata.get("timesignature") or timesignature or ""),
            "cot_text": cot_text,
            "audio_codes": audio_codes,
            "audio_code_count": int(target_code_count),
            "lm_model": str(self._lm_model_label or ""),
        }

    def _decode_audio_code_ids_to_lm_hints(self, *, audio_codes: Any, device: str, dtype: Any) -> Any:
        torch = _lazy_import_torch()
        if self._model is None:
            raise RuntimeError("ACE-Step model not loaded")
        tokenizer = getattr(self._model, "tokenizer", None)
        detokenizer = getattr(self._model, "detokenizer", None)
        quantizer = getattr(tokenizer, "quantizer", None) if tokenizer is not None else None
        if quantizer is None or detokenizer is None:
            raise RuntimeError("ACE-Step model is missing audio quantizer/detokenizer for audio-code conditioning")

        indices = torch.as_tensor(audio_codes, device=device, dtype=torch.long)
        if indices.ndim == 1:
            indices = indices.unsqueeze(0).unsqueeze(-1)
        elif indices.ndim == 2:
            indices = indices.unsqueeze(-1)
        elif indices.ndim != 3:
            raise ValueError(f"audio_codes must have rank 1/2/3; got shape={tuple(indices.shape)}")

        decode_device = device
        decode_dtype = dtype
        restore_modules = False
        if str(device) == "mps":
            print(
                "WARNING #FALLBACK : Running ACE-Step audio-code detokenizer on CPU float32 for MPS compatibility; "
                "planner hints are cast back to the model dtype/device.",
                file=sys.stderr,
            )
            decode_device = "cpu"
            decode_dtype = torch.float32
            restore_modules = True

        if restore_modules:
            try:
                tokenizer.to(device=decode_device, dtype=decode_dtype)
            except Exception:
                tokenizer.to(decode_device)
            try:
                detokenizer.to(device=decode_device, dtype=decode_dtype)
            except Exception:
                detokenizer.to(decode_device)

        try:
            indices = indices.to(device=decode_device)
            with torch.inference_mode():
                quantized = quantizer.get_output_from_indices(indices)
                if torch.is_floating_point(quantized) and quantized.dtype != decode_dtype:
                    quantized = quantized.to(dtype=decode_dtype)
                lm_hints = detokenizer(quantized)
        finally:
            if restore_modules:
                try:
                    tokenizer.to(device=device, dtype=dtype)
                except Exception:
                    tokenizer.to(device)
                try:
                    detokenizer.to(device=device, dtype=dtype)
                except Exception:
                    detokenizer.to(device)

        if torch.is_floating_point(lm_hints):
            lm_hints = lm_hints.to(device=device, dtype=dtype)
        else:
            lm_hints = lm_hints.to(device=device)
        return lm_hints

    def _should_segment_long_mps_generation(
        self,
        *,
        resolved_device: str,
        duration_s: float,
        planner: Optional[Dict[str, Any]],
        extra: Dict[str, Any],
    ) -> bool:
        if bool(extra.get("_acestep_segmented_run")):
            return False
        if str(resolved_device) != "mps":
            return False
        if planner is None or planner.get("audio_codes") is None:
            return False
        if float(duration_s) < float(self._config.mps_segment_generation_min_duration_s):
            return False
        segment_s = float(self._config.mps_segment_duration_s)
        return segment_s > 0.0 and segment_s < float(duration_s)

    def _generate_segmented_long_audio(
        self,
        *,
        request: AudioGenerationRequest,
        planner: Dict[str, Any],
        extra: Dict[str, Any],
    ) -> GeneratedAsset:
        torch = _lazy_import_torch()

        planner_codes = planner.get("audio_codes")
        if planner_codes is None:
            raise RuntimeError("Segmented long-generation requested without planner audio codes.")
        planner_codes = torch.as_tensor(planner_codes, dtype=torch.long)
        if planner_codes.ndim == 2:
            planner_codes = planner_codes.unsqueeze(-1)
        if planner_codes.ndim != 3 or int(planner_codes.shape[0]) != 1:
            raise ValueError(f"planner audio codes must have shape [1, T, 1]; got {tuple(planner_codes.shape)}")

        segment_duration_s = float(self._config.mps_segment_duration_s)
        codes_per_second = 5.0
        segment_code_count = max(1, int(round(segment_duration_s * codes_per_second)))
        total_codes = int(planner_codes.shape[1])
        crossfade_s = max(0.0, float(self._config.mps_segment_crossfade_s))
        base_seed = int(request.seed) if isinstance(request.seed, int) and int(request.seed) >= 0 else 123
        base_extra = {k: v for k, v in dict(extra).items() if not str(k).startswith("_acestep_")}

        segment_wavs = []
        segment_sample_rate = None
        final_meta: Dict[str, Any] = {}
        segment_count = 0

        for segment_index, code_start in enumerate(range(0, total_codes, segment_code_count)):
            code_end = min(total_codes, code_start + segment_code_count)
            segment_codes = planner_codes[:, code_start:code_end, :].clone()
            current_duration_s = float(segment_codes.shape[1]) / codes_per_second

            segment_planner = dict(planner)
            segment_planner["audio_codes"] = segment_codes
            segment_planner["audio_code_count"] = int(segment_codes.shape[1])

            segment_extra = dict(base_extra)
            segment_extra["_acestep_internal_planner"] = segment_planner
            segment_extra["_acestep_segmented_run"] = True

            segment_request = replace(
                request,
                duration_s=current_duration_s,
                seed=int(base_seed + segment_index),
                extra=segment_extra,
            )
            segment_asset = self.generate_audio(segment_request)
            segment_wav, sample_rate = _decode_wav_bytes(segment_asset.data)
            if segment_sample_rate is None:
                segment_sample_rate = int(sample_rate)
            elif int(sample_rate) != int(segment_sample_rate):
                raise RuntimeError(
                    f"Segmented ACE-Step generation produced inconsistent sample rates: "
                    f"{segment_sample_rate} vs {sample_rate}"
                )
            segment_wavs.append(segment_wav)
            final_meta = dict(segment_asset.metadata)
            segment_count += 1

        if not segment_wavs or segment_sample_rate is None:
            raise RuntimeError("Segmented ACE-Step generation produced no audio segments.")

        stitched = segment_wavs[0]
        overlap_samples = int(round(crossfade_s * float(segment_sample_rate)))
        for segment_wav in segment_wavs[1:]:
            if overlap_samples <= 0:
                stitched = torch.cat([stitched, segment_wav], dim=-1)
                continue
            overlap = min(overlap_samples, int(stitched.shape[-1]), int(segment_wav.shape[-1]))
            if overlap <= 0:
                stitched = torch.cat([stitched, segment_wav], dim=-1)
                continue
            fade_out = torch.linspace(1.0, 0.0, overlap, dtype=stitched.dtype)
            fade_in = 1.0 - fade_out
            mixed = stitched[:, -overlap:] * fade_out + segment_wav[:, :overlap] * fade_in
            stitched = torch.cat([stitched[:, :-overlap], mixed, segment_wav[:, overlap:]], dim=-1)

        stitched = _remove_dc_offset(stitched)
        if bool(self._config.enable_normalization):
            stitched = _peak_normalize(stitched, target_db=float(self._config.normalization_db))
        stitched = _ensure_finite_audio(stitched)

        final_meta.update(
            {
                "duration_s": float(stitched.shape[-1]) / float(segment_sample_rate),
                "segmented_long_generation": True,
                "segment_count": int(segment_count),
                "segment_duration_s": float(segment_duration_s),
                "segment_crossfade_s": float(crossfade_s),
            }
        )
        return GeneratedAsset(
            data=_encode_wav_bytes(stitched, sample_rate=int(segment_sample_rate)),
            mime_type="audio/wav",
            metadata=final_meta,
        )

    def generate_audio(self, request: AudioGenerationRequest) -> GeneratedAsset:
        torch = _lazy_import_torch()

        if request.guidance_scale is not None:
            raise ValueError(
                "ACE-Step v1.5 turbo does not support guidance_scale/CFG (CFG is disabled for turbo checkpoints). "
                "Omit guidance_scale."
            )
        if isinstance(request.negative_prompt, str) and request.negative_prompt.strip():
            raise ValueError("ACE-Step v1.5 backend does not support negative_prompt. Omit negative_prompt.")

        prompt = str(request.prompt or "").strip()
        if not prompt:
            prompt = ""

        lyrics = request.lyrics.strip() if isinstance(request.lyrics, str) and request.lyrics.strip() else None

        duration_s = float(request.duration_s) if request.duration_s is not None else float(self._config.default_duration_s)
        if duration_s <= 0:
            duration_s = float(self._config.default_duration_s)

        extra = dict(request.extra or {})
        requested_bpm = extra.get("bpm")
        try:
            bpm = int(requested_bpm) if requested_bpm is not None else None
        except Exception:
            bpm = None
        keyscale = str(extra.get("keyscale") or "").strip()
        timesignature = str(extra.get("timesignature") or "").strip()
        vocal_language = str(request.vocal_language or "unknown").strip() or "unknown"

        user_seed_flag = extra.get("_acestep_user_seed_provided")
        request_has_seed = isinstance(request.seed, int) and int(request.seed) >= 0
        if isinstance(user_seed_flag, bool):
            seed_was_provided = bool(user_seed_flag)
        else:
            seed_was_provided = request_has_seed
        seed = int(request.seed) if request_has_seed else random.randint(0, 2**32 - 1)
        extra.setdefault("_acestep_user_seed_provided", bool(seed_was_provided))

        planner = None
        planner_error = None
        injected_planner = extra.get("_acestep_internal_planner")
        if isinstance(injected_planner, dict):
            planner = dict(injected_planner)
        elif self._planner_enabled_for_duration(duration_s):
            try:
                planner = self._maybe_plan_audio_codes(
                    prompt=prompt or "music",
                    lyrics=lyrics or "",
                    duration_s=duration_s,
                    vocal_language=vocal_language,
                    bpm=bpm,
                    keyscale=keyscale,
                    timesignature=timesignature,
                    seed=int(seed),
                )
            except Exception as exc:
                planner_error = str(exc)
                if not bool(self._config.allow_direct_text_fallback):
                    raise
                print(
                    "WARNING #FALLBACK : Internal ACE-Step audio-code planner failed; "
                    f"falling back to direct text conditioning. (error={type(exc).__name__}: {exc})",
                    file=sys.stderr,
                )
            finally:
                # The internal 5 Hz LM is only needed to materialize planning
                # metadata + audio codes. Drop it before DiT/VAE work to keep
                # unified-memory pressure under control on Apple Silicon.
                self._release_lm_runtime()

        resolved_device = _resolve_device(torch, self._config.device)
        if self._should_segment_long_mps_generation(
            resolved_device=resolved_device,
            duration_s=duration_s,
            planner=planner,
            extra=extra,
        ):
            return self._generate_segmented_long_audio(
                request=request,
                planner=planner,
                extra=extra,
            )

        self._ensure_loaded()

        device = str(self._device or "cpu")
        dtype = self._dtype

        # Turbo schedules are calibrated for fix_nfe=8 (as shipped in the checkpoint repo).
        if int(self._config.fix_nfe) != 8:
            raise ValueError("ACE-Step v1.5 turbo backend only supports fix_nfe=8 (v1).")
        if request.num_inference_steps is not None and int(request.num_inference_steps) != int(self._config.fix_nfe):
            raise ValueError(
                f"ACE-Step v1.5 turbo backend only supports num_inference_steps={int(self._config.fix_nfe)} (v1)."
            )

        frames_per_second = int(self._config.frames_per_second)
        latent_len = int(round(duration_s * float(frames_per_second)))
        if latent_len < 1:
            latent_len = 1
        if latent_len > 15000:
            raise ValueError("ACE-Step v1.5 supports up to ~600s (15000 frames @ 25Hz). Reduce duration_s.")

        effective_caption = str(planner.get("caption") if isinstance(planner, dict) else (prompt or "music")) or "music"
        effective_vocal_language = str(
            planner.get("vocal_language") if isinstance(planner, dict) else vocal_language
        ).strip() or "unknown"
        effective_bpm = bpm
        if effective_bpm is None and isinstance(planner, dict):
            try:
                planned_bpm = planner.get("bpm")
                effective_bpm = int(planned_bpm) if planned_bpm is not None else None
            except Exception:
                effective_bpm = None
        effective_keyscale = str(planner.get("keyscale") if isinstance(planner, dict) else keyscale).strip()
        effective_timesignature = str(
            planner.get("timesignature") if isinstance(planner, dict) else timesignature
        ).strip()
        planner_audio_codes = planner.get("audio_codes") if isinstance(planner, dict) else None
        use_planner_conditioning = planner_audio_codes is not None

        # Build caption conditioning text.
        if use_planner_conditioning or bool(self._config.use_sft_prompt):
            instruction = (
                _DEFAULT_COVER_DIT_INSTRUCTION if use_planner_conditioning else _DEFAULT_DIT_INSTRUCTION
            )
            metas = _meta_string(
                duration_s=duration_s,
                bpm=effective_bpm,
                keyscale=effective_keyscale,
                timesignature=effective_timesignature,
            )
            caption_input = _sft_gen_prompt(instruction, effective_caption, metas)
        else:
            # Align with upstream inference: use raw prompt/tags without SFT wrapper.
            caption_input = prompt or "music"

        text_device = str(self._text_device or device)
        # Tokenize and encode caption on text runtime device.
        text_ids, text_mask = self._tokenize_text_input(caption_input, max_length=256)
        text_ids = text_ids.to(text_device)
        text_mask = text_mask.to(text_device)

        text_hidden = self._encode_text_hidden(text_ids, text_mask)

        planner_cover_strength = 1.0
        non_cover_text_hidden = None
        non_cover_text_mask = None
        if use_planner_conditioning:
            try:
                planner_cover_strength = float(self._config.planner_cover_strength)
            except Exception:
                planner_cover_strength = 1.0
            planner_cover_strength = max(0.0, min(1.0, planner_cover_strength))
            if planner_cover_strength < 1.0:
                if bool(self._config.use_sft_prompt):
                    non_cover_caption_input = _sft_gen_prompt(_DEFAULT_DIT_INSTRUCTION, effective_caption, metas)
                else:
                    non_cover_caption_input = prompt or "music"
                non_cover_ids, non_cover_mask = self._tokenize_text_input(non_cover_caption_input, max_length=256)
                non_cover_ids = non_cover_ids.to(text_device)
                non_cover_mask = non_cover_mask.to(text_device)
                non_cover_text_hidden = self._encode_text_hidden(non_cover_ids, non_cover_mask)
                non_cover_text_mask = non_cover_mask

        # Upstream always formats the lyric branch, even when the user leaves lyrics empty.
        lyrics_input = _format_lyrics(lyrics.strip(), effective_vocal_language) if isinstance(lyrics, str) else _format_lyrics("", effective_vocal_language)
        lyric_ids, lyric_mask = self._tokenize_text_input(lyrics_input, max_length=2048)
        lyric_ids = lyric_ids.to(text_device)
        lyric_mask = lyric_mask.to(text_device)
        lyric_hidden = self._embed_lyrics(lyric_ids)

        # Keep all floating conditioning tensors aligned with the DiT model dtype.
        # MPS can abort hard (LLVM error) on mixed f32/f16 arithmetic.
        model_dtype = dtype
        try:
            if self._model is not None:
                model_dtype = next(self._model.parameters()).dtype
        except Exception:
            model_dtype = dtype

        if isinstance(text_hidden, torch.Tensor):
            if torch.is_floating_point(text_hidden):
                text_hidden = text_hidden.to(device=device, dtype=model_dtype)
            else:
                text_hidden = text_hidden.to(device)
        if isinstance(non_cover_text_hidden, torch.Tensor):
            if torch.is_floating_point(non_cover_text_hidden):
                non_cover_text_hidden = non_cover_text_hidden.to(device=device, dtype=model_dtype)
            else:
                non_cover_text_hidden = non_cover_text_hidden.to(device)
        if isinstance(non_cover_text_mask, torch.Tensor):
            non_cover_text_mask = non_cover_text_mask.to(device=device)
        if isinstance(lyric_hidden, torch.Tensor):
            if torch.is_floating_point(lyric_hidden):
                lyric_hidden = lyric_hidden.to(device=device, dtype=model_dtype)
            else:
                lyric_hidden = lyric_hidden.to(device)

        # Reference timbre defaults to silence.
        silence = self._get_silence_latent_slice(latent_len, device=device, dtype=model_dtype)
        latent_channels = int(silence.shape[-1])

        refer_frames = int(self._config.refer_timbre_frames)
        reference_mode_value = "silence" if use_planner_conditioning else str(self._config.reference_mode).strip().lower()
        if reference_mode_value == "silence":
            refer_latents = self._get_silence_latent_slice(refer_frames, device=device, dtype=model_dtype)
            refer_latents = refer_latents.expand(1, -1, -1).contiguous()  # [1, 750, 64]
            refer_latents = refer_latents.to(dtype=model_dtype)
            reference_mode = "silence"
        else:
            refer_latents = torch.zeros((1, refer_frames, latent_channels), device=device, dtype=model_dtype)
            reference_mode = "zeros"
        refer_order_mask = torch.tensor([0], device=device, dtype=torch.long)
        planner_lm_hints = None
        if use_planner_conditioning:
            planner_lm_hints = self._decode_audio_code_ids_to_lm_hints(
                audio_codes=planner_audio_codes,
                device=device,
                dtype=model_dtype,
            )
            if int(planner_lm_hints.shape[1]) < latent_len:
                pad = self._get_silence_latent_slice(
                    latent_len - int(planner_lm_hints.shape[1]),
                    device=device,
                    dtype=model_dtype,
                )
                planner_lm_hints = torch.cat([planner_lm_hints, pad], dim=1)
            elif int(planner_lm_hints.shape[1]) > latent_len:
                planner_lm_hints = planner_lm_hints[:, :latent_len, :]
            planner_lm_hints = planner_lm_hints.contiguous()
            src_latents = planner_lm_hints.clone()
            src_latents_init = "audio_code_hints"
        elif bool(self._config.use_random_src_latents):
            try:
                gen = torch.Generator(device=device).manual_seed(int(seed))
            except Exception:
                gen = torch.Generator().manual_seed(int(seed))
            src_latents = torch.randn(
                (1, latent_len, latent_channels),
                generator=gen,
                device=device,
                dtype=model_dtype,
            )
            src_latents_init = "random"
        else:
            src_latents = silence.clone().expand(1, -1, -1).contiguous().to(dtype=model_dtype)
            src_latents_init = "silence"

        chunk_mask_mode = str(self._config.chunk_mask_mode).strip().lower()
        if use_planner_conditioning and chunk_mask_mode == "zeros":
            chunk_mask_mode = "auto"
        if chunk_mask_mode == "auto":
            chunk_masks = torch.full((1, latent_len, latent_channels), 2.0, device=device, dtype=model_dtype)
        elif chunk_mask_mode == "ones":
            chunk_masks = torch.ones((1, latent_len, latent_channels), device=device, dtype=model_dtype)
        elif chunk_mask_mode == "zeros":
            chunk_masks = torch.zeros((1, latent_len, latent_channels), device=device, dtype=model_dtype)
        else:
            raise ValueError(
                f"Unsupported chunk_mask_mode={self._config.chunk_mask_mode!r}. Use 'zeros', 'ones', or 'auto'."
            )
        is_covers = torch.ones((1,), device=device, dtype=torch.long) if use_planner_conditioning else torch.zeros((1,), device=device, dtype=torch.long)

        infer_method = str(self._config.infer_method)

        def _run_model_once(*, run_seed: int, run_infer_method: str) -> Any:
            return self._model.generate_audio(
                text_hidden_states=text_hidden,
                text_attention_mask=text_mask.to(device=device),
                lyric_hidden_states=lyric_hidden,
                lyric_attention_mask=lyric_mask.to(device=device),
                refer_audio_acoustic_hidden_states_packed=refer_latents,
                refer_audio_order_mask=refer_order_mask,
                src_latents=src_latents,
                chunk_masks=chunk_masks,
                is_covers=is_covers,
                silence_latent=silence,
                seed=int(run_seed),
                fix_nfe=int(self._config.fix_nfe),
                infer_method=str(run_infer_method),
                audio_cover_strength=float(planner_cover_strength if use_planner_conditioning else 1.0),
                non_cover_text_hidden_states=non_cover_text_hidden,
                non_cover_text_attention_mask=non_cover_text_mask,
                precomputed_lm_hints_25Hz=planner_lm_hints,
                shift=float(self._config.shift),
                dcw_enabled=bool(self._config.dcw_enabled),
                dcw_mode=str(self._config.dcw_mode),
                dcw_scaler=float(self._config.dcw_scaler),
                dcw_high_scaler=float(self._config.dcw_high_scaler),
            )

        # Generate latents with ACE-Step DiT.
        try:
            out = _run_model_once(run_seed=int(seed), run_infer_method=infer_method)
        except NotImplementedError as e:
            msg = str(e)
            is_mps = device == "mps"
            if is_mps and bool(self._config.auto_retry_cpu_on_mps_error):
                print(
                    "WARNING #FALLBACK : MPS op unsupported during ACE-Step diffusion; retrying on CPU float32. "
                    f"(error={msg})",
                    file=sys.stderr,
                )
                # Reload on CPU/float32 and retry.
                self._loaded = False
                self._model = None
                self._vae = None
                self._text_tokenizer = None
                self._text_encoder = None
                self._silence_latent = None
                self._config = replace(self._config, device="cpu", torch_dtype="float32", vae_torch_dtype="float32")
                self._ensure_loaded()
                return self.generate_audio(request)
            raise

        pred_latents = None
        if isinstance(out, dict):
            pred_latents = out.get("target_latents")
        if pred_latents is None:
            raise RuntimeError("ACE-Step generate_audio did not return target_latents")

        # Guard against non-finite latents (can happen on unstable dtype/scheduler combos).
        def _has_non_finite_latents(latents: Any) -> bool:
            try:
                return isinstance(latents, torch.Tensor) and (not bool(torch.isfinite(latents).all().item()))
            except Exception:
                return False

        if _has_non_finite_latents(pred_latents):
            alt_infer_method = "ode" if str(infer_method) != "ode" else "sde"
            retry_plan = [(int(seed) + 1, alt_infer_method)]
            if seed_was_provided:
                retry_plan.extend(
                    [
                        (int(seed) + 2, str(infer_method)),
                        (int(seed) + 3, alt_infer_method),
                    ]
                )
            else:
                retry_plan.extend(
                    [
                        (123, "ode"),
                        (124, "sde"),
                        (321, "ode"),
                        (322, "sde"),
                    ]
                )

            attempts_seen = {(int(seed), str(infer_method))}
            recovered = False
            for retry_seed, retry_method in retry_plan:
                retry_key = (int(retry_seed), str(retry_method))
                if retry_key in attempts_seen:
                    continue
                attempts_seen.add(retry_key)
                print(
                    "WARNING #FALLBACK : ACE-Step returned non-finite latents; "
                    f"retrying with infer_method={retry_method!r}, seed={retry_seed}.",
                    file=sys.stderr,
                )
                out = _run_model_once(run_seed=int(retry_seed), run_infer_method=str(retry_method))
                infer_method = str(retry_method)
                seed = int(retry_seed)
                if isinstance(out, dict):
                    pred_latents = out.get("target_latents")
                if pred_latents is None:
                    raise RuntimeError("ACE-Step retry did not return target_latents")
                if not _has_non_finite_latents(pred_latents):
                    recovered = True
                    break
            if not recovered:
                if device == "mps" and bool(self._config.auto_retry_cpu_on_mps_error):
                    cpu_seed = int(request.seed) if seed_was_provided else 123
                    print(
                        "WARNING #FALLBACK : ACE-Step still produced non-finite latents on MPS after retries; "
                        "retrying on CPU float32.",
                        file=sys.stderr,
                    )
                    self._loaded = False
                    self._model = None
                    self._vae = None
                    self._text_tokenizer = None
                    self._text_encoder = None
                    self._silence_latent = None
                    self._config = replace(
                        self._config,
                        device="cpu",
                        torch_dtype="float32",
                        vae_torch_dtype="float32",
                        infer_method="ode",
                    )
                    self._ensure_loaded()
                    return self.generate_audio(replace(request, seed=cpu_seed))
                raise RuntimeError("ACE-Step produced non-finite latents after retries; aborting.")

        # Decode to waveform via VAE.
        wav = self._decode_latents(pred_latents)
        wav = _remove_dc_offset(wav)

        if bool(self._config.enable_normalization):
            wav = _peak_normalize(wav, target_db=float(self._config.normalization_db))
        wav = _ensure_finite_audio(wav)

        # Encode WAV bytes.
        wav_bytes = _encode_wav_bytes(wav[0], sample_rate=48000)
        stats = inspect_music_signal_bytes(wav_bytes)
        harmonic_stats = None
        modulation_stats = None
        try:
            harmonic_stats = inspect_harmonic_diversity_bytes(wav_bytes)
        except Exception as exc:
            if bool(self._config.quality_retry_enabled):
                print(
                    "WARNING #FALLBACK : ACE-Step harmonic-diversity inspection failed; "
                    f"continuing with basic signal gate. (error={type(exc).__name__}: {exc})",
                    file=sys.stderr,
                )
        try:
            modulation_stats = inspect_spectrotemporal_modulation_bytes(wav_bytes)
        except Exception as exc:
            if bool(self._config.quality_retry_enabled):
                print(
                    "WARNING #FALLBACK : ACE-Step spectrotemporal inspection failed; "
                    f"continuing with basic signal gate. (error={type(exc).__name__}: {exc})",
                    file=sys.stderr,
                )
        quality_retry_count = 0
        try:
            quality_retry_count = int(extra.get("_acestep_quality_retry_count", 0) or 0)
        except Exception:
            quality_retry_count = 0
        harmonic_collapsed = bool(
            harmonic_stats is not None
            and (
                bool(harmonic_stats.is_probably_single_note_collapse)
                or bool(harmonic_stats.is_probably_low_pitch_variety)
            )
        )
        modulation_artifact = bool(
            modulation_stats is not None
            and bool(modulation_stats.is_probably_broadband_repetition_artifact)
        )
        quality_passed = (
            bool(stats.is_probably_music_like)
            and not bool(stats.is_probably_repetitive_noise)
            and not harmonic_collapsed
            and not modulation_artifact
        )
        if (
            bool(self._config.quality_retry_enabled)
            and not seed_was_provided
            and not quality_passed
            and quality_retry_count < int(self._config.quality_retry_max_attempts)
        ):
            attempted_raw = extra.get("_acestep_quality_retry_attempted_seeds")
            attempted_seeds = []
            if isinstance(attempted_raw, (list, tuple)):
                for value in attempted_raw:
                    try:
                        attempted_seeds.append(int(value))
                    except Exception:
                        continue
            if int(seed) not in attempted_seeds:
                attempted_seeds.append(int(seed))
            retry_seed = None
            for candidate in self._config.quality_retry_fallback_seeds:
                try:
                    candidate_seed = int(candidate)
                except Exception:
                    continue
                if candidate_seed not in attempted_seeds:
                    retry_seed = candidate_seed
                    break
            if retry_seed is None:
                retry_seed = int(seed) + quality_retry_count + 1
                while retry_seed in attempted_seeds:
                    retry_seed += 1
            print(
                "WARNING #FALLBACK : ACE-Step output failed the local music smoke test; "
                f"retrying with seed={retry_seed}.",
                file=sys.stderr,
            )
            retry_extra = dict(extra)
            retry_extra["_acestep_quality_retry_count"] = quality_retry_count + 1
            retry_extra["_acestep_quality_retry_attempted_seeds"] = attempted_seeds + [int(retry_seed)]
            retry_extra["_acestep_user_seed_provided"] = bool(seed_was_provided)
            return self.generate_audio(replace(request, seed=int(retry_seed), extra=retry_extra))

        meta: Dict[str, Any] = {
            "backend": self.backend_id,
            "repo_id": str(self._config.repo_id),
            "device": device,
            "dtype": str(getattr(self._dtype, "__name__", self._dtype)),
            "duration_s": float(duration_s),
            "frames_per_second": int(frames_per_second),
            "latent_frames": int(latent_len),
            "seed": int(seed),
            "infer_method": str(infer_method),
            "dcw_enabled": bool(self._config.dcw_enabled),
            "dcw_mode": str(self._config.dcw_mode),
            "dcw_scaler": float(self._config.dcw_scaler),
            "dcw_high_scaler": float(self._config.dcw_high_scaler),
            "src_latents_init": str(src_latents_init),
            "use_sft_prompt": bool(use_planner_conditioning or self._config.use_sft_prompt),
            "chunk_mask_mode": str(chunk_mask_mode),
            "reference_mode": str(reference_mode),
            "planner_mode": "lm_audio_codes" if use_planner_conditioning else "direct_text",
            "planner_cover_strength": float(planner_cover_strength if use_planner_conditioning else 0.0),
            "planner_error": planner_error,
            "quality_retry_count": int(quality_retry_count),
            "quality_gate_passed": bool(quality_passed),
            "audio_stats": asdict(stats),
            "harmonic_diversity_stats": asdict(harmonic_stats) if harmonic_stats is not None else None,
            "spectrotemporal_modulation_stats": asdict(modulation_stats) if modulation_stats is not None else None,
            "harmonic_collapse_detected": bool(harmonic_collapsed),
            "spectrotemporal_artifact_detected": bool(modulation_artifact),
        }
        if use_planner_conditioning and isinstance(planner, dict):
            meta["planner_audio_code_count"] = int(planner.get("audio_code_count") or 0)
            meta["planner_lm_model"] = str(planner.get("lm_model") or "")
            meta["caption"] = effective_caption
            meta["vocal_language"] = effective_vocal_language
            if effective_bpm is not None:
                meta["bpm"] = int(effective_bpm)
            if effective_keyscale:
                meta["keyscale"] = effective_keyscale
            if effective_timesignature:
                meta["timesignature"] = effective_timesignature

        return GeneratedAsset(data=bytes(wav_bytes), mime_type="audio/wav", metadata=meta)
