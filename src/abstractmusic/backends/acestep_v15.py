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
import subprocess
import sys
import wave
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

from ..errors import OptionalDependencyMissingError
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


def _default_meta_string(duration_s: float) -> str:
    # Mirrors upstream _create_default_meta.
    ds = max(1, int(round(float(duration_s))))
    return (
        "- bpm: N/A\n"
        "- timesignature: N/A\n"
        "- keyscale: N/A\n"
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
    cache_dir: Optional[str] = None

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
    enable_normalization: bool = True
    normalization_db: float = -1.0
    use_random_src_latents: bool = True
    use_sft_prompt: bool = False
    chunk_mask_mode: str = "zeros"  # zeros|ones
    reference_mode: str = "zeros"  # zeros|silence

    # VAE decode tiling (helps on MPS/unified memory).
    mps_decode_chunk_frames: int = 32
    mps_decode_overlap_frames: int = 8
    cpu_decode_chunk_frames: int = 256
    cpu_decode_overlap_frames: int = 64

    # When running on MPS, retry on CPU if an op is unsupported.
    auto_retry_cpu_on_mps_error: bool = True


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
        if self._loaded:
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
        hf_hub_download = _lazy_import_hf_hub_download()
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
        cache_dir = self._config.cache_dir

        # Load core DiT model (ACE-Step v1.5 turbo).
        try:
            self._model = _from_pretrained_transformers(
                AceStepConditionGenerationModel,
                dtype=dtype,
                pretrained_model_name_or_path=repo_id,
                revision=revision,
                cache_dir=cache_dir,
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
                cache_dir=cache_dir,
                subfolder=str(self._config.text_encoder_subfolder),
                trust_remote_code=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ACE-Step text tokenizer from {repo_id!r}: {e}") from e

        try:
            self._text_encoder = _from_pretrained_transformers(
                AutoModel,
                dtype=text_dtype,
                pretrained_model_name_or_path=repo_id,
                revision=revision,
                cache_dir=cache_dir,
                subfolder=str(self._config.text_encoder_subfolder),
                trust_remote_code=False,
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
                cache_dir=cache_dir,
                subfolder=str(self._config.vae_subfolder),
                low_cpu_mem_usage=False,
            )
            self._vae.to(device)
            self._vae.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load ACE-Step VAE from {repo_id!r}: {e}") from e

        # Load and transpose silence latent (stored as [1, C, T] in repo).
        try:
            sl_path = hf_hub_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=cache_dir,
                filename=f"{self._config.dit_subfolder}/silence_latent.pt",
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

    def _tokenize_no_truncation(self, text: str) -> Tuple[Any, Any]:
        """Tokenize without silent truncation. Raises if model_max_length is exceeded."""
        tok = self._text_tokenizer
        if tok is None:
            raise RuntimeError("Text tokenizer not loaded")

        out = tok(text, return_tensors="pt", padding=False, truncation=False)
        input_ids = getattr(out, "input_ids", None)
        attention_mask = getattr(out, "attention_mask", None)
        if input_ids is None:
            raise RuntimeError("Tokenizer did not return input_ids")
        if attention_mask is None:
            # Some tokenizers may omit attention_mask for single sequences; generate one.
            torch = _lazy_import_torch()
            attention_mask = torch.ones_like(input_ids)

        # Guard against extremely long prompts.
        max_len = getattr(tok, "model_max_length", None)
        try:
            max_len_int = int(max_len) if max_len is not None else None
        except Exception:
            max_len_int = None
        if max_len_int and max_len_int > 0:
            if int(input_ids.shape[-1]) > max_len_int:
                raise ValueError(
                    f"Prompt token length {int(input_ids.shape[-1])} exceeds model_max_length={max_len_int}. "
                    "Shorten the prompt/lyrics, or configure a smaller model."
                )

        return input_ids, attention_mask.bool()

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

        def _ensure_vae_cpu_float32() -> None:
            if self._vae is None:
                raise RuntimeError("VAE not loaded")
            AutoencoderOobleck = _lazy_import_diffusers_oobleck()
            _patch_oobleck_weight_norm(torch)
            self._vae = _from_pretrained_diffusers(
                AutoencoderOobleck,
                dtype=torch.float32,
                pretrained_model_name_or_path=str(self._config.repo_id),
                revision=self._config.revision,
                cache_dir=self._config.cache_dir,
                subfolder=str(self._config.vae_subfolder),
                low_cpu_mem_usage=False,
            )
            self._vae.to("cpu")
            self._vae.eval()

        def _offload_mps_after_oom() -> None:
            if str(device) != "mps":
                return
            try:
                if self._model is not None:
                    self._model.to("cpu")
            except Exception:
                pass
            try:
                if self._silence_latent is not None:
                    self._silence_latent = self._silence_latent.to("cpu")
            except Exception:
                pass
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
            try:
                import gc

                gc.collect()
            except Exception:
                pass

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
                    print(
                        "WARNING #FALLBACK : MPS VAE decode failed; retrying decode on CPU float32. "
                        f"(error={type(e).__name__}: {e})",
                        file=sys.stderr,
                    )
                    # Move VAE to CPU and decode the remainder on CPU to avoid repeated OOMs.
                    _ensure_vae_cpu_float32()
                    _offload_mps_after_oom()
                    try:
                        pred_latents_for_decode = pred_latents_for_decode.to("cpu", dtype=torch.float32)
                    except Exception as err:
                        _offload_mps_after_oom()
                        try:
                            pred_latents_for_decode = pred_latents_for_decode.to("cpu", dtype=torch.float32)
                        except Exception as err2:
                            raise RuntimeError(
                                "Failed to move latents to CPU after MPS OOM. "
                                "Try lowering duration or increasing --mps-max-memory-gb (<=16)."
                            ) from err2
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass
                    device = "cpu"
                    if wav_accum is not None:
                        try:
                            wav_accum = wav_accum.to("cpu", dtype=torch.float32)
                        except Exception:
                            wav_accum = wav_accum.to("cpu")
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

    def generate_audio(self, request: AudioGenerationRequest) -> GeneratedAsset:
        torch = _lazy_import_torch()
        self._ensure_loaded()

        device = str(self._device or "cpu")
        dtype = self._dtype

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

        # Build caption conditioning text.
        if bool(self._config.use_sft_prompt):
            instruction = "Fill the audio semantic mask based on the given conditions:"
            metas = _default_meta_string(duration_s)
            caption_input = _sft_gen_prompt(instruction, prompt, metas)
        else:
            # Align with upstream inference: use raw prompt/tags without SFT wrapper.
            caption_input = prompt or "music"

        text_device = str(self._text_device or device)
        # Tokenize and encode caption on text runtime device.
        text_ids, text_mask = self._tokenize_no_truncation(caption_input)
        text_ids = text_ids.to(text_device)
        text_mask = text_mask.to(text_device)

        text_hidden = self._encode_text_hidden(text_ids, text_mask)

        # Lyric conditioning:
        # - if lyrics are provided, encode them
        # - otherwise use a null lyric condition (mask=0) instead of synthetic text
        if isinstance(lyrics, str) and lyrics.strip():
            language = str(request.vocal_language or "unknown").strip() or "unknown"
            lyrics_input = _format_lyrics(lyrics.strip(), language)
            lyric_ids, lyric_mask = self._tokenize_no_truncation(lyrics_input)
            lyric_ids = lyric_ids.to(text_device)
            lyric_mask = lyric_mask.to(text_device)
        else:
            lyric_ids = torch.zeros((int(text_ids.shape[0]), 1), device=text_device, dtype=torch.long)
            lyric_mask = torch.zeros((int(text_ids.shape[0]), 1), device=text_device, dtype=torch.bool)
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
        if isinstance(lyric_hidden, torch.Tensor):
            if torch.is_floating_point(lyric_hidden):
                lyric_hidden = lyric_hidden.to(device=device, dtype=model_dtype)
            else:
                lyric_hidden = lyric_hidden.to(device)

        # Reference timbre defaults to silence.
        silence = self._silence_latent
        if silence is None:
            raise RuntimeError("silence_latent not loaded")
        silence = silence.to(device=device, dtype=model_dtype)
        latent_channels = int(silence.shape[-1])

        seed = request.seed
        if not isinstance(seed, int) or seed < 0:
            seed = random.randint(0, 2**32 - 1)

        refer_frames = int(self._config.refer_timbre_frames)
        if str(self._config.reference_mode).strip().lower() == "silence":
            refer_latents = silence[:, :refer_frames, :].expand(1, -1, -1).contiguous()  # [1, 750, 64]
            refer_latents = refer_latents.to(dtype=model_dtype)
            reference_mode = "silence"
        else:
            refer_latents = torch.zeros((1, refer_frames, latent_channels), device=device, dtype=model_dtype)
            reference_mode = "zeros"
        refer_order_mask = torch.tensor([0], device=device, dtype=torch.long)
        if bool(self._config.use_random_src_latents):
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
            src_latents = silence[:, :latent_len, :].expand(1, -1, -1).contiguous().to(dtype=model_dtype)
            src_latents_init = "silence"

        chunk_mask_mode = str(self._config.chunk_mask_mode).strip().lower()
        if chunk_mask_mode == "ones":
            chunk_masks = torch.ones((1, latent_len, latent_channels), device=device, dtype=model_dtype)
        elif chunk_mask_mode == "zeros":
            chunk_masks = torch.zeros((1, latent_len, latent_channels), device=device, dtype=model_dtype)
        else:
            raise ValueError(f"Unsupported chunk_mask_mode={self._config.chunk_mask_mode!r}. Use 'zeros' or 'ones'.")
        is_covers = torch.zeros((1,), device=device, dtype=torch.long)

        infer_method = str(self._config.infer_method)

        def _run_model_once(*, run_seed: int, run_infer_method: str) -> Any:
            return self._model.generate_audio(
                text_hidden_states=text_hidden,
                text_attention_mask=text_mask.to(device=device, dtype=text_hidden.dtype),
                lyric_hidden_states=lyric_hidden,
                lyric_attention_mask=lyric_mask.to(device=device, dtype=lyric_hidden.dtype),
                refer_audio_acoustic_hidden_states_packed=refer_latents,
                refer_audio_order_mask=refer_order_mask,
                src_latents=src_latents,
                chunk_masks=chunk_masks,
                is_covers=is_covers,
                silence_latent=silence[:, :latent_len, :],
                seed=int(run_seed),
                fix_nfe=int(self._config.fix_nfe),
                infer_method=str(run_infer_method),
                shift=float(self._config.shift),
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
        has_non_finite_latents = False
        try:
            has_non_finite_latents = isinstance(pred_latents, torch.Tensor) and (not bool(torch.isfinite(pred_latents).all().item()))
        except Exception:
            has_non_finite_latents = False

        if has_non_finite_latents:
            alt_infer_method = "ode" if str(infer_method) != "ode" else "sde"
            retry_seed = int(seed) + 1
            print(
                "WARNING #FALLBACK : ACE-Step returned non-finite latents; "
                f"retrying with infer_method={alt_infer_method!r}, seed={retry_seed}.",
                file=sys.stderr,
            )
            out = _run_model_once(run_seed=retry_seed, run_infer_method=alt_infer_method)
            infer_method = alt_infer_method
            seed = retry_seed
            if isinstance(out, dict):
                pred_latents = out.get("target_latents")
            if pred_latents is None:
                raise RuntimeError("ACE-Step retry did not return target_latents")
            try:
                still_non_finite = isinstance(pred_latents, torch.Tensor) and (not bool(torch.isfinite(pred_latents).all().item()))
            except Exception:
                still_non_finite = False
            if still_non_finite:
                raise RuntimeError("ACE-Step produced non-finite latents after retry; aborting.")

        # Decode to waveform via VAE.
        wav = self._decode_latents(pred_latents)
        wav = _remove_dc_offset(wav)

        if bool(self._config.enable_normalization):
            wav = _peak_normalize(wav, target_db=float(self._config.normalization_db))
        wav = _ensure_finite_audio(wav)

        # Encode WAV bytes.
        wav_bytes = _encode_wav_bytes(wav[0], sample_rate=48000)

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
            "src_latents_init": str(src_latents_init),
            "use_sft_prompt": bool(self._config.use_sft_prompt),
            "chunk_mask_mode": str(chunk_mask_mode),
            "reference_mode": str(reference_mode),
        }

        return GeneratedAsset(data=bytes(wav_bytes), mime_type="audio/wav", metadata=meta)
