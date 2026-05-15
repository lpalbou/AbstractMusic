"""
Diffusers-based local text-to-audio backend.

This backend loads a Diffusers audio pipeline from a Hugging Face model id and
generates audio in-process. Output is encoded as WAV bytes using the stdlib
`wave` module (no external codecs required).
"""

from __future__ import annotations

import inspect
import io
import sys
import wave
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

from ..errors import OptionalDependencyMissingError
from ..types import AudioGenerationRequest, GeneratedAsset, MusicBackendCapabilities


def _lazy_import_torch():
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): torch. Install via: pip install 'torch'"
        ) from e
    return torch


def _lazy_import_diffusers():
    try:
        import diffusers  # type: ignore
        from diffusers import DiffusionPipeline  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): diffusers. Install via: pip install 'diffusers'"
        ) from e
    return diffusers, DiffusionPipeline


def _lazy_import_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): numpy. Install via: pip install 'numpy'"
        ) from e
    return np


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
    return "cpu"


def _resolve_dtype(torch_mod: Any, dtype: str, *, device: str) -> Any:
    d = str(dtype or "").strip().lower() or "float32"
    dev = str(device or "").strip().lower() or "cpu"

    if d == "auto":
        # Practical default:
        # - float16 on CUDA/MPS accelerators
        # - float32 on CPU for broadest correctness
        d = "float16" if dev in {"cuda", "mps"} else "float32"

    if d in {"float16", "fp16"}:
        if dev == "cpu":
            # float16 on CPU is not broadly supported and Diffusers warns loudly.
            return torch_mod.float32
        return torch_mod.float16
    if d in {"bfloat16", "bf16"}:
        # bfloat16 is typically only viable on CUDA-class accelerators.
        return torch_mod.bfloat16 if dev in {"cuda", "xpu"} else torch_mod.float32
    return torch_mod.float32


def _coerce_waveform_to_np(audio: Any) -> Tuple[Any, int]:
    """Return (np.ndarray waveform, inferred_channels_hint).

    The channel hint is the number of channels when inferable, otherwise 1.
    """
    np = _lazy_import_numpy()

    if hasattr(audio, "detach") and callable(getattr(audio, "detach", None)):
        # torch tensor
        audio = audio.detach().cpu().float().numpy()
    else:
        audio = np.asarray(audio)

    if audio.ndim == 1:
        return audio, 1
    if audio.ndim == 2:
        # Heuristic: many libraries represent audio as (channels, samples).
        # If first dim is small, assume channels-first.
        if int(audio.shape[0]) <= 8 and int(audio.shape[1]) > int(audio.shape[0]):
            return audio.T, int(audio.shape[0])
        return audio, int(audio.shape[1])
    raise ValueError("Unsupported audio array shape; expected 1D or 2D waveform")


def _encode_wav_bytes(audio_np: Any, *, sample_rate: int) -> bytes:
    np = _lazy_import_numpy()
    x, _ = _coerce_waveform_to_np(audio_np)
    x = np.asarray(x, dtype=np.float32)

    # Ensure (samples, channels)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim != 2:
        raise ValueError("Unsupported audio array shape after coercion")

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


@dataclass(frozen=True)
class DiffusersAudioBackendConfig:
    model_id: str
    device: str = "auto"
    torch_dtype: str = "float16"
    pipeline_class: Optional[str] = None  # e.g. "AudioLDMPipeline", "AudioLDM2Pipeline", "StableAudioPipeline"
    num_inference_steps: int = 50
    guidance_scale: Optional[float] = None
    duration_s: Optional[float] = 10.0
    sampling_rate: Optional[int] = None
    # When running on Apple Silicon MPS, some vocoder convs can raise:
    # "Output channels > 65536 not supported at the MPS device."
    # We retry on CPU with an explicit warning. #FALLBACK : MPS limitation
    auto_retry_cpu_on_mps_error: bool = True


class DiffusersAudioBackend:
    """Local text-to-audio backend powered by Diffusers."""

    backend_id = "abstractmusic:diffusers"

    def __init__(self, *, config: DiffusersAudioBackendConfig) -> None:
        self._config = config
        self._pipe = None
        self._pipe_device = None

        mid = str(config.model_id or "").strip()
        if not mid:
            raise ValueError("model_id is required for DiffusersAudioBackend")

    def get_capabilities(self) -> MusicBackendCapabilities:
        return MusicBackendCapabilities(
            supported_tasks=("text_to_music", "text_to_audio"),
            output_formats=("wav",),
            supports_lyrics=False,
            supports_negative_prompt=None,
            supports_guidance_scale=None,
            model_id=str(self._config.model_id),
            official_8bit_available=False,
            preferred_precision="official pipeline dtype; prefer official 8-bit artifacts when available",
        )

    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe

        torch = _lazy_import_torch()
        diffusers, DiffusionPipeline = _lazy_import_diffusers()

        device = _resolve_device(torch, self._config.device)
        dtype_s = str(self._config.torch_dtype or "").strip() or "auto"
        dtype = _resolve_dtype(torch, dtype_s, device=device)
        if device == "cpu" and str(dtype_s).strip().lower() in {"float16", "fp16", "auto"} and dtype == torch.float32:
            # float16 on CPU is not broadly supported and Diffusers warns loudly.
            # Use float32 for correctness and predictable behavior.
            print(
                "WARNING #FALLBACK : CPU device does not reliably support float16 for Diffusers audio pipelines. "
                "Reloading with torch_dtype=float32. (To silence this, set --dtype float32 / ABSTRACTMUSIC_TORCH_DTYPE=float32.)",
                file=sys.stderr,
            )
            dtype = torch.float32
            self._config = replace(self._config, torch_dtype="float32")

        pipe_cls = None
        if isinstance(self._config.pipeline_class, str) and self._config.pipeline_class.strip():
            pipe_cls = getattr(diffusers, self._config.pipeline_class.strip(), None)
            if pipe_cls is None:
                raise ValueError(f"Unknown diffusers pipeline class: {self._config.pipeline_class!r}")
        else:
            pipe_cls = DiffusionPipeline

        # Load pipeline (model weights are cached by HF).
        pipe = pipe_cls.from_pretrained(str(self._config.model_id), torch_dtype=dtype)  # type: ignore[attr-defined]

        # Move to device when supported.
        to_fn = getattr(pipe, "to", None)
        if callable(to_fn):
            pipe = to_fn(device)

        self._pipe = pipe
        self._pipe_device = device
        return self._pipe

    def _infer_sampling_rate(self, pipe: Any) -> int:
        if isinstance(self._config.sampling_rate, int) and self._config.sampling_rate > 0:
            return int(self._config.sampling_rate)
        # StableAudioPipeline exposes `pipe.vae.sampling_rate`.
        try:
            vae = getattr(pipe, "vae", None)
            sr = getattr(vae, "sampling_rate", None) if vae is not None else None
            if isinstance(sr, int) and sr > 0:
                return int(sr)
        except Exception:
            pass
        # Common default for AudioLDM family.
        return 16000

    def generate_audio(self, request: AudioGenerationRequest) -> GeneratedAsset:
        torch = _lazy_import_torch()
        pipe = self._load_pipe()

        prompt = str(request.prompt or "")
        if not prompt.strip():
            prompt = ""

        steps = int(request.num_inference_steps) if request.num_inference_steps is not None else int(self._config.num_inference_steps)
        duration_s = float(request.duration_s) if request.duration_s is not None else (float(self._config.duration_s) if self._config.duration_s is not None else None)
        guidance_scale = float(request.guidance_scale) if request.guidance_scale is not None else (float(self._config.guidance_scale) if self._config.guidance_scale is not None else None)
        seed = request.seed

        call_sig = None
        try:
            call_sig = inspect.signature(pipe.__call__)
        except Exception:
            call_sig = None
        params = set(call_sig.parameters.keys()) if call_sig is not None else set()

        gen = None
        if isinstance(seed, int) and seed >= 0 and "generator" in params:
            try:
                gen = torch.Generator(device=str(self._pipe_device or "cpu")).manual_seed(int(seed))
            except Exception:
                gen = torch.Generator().manual_seed(int(seed))

        kwargs: Dict[str, Any] = {}
        if "num_inference_steps" in params:
            kwargs["num_inference_steps"] = int(steps)
        if guidance_scale is not None and "guidance_scale" in params:
            kwargs["guidance_scale"] = float(guidance_scale)
        if isinstance(request.negative_prompt, str) and request.negative_prompt.strip() and "negative_prompt" in params:
            kwargs["negative_prompt"] = str(request.negative_prompt)
        if gen is not None:
            kwargs["generator"] = gen

        if duration_s is not None:
            if "audio_length_in_s" in params:
                kwargs["audio_length_in_s"] = float(duration_s)
            elif "audio_end_in_s" in params:
                kwargs["audio_end_in_s"] = float(duration_s)

        # Backend-specific passthrough (only for known parameters).
        extra = request.extra or {}
        for k, v in extra.items():
            ks = str(k or "").strip()
            if not ks:
                continue
            if params and ks in params and ks not in kwargs:
                kwargs[ks] = v

        def _run_pipe() -> Any:
            return pipe(prompt, **kwargs)

        try:
            out = _run_pipe()
        except NotImplementedError as e:
            msg = str(e)
            is_mps = str(self._pipe_device or "").strip().lower() == "mps"
            is_known_mps_limit = "Output channels > 65536 not supported at the MPS device" in msg
            if is_mps and is_known_mps_limit and bool(self._config.auto_retry_cpu_on_mps_error):
                print(
                    "WARNING #FALLBACK : PyTorch MPS limitation in vocoder (Output channels > 65536). "
                    "Retrying music generation on CPU with torch_dtype=float32. "
                    "To avoid this, pass --device cpu (and optionally --dtype float32).",
                    file=sys.stderr,
                )

                # Reload on CPU in float32 (do NOT move a float16 pipeline to CPU).
                self._pipe = None
                self._pipe_device = None
                self._config = replace(self._config, device="cpu", torch_dtype="float32")
                pipe = self._load_pipe()

                # After switching pipes, remove any kwargs not supported by the new call signature.
                try:
                    new_sig = inspect.signature(pipe.__call__)
                    new_params = set(new_sig.parameters.keys())
                    kwargs = {k: v for k, v in kwargs.items() if k in new_params}
                    params = new_params
                except Exception:
                    pass

                # Rebuild generator for CPU if needed.
                if isinstance(seed, int) and seed >= 0 and "generator" in params:
                    try:
                        kwargs["generator"] = torch.Generator(device="cpu").manual_seed(int(seed))
                    except Exception:
                        kwargs["generator"] = torch.Generator().manual_seed(int(seed))

                out = pipe(prompt, **kwargs)
            else:
                raise

        audios = getattr(out, "audios", None)
        if audios is None and isinstance(out, dict):
            audios = out.get("audios") or out.get("audio")
        if audios is None:
            raise ValueError("Diffusers pipeline did not return .audios")

        # Pick first waveform.
        audio0 = audios[0] if isinstance(audios, (list, tuple)) else audios

        sr = self._infer_sampling_rate(pipe)
        wav_bytes = _encode_wav_bytes(audio0, sample_rate=sr)

        meta: Dict[str, Any] = {
            "backend": self.backend_id,
            "model_id": str(self._config.model_id),
            "sampling_rate": int(sr),
            "num_inference_steps": int(steps),
        }
        if duration_s is not None:
            meta["duration_s"] = float(duration_s)
        if seed is not None:
            meta["seed"] = int(seed)

        return GeneratedAsset(data=bytes(wav_bytes), mime_type="audio/wav", metadata=meta)
