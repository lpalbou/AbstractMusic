"""
ACE-Step Diffusers backend.

This adapter is intentionally ACE-Step-specific. The generic Diffusers audio
backend cannot safely infer ACE-Step's `audio_duration`, `lyrics`, and
`vocal_language` parameters.
"""

from __future__ import annotations

import inspect
import io
import sys
import wave
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

from ..audio_analysis import inspect_wav_bytes
from ..errors import OptionalDependencyMissingError
from ..types import AudioGenerationRequest, GeneratedAsset, MusicBackendCapabilities


def _lazy_import_torch():
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): torch. Install with the appropriate "
            "local generation extra, for example: pip install 'abstractmusic[acestep-diffusers]'."
        ) from e
    return torch


def _lazy_import_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): numpy. Install with the appropriate "
            "local generation extra, for example: pip install 'abstractmusic[acestep-diffusers]'."
        ) from e
    return np


def _lazy_import_acestep_pipeline():
    try:
        import diffusers  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): diffusers. Install with the appropriate "
            "local generation extra, for example: pip install 'abstractmusic[acestep-diffusers]'."
        ) from e
    pipe_cls = getattr(diffusers, "AceStepPipeline", None)
    if pipe_cls is None:
        raise OptionalDependencyMissingError(
            "Installed diffusers does not expose AceStepPipeline. Install a Diffusers release that includes "
            "AceStepPipeline, or the official Diffusers main branch until that release is available."
        )
    return pipe_cls


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
    try:
        xpu = getattr(torch_mod, "xpu", None)
        if xpu is not None and bool(getattr(xpu, "is_available", lambda: False)()):
            return "xpu"
    except Exception:
        pass
    return "cpu"


def _resolve_dtype(torch_mod: Any, dtype: str, *, device: str) -> Any:
    d = str(dtype or "").strip().lower() or "auto"
    dev = str(device or "").strip().lower() or "cpu"
    if d == "auto":
        if dev in {"cuda", "xpu"}:
            return torch_mod.bfloat16
        if dev == "mps":
            return torch_mod.float16
        return torch_mod.float32
    if d in {"bfloat16", "bf16"}:
        if dev in {"cuda", "xpu"}:
            return torch_mod.bfloat16
        print(
            "WARNING #FALLBACK : bfloat16 is not a supported ACE-Step Diffusers dtype on this device; "
            "using float16 on MPS or float32 on CPU.",
            file=sys.stderr,
        )
        return torch_mod.float16 if dev == "mps" else torch_mod.float32
    if d in {"float16", "fp16"}:
        if dev == "cpu":
            print(
                "WARNING #FALLBACK : float16 on CPU is not reliable for ACE-Step Diffusers; using float32.",
                file=sys.stderr,
            )
            return torch_mod.float32
        return torch_mod.float16
    return torch_mod.float32


def _coerce_waveform_to_np(audio: Any) -> Tuple[Any, int]:
    np = _lazy_import_numpy()
    if hasattr(audio, "detach") and callable(getattr(audio, "detach", None)):
        audio = audio.detach().cpu().float().numpy()
    else:
        audio = np.asarray(audio)
    if audio.ndim == 1:
        return audio, 1
    if audio.ndim == 2:
        if int(audio.shape[0]) <= 8 and int(audio.shape[1]) > int(audio.shape[0]):
            return audio.T, int(audio.shape[0])
        return audio, int(audio.shape[1])
    if audio.ndim == 3:
        # Diffusers AceStepPipeline may return a batched tensor shaped
        # [batch, channels, samples] or [batch, samples, channels]. The CLI
        # writes one file, so select the first batch item and reuse 2D handling.
        if int(audio.shape[0]) >= 1:
            return _coerce_waveform_to_np(audio[0])
        if int(audio.shape[1]) >= 1:
            return _coerce_waveform_to_np(audio[:, 0, :])
    raise ValueError("Unsupported ACE-Step audio array shape; expected 1D, 2D, or batched 3D waveform")


def _encode_wav_bytes(audio: Any, *, sample_rate: int) -> bytes:
    np = _lazy_import_numpy()
    x, _ = _coerce_waveform_to_np(audio)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim != 2:
        raise ValueError("Unsupported audio array shape after coercion")
    if not bool(np.isfinite(x).all()):
        raise ValueError("ACE-Step Diffusers pipeline returned non-finite audio")
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2", copy=False)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(int(pcm.shape[1]))
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes(order="C"))
    return buf.getvalue()


@dataclass(frozen=True)
class AceStepDiffusersBackendConfig:
    model_id: str = "ACE-Step/acestep-v15-xl-turbo-diffusers"
    device: str = "auto"
    torch_dtype: str = "auto"
    num_inference_steps: int = 8
    duration_s: float = 10.0
    vocal_language: str = "en"
    guidance_scale: Optional[float] = None
    shift: Optional[float] = 3.0
    enable_vae_tiling: bool = True
    auto_retry_cpu_on_mps_error: bool = True


class AceStepDiffusersBackend:
    """Local ACE-Step XL Turbo backend powered by Diffusers AceStepPipeline."""

    backend_id = "abstractmusic:acestep-diffusers"

    def __init__(self, *, config: AceStepDiffusersBackendConfig) -> None:
        self._config = config
        self._pipe = None
        self._pipe_device = None
        self._pipe_dtype = None

    def get_capabilities(self) -> MusicBackendCapabilities:
        return MusicBackendCapabilities(
            supported_tasks=("text_to_music",),
            output_formats=("wav",),
            supports_lyrics=True,
            supports_negative_prompt=False,
            supports_guidance_scale=False,
            supports_reference_audio=None,
            supports_video=False,
            model_id=str(self._config.model_id),
            license="MIT",
            commercial_allowed=True,
            official_8bit_available=False,
            preferred_precision="official bf16/fp16 Diffusers checkpoint; no official 8-bit artifact reviewed",
        )

    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe

        torch = _lazy_import_torch()
        pipe_cls = _lazy_import_acestep_pipeline()
        device = _resolve_device(torch, self._config.device)
        dtype = _resolve_dtype(torch, self._config.torch_dtype, device=device)

        pipe = pipe_cls.from_pretrained(str(self._config.model_id), torch_dtype=dtype)
        to_fn = getattr(pipe, "to", None)
        if callable(to_fn):
            pipe = to_fn(device)

        if bool(self._config.enable_vae_tiling):
            vae = getattr(pipe, "vae", None)
            enable_tiling = getattr(vae, "enable_tiling", None) if vae is not None else None
            if callable(enable_tiling):
                enable_tiling()

        self._pipe = pipe
        self._pipe_device = device
        self._pipe_dtype = dtype
        return self._pipe

    def _run_pipe(self, pipe: Any, kwargs: Dict[str, Any]) -> Any:
        return pipe(**kwargs)

    def generate_audio(self, request: AudioGenerationRequest) -> GeneratedAsset:
        if request.negative_prompt:
            raise ValueError("ACE-Step Diffusers XL Turbo does not support negative_prompt.")
        if request.guidance_scale is not None:
            raise ValueError("ACE-Step Diffusers XL Turbo is guidance-distilled; omit guidance_scale.")

        torch = _lazy_import_torch()
        pipe = self._load_pipe()

        try:
            call_sig = inspect.signature(pipe.__call__)
            params = set(call_sig.parameters.keys())
        except Exception:
            params = set()

        duration_s = float(request.duration_s) if request.duration_s is not None else float(self._config.duration_s)
        steps = (
            int(request.num_inference_steps)
            if request.num_inference_steps is not None
            else int(self._config.num_inference_steps)
        )
        vocal_language = str(request.vocal_language or self._config.vocal_language or "en")
        lyrics = str(request.lyrics or "")

        kwargs: Dict[str, Any] = {
            "prompt": str(request.prompt or ""),
            "lyrics": lyrics,
            "audio_duration": float(duration_s),
            "vocal_language": vocal_language,
            "num_inference_steps": int(steps),
        }
        if self._config.guidance_scale is not None:
            kwargs["guidance_scale"] = float(self._config.guidance_scale)
        if self._config.shift is not None:
            kwargs["shift"] = float(self._config.shift)

        if isinstance(request.seed, int) and request.seed >= 0:
            try:
                kwargs["generator"] = torch.Generator(device=str(self._pipe_device or "cpu")).manual_seed(int(request.seed))
            except Exception:
                kwargs["generator"] = torch.Generator().manual_seed(int(request.seed))

        extra = request.extra or {}
        for key, value in extra.items():
            key_s = str(key or "").strip()
            if key_s and (not params or key_s in params) and key_s not in kwargs:
                kwargs[key_s] = value

        if params:
            kwargs = {k: v for k, v in kwargs.items() if k in params}

        fallback_events = []
        try:
            out = self._run_pipe(pipe, kwargs)
        except NotImplementedError as e:
            msg = str(e)
            is_mps = str(self._pipe_device or "").strip().lower() == "mps"
            is_known_mps_limit = "Output channels > 65536 not supported at the MPS device" in msg
            if is_mps and is_known_mps_limit and bool(self._config.auto_retry_cpu_on_mps_error):
                print(
                    "WARNING #FALLBACK : PyTorch MPS limitation in ACE-Step Diffusers decode. "
                    "Retrying on CPU with torch_dtype=float32.",
                    file=sys.stderr,
                )
                fallback_events.append("mps_decode_cpu_retry")
                self._pipe = None
                self._pipe_device = None
                self._pipe_dtype = None
                self._config = replace(self._config, device="cpu", torch_dtype="float32")
                pipe = self._load_pipe()
                kwargs = dict(kwargs)
                if "generator" in kwargs and isinstance(request.seed, int) and request.seed >= 0:
                    kwargs["generator"] = torch.Generator(device="cpu").manual_seed(int(request.seed))
                out = self._run_pipe(pipe, kwargs)
            else:
                raise

        sr = getattr(pipe, "sample_rate", None)
        sample_rate = int(sr) if isinstance(sr, int) and sr > 0 else 48000

        def _first_audio(output: Any) -> Any:
            audios = getattr(output, "audios", None)
            if audios is None and isinstance(output, dict):
                audios = output.get("audios") or output.get("audio")
            if audios is None:
                raise ValueError("AceStepPipeline did not return .audios")
            return audios[0] if isinstance(audios, (list, tuple)) else audios

        try:
            wav_bytes = _encode_wav_bytes(_first_audio(out), sample_rate=sample_rate)
        except ValueError as e:
            msg = str(e)
            is_mps = str(self._pipe_device or "").strip().lower() == "mps"
            if is_mps and "non-finite audio" in msg and bool(self._config.auto_retry_cpu_on_mps_error):
                print(
                    "WARNING #FALLBACK : ACE-Step Diffusers produced non-finite audio on MPS; "
                    "retrying on CPU float32.",
                    file=sys.stderr,
                )
                fallback_events.append("mps_nonfinite_audio_cpu_retry")
                self._pipe = None
                self._pipe_device = None
                self._pipe_dtype = None
                self._config = replace(self._config, device="cpu", torch_dtype="float32")
                pipe = self._load_pipe()
                kwargs = dict(kwargs)
                if "generator" in kwargs and isinstance(request.seed, int) and request.seed >= 0:
                    kwargs["generator"] = torch.Generator(device="cpu").manual_seed(int(request.seed))
                out = self._run_pipe(pipe, kwargs)
                sr = getattr(pipe, "sample_rate", None)
                sample_rate = int(sr) if isinstance(sr, int) and sr > 0 else 48000
                wav_bytes = _encode_wav_bytes(_first_audio(out), sample_rate=sample_rate)
            else:
                raise
        stats = inspect_wav_bytes(wav_bytes)

        metadata: Dict[str, Any] = {
            "backend": self.backend_id,
            "model_id": str(self._config.model_id),
            "sampling_rate": int(sample_rate),
            "duration_s": float(duration_s),
            "num_inference_steps": int(steps),
            "lyrics": bool(lyrics.strip()),
            "vocal_language": vocal_language,
            "device": str(self._pipe_device or ""),
            "dtype": str(self._pipe_dtype),
            "fallback_events": tuple(fallback_events),
            "audio_stats": {
                "channels": int(stats.channels),
                "frames": int(stats.frames),
                "duration_s": float(stats.duration_s),
                "peak": float(stats.peak),
                "rms": float(stats.rms),
                "dc_offset": float(stats.dc_offset),
                "clipped_ratio": float(stats.clipped_ratio),
                "zero_crossing_rate": float(stats.zero_crossing_rate),
                "probably_noise_or_invalid": bool(stats.is_probably_noise_or_invalid),
            },
        }
        if request.seed is not None:
            metadata["seed"] = int(request.seed)

        return GeneratedAsset(data=bytes(wav_bytes), mime_type="audio/wav", metadata=metadata)
