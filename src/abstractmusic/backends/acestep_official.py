"""
Official ACE-Step backend adapter.

This backend wraps the upstream ACE-Step runtime when it is installed or when
its source tree is provided via ``ABSTRACTMUSIC_ACESTEP_SOURCE_DIR``. The heavy
ACE-Step dependencies stay optional; AbstractMusic only owns the unified request
contract, checkpoint layout glue, and result normalization.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
import tempfile
import wave
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..audio_analysis import inspect_music_signal_bytes
from ..types import AudioGenerationRequest, GeneratedAsset, MusicBackendCapabilities, ProviderModelInfo


DEFAULT_ACESTEP_LM_MODEL_PATH = "acestep-5Hz-lm-1.7B"


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(str(key))
    if value is None:
        return default
    value = str(value).strip()
    return value or default


def _env_flag(key: str, default: bool = False) -> bool:
    value = _env(key)
    if value is None:
        return bool(default)
    return value.lower() in {"1", "on", "true", "yes"}


def _extra_str(extra: Dict[str, Any], key: str, default: str) -> str:
    value = extra.get(key, default)
    if value is None:
        return str(default)
    text = str(value).strip()
    return text or str(default)


def _extra_float(extra: Dict[str, Any], key: str, default: float) -> float:
    value = extra.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def _extra_int(extra: Dict[str, Any], key: str, default: int) -> int:
    value = extra.get(key, default)
    if value is None:
        return int(default)
    return int(value)


def _extra_optional_int(extra: Dict[str, Any], key: str, default: Optional[int] = None) -> Optional[int]:
    value = extra.get(key, default)
    if value is None or value == "":
        return default
    return int(value)


def _extra_bool(extra: Dict[str, Any], key: str, default: bool) -> bool:
    value = extra.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "on", "true", "yes"}
    return bool(value)


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _default_lm_backend() -> str:
    if _is_apple_silicon() and _has_module("mlx_lm"):
        return "mlx"
    try:
        import torch

        if torch.cuda.is_available():
            return "vllm"
    except Exception:
        pass
    return "pt"


def _copy_or_link_tree(src: Path, dst: Path, *, copy_suffixes: Sequence[str] = ()) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"Missing ACE-Step checkpoint component: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    copy_suffixes = tuple(copy_suffixes)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            _copy_or_link_tree(item, target, copy_suffixes=copy_suffixes)
            continue
        if target.exists() or target.is_symlink():
            continue
        if copy_suffixes and item.suffix in copy_suffixes:
            target.write_bytes(item.read_bytes())
        else:
            target.symlink_to(item)


def _copy_or_link_checkpoint_component(
    src: Path,
    dst: Path,
    *,
    marker: str = "model.safetensors",
    copy_suffixes: Sequence[str] = (),
) -> None:
    marker_src = src / marker
    marker_dst = dst / marker
    if marker_src.exists() and marker_dst.exists():
        try:
            if marker_src.stat().st_size != marker_dst.stat().st_size:
                shutil.rmtree(dst)
        except OSError:
            pass
    _copy_or_link_tree(src, dst, copy_suffixes=copy_suffixes)


def _latest_snapshot_dir(repo_cache_dir: Path) -> Optional[Path]:
    snapshots = repo_cache_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = [p for p in snapshots.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_local_hf_snapshot(repo_id: str) -> Optional[Path]:
    cache_home = Path(_env("HF_HOME", str(Path.home() / ".cache" / "huggingface")) or "").expanduser()
    repo_cache_name = "models--" + repo_id.replace("/", "--")
    return _latest_snapshot_dir(cache_home / "hub" / repo_cache_name)


def _snapshot_has_required_paths(snapshot: Path, required_paths: Optional[Sequence[str]]) -> bool:
    if not required_paths:
        return True
    return all((snapshot / item).exists() for item in required_paths)


def _snapshot_download(
    repo_id: str,
    *,
    allow_patterns: Optional[Sequence[str]] = None,
    required_paths: Optional[Sequence[str]] = None,
) -> Path:
    local = _find_local_hf_snapshot(repo_id)
    if local is not None and _snapshot_has_required_paths(local, required_paths):
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id, allow_patterns=list(allow_patterns) if allow_patterns else None))


def _tensor_to_wav_bytes(tensor: Any, sample_rate: int) -> bytes:
    if tensor is None:
        raise ValueError("ACE-Step generation returned neither a tensor nor a readable audio file.")
    arr = tensor
    if hasattr(arr, "detach"):
        arr = arr.detach()
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    data = np.asarray(arr, dtype=np.float32)
    if data.ndim == 2 and data.shape[0] <= 8:
        data = data.T
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError(f"Unsupported ACE-Step audio tensor shape: {data.shape!r}")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    pcm = np.clip(data, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype("<i2")
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(int(pcm_i16.shape[1]))
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm_i16.tobytes())
        return Path(tmp.name).read_bytes()


def _audio_result_to_wav_bytes(audio: Dict[str, Any]) -> tuple[bytes, str]:
    sample_rate = int(audio.get("sample_rate") or 48000)
    tensor = audio.get("tensor")
    if tensor is not None:
        return _tensor_to_wav_bytes(tensor, sample_rate), "tensor_pcm16"

    path = Path(str(audio.get("path") or ""))
    if path.exists():
        return path.read_bytes(), "file"

    raise RuntimeError("ACE-Step generation returned no usable audio payload.")


def _configure_upstream_runtime_output(*, verbose: bool) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    if not verbose:
        os.environ["ACESTEP_DISABLE_TQDM"] = "1"
        os.environ.setdefault("TQDM_DISABLE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    else:
        os.environ["ACESTEP_DISABLE_TQDM"] = "0"
        os.environ.pop("TQDM_DISABLE", None)
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)

    try:
        from loguru import logger

        logger.remove()
        if verbose:
            logger.add(sys.stderr, level="INFO")
    except Exception:
        pass


@contextmanager
def _suppress_upstream_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


def _is_meta_init_context(ctx: Any) -> bool:
    try:
        import torch

        if isinstance(ctx, torch.device) and ctx.type == "meta":
            return True
    except Exception:
        pass
    try:
        func = getattr(ctx, "func", None)
        return func is not None and getattr(func, "__name__", "") == "init_empty_weights"
    except Exception:
        return False


@contextmanager
def _without_transformers_meta_init():
    """Temporarily remove Transformers meta init contexts for ACE-Step model load."""

    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        yield
        return

    original_descriptor = PreTrainedModel.__dict__.get("get_init_context")
    if isinstance(original_descriptor, classmethod):
        original_func = original_descriptor.__func__
    else:
        original_func = original_descriptor
    if not callable(original_func):
        yield
        return

    def filtered_get_init_context(cls, *args, **kwargs):
        ctxs = list(original_func(cls, *args, **kwargs))
        return [ctx for ctx in ctxs if not _is_meta_init_context(ctx)]

    PreTrainedModel.get_init_context = classmethod(filtered_get_init_context)  # type: ignore[method-assign]
    try:
        yield
    finally:
        if original_descriptor is not None:
            PreTrainedModel.get_init_context = original_descriptor  # type: ignore[method-assign]


@dataclass
class AceStepOfficialBackendConfig:
    repo_id: str = "ACE-Step/Ace-Step1.5"
    source_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    dit_model: str = "acestep-v15-turbo"
    lm_model_path: str = DEFAULT_ACESTEP_LM_MODEL_PATH
    lm_backend: str = "auto"
    device: str = "auto"
    default_duration_s: float = 10.0
    num_inference_steps: int = 8
    guidance_scale: float = 1.0
    shift: float = 3.0
    infer_method: str = "ode"
    sampler_mode: str = "euler"
    lm_temperature: float = 0.85
    lm_cfg_scale: float = 2.0
    lm_top_k: int = 0
    lm_top_p: float = 0.9
    use_constrained_decoding: bool = True
    audio_cover_strength: float = 1.0
    cover_noise_strength: float = 0.0
    offload_to_cpu: bool = False
    offload_dit_to_cpu: bool = False
    use_mlx_dit: Optional[bool] = None
    audio_format: str = "wav"
    verbose: bool = False


class AceStepOfficialBackend:
    """Adapter for the upstream ACE-Step handler and 5Hz LM pipeline."""

    backend_id = "abstractmusic:acestep-official"

    def __init__(self, *, config: Optional[AceStepOfficialBackendConfig] = None) -> None:
        self._config = config or AceStepOfficialBackendConfig()
        self._dit_handler = None
        self._llm_handler = None
        self._checkpoint_dir: Optional[Path] = None
        self._lm_backend: Optional[str] = None
        self._device: Optional[str] = None
        self._init_status: Dict[str, str] = {}

    def get_capabilities(self) -> MusicBackendCapabilities:
        is_turbo = "turbo" in str(self._config.dit_model or "").lower()
        return MusicBackendCapabilities(
            supported_tasks=("text_to_music",),
            output_formats=("wav",),
            supports_lyrics=True,
            supports_negative_prompt=True,
            supports_guidance_scale=not is_turbo,
            supports_reference_audio=False,
            max_duration_s=600.0,
            sample_rates_hz=(48000,),
            model_id=self._config.repo_id,
            license="MIT",
            commercial_allowed=True,
            official_8bit_available=False,
            preferred_precision="official model/default precision; MLX on Apple Silicon, vLLM/PyTorch on CUDA, CPU only as fallback",
        )

    def list_provider_models(self, *, task: Optional[str] = None) -> Sequence[ProviderModelInfo]:
        if task is not None and task != "text_to_music":
            return ()
        return (
            ProviderModelInfo(
                id=self._config.repo_id,
                owned_by="ACE-Step",
                capabilities=("text_to_music", "lyrics", "official_upstream_runtime"),
            ),
        )

    def _resolve_source_dir(self) -> Optional[Path]:
        configured = self._config.source_dir or _env("ABSTRACTMUSIC_ACESTEP_SOURCE_DIR")
        candidates = []
        if configured:
            candidates.append(Path(configured).expanduser())
        candidates.extend(
            [
                Path.cwd() / "ACE-Step-1.5-main",
                Path.cwd().parent / "ACE-Step-1.5-main",
                Path("/tmp/ACE-Step-1.5-main"),
                Path("/tmp/ace-step-raw"),
            ]
        )
        for candidate in candidates:
            if (candidate / "acestep" / "handler.py").exists():
                source = candidate.resolve()
                if str(source) not in sys.path:
                    sys.path.insert(0, str(source))
                return source
        if _has_module("acestep"):
            return None
        raise ImportError(
            "Official ACE-Step runtime not found. Install ACE-Step or set "
            "ABSTRACTMUSIC_ACESTEP_SOURCE_DIR to the upstream source tree."
        )

    def _ensure_checkpoint_dir(self) -> Path:
        configured = self._config.checkpoint_dir or _env("ABSTRACTMUSIC_ACESTEP_CHECKPOINT_DIR") or _env("ACESTEP_CHECKPOINTS_DIR")
        if configured:
            return Path(configured).expanduser().resolve()

        dst = Path.home() / ".cache" / "abstractmusic" / "acestep-official-checkpoints"
        main_snapshot = _snapshot_download(
            self._config.repo_id,
            allow_patterns=(
                f"{self._config.dit_model}/*",
                "vae/*",
                "Qwen3-Embedding-0.6B/*",
            ),
            required_paths=(
                f"{self._config.dit_model}/model.safetensors",
                "vae/diffusion_pytorch_model.safetensors",
                "Qwen3-Embedding-0.6B/model.safetensors",
            ),
        )
        _copy_or_link_checkpoint_component(
            main_snapshot / self._config.dit_model,
            dst / self._config.dit_model,
            copy_suffixes=(".py",),
        )
        _copy_or_link_checkpoint_component(
            main_snapshot / "vae",
            dst / "vae",
            marker="diffusion_pytorch_model.safetensors",
        )
        _copy_or_link_checkpoint_component(main_snapshot / "Qwen3-Embedding-0.6B", dst / "Qwen3-Embedding-0.6B")

        lm_name = str(self._config.lm_model_path or "").strip() or DEFAULT_ACESTEP_LM_MODEL_PATH
        if lm_name == "acestep-5Hz-lm-0.6B":
            lm_snapshot = _snapshot_download(
                "ACE-Step/acestep-5Hz-lm-0.6B",
                required_paths=("model.safetensors",),
            )
            _copy_or_link_checkpoint_component(lm_snapshot, dst / lm_name)
        else:
            lm_snapshot = _snapshot_download(
                self._config.repo_id,
                allow_patterns=(f"{lm_name}/*",),
                required_paths=(f"{lm_name}/model.safetensors",),
            )
            _copy_or_link_checkpoint_component(lm_snapshot / lm_name, dst / lm_name)

        return dst.resolve()

    def preload(self) -> None:
        if self._dit_handler is not None and self._llm_handler is not None:
            return

        verbose = bool(self._config.verbose or _env_flag("ABSTRACTMUSIC_VERBOSE") or _env_flag("ABSTRACTMUSIC_ACESTEP_VERBOSE"))
        quiet = not verbose
        _configure_upstream_runtime_output(verbose=verbose)
        source_dir = self._resolve_source_dir()
        checkpoint_dir = self._ensure_checkpoint_dir()
        os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(checkpoint_dir)

        with _suppress_upstream_output(quiet):
            from acestep.handler import AceStepHandler
            from acestep.llm_inference import LLMHandler

        dit = AceStepHandler()
        if hasattr(dit, "disable_tqdm"):
            setattr(dit, "disable_tqdm", quiet)
        use_mlx_dit = self._config.use_mlx_dit
        if use_mlx_dit is None:
            use_mlx_dit = _is_apple_silicon()
        project_root = str(source_dir) if source_dir is not None else ""
        with _suppress_upstream_output(quiet), _without_transformers_meta_init():
            dit_status, dit_ok = dit.initialize_service(
                project_root=project_root,
                config_path=self._config.dit_model,
                device=self._config.device,
                offload_to_cpu=bool(self._config.offload_to_cpu),
                offload_dit_to_cpu=bool(self._config.offload_dit_to_cpu),
                use_mlx_dit=bool(use_mlx_dit),
            )
        if not dit_ok:
            raise RuntimeError(f"Failed to initialize official ACE-Step DiT: {dit_status}")

        lm_backend = self._config.lm_backend
        if not lm_backend or lm_backend == "auto":
            lm_backend = _default_lm_backend()
        os.environ["ACESTEP_LM_BACKEND"] = lm_backend
        llm = LLMHandler()
        if hasattr(llm, "disable_tqdm"):
            setattr(llm, "disable_tqdm", quiet)
        with _suppress_upstream_output(quiet):
            llm_status, llm_ok = llm.initialize(
                checkpoint_dir=str(checkpoint_dir),
                lm_model_path=self._config.lm_model_path,
                backend=lm_backend,
                device=self._config.device,
                offload_to_cpu=bool(self._config.offload_to_cpu),
                dtype=None,
            )
        if hasattr(llm, "disable_tqdm"):
            setattr(llm, "disable_tqdm", quiet)
        if not llm_ok:
            raise RuntimeError(f"Failed to initialize official ACE-Step 5Hz LM: {llm_status}")

        self._dit_handler = dit
        self._llm_handler = llm
        self._checkpoint_dir = checkpoint_dir
        self._lm_backend = str(getattr(llm, "llm_backend", None) or lm_backend)
        self._device = str(getattr(dit, "device", None) or getattr(llm, "device", None) or self._config.device)
        self._init_status = {"dit": str(dit_status), "lm": str(llm_status)}

    def unload(self) -> None:
        self._dit_handler = None
        self._llm_handler = None

    def generate_audio(self, request: AudioGenerationRequest) -> GeneratedAsset:
        self.preload()
        assert self._dit_handler is not None
        assert self._llm_handler is not None

        verbose = bool(self._config.verbose or _env_flag("ABSTRACTMUSIC_VERBOSE") or _env_flag("ABSTRACTMUSIC_ACESTEP_VERBOSE"))
        with _suppress_upstream_output(not verbose):
            from acestep.inference import GenerationConfig, GenerationParams, generate_music

        duration = float(request.duration_s or self._config.default_duration_s)
        steps = int(request.num_inference_steps or self._config.num_inference_steps)
        seed = int(request.seed) if request.seed is not None else -1
        extra = request.extra or {}
        lyrics = "" if request.lyrics is None else str(request.lyrics)
        instrumental = _extra_bool(extra, "instrumental", False) or str(lyrics).strip().lower() == "[instrumental]"
        bpm = _extra_optional_int(extra, "bpm")
        keyscale = _extra_str(extra, "keyscale", "")
        timesignature = _extra_str(extra, "timesignature", "")
        guidance_scale = _extra_float(extra, "guidance_scale", request.guidance_scale if request.guidance_scale is not None else self._config.guidance_scale)
        shift = _extra_float(extra, "shift", self._config.shift)
        infer_method = _extra_str(extra, "infer_method", self._config.infer_method)
        sampler_mode = _extra_str(extra, "sampler_mode", self._config.sampler_mode)
        lm_temperature = _extra_float(extra, "lm_temperature", self._config.lm_temperature)
        lm_cfg_scale = _extra_float(extra, "lm_cfg_scale", self._config.lm_cfg_scale)
        lm_top_k = _extra_int(extra, "lm_top_k", self._config.lm_top_k)
        lm_top_p = _extra_float(extra, "lm_top_p", self._config.lm_top_p)
        audio_cover_strength = _extra_float(extra, "audio_cover_strength", self._config.audio_cover_strength)
        cover_noise_strength = _extra_float(extra, "cover_noise_strength", self._config.cover_noise_strength)
        use_constrained_decoding = _extra_bool(
            extra,
            "use_constrained_decoding",
            bool(self._config.use_constrained_decoding),
        )

        params = GenerationParams(
            task_type="text2music",
            thinking=True,
            caption=str(request.prompt or ""),
            lyrics=str(lyrics or ""),
            instrumental=instrumental,
            vocal_language=str(request.vocal_language or "unknown"),
            bpm=bpm,
            keyscale=keyscale,
            timesignature=timesignature,
            duration=duration,
            inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            shift=shift,
            infer_method=infer_method,
            sampler_mode=sampler_mode,
            lm_temperature=lm_temperature,
            lm_cfg_scale=lm_cfg_scale,
            lm_top_k=lm_top_k,
            lm_top_p=lm_top_p,
            lm_negative_prompt=str(request.negative_prompt or "NO USER INPUT"),
            audio_cover_strength=audio_cover_strength,
            cover_noise_strength=cover_noise_strength,
            use_cot_metas=True,
            use_cot_caption=True,
            use_cot_language=True,
            use_constrained_decoding=use_constrained_decoding,
        )
        config = GenerationConfig(
            batch_size=1,
            use_random_seed=request.seed is None,
            seeds=[seed] if request.seed is not None else None,
            audio_format=self._config.audio_format or "wav",
        )

        with tempfile.TemporaryDirectory(prefix="abstractmusic-acestep-official-") as tmp:
            with _suppress_upstream_output(not verbose):
                result = generate_music(
                    self._dit_handler,
                    self._llm_handler,
                    params=params,
                    config=config,
                    save_dir=tmp,
                )
            if not getattr(result, "success", False):
                raise RuntimeError(getattr(result, "status_message", None) or getattr(result, "error", None) or "ACE-Step generation failed")
            audios = list(getattr(result, "audios", []) or [])
            if not audios:
                raise RuntimeError("ACE-Step generation returned no audio.")
            audio = audios[0]
            data, audio_source = _audio_result_to_wav_bytes(audio)

        stats = inspect_music_signal_bytes(data)
        metadata: Dict[str, Any] = {
            "backend": self.backend_id,
            "model_id": self._config.repo_id,
            "dit_model": self._config.dit_model,
            "lm_model_path": self._config.lm_model_path,
            "lm_backend": self._lm_backend,
            "device": self._device or self._config.device,
            "duration_s": duration,
            "num_inference_steps": steps,
            "bpm": bpm,
            "keyscale": keyscale,
            "timesignature": timesignature,
            "instrumental": instrumental,
            "guidance_scale": guidance_scale,
            "shift": shift,
            "infer_method": infer_method,
            "sampler_mode": sampler_mode,
            "lm_temperature": lm_temperature,
            "lm_cfg_scale": lm_cfg_scale,
            "lm_top_k": lm_top_k,
            "lm_top_p": lm_top_p,
            "audio_cover_strength": audio_cover_strength,
            "cover_noise_strength": cover_noise_strength,
            "use_constrained_decoding": use_constrained_decoding,
            "audio_source": audio_source,
            "output_encoding": "pcm_s16le" if audio_source == "tensor_pcm16" else "upstream_file",
            "sample_rate": stats.wav.sample_rate_hz,
            "audio_stats": asdict(stats),
        }
        if request.seed is not None:
            metadata["seed"] = int(request.seed)
        extra = getattr(result, "extra_outputs", None)
        if isinstance(extra, dict):
            metadata["extra_outputs"] = {
                key: value
                for key, value in extra.items()
                if key in {"lm_metadata", "time_costs"} and value is not None
            }
        if self._init_status:
            metadata["init_status"] = dict(self._init_status)
        return GeneratedAsset(data=data, mime_type="audio/wav", metadata=metadata)
