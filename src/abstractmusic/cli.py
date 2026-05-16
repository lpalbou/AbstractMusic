"""
AbstractMusic CLI.

Goals:
- Provide a minimal, practical interface similar to AbstractVision's CLI.
- Support a simple REPL loop for iterative prompt→WAV generation.
- Keep module import light; heavy ML imports happen only when generating.
"""

from __future__ import annotations

import argparse
import cmd
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Optional

if os.environ.get("DIFFUSERS_SLOW_IMPORT", "").strip().upper() in {"1", "ON", "YES", "TRUE"}:
    os.environ["DIFFUSERS_SLOW_IMPORT"] = "0"


SUPPORTED_BACKENDS = ("acestep-official", "acestep", "acestep-diffusers", "diffusers", "musicgen", "stable-audio")
DEFAULT_OFFICIAL_LM_MODEL_PATH = "acestep-5Hz-lm-1.7B"
BACKEND_ALIASES = {
    "official": "acestep-official",
    "upstream": "acestep-official",
    "mlx": "acestep-official",
    "acestep-mlx": "acestep-official",
    "acestep_official": "acestep-official",
    "ace": "acestep",
    "acestep-v15": "acestep",
    "acestep_v15": "acestep",
    "v15": "acestep",
    "xl": "acestep-diffusers",
    "ace-xl": "acestep-diffusers",
    "acestep-xl": "acestep-diffusers",
    "acestep_diffusers": "acestep-diffusers",
    "diffusers-xl": "acestep-diffusers",
    "musicgen-small": "musicgen",
    "facebook-musicgen-small": "musicgen",
    "stable": "stable-audio",
    "stableaudio": "stable-audio",
    "stable-audio-open-small": "stable-audio",
    "hf": "diffusers",
    "generic": "diffusers",
}


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(str(key), None)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _env_flag(key: str, default: bool = False) -> bool:
    value = _env(key)
    if value is None:
        return bool(default)
    return value.lower() in {"1", "on", "true", "yes"}


def _timestamp_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _write_bytes(path: Path, content: bytes) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(bytes(content))


def _normalize_backend(value: str) -> str:
    raw = str(value or "").strip().lower()
    backend = BACKEND_ALIASES.get(raw, raw)
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unknown engine {value!r}; expected one of: {', '.join(SUPPORTED_BACKENDS)}")
    return backend


def _arg_float(args: argparse.Namespace, name: str, default: float) -> float:
    value = getattr(args, name, default)
    return float(default) if value is None else float(value)


def _arg_int(args: argparse.Namespace, name: str, default: int) -> int:
    value = getattr(args, name, default)
    return int(default) if value is None else int(value)


def _arg_str(args: argparse.Namespace, name: str, default: str) -> str:
    value = getattr(args, name, default)
    if value is None:
        return str(default)
    text = str(value).strip()
    return text or str(default)


def _parse_backend_arg(value: str) -> str:
    try:
        return _normalize_backend(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def _copy_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(args))


def _configure_mps_env(args: argparse.Namespace) -> None:
    backend_kind = str(getattr(args, "backend", "acestep") or "acestep").strip().lower()
    if backend_kind != "acestep":
        return
    device = str(getattr(args, "device", "auto") or "auto").strip().lower()
    if device not in {"auto", "mps"}:
        return
    try:
        from .backends.acestep_v15 import _maybe_configure_mps_high_watermark
    except Exception:
        return
    mps_max = getattr(args, "mps_max_memory_gb", None)
    mps_ratio = getattr(args, "mps_high_watermark_ratio", None)
    _maybe_configure_mps_high_watermark(
        mps_max_memory_gb=mps_max,
        mps_high_watermark_ratio=mps_ratio,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="abstractmusic", description="AbstractMusic — local text-to-audio/music generation")

    def _add_common_args(parser: argparse.ArgumentParser, *, use_defaults: bool) -> None:
        """
        Add common args to a parser.

        Why:
        - `argparse` subparsers do not accept top-level options placed AFTER the subcommand.
        - We want both of these to work:
          - abstractmusic --backend acestep t2m "..." --duration 10
          - abstractmusic t2m "..." --backend acestep --duration 10
        """

        default_suppress = argparse.SUPPRESS if not use_defaults else None
        mps_max_env = _env("ABSTRACTMUSIC_MPS_MAX_MEMORY_GB")
        mps_ratio_env = _env("ABSTRACTMUSIC_MPS_HIGH_WATERMARK_RATIO")
        mps_max_default = float(mps_max_env) if mps_max_env is not None else 16.0
        mps_ratio_default = float(mps_ratio_env) if mps_ratio_env is not None else None
        verbose_default = _env_flag("ABSTRACTMUSIC_VERBOSE") or _env_flag("ABSTRACTMUSIC_ACESTEP_VERBOSE")

        parser.add_argument(
            "--backend",
            "--engine",
            dest="backend",
            type=_parse_backend_arg,
            default=_normalize_backend(_env("ABSTRACTMUSIC_BACKEND", "acestep-official") or "acestep-official")
            if use_defaults
            else default_suppress,
            help="Local generation engine/backend (acestep-official|acestep|acestep-diffusers|diffusers|musicgen|stable-audio; aliases: official, xl, ace, hf). Default: acestep-official.",
        )
        parser.add_argument(
            "--model-id",
            default=_env("ABSTRACTMUSIC_MODEL_ID") if use_defaults else default_suppress,
            help="Model repo id (or set $ABSTRACTMUSIC_MODEL_ID). "
            "Diffusers: any Diffusers audio checkpoint id (license varies). "
            "ACE-Step: ACE-Step/Ace-Step1.5. ACE-Step Diffusers: ACE-Step/acestep-v15-xl-turbo-diffusers. "
            "MusicGen: facebook/musicgen-small. Stable Audio: stabilityai/stable-audio-open-small.",
        )
        parser.add_argument(
            "--revision",
            default=_env("ABSTRACTMUSIC_REVISION") if use_defaults else default_suppress,
            help="Optional Hugging Face revision (commit/tag).",
        )
        parser.add_argument(
            "--cache-dir",
            default=_env("ABSTRACTMUSIC_CACHE_DIR") if use_defaults else default_suppress,
            help="Optional Hugging Face cache directory.",
        )
        parser.add_argument(
            "--device",
            default=_env("ABSTRACTMUSIC_DEVICE", "auto") if use_defaults else default_suppress,
            help="auto|cuda|mps|cpu",
        )
        parser.add_argument(
            "--mps-max-memory-gb",
            type=float,
            default=mps_max_default if use_defaults else default_suppress,
            help="Max MPS memory budget in GiB (uses PYTORCH_MPS_HIGH_WATERMARK_RATIO). "
            "Use 0 to disable the cap.",
        )
        parser.add_argument(
            "--mps-high-watermark-ratio",
            type=float,
            default=mps_ratio_default if use_defaults else default_suppress,
            help="Explicit MPS high-watermark ratio override (0-1). Overrides --mps-max-memory-gb.",
        )
        parser.add_argument(
            "--dtype",
            default=_env("ABSTRACTMUSIC_TORCH_DTYPE", "auto") if use_defaults else default_suppress,
            help="auto|float16|bfloat16|float32",
        )
        parser.add_argument(
            "--pipeline-class",
            default=_env("ABSTRACTMUSIC_PIPELINE_CLASS") if use_defaults else default_suppress,
            help="Optional diffusers pipeline class name (e.g. AudioLDMPipeline). Usually not needed.",
        )
        parser.add_argument(
            "--lm-model-path",
            default=_env("ABSTRACTMUSIC_ACESTEP_LM_MODEL_PATH", DEFAULT_OFFICIAL_LM_MODEL_PATH) if use_defaults else default_suppress,
            help=f"Official ACE-Step 5Hz LM checkpoint name. Default: {DEFAULT_OFFICIAL_LM_MODEL_PATH}.",
        )
        parser.add_argument(
            "--lm-backend",
            default=_env("ABSTRACTMUSIC_ACESTEP_LM_BACKEND", "auto") if use_defaults else default_suppress,
            help="Official ACE-Step LM runtime: auto|mlx|vllm|pt.",
        )
        parser.add_argument(
            "--acestep-source-dir",
            default=_env("ABSTRACTMUSIC_ACESTEP_SOURCE_DIR") if use_defaults else default_suppress,
            help="Optional upstream ACE-Step source tree for the official backend.",
        )
        parser.add_argument(
            "--acestep-checkpoint-dir",
            default=_env("ABSTRACTMUSIC_ACESTEP_CHECKPOINT_DIR") if use_defaults else default_suppress,
            help="Optional upstream ACE-Step checkpoint directory.",
        )
        parser.add_argument(
            "--steps",
            type=int,
            default=None if use_defaults else default_suppress,
            help="Inference steps. Diffusers: num_inference_steps (default 50). ACE-Step turbo: fixed schedule (default 8).",
        )
        parser.add_argument(
            "--duration",
            type=float,
            default=float(_env("ABSTRACTMUSIC_DURATION_S", "10") or "10") if use_defaults else default_suppress,
            help="seconds",
        )
        bpm_env = _env("ABSTRACTMUSIC_BPM")
        parser.add_argument(
            "--bpm",
            type=int,
            default=int(bpm_env) if use_defaults and bpm_env is not None else default_suppress,
            help="Optional target BPM for engines that support music metadata.",
        )
        parser.add_argument(
            "--keyscale",
            default=_env("ABSTRACTMUSIC_KEYSCALE") if use_defaults else default_suppress,
            help="Optional musical key/scale, e.g. C major or A minor.",
        )
        parser.add_argument(
            "--timesignature",
            default=_env("ABSTRACTMUSIC_TIMESIGNATURE") if use_defaults else default_suppress,
            help="Optional time signature metadata, e.g. 4.",
        )
        parser.add_argument(
            "--guidance-scale",
            type=float,
            default=None if use_defaults else default_suppress,
        )
        parser.add_argument(
            "--shift",
            type=float,
            default=float(_env("ABSTRACTMUSIC_ACESTEP_SHIFT", "3.0") or "3.0") if use_defaults else default_suppress,
            help="Official ACE-Step timestep shift. Default: 3.0 for turbo checkpoints.",
        )
        parser.add_argument(
            "--infer-method",
            choices=("ode", "sde"),
            default=_env("ABSTRACTMUSIC_ACESTEP_INFER_METHOD", "ode") if use_defaults else default_suppress,
            help="Official ACE-Step diffusion inference method. Default: ode.",
        )
        parser.add_argument(
            "--sampler-mode",
            choices=("euler", "heun"),
            default=_env("ABSTRACTMUSIC_ACESTEP_SAMPLER_MODE", "euler") if use_defaults else default_suppress,
            help="Official ACE-Step sampler mode. Default: euler.",
        )
        parser.add_argument(
            "--lm-temperature",
            type=float,
            default=float(_env("ABSTRACTMUSIC_ACESTEP_LM_TEMPERATURE", "0.85") or "0.85") if use_defaults else default_suppress,
            help="Official ACE-Step 5Hz LM sampling temperature. Default: 0.85.",
        )
        parser.add_argument(
            "--lm-cfg-scale",
            type=float,
            default=float(_env("ABSTRACTMUSIC_ACESTEP_LM_CFG_SCALE", "2.0") or "2.0") if use_defaults else default_suppress,
            help="Official ACE-Step 5Hz LM CFG scale. Default: 2.0.",
        )
        parser.add_argument(
            "--lm-top-k",
            type=int,
            default=int(_env("ABSTRACTMUSIC_ACESTEP_LM_TOP_K", "0") or "0") if use_defaults else default_suppress,
            help="Official ACE-Step 5Hz LM top-k sampling. Default: 0 disables top-k.",
        )
        parser.add_argument(
            "--lm-top-p",
            type=float,
            default=float(_env("ABSTRACTMUSIC_ACESTEP_LM_TOP_P", "0.9") or "0.9") if use_defaults else default_suppress,
            help="Official ACE-Step 5Hz LM top-p sampling. Default: 0.9.",
        )
        parser.add_argument(
            "--audio-cover-strength",
            type=float,
            default=float(_env("ABSTRACTMUSIC_ACESTEP_AUDIO_COVER_STRENGTH", "1.0") or "1.0")
            if use_defaults
            else default_suppress,
            help="Official ACE-Step strength for LM audio-code/reference conditioning. Default: 1.0.",
        )
        parser.add_argument(
            "--cover-noise-strength",
            type=float,
            default=float(_env("ABSTRACTMUSIC_ACESTEP_COVER_NOISE_STRENGTH", "0.0") or "0.0")
            if use_defaults
            else default_suppress,
            help="Official ACE-Step source/code noise mixing strength. Default: 0.0.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None if use_defaults else default_suppress,
        )
        parser.add_argument(
            "--negative",
            default=None if use_defaults else default_suppress,
            help="Negative prompt (backend-supported)",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            default=verbose_default if use_defaults else default_suppress,
            help="Show upstream backend logs/progress. Default keeps the REPL output quiet.",
        )

    _add_common_args(p, use_defaults=True)

    sub = p.add_subparsers(dest="cmd", required=True)

    t2m = sub.add_parser("t2m", help="Generate a WAV file from a single prompt")
    _add_common_args(t2m, use_defaults=False)
    t2m.add_argument("prompt", help="Text prompt")
    t2m.add_argument("--lyrics", default=None, help="Optional lyrics (ACE-Step supports lyrics natively).")
    t2m.add_argument("--out", default="out.wav", help="Output WAV path")

    repl = sub.add_parser("repl", help="Interactive modular prompt→WAV loop")
    _add_common_args(repl, use_defaults=False)
    repl.add_argument("--out-dir", default=".", help="Directory to write WAV files into")
    repl.add_argument("--prefix", default="music", help="Filename prefix for outputs")
    repl.add_argument("--prompt", default=None, help="Initial prompt for /run.")
    repl.add_argument("--lyrics", default=None, help="Optional lyrics applied to every prompt in this REPL session.")
    repl.add_argument("--open", action="store_true", help="Open output file after each generation (best-effort)")

    return p


def _open_file(path: Path) -> None:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return
    if sys.platform == "darwin":
        os.system(f"open {str(p)!r}")  # nosec - best-effort UX
        return
    if sys.platform.startswith("win"):
        os.system(f'start "" {str(p)!r}')  # nosec - best-effort UX
        return
    os.system(f"xdg-open {str(p)!r}")  # nosec - best-effort UX


def _make_manager_from_args(args: argparse.Namespace):
    # Import lazily to avoid pulling torch/diffusers during `abstractmusic --help`.
    from .music_manager import MusicManager

    backend_kind = str(getattr(args, "backend", "diffusers") or "diffusers").strip().lower()
    model_id = str(getattr(args, "model_id", "") or "").strip()

    if backend_kind == "acestep-official":
        from .backends.acestep_official import AceStepOfficialBackend, AceStepOfficialBackendConfig

        if not model_id:
            model_id = "ACE-Step/Ace-Step1.5"
        cfg = AceStepOfficialBackendConfig(
            repo_id=model_id,
            source_dir=str(getattr(args, "acestep_source_dir", "") or "").strip() or None,
            checkpoint_dir=str(getattr(args, "acestep_checkpoint_dir", "") or "").strip() or None,
            lm_model_path=str(getattr(args, "lm_model_path", DEFAULT_OFFICIAL_LM_MODEL_PATH) or DEFAULT_OFFICIAL_LM_MODEL_PATH),
            lm_backend=str(getattr(args, "lm_backend", "auto") or "auto"),
            device=str(getattr(args, "device", "auto") or "auto"),
            default_duration_s=float(getattr(args, "duration", 10.0)),
            num_inference_steps=int(getattr(args, "steps", None) or 8),
            guidance_scale=float(getattr(args, "guidance_scale")) if getattr(args, "guidance_scale", None) is not None else 1.0,
            shift=_arg_float(args, "shift", 3.0),
            infer_method=_arg_str(args, "infer_method", "ode"),
            sampler_mode=_arg_str(args, "sampler_mode", "euler"),
            lm_temperature=_arg_float(args, "lm_temperature", 0.85),
            lm_cfg_scale=_arg_float(args, "lm_cfg_scale", 2.0),
            lm_top_k=_arg_int(args, "lm_top_k", 0),
            lm_top_p=_arg_float(args, "lm_top_p", 0.9),
            audio_cover_strength=_arg_float(args, "audio_cover_strength", 1.0),
            cover_noise_strength=_arg_float(args, "cover_noise_strength", 0.0),
            verbose=bool(getattr(args, "verbose", False)),
        )
        backend = AceStepOfficialBackend(config=cfg)
        return MusicManager(backend=backend)

    if backend_kind == "acestep-diffusers":
        from .backends.acestep_diffusers import AceStepDiffusersBackend, AceStepDiffusersBackendConfig

        if not model_id:
            model_id = "ACE-Step/acestep-v15-xl-turbo-diffusers"
        cfg = AceStepDiffusersBackendConfig(
            model_id=model_id,
            device=str(getattr(args, "device", "auto") or "auto"),
            torch_dtype=str(getattr(args, "dtype", "auto") or "auto"),
            num_inference_steps=int(getattr(args, "steps", None) or 8),
            duration_s=float(getattr(args, "duration", 10.0)),
            guidance_scale=float(getattr(args, "guidance_scale")) if getattr(args, "guidance_scale", None) is not None else None,
        )
        backend = AceStepDiffusersBackend(config=cfg)
        return MusicManager(backend=backend)

    if backend_kind == "diffusers":
        from .backends.diffusers_audio import DiffusersAudioBackend, DiffusersAudioBackendConfig

        if not model_id:
            raise SystemExit("Missing --model-id (or $ABSTRACTMUSIC_MODEL_ID) for backend=diffusers.")
        cfg = DiffusersAudioBackendConfig(
            model_id=model_id,
            device=str(getattr(args, "device", "auto") or "auto"),
            torch_dtype=str(getattr(args, "dtype", "auto") or "auto"),
            pipeline_class=str(getattr(args, "pipeline_class") or "").strip() or None,
            num_inference_steps=int(getattr(args, "steps", None) or 50),
            duration_s=float(getattr(args, "duration", 10.0)),
            guidance_scale=float(getattr(args, "guidance_scale")) if getattr(args, "guidance_scale", None) is not None else None,
        )
        backend = DiffusersAudioBackend(config=cfg)
        return MusicManager(backend=backend)

    if backend_kind == "musicgen":
        from .backends.musicgen import MusicGenBackend, MusicGenBackendConfig

        if not model_id:
            model_id = "facebook/musicgen-small"
        cfg = MusicGenBackendConfig(
            model_id=model_id,
            device=str(getattr(args, "device", "auto") or "auto"),
            torch_dtype=str(getattr(args, "dtype", "auto") or "auto"),
            duration_s=float(getattr(args, "duration", 10.0)),
            guidance_scale=float(getattr(args, "guidance_scale")) if getattr(args, "guidance_scale", None) is not None else 3.0,
        )
        backend = MusicGenBackend(config=cfg)
        return MusicManager(backend=backend)

    if backend_kind == "stable-audio":
        from .backends.stable_audio import StableAudioBackend, StableAudioBackendConfig

        if not model_id:
            model_id = "stabilityai/stable-audio-open-small"
        cfg = StableAudioBackendConfig(
            model_id=model_id,
            device=str(getattr(args, "device", "auto") or "auto"),
            duration_s=float(getattr(args, "duration", 10.0)),
            num_inference_steps=int(getattr(args, "steps", None) or 8),
            guidance_scale=float(getattr(args, "guidance_scale")) if getattr(args, "guidance_scale", None) is not None else 1.0,
        )
        backend = StableAudioBackend(config=cfg)
        return MusicManager(backend=backend)

    if backend_kind == "acestep":
        from .backends import AceStepV15Backend, AceStepV15BackendConfig

        if not model_id:
            model_id = "ACE-Step/Ace-Step1.5"
        cfg_kwargs = {
            "repo_id": model_id,
            "device": str(getattr(args, "device", "auto") or "auto"),
            "torch_dtype": str(getattr(args, "dtype", "auto") or "auto"),
            "vae_torch_dtype": str(getattr(args, "dtype", "auto") or "auto"),
            "default_duration_s": float(getattr(args, "duration", 10.0)),
            "fix_nfe": int(getattr(args, "steps", None) or 8),
        }
        mps_max = getattr(args, "mps_max_memory_gb", None)
        if mps_max is not None:
            try:
                mps_max = float(mps_max)
            except Exception:
                mps_max = None
            if mps_max is not None and mps_max <= 0:
                mps_max = None
            cfg_kwargs["mps_max_memory_gb"] = mps_max
        mps_ratio = getattr(args, "mps_high_watermark_ratio", None)
        if mps_ratio is not None:
            try:
                mps_ratio = float(mps_ratio)
            except Exception:
                mps_ratio = None
            if mps_ratio is not None and mps_ratio <= 0:
                mps_ratio = None
            cfg_kwargs["mps_high_watermark_ratio"] = mps_ratio
        rev = getattr(args, "revision", None)
        rev = str(rev).strip() if isinstance(rev, str) and rev.strip() else None
        if rev is not None:
            cfg_kwargs["revision"] = rev
        cache_dir = getattr(args, "cache_dir", None)
        cache_dir = str(cache_dir).strip() if isinstance(cache_dir, str) and cache_dir.strip() else None
        if cache_dir is not None:
            cfg_kwargs["cache_dir"] = cache_dir
        cfg = AceStepV15BackendConfig(**cfg_kwargs)
        backend = AceStepV15Backend(config=cfg)
        return MusicManager(backend=backend)

    raise SystemExit(f"Unknown --backend: {backend_kind!r}")


def _official_generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    backend_kind = str(getattr(args, "backend", "acestep") or "acestep").strip().lower()
    if backend_kind != "acestep-official":
        return {}
    kwargs: dict[str, Any] = {
        "shift": _arg_float(args, "shift", 3.0),
        "infer_method": _arg_str(args, "infer_method", "ode"),
        "sampler_mode": _arg_str(args, "sampler_mode", "euler"),
        "lm_temperature": _arg_float(args, "lm_temperature", 0.85),
        "lm_cfg_scale": _arg_float(args, "lm_cfg_scale", 2.0),
        "lm_top_k": _arg_int(args, "lm_top_k", 0),
        "lm_top_p": _arg_float(args, "lm_top_p", 0.9),
        "audio_cover_strength": _arg_float(args, "audio_cover_strength", 1.0),
        "cover_noise_strength": _arg_float(args, "cover_noise_strength", 0.0),
    }
    if getattr(args, "bpm", None) is not None:
        kwargs["bpm"] = int(getattr(args, "bpm"))
    if getattr(args, "keyscale", None):
        kwargs["keyscale"] = str(getattr(args, "keyscale"))
    if getattr(args, "timesignature", None):
        kwargs["timesignature"] = str(getattr(args, "timesignature"))
    return kwargs


def _cmd_t2m(args: argparse.Namespace) -> int:
    mm = _make_manager_from_args(args)
    backend_kind = str(getattr(args, "backend", "acestep") or "acestep").strip().lower()
    prompt = str(args.prompt)
    lyrics = getattr(args, "lyrics", None)
    if backend_kind in {"diffusers", "musicgen", "stable-audio"} and isinstance(lyrics, str) and lyrics.strip():
        # Avoid silently ignoring lyrics for backends that do not support them.
        prompt = f"{prompt}\n\nLyrics:\n{lyrics.strip()}"
        lyrics = None

    wav = mm.t2m(
        prompt,
        duration_s=float(args.duration),
        num_inference_steps=int(args.steps) if args.steps is not None else None,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        negative_prompt=args.negative,
        lyrics=lyrics,
        **_official_generation_kwargs(args),
    )
    out_path = Path(str(args.out)).expanduser()
    _write_bytes(out_path, wav)
    print(str(out_path))
    return 0


class MusicREPL(cmd.Cmd):
    """Interactive music generation shell with switchable engines."""

    prompt = "music> "
    ruler = ""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = _copy_namespace(args)
        self.args.backend = _normalize_backend(str(getattr(self.args, "backend", "acestep") or "acestep"))
        self.out_dir = Path(str(getattr(self.args, "out_dir", ".") or ".")).expanduser()
        self.prefix = str(getattr(self.args, "prefix", "music") or "music").strip() or "music"
        self.open_outputs = bool(getattr(self.args, "open", False))
        self.current_prompt = str(getattr(self.args, "prompt", "") or "").strip()
        self._manager = None
        self._manager_dirty = True
        self._render_count = 0
        self._init_readline()
        self.intro = (
            "AbstractMusic REPL\n"
            "Type a prompt to generate music, or use /prompt then /run. Commands: /help, /params, /engine, /models, /exit"
        )

    def _init_readline(self) -> None:
        try:
            import atexit
            import readline  # type: ignore
        except Exception:
            return
        hist_dir = Path.home() / ".abstractmusic"
        hist_dir.mkdir(parents=True, exist_ok=True)
        hist_path = hist_dir / "repl_history"
        try:
            readline.read_history_file(str(hist_path))
        except FileNotFoundError:
            pass
        except Exception:
            pass
        try:
            readline.set_history_length(2000)
        except Exception:
            pass

        def _save_history() -> None:
            try:
                readline.write_history_file(str(hist_path))
            except Exception:
                pass

        atexit.register(_save_history)

    def parseline(self, line: str):  # type: ignore[override]
        line = str(line or "").strip()
        if line.startswith("/"):
            line = line[1:].lstrip()
            if not line:
                return None, None, line
            pieces = line.split(None, 1)
            cmd_name = pieces[0].replace("-", "_")
            arg = pieces[1] if len(pieces) > 1 else ""
            parsed_line = f"{cmd_name} {arg}".strip()
            return cmd_name, arg, parsed_line
        return super().parseline(line)

    def emptyline(self) -> bool:
        return False

    def _mark_dirty(self) -> None:
        self._manager_dirty = True

    def _get_manager(self):
        if self._manager is None or self._manager_dirty:
            _configure_mps_env(self.args)
            self._manager = _make_manager_from_args(self.args)
            self._manager_dirty = False
        return self._manager

    def _next_output_path(self) -> Path:
        self._render_count += 1
        stamp = _timestamp_id()
        return self.out_dir / f"{self.prefix}-{stamp}-{self._render_count:03d}.wav"

    @staticmethod
    def _tokens(arg: str) -> list[str]:
        try:
            return shlex.split(str(arg or ""))
        except ValueError as e:
            raise ValueError(f"Could not parse command: {e}") from e

    def _set_optional_text(self, name: str, arg: str, label: str) -> None:
        value = str(arg or "").strip()
        if value.lower() in {"", "clear", "none", "null", "off"}:
            setattr(self.args, name, None)
            print(f"{label} cleared")
        else:
            setattr(self.args, name, value)
            print(f"{label}: {value}")

    def _set_optional_int(self, name: str, arg: str, label: str) -> None:
        value = str(arg or "").strip()
        if value.lower() in {"", "clear", "none", "null", "auto", "off"}:
            setattr(self.args, name, None)
            print(f"{label}: auto")
            return
        parsed = int(value)
        setattr(self.args, name, parsed)
        print(f"{label}: {parsed}")

    def _set_optional_float(self, name: str, arg: str, label: str, *, clear_to_none: bool = True) -> None:
        value = str(arg or "").strip()
        if not value and not clear_to_none:
            print(f"{label}: {getattr(self.args, name, None)}")
            return
        if clear_to_none and value.lower() in {"", "clear", "none", "null", "auto", "off"}:
            setattr(self.args, name, None)
            print(f"{label}: auto")
            return
        parsed = float(value)
        setattr(self.args, name, parsed)
        print(f"{label}: {parsed:g}")

    def _generate(self, prompt: str | None = None) -> None:
        prompt = str(prompt or "").strip()
        if prompt:
            self.current_prompt = prompt
        else:
            prompt = self.current_prompt
        if not prompt:
            print("Usage: /prompt <text>, then /run; or /generate <prompt>")
            return
        mm = self._get_manager()
        backend_kind = str(getattr(self.args, "backend", "acestep") or "acestep").strip().lower()
        lyrics = getattr(self.args, "lyrics", None)
        request_prompt = prompt
        request_lyrics = lyrics
        if backend_kind in {"diffusers", "musicgen", "stable-audio"} and isinstance(lyrics, str) and lyrics.strip():
            request_prompt = f"{prompt}\n\nLyrics:\n{lyrics.strip()}"
            request_lyrics = None
        duration = float(getattr(self.args, "duration", 10.0))
        print(f"Generating with {backend_kind} ({duration:g}s)...")
        wav = mm.t2m(
            request_prompt,
            duration_s=duration,
            num_inference_steps=int(getattr(self.args, "steps")) if getattr(self.args, "steps", None) is not None else None,
            guidance_scale=getattr(self.args, "guidance_scale", None),
            seed=getattr(self.args, "seed", None),
            negative_prompt=getattr(self.args, "negative", None),
            lyrics=request_lyrics,
            **_official_generation_kwargs(self.args),
        )
        out_path = self._next_output_path()
        _write_bytes(out_path, wav)
        print(f"Saved: {out_path}")
        if self.open_outputs:
            _open_file(out_path)

    def default(self, line: str) -> None:
        line = str(line or "").strip()
        if not line:
            return
        if line.startswith("!"):
            print("Shell escapes are not supported in this REPL.")
            return
        try:
            self.current_prompt = line
            self._generate(line)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)

    def do_help(self, arg: str = "") -> None:  # noqa: D401
        """Show REPL commands."""
        _ = arg
        print("Commands:")
        print("  /prompt [text|clear]     Show, set, or clear the session prompt")
        print("  /run                     Generate from the current prompt")
        print("  /generate [prompt]       Generate from a prompt; bare prompt also works")
        print("  /engine [name]           Show or set engine: acestep-official, acestep, xl, diffusers, musicgen, stable-audio")
        print("  /model [id|clear]        Show or set model id")
        print("  /lm [name]               Show or set official ACE-Step 5Hz LM checkpoint")
        print("  /lm-backend [auto|mlx|vllm|pt]")
        print("  /device [auto|mps|cuda|cpu]")
        print("  /dtype [auto|float16|bfloat16|float32]")
        print("  /duration <seconds>      Set generation duration")
        print("  /bpm <n|auto>            Set target BPM metadata")
        print("  /keyscale <text|clear>   Set target key/scale metadata")
        print("  /timesignature <n|clear> Set target time signature metadata")
        print("  /steps <n|auto>          Set inference steps")
        print("  /seed <n|auto>           Set seed")
        print("  /guidance <value|auto>   Set guidance scale")
        print("  /shift <value>           Set ACE-Step timestep shift")
        print("  /infer-method <ode|sde>  Set ACE-Step inference method")
        print("  /sampler-mode <euler|heun>")
        print("  /lm-temperature <value>  Set ACE-Step 5Hz LM temperature")
        print("  /lm-cfg-scale <value>    Set ACE-Step 5Hz LM CFG scale")
        print("  /audio-cover-strength <value>")
        print("  /verbose [on|off]        Show or hide upstream backend logs")
        print("  /lyrics <text|clear>     Set session lyrics, e.g. [Instrumental]")
        print("  /negative <text|clear>   Set negative prompt")
        print("  /out <dir>               Set output directory")
        print("  /prefix <name>           Set output filename prefix")
        print("  /models                  List packaged model registry")
        print("  /params                  Show current settings")
        print("  /exit                    Quit")

    def do_exit(self, arg: str = "") -> bool:
        _ = arg
        return True

    def do_quit(self, arg: str = "") -> bool:
        return self.do_exit(arg)

    def do_q(self, arg: str = "") -> bool:
        return self.do_exit(arg)

    def do_EOF(self, arg: str = "") -> bool:  # noqa: N802
        print("")
        return self.do_exit(arg)

    def do_generate(self, arg: str) -> None:
        try:
            self._generate(arg)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)

    def do_run(self, arg: str = "") -> None:
        _ = arg
        self.do_generate("")

    def do_prompt(self, arg: str = "") -> None:
        value = str(arg or "").strip()
        if not value:
            print(self.current_prompt or "none")
            return
        if value.lower() in {"clear", "none", "null", "off"}:
            self.current_prompt = ""
            print("prompt cleared")
            return
        self.current_prompt = value
        print(f"prompt: {self.current_prompt}")

    def do_engine(self, arg: str) -> None:
        value = str(arg or "").strip()
        if not value:
            print(str(getattr(self.args, "backend", "acestep")))
            return
        backend = _normalize_backend(value)
        setattr(self.args, "backend", backend)
        self._mark_dirty()
        print(f"engine: {backend}")

    def do_backend(self, arg: str) -> None:
        self.do_engine(arg)

    def complete_engine(self, text: str, line: str, begidx: int, endidx: int) -> list[str]:
        _ = line, begidx, endidx
        options = list(SUPPORTED_BACKENDS) + sorted(BACKEND_ALIASES)
        return [item for item in options if item.startswith(text)]

    complete_backend = complete_engine

    def do_model(self, arg: str) -> None:
        value = str(arg or "").strip()
        if not value:
            print(str(getattr(self.args, "model_id", None) or "default"))
            return
        if value.lower() in {"clear", "none", "default"}:
            setattr(self.args, "model_id", None)
            print("model: default")
        else:
            setattr(self.args, "model_id", value)
            print(f"model: {value}")
        self._mark_dirty()

    def do_lm(self, arg: str) -> None:
        value = str(arg or "").strip()
        if not value:
            print(str(getattr(self.args, "lm_model_path", None) or DEFAULT_OFFICIAL_LM_MODEL_PATH))
            return
        setattr(self.args, "lm_model_path", value)
        self._mark_dirty()
        print(f"lm: {value}")

    def do_lm_model_path(self, arg: str) -> None:
        self.do_lm(arg)

    def do_lm_backend(self, arg: str) -> None:
        value = str(arg or "").strip()
        if not value:
            print(str(getattr(self.args, "lm_backend", "auto") or "auto"))
            return
        setattr(self.args, "lm_backend", value)
        self._mark_dirty()
        print(f"lm-backend: {value}")

    def do_device(self, arg: str) -> None:
        value = str(arg or "").strip()
        if not value:
            print(str(getattr(self.args, "device", "auto") or "auto"))
            return
        setattr(self.args, "device", value)
        self._mark_dirty()
        print(f"device: {value}")

    def do_dtype(self, arg: str) -> None:
        value = str(arg or "").strip()
        if not value:
            print(str(getattr(self.args, "dtype", "auto") or "auto"))
            return
        setattr(self.args, "dtype", value)
        self._mark_dirty()
        print(f"dtype: {value}")

    def do_duration(self, arg: str) -> None:
        self._set_optional_float("duration", arg, "duration", clear_to_none=False)

    def do_bpm(self, arg: str) -> None:
        self._set_optional_int("bpm", arg, "bpm")

    def do_keyscale(self, arg: str) -> None:
        self._set_optional_text("keyscale", arg, "keyscale")

    def do_timesignature(self, arg: str) -> None:
        self._set_optional_text("timesignature", arg, "timesignature")

    def do_steps(self, arg: str) -> None:
        self._set_optional_int("steps", arg, "steps")

    def do_seed(self, arg: str) -> None:
        self._set_optional_int("seed", arg, "seed")

    def do_guidance(self, arg: str) -> None:
        self._set_optional_float("guidance_scale", arg, "guidance")

    def do_guidance_scale(self, arg: str) -> None:
        self.do_guidance(arg)

    def do_shift(self, arg: str) -> None:
        self._set_optional_float("shift", arg, "shift", clear_to_none=False)

    def do_infer_method(self, arg: str) -> None:
        value = str(arg or "").strip().lower()
        if not value:
            print(str(getattr(self.args, "infer_method", "ode") or "ode"))
            return
        if value not in {"ode", "sde"}:
            print("Usage: /infer-method <ode|sde>")
            return
        setattr(self.args, "infer_method", value)
        print(f"infer-method: {value}")

    def do_sampler_mode(self, arg: str) -> None:
        value = str(arg or "").strip().lower()
        if not value:
            print(str(getattr(self.args, "sampler_mode", "euler") or "euler"))
            return
        if value not in {"euler", "heun"}:
            print("Usage: /sampler-mode <euler|heun>")
            return
        setattr(self.args, "sampler_mode", value)
        print(f"sampler-mode: {value}")

    def do_lm_temperature(self, arg: str) -> None:
        self._set_optional_float("lm_temperature", arg, "lm-temperature", clear_to_none=False)

    def do_lm_cfg_scale(self, arg: str) -> None:
        self._set_optional_float("lm_cfg_scale", arg, "lm-cfg-scale", clear_to_none=False)

    def do_lm_top_k(self, arg: str) -> None:
        self._set_optional_int("lm_top_k", arg, "lm-top-k")

    def do_lm_top_p(self, arg: str) -> None:
        self._set_optional_float("lm_top_p", arg, "lm-top-p", clear_to_none=False)

    def do_audio_cover_strength(self, arg: str) -> None:
        self._set_optional_float("audio_cover_strength", arg, "audio-cover-strength", clear_to_none=False)

    def do_cover_noise_strength(self, arg: str) -> None:
        self._set_optional_float("cover_noise_strength", arg, "cover-noise-strength", clear_to_none=False)

    def do_verbose(self, arg: str = "") -> None:
        value = str(arg or "").strip().lower()
        if not value:
            print("on" if bool(getattr(self.args, "verbose", False)) else "off")
            return
        if value in {"1", "on", "true", "yes"}:
            setattr(self.args, "verbose", True)
            self._mark_dirty()
            print("verbose: on")
            return
        if value in {"0", "off", "false", "no"}:
            setattr(self.args, "verbose", False)
            self._mark_dirty()
            print("verbose: off")
            return
        print("Usage: /verbose [on|off]")

    def do_negative(self, arg: str) -> None:
        self._set_optional_text("negative", arg, "negative")

    def do_lyrics(self, arg: str) -> None:
        self._set_optional_text("lyrics", arg, "lyrics")

    def do_out(self, arg: str) -> None:
        value = str(arg or "").strip()
        if not value:
            print(str(self.out_dir))
            return
        self.out_dir = Path(value).expanduser()
        print(f"out: {self.out_dir}")

    def do_out_dir(self, arg: str) -> None:
        self.do_out(arg)

    def do_prefix(self, arg: str) -> None:
        value = str(arg or "").strip()
        if not value:
            print(self.prefix)
            return
        self.prefix = value
        print(f"prefix: {self.prefix}")

    def do_params(self, arg: str = "") -> None:
        _ = arg
        fields: list[tuple[str, Any]] = [
            ("engine", getattr(self.args, "backend", None)),
            ("model", getattr(self.args, "model_id", None) or "default"),
            ("lm", getattr(self.args, "lm_model_path", None) or "default"),
            ("lm_backend", getattr(self.args, "lm_backend", None) or "auto"),
            ("prompt", self.current_prompt or "none"),
            ("device", getattr(self.args, "device", "auto")),
            ("dtype", getattr(self.args, "dtype", "auto")),
            ("duration", getattr(self.args, "duration", None)),
            ("bpm", getattr(self.args, "bpm", None) if getattr(self.args, "bpm", None) is not None else "auto"),
            ("keyscale", getattr(self.args, "keyscale", None) or "auto"),
            ("timesignature", getattr(self.args, "timesignature", None) or "auto"),
            ("steps", getattr(self.args, "steps", None) or "auto"),
            ("seed", getattr(self.args, "seed", None) if getattr(self.args, "seed", None) is not None else "auto"),
            ("guidance", getattr(self.args, "guidance_scale", None) if getattr(self.args, "guidance_scale", None) is not None else "auto"),
            ("shift", getattr(self.args, "shift", 3.0)),
            ("infer_method", getattr(self.args, "infer_method", "ode")),
            ("sampler_mode", getattr(self.args, "sampler_mode", "euler")),
            ("lm_temperature", getattr(self.args, "lm_temperature", 0.85)),
            ("lm_cfg_scale", getattr(self.args, "lm_cfg_scale", 2.0)),
            ("lm_top_k", getattr(self.args, "lm_top_k", 0)),
            ("lm_top_p", getattr(self.args, "lm_top_p", 0.9)),
            ("audio_cover_strength", getattr(self.args, "audio_cover_strength", 1.0)),
            ("cover_noise_strength", getattr(self.args, "cover_noise_strength", 0.0)),
            ("verbose", "on" if bool(getattr(self.args, "verbose", False)) else "off"),
            ("lyrics", getattr(self.args, "lyrics", None) or "none"),
            ("negative", getattr(self.args, "negative", None) or "none"),
            ("out", self.out_dir),
            ("prefix", self.prefix),
        ]
        width = max(len(k) for k, _v in fields)
        for key, value in fields:
            print(f"{key.rjust(width)}: {value}")

    def do_models(self, arg: str = "") -> None:
        from .model_capabilities import MusicModelCapabilitiesRegistry

        tokens = self._tokens(arg)
        task = tokens[0] if tokens else None
        reg = MusicModelCapabilitiesRegistry()
        for spec in reg.list_models(task=task):
            backends = ",".join(spec.backend_kinds)
            rec = "recommended" if spec.recommended else "candidate"
            print(f"{spec.id} [{spec.provider}] {rec} status={spec.status} backends={backends}")


def _cmd_repl(args: argparse.Namespace) -> int:
    MusicREPL(args).cmdloop()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_mps_env(args)

    if args.cmd == "t2m":
        return _cmd_t2m(args)
    if args.cmd == "repl":
        return _cmd_repl(args)
    raise SystemExit(f"Unknown command: {args.cmd}")
