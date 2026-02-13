"""
AbstractMusic CLI.

Goals:
- Provide a minimal, practical interface similar to AbstractVision's CLI.
- Support a simple REPL loop for iterative prompt→WAV generation.
- Keep module import light; heavy ML imports happen only when generating.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

if os.environ.get("DIFFUSERS_SLOW_IMPORT", "").strip().upper() in {"1", "ON", "YES", "TRUE"}:
    os.environ["DIFFUSERS_SLOW_IMPORT"] = "0"


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(str(key), None)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _timestamp_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _write_bytes(path: Path, content: bytes) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(bytes(content))


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

        parser.add_argument(
            "--backend",
            default=_env("ABSTRACTMUSIC_BACKEND", "acestep") if use_defaults else default_suppress,
            choices=["diffusers", "acestep"],
            help="Local backend kind (diffusers|acestep). Default: acestep.",
        )
        parser.add_argument(
            "--model-id",
            default=_env("ABSTRACTMUSIC_MODEL_ID") if use_defaults else default_suppress,
            help="Model repo id (or set $ABSTRACTMUSIC_MODEL_ID). "
            "Diffusers: any Diffusers audio checkpoint id (license varies). ACE-Step: ACE-Step/Ace-Step1.5",
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
        parser.add_argument(
            "--guidance-scale",
            type=float,
            default=None if use_defaults else default_suppress,
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

    _add_common_args(p, use_defaults=True)

    sub = p.add_subparsers(dest="cmd", required=True)

    t2m = sub.add_parser("t2m", help="Generate a WAV file from a single prompt")
    _add_common_args(t2m, use_defaults=False)
    t2m.add_argument("prompt", help="Text prompt")
    t2m.add_argument("--lyrics", default=None, help="Optional lyrics (ACE-Step supports lyrics natively).")
    t2m.add_argument("--out", default="out.wav", help="Output WAV path")

    repl = sub.add_parser("repl", help="Interactive prompt→WAV loop")
    _add_common_args(repl, use_defaults=False)
    repl.add_argument("--out-dir", default=".", help="Directory to write WAV files into")
    repl.add_argument("--prefix", default="music", help="Filename prefix for outputs")
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
    from .backends import AceStepV15Backend, AceStepV15BackendConfig

    backend_kind = str(getattr(args, "backend", "diffusers") or "diffusers").strip().lower()
    model_id = str(getattr(args, "model_id", "") or "").strip()

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

    if backend_kind == "acestep":
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


def _cmd_t2m(args: argparse.Namespace) -> int:
    mm = _make_manager_from_args(args)
    backend_kind = str(getattr(args, "backend", "acestep") or "acestep").strip().lower()
    prompt = str(args.prompt)
    lyrics = getattr(args, "lyrics", None)
    if backend_kind == "diffusers" and isinstance(lyrics, str) and lyrics.strip():
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
    )
    out_path = Path(str(args.out)).expanduser()
    _write_bytes(out_path, wav)
    print(str(out_path))
    return 0


def _cmd_repl(args: argparse.Namespace) -> int:
    mm = _make_manager_from_args(args)
    out_dir = Path(str(args.out_dir)).expanduser()
    prefix = str(args.prefix or "music").strip() or "music"
    backend_kind = str(getattr(args, "backend", "acestep") or "acestep").strip().lower()
    lyrics = getattr(args, "lyrics", None)
    if backend_kind == "diffusers" and isinstance(lyrics, str) and lyrics.strip():
        # Avoid silently ignoring lyrics for backends that do not support them.
        lyrics = None

    print("AbstractMusic REPL")
    print("Type a prompt and press Enter to generate. Commands: /help, /exit")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0
        if not line:
            continue
        if line in {"/exit", "/quit", "/q"}:
            return 0
        if line == "/help":
            print("Commands:")
            print("  /help        Show this help")
            print("  /exit        Quit (aliases: /quit, /q)")
            print("  <prompt...>  Generate a WAV file from the prompt")
            continue

        try:
            prompt = line
            if backend_kind == "diffusers":
                lyr = getattr(args, "lyrics", None)
                if isinstance(lyr, str) and lyr.strip():
                    prompt = f"{prompt}\n\nLyrics:\n{lyr.strip()}"
            wav = mm.t2m(
                prompt,
                duration_s=float(args.duration),
                num_inference_steps=int(args.steps) if args.steps is not None else None,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                negative_prompt=args.negative,
                lyrics=lyrics,
            )
            name = f"{prefix}-{_timestamp_id()}.wav"
            out_path = out_dir / name
            _write_bytes(out_path, wav)
            print(str(out_path))
            if bool(getattr(args, "open", False)):
                _open_file(out_path)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            continue


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_mps_env(args)

    if args.cmd == "t2m":
        return _cmd_t2m(args)
    if args.cmd == "repl":
        return _cmd_repl(args)
    raise SystemExit(f"Unknown command: {args.cmd}")

