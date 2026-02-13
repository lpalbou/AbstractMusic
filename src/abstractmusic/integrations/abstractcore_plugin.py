"""
AbstractCore capability plugin for AbstractMusic.

This registers a `music` capability backend discovered by AbstractCore via the
`abstractcore.capabilities_plugins` entry point group.

Default backend:
- Local ACE-Step v1.5 pipeline (in-process; no external server required).
- Local Diffusers audio pipeline (alternative; in-process).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from ..artifacts import RuntimeArtifactStoreAdapter
from ..errors import AbstractMusicError, CapabilityNotSupportedError
from ..music_manager import MusicManager


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(str(key), None)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _owner_cfg(owner: Any, key: str) -> Optional[str]:
    try:
        cfg = getattr(owner, "config", None)
        if isinstance(cfg, dict):
            v = cfg.get(key)
            if v is None:
                return None
            s = str(v).strip()
            return s if s else None
    except Exception:
        return None
    return None


def _owner_cfg_any(owner: Any, key: str) -> Any:
    try:
        cfg = getattr(owner, "config", None)
        if isinstance(cfg, dict):
            return cfg.get(key)
    except Exception:
        return None
    return None


def _require_model_id(owner: Any) -> str:
    model_id = _owner_cfg(owner, "music_model_id") or _env("ABSTRACTMUSIC_MODEL_ID")
    return str(model_id) if model_id else ""


class _AbstractMusicCapabilityBase:
    """Shared capability wrapper (handles ArtifactStore persistence)."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner
        self._backend = None

    def _get_backend(self):
        if self._backend is not None:
            return self._backend

        # Injection hook (tests / advanced embedding).
        try:
            cfg = getattr(self._owner, "config", None)
            if isinstance(cfg, dict):
                inst = cfg.get("music_backend_instance")
                if inst is not None:
                    self._backend = inst
                    return self._backend
                factory = cfg.get("music_backend_factory")
                if callable(factory):
                    self._backend = factory(self._owner)
                    return self._backend
        except Exception:
            pass

        raise NotImplementedError

    def _make_manager(self) -> MusicManager:
        # Keep capability execution predictable: always generate locally in-process
        # and let the capability layer handle ArtifactStore persistence (run_id/tags/metadata).
        return MusicManager(backend=self._get_backend(), store=None)

    def t2m(
        self,
        prompt: str,
        *,
        lyrics: Optional[str] = None,
        format: str = "wav",
        artifact_store: Any = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        fmt = str(format or "wav").strip().lower() or "wav"
        if fmt != "wav":
            raise CapabilityNotSupportedError(
                "Only format='wav' is supported (baseline, no external codecs required)."
            )

        mm = self._make_manager()
        out = mm.generate_audio(str(prompt or ""), lyrics=lyrics, **kwargs)

        if isinstance(out, dict):
            # MusicManager should not return artifact refs here (store=None). Defensive check.
            raise TypeError("Unexpected artifact-ref output; AbstractMusic capability expects bytes output before storage.")

        audio_bytes = bytes(out.data)
        if artifact_store is None:
            return audio_bytes

        # Store with AbstractCore-style tags for durability.
        store = RuntimeArtifactStoreAdapter(artifact_store)
        merged_tags: Dict[str, str] = {"kind": "generated_media", "modality": "audio", "task": "text2music"}
        if isinstance(tags, dict):
            merged_tags.update({str(k): str(v) for k, v in tags.items()})

        merged_metadata: Dict[str, Any] = {}
        if isinstance(out.metadata, dict):
            merged_metadata.update(out.metadata)
        if isinstance(metadata, dict):
            merged_metadata.update(metadata)

        return store.store_bytes(
            audio_bytes,
            content_type="audio/wav",
            filename="music.wav",
            run_id=str(run_id) if run_id else None,
            tags=merged_tags,
            metadata=merged_metadata or None,
        )


class _AbstractMusicDiffusersCapability(_AbstractMusicCapabilityBase):
    """AbstractCore MusicCapability using Diffusers local pipelines."""

    backend_id = "abstractmusic:diffusers"

    def _get_backend(self):
        if self._backend is not None:
            return self._backend

        # Respect test injection first.
        try:
            return super()._get_backend()
        except NotImplementedError:
            pass

        model_id = _require_model_id(self._owner)
        if not model_id:
            raise AbstractMusicError(
                "Missing music_model_id / ABSTRACTMUSIC_MODEL_ID. "
                "Configure a local Diffusers audio model id (checkpoint license varies)."
            )

        device = _owner_cfg(self._owner, "music_device") or _env("ABSTRACTMUSIC_DEVICE", "auto")
        dtype = _owner_cfg(self._owner, "music_torch_dtype") or _env("ABSTRACTMUSIC_TORCH_DTYPE", "auto")
        pipeline_class = _owner_cfg(self._owner, "music_pipeline_class") or _env("ABSTRACTMUSIC_PIPELINE_CLASS")

        # Optional numeric config.
        steps = _owner_cfg_any(self._owner, "music_num_inference_steps") or _env("ABSTRACTMUSIC_NUM_INFERENCE_STEPS")
        duration_s = _owner_cfg_any(self._owner, "music_duration_s") or _env("ABSTRACTMUSIC_DURATION_S")
        guidance_scale = _owner_cfg_any(self._owner, "music_guidance_scale") or _env("ABSTRACTMUSIC_GUIDANCE_SCALE")

        def _to_int(v: Any, default: int) -> int:
            try:
                return int(v)
            except Exception:
                return int(default)

        def _to_float_or_none(v: Any) -> Optional[float]:
            if v is None:
                return None
            try:
                return float(v)
            except Exception:
                return None

        from ..backends.diffusers_audio import DiffusersAudioBackend, DiffusersAudioBackendConfig

        cfg = DiffusersAudioBackendConfig(
            model_id=str(model_id),
            device=str(device or "auto"),
            torch_dtype=str(dtype or "auto"),
            pipeline_class=str(pipeline_class).strip() if isinstance(pipeline_class, str) and pipeline_class.strip() else None,
            num_inference_steps=_to_int(steps, 50),
            duration_s=_to_float_or_none(duration_s) if duration_s is not None else 10.0,
            guidance_scale=_to_float_or_none(guidance_scale),
        )
        self._backend = DiffusersAudioBackend(config=cfg)
        return self._backend

    def t2m(self, prompt: str, *, lyrics: Optional[str] = None, **kwargs: Any):
        # Diffusers pipelines do not expose a first-class lyrics channel; avoid silently ignoring.
        full_prompt = str(prompt or "")
        if isinstance(lyrics, str) and lyrics.strip():
            full_prompt = f"{full_prompt}\n\nLyrics:\n{lyrics.strip()}"
        return super().t2m(full_prompt, lyrics=None, **kwargs)


class _AbstractMusicAceStepV15Capability(_AbstractMusicCapabilityBase):
    """AbstractCore MusicCapability using ACE-Step v1.5 (local, in-process)."""

    backend_id = "abstractmusic:acestep-v15"

    def _get_backend(self):
        if self._backend is not None:
            return self._backend

        # Respect test injection first.
        try:
            return super()._get_backend()
        except NotImplementedError:
            pass

        repo_id = _require_model_id(self._owner) or "ACE-Step/Ace-Step1.5"
        device = _owner_cfg(self._owner, "music_device") or _env("ABSTRACTMUSIC_DEVICE", "auto")
        dtype = _owner_cfg(self._owner, "music_torch_dtype") or _env("ABSTRACTMUSIC_TORCH_DTYPE", "auto")
        revision = _owner_cfg(self._owner, "music_revision") or _env("ABSTRACTMUSIC_REVISION")
        cache_dir = _owner_cfg(self._owner, "music_cache_dir") or _env("ABSTRACTMUSIC_CACHE_DIR")

        from ..backends.acestep_v15 import AceStepV15Backend, AceStepV15BackendConfig

        cfg_kwargs = {
            "repo_id": str(repo_id),
            "device": str(device or "auto"),
            "torch_dtype": str(dtype or "auto"),
            "vae_torch_dtype": str(dtype or "auto"),
        }
        if isinstance(revision, str) and revision.strip():
            cfg_kwargs["revision"] = str(revision).strip()
        if isinstance(cache_dir, str) and cache_dir.strip():
            cfg_kwargs["cache_dir"] = str(cache_dir).strip()
        cfg = AceStepV15BackendConfig(**cfg_kwargs)
        self._backend = AceStepV15Backend(config=cfg)
        return self._backend


def register(registry: Any) -> None:
    """Register AbstractMusic as an AbstractCore capability plugin."""

    registry.register_music_backend(
        backend_id=_AbstractMusicAceStepV15Capability.backend_id,
        factory=lambda owner: _AbstractMusicAceStepV15Capability(owner),
        priority=10,
        description="AbstractMusic local generation via ACE-Step v1.5 (in-process).",
        config_hint="Optional: set music_model_id to a HF repo id (default: 'ACE-Step/Ace-Step1.5'). "
        "Optionally set music_device='auto'/'cuda'/'mps'/'cpu' and music_torch_dtype='auto'/'float32'/'bfloat16'.",
    )

    registry.register_music_backend(
        backend_id=_AbstractMusicDiffusersCapability.backend_id,
        factory=lambda owner: _AbstractMusicDiffusersCapability(owner),
        priority=0,
        description="AbstractMusic local generation via Diffusers audio pipeline.",
        config_hint="Set music_model_id (or ABSTRACTMUSIC_MODEL_ID) to a Diffusers audio model id "
        "(checkpoint license varies). Optionally set music_device='auto'/'cuda'/'mps'/'cpu'.",
    )
