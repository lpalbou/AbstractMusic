"""
AbstractCore capability plugin for AbstractMusic.

This registers a `music` capability backend discovered by AbstractCore via the
`abstractcore.capabilities_plugins` entry point group.

Built-in backends:
- Remote ACE Music API backend (default light/base install path).
- Remote ElevenLabs Music API backend (music endpoints only).
- Local ACE-Step pipeline (in-process, optional extra).
- Local Stable Audio 3 internal runtime (gated weights; optional extra).
- Local Diffusers audio pipeline (alternative; in-process).
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import re
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from ..artifacts import RuntimeArtifactStoreAdapter
from ..availability import RemoteEndpoint, cached_model_ids, is_model_cached, probe_endpoints
from ..errors import AbstractMusicError, CapabilityNotSupportedError
from ..huggingface import require_hf_repo_id
from ..model_capabilities import MusicModelCapabilitiesRegistry, MusicModelSpec
from ..music_manager import MusicManager

_TASK_ALIASES = {
    "t2m": "text_to_music",
    "text2music": "text_to_music",
    "text-to-music": "text_to_music",
    "music": "text_to_music",
    "l2m": "lyrics_to_music",
    "lyrics-to-music": "lyrics_to_music",
    "t2a": "text_to_audio",
    "text-to-audio": "text_to_audio",
}

_BACKEND_ID_TO_KIND = {
    "abstractmusic:acemusic": "acemusic",
    "abstractmusic:elevenlabs-music": "elevenlabs",
    "abstractmusic:acestep": "acestep",
    "abstractmusic:stable-audio": "stable-audio",
    "abstractmusic:stable-audio-3": "stable-audio-3",
    "abstractmusic:diffusers": "diffusers",
}

_BACKEND_KIND_TO_ID = {kind: backend_id for backend_id, kind in _BACKEND_ID_TO_KIND.items()}

_NON_RUNNABLE_MODEL_STATUS_PREFIXES = ("planned", "research")

_REMOTE_BACKEND_KINDS = frozenset({"acemusic", "elevenlabs"})

# Where `diffusers` keeps `AceStepPipeline`, relative to the package root.
_ACESTEP_PIPELINE_MODULE = ("pipelines", "ace_step")

# Extra to report for a provider that is discoverable without a registry entry.
_DEFAULT_DEPENDENCY_EXTRA = {"diffusers": "diffusers"}


# Owner-config prefix per remote backend kind. The environment variables behind
# each provider stay owned by its own `BackendConfig.from_env()`.
_REMOTE_CONFIG_PREFIX = {"acemusic": "music_acemusic", "elevenlabs": "music_elevenlabs"}


@dataclass(frozen=True)
class _ProviderState:
    """What one provider can do on this machine, right now.

    `installed` is about the Python runtime, `configured` is about what the
    provider needs in order to answer — credentials for a remote API, cached
    weights for a local one — and `usable` is what discovery filters on.
    """

    backend_kind: str
    provider_id: str
    backend_id: str
    remote: bool
    installed: Optional[bool]
    configured: bool
    usable: bool
    status: str
    detail: str = ""
    models: Tuple[MusicModelSpec, ...] = ()
    #: Everything this provider knows about, runnable or not, for `provider_details`.
    known_models: Tuple[MusicModelSpec, ...] = ()
    cached_model_ids: FrozenSet[str] = frozenset()
    configured_model_id: Optional[str] = None
    latency_ms: Optional[int] = None

_RUNTIME_IMPORTS_BY_EXTRA = {
    "remote": (),
    "acestep": ("torch", "numpy", "diffusers", "transformers", "accelerate", "safetensors", "huggingface_hub"),
    "diffusers": ("torch", "numpy", "diffusers", "transformers", "accelerate", "safetensors", "huggingface_hub"),
    "musicgen": ("torch", "transformers", "safetensors", "huggingface_hub"),
    "stable-audio": (
        "torch",
        "torchaudio",
        "numpy",
        "transformers",
        "safetensors",
        "huggingface_hub",
        "einops",
        "einops_exts",
        "alias_free_torch",
        "vector_quantize_pytorch",
        "pywt",
    ),
    "stable-audio-3": ("torch", "transformers", "safetensors", "huggingface_hub", "einops"),
}


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
    if not model_id:
        return ""
    try:
        return require_hf_repo_id(str(model_id), field_name="music_model_id / ABSTRACTMUSIC_MODEL_ID")
    except ValueError as e:
        raise AbstractMusicError(str(e)) from e


def _lookup_key_or_attr(value: Any, *names: str) -> Any:
    if isinstance(value, dict):
        for name in names:
            item = value.get(name)
            if item is not None:
                return item
    for name in names:
        try:
            item = getattr(value, name, None)
        except Exception:
            item = None
        if item is not None:
            return item
    return None


def _service_from_context(value: Any) -> Any:
    service = _lookup_key_or_attr(
        value,
        "text",
        "text_generation",
        "text_generation_service",
        "core_text_generation_service",
        "core_text_service",
        "host_text_service",
        "host_text_generation_service",
    )
    if _is_host_text_generation_service(service):
        return service

    get_service = getattr(value, "service", None)
    if callable(get_service):
        for name in ("text", "text_generation", "core_text", "abstractcore.text"):
            try:
                service = get_service(name)
            except Exception:
                continue
            if _is_host_text_generation_service(service):
                return service
    return None


def _is_host_text_generation_service(value: Any) -> bool:
    return callable(getattr(value, "generate_structured", None)) or callable(getattr(value, "generate_text", None))


def _call_host_text_method(method: Any, prompt: str, kwargs: Dict[str, Any]) -> Any:
    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return method(prompt, **kwargs)

    params = sig.parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params):
        return method(prompt, **kwargs)

    accepted = {
        name
        for name, param in sig.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered = {key: value for key, value in kwargs.items() if key in accepted}
    return method(prompt, **filtered)


def _get_host_text_generation_service(owner: Any) -> Any:
    cfg_service = _lookup_key_or_attr(
        getattr(owner, "config", None),
        "music_host_text_service",
        "music_host_text_generation_service",
        "music_text_generation_service",
        "core_text_generation_service",
        "core_text_service",
        "host_text_service",
        "host_text_generation_service",
        "text_generation_service",
    )
    if _is_host_text_generation_service(cfg_service):
        return cfg_service

    for ctx_name in ("capability_host_context", "host_context", "capability_context"):
        ctx = getattr(owner, ctx_name, None)
        service = _service_from_context(ctx)
        if service is not None:
            return service

    owner_service = _lookup_key_or_attr(
        owner,
        "text_generation_service",
        "core_text_generation_service",
        "core_text_service",
        "host_text_service",
        "host_text_generation_service",
    )
    return owner_service if _is_host_text_generation_service(owner_service) else None


def _normalize_task(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower().replace(" ", "_")
    if not text:
        return None
    return _TASK_ALIASES.get(text, text.replace("-", "_"))


def _canonical_provider_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text


def _display_name(provider_id: str) -> str:
    return str(provider_id or "").strip()


def _selector_text(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _dedupe_strings(values: List[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _registered_backend_kind_for_spec(spec: MusicModelSpec) -> Optional[str]:
    for kind in spec.backend_kinds:
        normalized = str(kind or "").strip()
        if normalized in _BACKEND_KIND_TO_ID:
            return normalized
    return None


def _provider_id_for_backend_kind(kind: Any) -> str:
    backend_id = _BACKEND_KIND_TO_ID.get(str(kind or "").strip())
    if not backend_id:
        return ""
    return str(backend_id).split(":", 1)[-1].strip()


def _provider_filter_backend_kind(value: Any) -> Optional[str]:
    text = _selector_text(value)
    if not text:
        return None
    for backend_kind, backend_id in _BACKEND_KIND_TO_ID.items():
        provider_id = _provider_id_for_backend_kind(backend_kind)
        if text in {_selector_text(provider_id), _selector_text(backend_id)}:
            return backend_kind
    return None


def _model_status_is_runnable(value: Any) -> bool:
    status = str(value or "").strip().lower()
    if not status:
        return False
    return not any(status.startswith(prefix) for prefix in _NON_RUNNABLE_MODEL_STATUS_PREFIXES)


def _package_ships_module(package: str, *relative_parts: str) -> bool:
    """Answer "does this installed package contain this submodule" without importing it.

    ``find_spec`` on a *dotted* name imports every parent package first, which
    for ``diffusers`` means importing ``torch`` — the cost this whole path
    exists to avoid. Locating the top-level package and looking at the
    filesystem does not, and unlike a version comparison it survives
    pre-releases, editable installs, and source checkouts with no metadata.
    """

    try:
        spec = importlib.util.find_spec(str(package))
    except (ImportError, ValueError):
        return False
    for location in getattr(spec, "submodule_search_locations", None) or ():
        candidate = Path(location).joinpath(*relative_parts)
        if candidate.is_dir() or candidate.with_suffix(".py").is_file():
            return True
    return False


def _runtime_installed(extra: Optional[str]) -> Optional[bool]:
    extra_name = str(extra or "").strip()
    imports = _RUNTIME_IMPORTS_BY_EXTRA.get(extra_name)
    if not imports:
        return None
    if not all(importlib.util.find_spec(name) is not None for name in imports):
        return False

    if extra_name == "acestep":
        return _package_ships_module("diffusers", *_ACESTEP_PIPELINE_MODULE)

    return True


def _selected_backend_kind(backend_id: str) -> Optional[str]:
    return _BACKEND_ID_TO_KIND.get(str(backend_id or ""))


def _backend_id_for_spec(spec: MusicModelSpec) -> Optional[str]:
    for kind in spec.backend_kinds:
        backend_id = _BACKEND_KIND_TO_ID.get(str(kind))
        if backend_id:
            return backend_id
    return None


def _model_record_from_spec(
    spec: MusicModelSpec,
    *,
    backend_id: str,
    installed: Optional[bool],
    cached: Optional[bool],
) -> Dict[str, Any]:
    backend_kind = _registered_backend_kind_for_spec(spec)
    provider_id = _provider_id_for_backend_kind(backend_kind)
    remote = bool(spec.raw.get("remote", False))
    local = bool(spec.raw.get("local", not remote))
    routed_backend_id = _backend_id_for_spec(spec)
    return {
        "model_id": spec.id,
        "provider_id": provider_id,
        "capability": "music",
        "tasks": list(spec.tasks),
        "modalities": [*list(spec.input_modalities), "audio"],
        "local": local,
        "remote": remote,
        "status": spec.status,
        "backend_id": routed_backend_id,
        "routed_model": spec.id,
        "formats": list(spec.output_formats),
        "source": spec.source_url,
        "recommended": bool(spec.recommended),
        "license": spec.license,
        "commercial_allowed": spec.commercial_allowed,
        "metadata": {
            "provider": spec.provider,
            "backend_kind": backend_kind,
            "backend_kinds": list(spec.backend_kinds),
            "dependency_extra": spec.dependency_extra,
            "installed": installed,
            "cached": cached,
            "supports_lyrics": bool(spec.supports_lyrics),
            "supports_negative_prompt": bool(spec.supports_negative_prompt),
            "supports_guidance_scale": bool(spec.supports_guidance_scale),
            "max_duration_s": spec.max_duration_s,
            "sample_rate_hz": spec.sample_rate_hz,
            "official_8bit_available": bool(spec.official_8bit_available),
            "preferred_precision": spec.preferred_precision,
        },
    }


def _configured_diffusers_model_record(
    owner: Any,
    *,
    backend_id: str,
    task: Optional[str],
    installed: Optional[bool],
    cached: Optional[bool],
) -> Optional[Dict[str, Any]]:
    model_id = _owner_cfg(owner, "music_model_id") or _env("ABSTRACTMUSIC_MODEL_ID")
    if not model_id:
        return None
    task_s = _normalize_task(task) or "text_to_audio"
    return {
        "model_id": str(model_id),
        "provider_id": "diffusers",
        "capability": "music",
        "tasks": [task_s],
        "modalities": ["text", "audio"],
        "local": True,
        "remote": False,
        "status": "configured",
        "backend_id": backend_id,
        "routed_model": str(model_id),
        "formats": ["wav"],
        "source": f"https://huggingface.co/{model_id}" if "/" in str(model_id) else None,
        "recommended": False,
        "metadata": {
            "provider": "diffusers",
            "backend_kind": "diffusers",
            "backend_kinds": ["diffusers"],
            "dependency_extra": "diffusers",
            "installed": installed,
            "cached": cached,
        },
    }


def _music_plan_json_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "prompt": {"type": "string"},
            "lyrics": {"type": ["string", "null"]},
            "vocal_language": {"type": ["string", "null"]},
            "bpm": {"type": ["integer", "null"], "minimum": 20, "maximum": 300},
            "keyscale": {"type": ["string", "null"]},
            "timesignature": {"type": ["string", "null"]},
            "positive_styles": {"type": "array", "items": {"type": "string"}},
            "negative_styles": {"type": "array", "items": {"type": "string"}},
            "instrumental": {"type": "boolean"},
            "enhanced_prompt": {"type": "boolean"},
            "structured_prompt": {"type": "boolean"},
            "generated_lyrics": {"type": "boolean"},
            "generated_fields": {"type": "array", "items": {"type": "string"}},
            "warnings": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
            "composition_plan": {"type": ["object", "null"], "additionalProperties": True},
        },
        "required": ["prompt"],
    }


def _planner_system_prompt() -> str:
    return (
        "You are a music planning service for a local text-to-music backend. "
        "Return only structured fields for the audio model. Create an original, highly specific music caption "
        "from the user's request; infer useful BPM, key/scale, time signature, instrumental intent, "
        "and optional lyrics only when requested. When the user specifies instruments, vocalist traits, "
        "or production style, include concise tag-style hints in positive_styles / negative_styles. "
        "For references to named games, films, artists, or songs, "
        "translate the reference into generic musical traits instead of claiming exact imitation. "
        "Favor clear rhythmic structure, instrumentation, section movement, and continuity constraints."
    )


def _planner_user_prompt(request_dict: Dict[str, Any]) -> str:
    return (
        "Create an AbstractMusic prompt plan as JSON for this request.\n"
        "Preserve explicit lyrics unless they are 'auto'. If instrumental is true or the request is clearly "
        "game/action background music, set lyrics to '[Instrumental]'. For long durations, include a concise "
        "section plan in the prompt and avoid empty gaps or long fade-outs. "
        "Use positive_styles / negative_styles for short comma-style tags (instruments, vocalist traits, mixing notes).\n\n"
        f"Request JSON:\n{json.dumps(request_dict, ensure_ascii=True, sort_keys=True)}"
    )


def _object_to_mapping(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dict(dumped)
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        dumped = as_dict()
        if isinstance(dumped, dict):
            return dict(dumped)
    return None


def _extract_text_result_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    for name in ("text", "content", "output_text", "response", "message"):
        item = getattr(value, name, None)
        if isinstance(item, str) and item.strip():
            return item
    if isinstance(value, dict):
        for name in ("text", "content", "output_text", "response", "message"):
            item = value.get(name)
            if isinstance(item, str) and item.strip():
                return item
    return str(value or "")


def _parse_json_object_text(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise TypeError("host text planner JSON output must be an object")
    return dict(parsed)


class _CoreTextServiceMusicPlanner:
    """Adapter from AbstractCore's narrow host text service to MusicPromptPlan mappings."""

    def __init__(self, service: Any) -> None:
        self._service = service

    def create_plan(self, request: Any) -> Dict[str, Any]:
        request_dict = request.to_dict() if hasattr(request, "to_dict") else dict(request)
        prompt = _planner_user_prompt(dict(request_dict))
        system_prompt = _planner_system_prompt()
        metadata = {
            "capability": "music",
            "task": "text_planning",
            "backend": str(request_dict.get("backend") or ""),
            "model_id": str(request_dict.get("model_id") or ""),
        }

        generate_structured = getattr(self._service, "generate_structured", None)
        structured_error: Optional[Exception] = None
        if callable(generate_structured):
            try:
                value = _call_host_text_method(
                    generate_structured,
                    prompt,
                    {
                        "json_schema": _music_plan_json_schema(),
                        "system_prompt": system_prompt,
                        "purpose": "abstractmusic.text_planning",
                        "metadata": metadata,
                    },
                )
                mapping = _object_to_mapping(value)
                if mapping is not None:
                    return self._finalize_mapping(mapping, value)
                mapping = _parse_json_object_text(_extract_text_result_text(value))
                return self._finalize_mapping(mapping, value)
            except Exception as exc:
                structured_error = exc

        generate_text = getattr(self._service, "generate_text", None)
        if not callable(generate_text):
            if structured_error is not None:
                raise structured_error
            raise TypeError("host text generation service must expose generate_structured(...) or generate_text(...)")
        value = _call_host_text_method(
            generate_text,
            prompt,
            {
                "system_prompt": system_prompt,
                "max_output_tokens": 900,
                "temperature": 0.2,
                "purpose": "abstractmusic.text_planning",
                "metadata": metadata,
            },
        )
        mapping = _parse_json_object_text(_extract_text_result_text(value))
        return self._finalize_mapping(mapping, value)

    @staticmethod
    def _finalize_mapping(mapping: Dict[str, Any], raw_value: Any) -> Dict[str, Any]:
        out = dict(mapping)
        out.setdefault("planner_backend", "abstractcore-host-text-service")
        model = _lookup_key_or_attr(raw_value, "model", "model_id", "provider_model")
        if isinstance(model, str) and model.strip():
            out.setdefault("planner_model", model.strip())
        if not out.get("generated_fields"):
            out["generated_fields"] = [
                key
                for key in (
                    "prompt",
                    "lyrics",
                    "bpm",
                    "keyscale",
                    "timesignature",
                    "vocal_language",
                    "positive_styles",
                    "negative_styles",
                )
                if out.get(key) not in (None, "", [])
            ]
        return out


class _AbstractMusicCapabilityBase:
    """Shared capability wrapper (handles ArtifactStore persistence)."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner
        self._backend = None
        self._text_planner = None
        self._state_lock = threading.RLock()
        self._loaded_models: Dict[str, Dict[str, Any]] = {}

    def _residency_kind(self) -> str:
        return str(_selected_backend_kind(str(getattr(self, "backend_id", ""))) or "").strip()

    def _residency_is_remote(self) -> bool:
        return self._residency_kind() in {"acemusic", "elevenlabs"}

    def _residency_model_id(self) -> Optional[str]:
        backend = self._get_backend()
        caps = None
        get_caps = getattr(backend, "get_capabilities", None)
        if callable(get_caps):
            try:
                caps = get_caps()
            except Exception:
                caps = None
        model_id = getattr(caps, "model_id", None) if caps is not None else None
        if isinstance(model_id, str) and model_id.strip():
            return model_id.strip()
        configured = _owner_cfg(self._owner, "music_model_id") or _env("ABSTRACTMUSIC_MODEL_ID")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        return None

    def _load_id(self, *, provider: str, model: Optional[str]) -> str:
        model_s = str(model or "").strip()
        provider_s = str(provider or "").strip()
        if not provider_s:
            return model_s
        return f"{provider_s}/{model_s}" if model_s else provider_s

    def _record_loaded(
        self,
        *,
        task: Optional[str],
        resident: bool,
        source: str,
        error: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        kind = self._residency_kind()
        model_id = self._residency_model_id()
        now = time.time()
        load_id = self._load_id(provider=kind, model=model_id)
        with self._state_lock:
            existing = dict(self._loaded_models.get(load_id, {}))
            loaded_at = existing.get("loaded_at")
            if loaded_at is None:
                loaded_at = now
            record = {
                "task": str(task or ""),
                "provider": kind or None,
                "model": model_id,
                "load_id": load_id,
                "backend_kind": kind or None,
                "scope": "process",
                "state": "resident" if resident else "active",
                "resident": bool(resident),
                "loaded": True,
                "unloadable": not self._residency_is_remote(),
                "source": str(source),
                "loaded_at": float(loaded_at),
                "last_used_at": float(now),
                "error": dict(error) if isinstance(error, dict) else None,
            }
            if existing.get("resident"):
                record["resident"] = True
                record["state"] = "resident"
                record["source"] = "explicit_preload"
            self._loaded_models[load_id] = record
            return dict(record)

    def _clear_loaded(self, load_id: str) -> Optional[Dict[str, Any]]:
        with self._state_lock:
            existing = self._loaded_models.pop(str(load_id or ""), None)
        return dict(existing) if isinstance(existing, dict) else None

    def load_resident_model(self, request: Any) -> Dict[str, Any]:
        payload = dict(request) if isinstance(request, dict) else {}
        task = _normalize_task(payload.get("task")) or "text_to_music"
        kind = self._residency_kind()
        if self._residency_is_remote():
            return {
                "task": task,
                "provider": kind or None,
                "model": self._residency_model_id(),
                "load_id": self._load_id(provider=kind, model=self._residency_model_id()),
                "backend_kind": kind or None,
                "scope": "process",
                "state": "stateless",
                "resident": False,
                "loaded": False,
                "unloadable": False,
                "source": "remote_stateless",
                "loaded_at": None,
                "last_used_at": None,
                "error": None,
            }

        backend = self._get_backend()
        preload = getattr(backend, "preload", None)
        try:
            if callable(preload):
                preload()
        except Exception as exc:
            return self._record_loaded(
                task=task,
                resident=False,
                source="explicit_preload",
                error={"code": "load_failed", "message": str(exc)},
            )
        return self._record_loaded(task=task, resident=True, source="explicit_preload")

    def list_loaded_models(self, filters: Any | None = None) -> List[Dict[str, Any]]:
        filter_map = dict(filters) if isinstance(filters, dict) else {}
        wanted_load_id = str(filter_map.get("load_id") or filter_map.get("id") or "").strip() or None
        wanted_provider = str(filter_map.get("provider") or filter_map.get("backend_kind") or "").strip() or None
        wanted_model = str(filter_map.get("model") or "").strip() or None
        wanted_resident = filter_map.get("resident")

        with self._state_lock:
            records = [dict(item) for item in self._loaded_models.values()]

        out: List[Dict[str, Any]] = []
        for record in records:
            if wanted_load_id and str(record.get("load_id") or "") != wanted_load_id:
                continue
            if wanted_provider and str(record.get("provider") or "") != wanted_provider:
                continue
            if wanted_model and str(record.get("model") or "") != wanted_model:
                continue
            if wanted_resident is not None and bool(record.get("resident")) is not bool(wanted_resident):
                continue
            out.append(record)

        out.sort(key=lambda item: (str(item.get("provider") or ""), str(item.get("model") or "")))
        return out

    def list_resident_models(self, filters: Any | None = None) -> List[Dict[str, Any]]:
        filter_map = dict(filters) if isinstance(filters, dict) else {}
        filter_map["resident"] = True
        return self.list_loaded_models(filter_map)

    def unload_resident_model(self, request: Any) -> Dict[str, Any]:
        payload = dict(request) if isinstance(request, dict) else {}
        load_id = str(payload.get("load_id") or payload.get("id") or "").strip()
        provider = str(payload.get("provider") or payload.get("backend_kind") or "").strip()
        model = str(payload.get("model") or "").strip()

        if not load_id:
            kind = self._residency_kind()
            model_id = self._residency_model_id()
            if provider or model:
                load_id = self._load_id(provider=(provider or kind), model=(model or model_id))
            else:
                load_id = self._load_id(provider=kind, model=model_id)

        existing = None
        with self._state_lock:
            existing = dict(self._loaded_models.get(load_id, {})) if load_id in self._loaded_models else None

        if not existing:
            return {
                "task": _normalize_task(payload.get("task")) or "text_to_music",
                "provider": provider or self._residency_kind() or None,
                "model": model or self._residency_model_id(),
                "load_id": load_id or None,
                "backend_kind": provider or self._residency_kind() or None,
                "scope": "process",
                "state": "not_loaded",
                "resident": False,
                "loaded": False,
                "unloadable": not self._residency_is_remote(),
                "source": None,
                "loaded_at": None,
                "last_used_at": None,
                "error": None,
            }

        backend = self._get_backend()
        unload = getattr(backend, "unload", None)
        try:
            if callable(unload):
                unload()
        except Exception as exc:
            existing["state"] = "failed"
            existing["resident"] = False
            existing["loaded"] = True
            existing["error"] = {"code": "unload_failed", "message": str(exc)}
            return dict(existing)

        self._clear_loaded(load_id)
        # Drop cached backend instance so the next call is forced to reload.
        self._backend = None
        return {
            "task": existing.get("task"),
            "provider": existing.get("provider"),
            "model": existing.get("model"),
            "load_id": existing.get("load_id"),
            "backend_kind": existing.get("backend_kind"),
            "scope": "process",
            "state": "unloaded",
            "resident": False,
            "loaded": False,
            "unloadable": bool(existing.get("unloadable", True)),
            "source": existing.get("source"),
            "loaded_at": None,
            "last_used_at": existing.get("last_used_at"),
            "error": None,
        }

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

    def _get_text_planner(self):
        if self._text_planner is not None:
            return self._text_planner
        mode = self._get_text_planner_mode().strip().lower()
        if mode in {"0", "false", "no", "none", "off", "raw"}:
            return None
        try:
            cfg = getattr(self._owner, "config", None)
            if isinstance(cfg, dict):
                inst = cfg.get("music_text_planner")
                if inst is None:
                    inst = cfg.get("music_text_planner_instance")
                if inst is not None:
                    self._text_planner = inst
                    return self._text_planner
                factory = cfg.get("music_text_planner_factory")
                if callable(factory):
                    try:
                        self._text_planner = factory(self._owner)
                        return self._text_planner
                    except Exception:
                        if self._get_text_planner_mode().strip().lower() == "required":
                            raise
                        return None
            service = _get_host_text_generation_service(self._owner)
            if service is not None:
                self._text_planner = _CoreTextServiceMusicPlanner(service)
                return self._text_planner
        except Exception:
            if mode == "required":
                raise
            return None
        return None

    def _get_text_planner_mode(self) -> str:
        return (
            _owner_cfg(self._owner, "music_text_planner_mode")
            or _env("ABSTRACTMUSIC_TEXT_PLANNER")
            or "auto"
        )

    def _make_manager(self) -> MusicManager:
        # Keep capability execution predictable: always generate locally in-process
        # and let the capability layer handle ArtifactStore persistence (run_id/tags/metadata).
        return MusicManager(
            backend=self._get_backend(),
            store=None,
            text_planner=self._get_text_planner(),
            text_planner_mode=self._get_text_planner_mode(),
        )

    def _remote_health_endpoint(self, backend_kind: str) -> Optional[RemoteEndpoint]:
        """Return the probe endpoint for a remote provider, or None when unconfigured.

        Discovery reads configuration, never a backend instance: it answers for
        every provider, not just the selected one. Environment resolution is left
        to each backend's own `from_env()`; only owner config is layered on top.
        """

        kind = str(backend_kind or "").strip()
        prefix = _REMOTE_CONFIG_PREFIX.get(kind)
        if prefix is None:
            return None
        if kind == "acemusic":
            from ..backends.acemusic import AceMusicBackendConfig as ConfigClass
        else:
            from ..backends.elevenlabs_music import ElevenLabsMusicBackendConfig as ConfigClass

        config = ConfigClass.from_env()
        overrides: Dict[str, Any] = {}
        for field_name in ("base_url", "api_key"):
            value = _owner_cfg(self._owner, f"{prefix}_{field_name}")
            if value:
                overrides[field_name] = value
        if overrides:
            config = replace(config, **overrides)
        return config.health_endpoint()

    def _provider_states(self, *, task: Optional[str] = None) -> Dict[str, _ProviderState]:
        """Resolve what every provider can do on this machine, right now.

        Local providers are answered from the filesystem (are the weights here?)
        and remote providers from one parallel round of short HTTP probes. No
        model runtime is imported and no weights are read.
        """

        specs_by_kind: Dict[str, List[MusicModelSpec]] = {}
        for spec in self._registry_models(task=task):
            backend_kind = _registered_backend_kind_for_spec(spec)
            if not backend_kind or not _model_status_is_runnable(spec.status):
                continue
            specs_by_kind.setdefault(backend_kind, []).append(spec)

        endpoints: Dict[str, RemoteEndpoint] = {}
        for kind in specs_by_kind:
            if kind not in _REMOTE_BACKEND_KINDS:
                continue
            endpoint = self._remote_health_endpoint(kind)
            if endpoint is not None:
                endpoints[kind] = endpoint
        probes = probe_endpoints(endpoints) if endpoints else {}

        states: Dict[str, _ProviderState] = {}
        for kind, specs in specs_by_kind.items():
            if kind in _REMOTE_BACKEND_KINDS:
                states[kind] = self._remote_provider_state(kind, specs, probes.get(kind))
            else:
                states[kind] = self._local_provider_state(kind, specs)

        if "diffusers" not in states:
            # A user-configured checkpoint has no registry entry; a registry-derived
            # diffusers provider, if one ever exists, is the more informative answer.
            diffusers_state = self._configured_diffusers_state()
            if diffusers_state is not None:
                states["diffusers"] = diffusers_state
        return states

    def _remote_provider_state(
        self,
        backend_kind: str,
        specs: List[MusicModelSpec],
        probe: Optional[Any],
    ) -> _ProviderState:
        configured = probe is not None
        if probe is None:
            status, detail, usable = "not-configured", "no API key configured", False
        else:
            status, detail, usable = probe.status, probe.detail, probe.usable
        return _ProviderState(
            backend_kind=backend_kind,
            provider_id=_provider_id_for_backend_kind(backend_kind),
            backend_id=str(_BACKEND_KIND_TO_ID.get(backend_kind) or ""),
            remote=True,
            installed=True,
            configured=configured,
            usable=usable,
            status=status,
            detail=detail,
            models=tuple(specs) if usable else (),
            known_models=tuple(specs),
            latency_ms=getattr(probe, "latency_ms", None),
        )

    def _local_provider_state(self, backend_kind: str, specs: List[MusicModelSpec]) -> _ProviderState:
        installed_by_extra: Dict[str, Optional[bool]] = {}
        runnable: List[MusicModelSpec] = []
        for spec in specs:
            extra = str(spec.dependency_extra or "")
            if extra not in installed_by_extra:
                installed_by_extra[extra] = _runtime_installed(spec.dependency_extra)
            if installed_by_extra[extra] is True:
                runnable.append(spec)

        installed_values = [value for value in installed_by_extra.values() if value is not None]
        installed = any(installed_values) if installed_values else None
        present = cached_model_ids(spec.id for spec in runnable) if runnable else frozenset()
        models = tuple(spec for spec in runnable if spec.id in present)

        if installed is not True:
            status, detail = "not-installed", "runtime dependencies are not installed"
        elif not models:
            status, detail = "no-local-weights", "no model weights found in the Hugging Face cache"
        else:
            status, detail = "available", ""

        return _ProviderState(
            backend_kind=backend_kind,
            provider_id=_provider_id_for_backend_kind(backend_kind),
            backend_id=str(_BACKEND_KIND_TO_ID.get(backend_kind) or ""),
            remote=False,
            installed=installed,
            configured=bool(models),
            usable=bool(models),
            status=status,
            detail=detail,
            models=models,
            known_models=tuple(specs),
            cached_model_ids=frozenset(present),
        )

    def _configured_diffusers_state(self) -> Optional[_ProviderState]:
        """State for a user-configured Diffusers checkpoint, which has no registry entry."""

        model_id = _owner_cfg(self._owner, "music_model_id") or _env("ABSTRACTMUSIC_MODEL_ID")
        if not model_id:
            return None
        installed = _runtime_installed("diffusers")
        cached = bool(installed) and is_model_cached(str(model_id))
        if installed is not True:
            status, detail = "not-installed", "runtime dependencies are not installed"
        elif not cached:
            status, detail = "no-local-weights", f"{model_id} is not in the Hugging Face cache"
        else:
            status, detail = "available", ""
        return _ProviderState(
            backend_kind="diffusers",
            provider_id=_provider_id_for_backend_kind("diffusers"),
            backend_id=str(_BACKEND_KIND_TO_ID["diffusers"]),
            remote=False,
            installed=installed,
            configured=True,
            usable=bool(cached),
            status=status,
            detail=detail,
            models=(),
            configured_model_id=str(model_id),
            cached_model_ids=frozenset({str(model_id)}) if cached else frozenset(),
        )

    def _registry_models(self, *, task: Optional[str] = None) -> List[MusicModelSpec]:
        return list(MusicModelCapabilitiesRegistry().list_models(task=_normalize_task(task)))

    def available_providers(
        self,
        *,
        task: Optional[str] = None,
        _states: Optional[Dict[str, "_ProviderState"]] = None,
    ) -> List[Dict[str, Any]]:
        """Return the music providers that are usable on this machine right now.

        Local providers are usable when their weights are already cached; remote
        providers when their API answers. Nothing here loads a model runtime.
        Use `provider_details(...)` to see the providers that were left out and
        why.
        """

        task_s = _normalize_task(task)
        states = self._provider_states(task=task_s) if _states is None else _states
        providers = [
            self._provider_record(state, task=task_s)
            for state in states.values()
            if state.usable
        ]
        return sorted(providers, key=lambda item: (not bool(item.get("selected")), str(item["provider_id"])))

    def list_available_providers(self, *, task: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.available_providers(task=task)

    def provider_details(
        self,
        *,
        task: Optional[str] = None,
        _states: Optional[Dict[str, "_ProviderState"]] = None,
    ) -> List[Dict[str, Any]]:
        """Return every known provider with why it is or is not usable.

        `available_providers` answers "what can I run"; this answers "what else
        is there, and what is missing" — a rejected API key, an uninstalled
        extra, or weights that have not been downloaded — so an empty provider
        list is never a dead end. Costs nothing extra: the same probe round
        backs both.
        """

        task_s = _normalize_task(task)
        states = self._provider_states(task=task_s) if _states is None else _states
        details = []
        for state in states.values():
            record = self._provider_record(state, task=task_s, include_all_models=True)
            record["usable"] = state.usable
            record["metadata"]["reason"] = state.detail
            record["metadata"]["cached_models"] = sorted(state.cached_model_ids)
            if state.latency_ms is not None:
                record["metadata"]["latency_ms"] = state.latency_ms
            details.append(record)
        return sorted(details, key=lambda item: (not bool(item["usable"]), str(item["provider_id"])))

    def _provider_record(
        self,
        state: "_ProviderState",
        *,
        task: Optional[str],
        include_all_models: bool = False,
    ) -> Dict[str, Any]:
        specs = state.known_models if include_all_models else state.models
        tasks: List[str] = []
        models: List[str] = []
        for spec in specs:
            for value in spec.tasks:
                if value not in tasks:
                    tasks.append(value)
            if spec.id not in models:
                models.append(spec.id)
        if state.configured_model_id and state.configured_model_id not in models:
            models.append(state.configured_model_id)
        if not tasks and task:
            tasks = [task]

        extras = _dedupe_strings([spec.dependency_extra for spec in specs])
        if not extras:
            default_extra = _DEFAULT_DEPENDENCY_EXTRA.get(state.backend_kind)
            extras = [default_extra] if default_extra else []

        return {
            "provider_id": state.provider_id,
            "display_name": _display_name(state.provider_id),
            "capability": "music",
            "tasks": tasks,
            "local": not state.remote,
            "remote": state.remote,
            "status": state.status,
            "backend_id": state.backend_id,
            "installed": state.installed,
            "configured": state.configured,
            "selected": _selected_backend_kind(str(getattr(self, "backend_id", ""))) == state.backend_kind,
            "metadata": {
                "models": models,
                "backend_kind": state.backend_kind,
                "backend_kinds": [state.backend_kind],
                "dependency_extras": extras,
                "detail": state.detail,
            },
        }

    def list_models(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        provider_id: Optional[str] = None,
        _states: Optional[Dict[str, "_ProviderState"]] = None,
    ) -> List[Dict[str, Any]]:
        """Return the music models that can run right now, for usable providers."""

        task_s = _normalize_task(task)
        provider_s = provider_id or provider
        wanted_backend_kind = _provider_filter_backend_kind(provider_s)
        if provider_s is not None and wanted_backend_kind is None:
            return []

        states = self._provider_states(task=task_s) if _states is None else _states
        records: List[Dict[str, Any]] = []
        for state in states.values():
            if not state.usable:
                continue
            if wanted_backend_kind is not None and state.backend_kind != wanted_backend_kind:
                continue
            if state.configured_model_id:
                configured = _configured_diffusers_model_record(
                    self._owner,
                    backend_id=state.backend_id,
                    task=task_s,
                    installed=state.installed,
                    cached=True,
                )
                if configured is not None:
                    records.append(configured)
            records.extend(
                _model_record_from_spec(
                    spec,
                    backend_id=state.backend_id,
                    installed=state.installed,
                    cached=None if state.remote else True,
                )
                for spec in state.models
            )
        return records

    def list_provider_models(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.list_models(task=task, provider=provider, provider_id=provider_id)

    def list_operations(self, *, task: Optional[str] = None) -> List[Dict[str, Any]]:
        task_s = _normalize_task(task)
        tasks = sorted({t for spec in self._registry_models(task=task_s) for t in spec.tasks})
        if task_s is not None:
            if task_s not in tasks:
                return []
            tasks = [task_s]
        formats = self._supported_output_formats()
        schema = {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "prompt": {"type": "string"},
                "lyrics": {"type": ["string", "null"]},
                "positive_styles": {"type": ["array", "string", "null"], "items": {"type": "string"}},
                "negative_styles": {"type": ["array", "string", "null"], "items": {"type": "string"}},
                "duration_s": {"type": ["number", "null"], "minimum": 0},
                "seed": {"type": ["integer", "null"]},
                "format": {"type": "string", "enum": formats},
                "num_inference_steps": {"type": ["integer", "null"], "minimum": 1},
                "guidance_scale": {"type": ["number", "null"]},
                "instrumental": {"type": "boolean"},
                "enhance_prompt": {"type": "boolean"},
                "structure_prompt": {"type": "boolean"},
                "auto_lyrics": {"type": "boolean"},
                "text_planner_mode": {"type": "string"},
            },
            "required": ["prompt"],
        }
        return [
            {
                "operation_id": current_task,
                "capability": "music",
                "task": current_task,
                "input_modalities": ["text", "lyrics"] if current_task == "lyrics_to_music" else ["text"],
                "output_modalities": ["audio"],
                "parameter_schema": schema,
                "required_parameters": ["prompt"],
                "artifact_output": True,
                "metadata": {
                    "typed_method": "t2m",
                    "backend_id": str(getattr(self, "backend_id", "")),
                    "formats": formats,
                },
            }
            for current_task in tasks
        ]

    def _supported_output_formats(self) -> List[str]:
        if _selected_backend_kind(str(getattr(self, "backend_id", ""))) == "acemusic":
            return ["wav", "mp3", "flac"]
        if _selected_backend_kind(str(getattr(self, "backend_id", ""))) == "elevenlabs":
            return ["wav", "mp3"]
        return ["wav"]

    def capability_catalog(self, *, task: Optional[str] = None) -> Dict[str, Any]:
        # Resolve availability once: probing remote providers per sub-call would
        # multiply the wall-clock cost of an unresponsive provider.
        states = self._provider_states(task=_normalize_task(task))
        return {
            "capability": "music",
            "backend_id": str(getattr(self, "backend_id", "")),
            "task": _normalize_task(task),
            "providers": self.available_providers(task=task, _states=states),
            "provider_details": self.provider_details(task=task, _states=states),
            "models": self.list_models(task=task, _states=states),
            "operations": self.list_operations(task=task),
        }

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
        allowed_formats = set(self._supported_output_formats())
        if fmt not in allowed_formats:
            raise CapabilityNotSupportedError(
                f"format={fmt!r} is not supported by this backend; supported formats: {sorted(allowed_formats)}"
            )

        mm = self._make_manager()
        planner_mode = self._get_text_planner_mode()
        if "planning" not in kwargs and "plan_text" not in kwargs:
            kwargs["planning"] = str(planner_mode or "").strip().lower() not in {"0", "false", "no", "none", "off"}
        if "text_planner_mode" not in kwargs:
            kwargs["text_planner_mode"] = planner_mode
        out = mm.generate_audio(str(prompt or ""), lyrics=lyrics, format=fmt, **kwargs)

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
            content_type=str(out.mime_type or ("audio/wav" if fmt == "wav" else "application/octet-stream")),
            filename=f"music.{fmt}",
            run_id=str(run_id) if run_id else None,
            tags=merged_tags,
            metadata=merged_metadata or None,
        )


class _AbstractMusicAceMusicCapability(_AbstractMusicCapabilityBase):
    """AbstractCore MusicCapability using the lightweight ACE Music remote API."""

    backend_id = "abstractmusic:acemusic"

    def _get_backend(self):
        if self._backend is not None:
            return self._backend

        try:
            return super()._get_backend()
        except NotImplementedError:
            pass

        base_url = (
            _owner_cfg(self._owner, "music_acemusic_base_url")
            or _env("ACEMUSIC_BASE_URL")
            or "https://api.acemusic.ai"
        )
        api_key = (
            _owner_cfg(self._owner, "music_acemusic_api_key")
            or _env("ACEMUSIC_API_KEY")
        )
        model = (
            _owner_cfg(self._owner, "music_acemusic_model")
        )
        timeout_s = _owner_cfg_any(self._owner, "music_acemusic_timeout_s")

        def _to_float(v: Any, default: float) -> float:
            try:
                return float(v)
            except Exception:
                return float(default)

        from ..backends.acemusic import AceMusicBackend, AceMusicBackendConfig

        cfg = AceMusicBackendConfig(
            base_url=str(base_url or "https://api.acemusic.ai"),
            api_key=str(api_key) if api_key is not None else None,
            model=str(model).strip() if isinstance(model, str) and model.strip() else None,
            timeout_s=_to_float(timeout_s, 600.0),
        )
        self._backend = AceMusicBackend(config=cfg)
        return self._backend


class _AbstractMusicElevenLabsMusicCapability(_AbstractMusicCapabilityBase):
    """AbstractCore MusicCapability using ElevenLabs Music endpoints only."""

    backend_id = "abstractmusic:elevenlabs-music"

    def _get_backend(self):
        if self._backend is not None:
            return self._backend

        try:
            return super()._get_backend()
        except NotImplementedError:
            pass

        base_url = (
            _owner_cfg(self._owner, "music_elevenlabs_base_url")
            or _env("ELEVENLABS_BASE_URL")
            or "https://api.elevenlabs.io"
        )
        api_key = (
            _owner_cfg(self._owner, "music_elevenlabs_api_key")
            or _env("ELEVENLABS_API_KEY")
        )
        model = _owner_cfg(self._owner, "music_elevenlabs_model") or "music_v1"
        output_format = _owner_cfg(self._owner, "music_elevenlabs_output_format")
        composition_mode = _owner_cfg(self._owner, "music_composition_mode") or "auto"
        timeout_s = _owner_cfg_any(self._owner, "music_elevenlabs_timeout_s")

        def _to_float(v: Any, default: float) -> float:
            try:
                return float(v)
            except Exception:
                return float(default)

        from ..backends.elevenlabs_music import ElevenLabsMusicBackend, ElevenLabsMusicBackendConfig

        cfg = ElevenLabsMusicBackendConfig(
            base_url=str(base_url or "https://api.elevenlabs.io"),
            api_key=str(api_key) if api_key is not None else None,
            model=str(model or "music_v1"),
            timeout_s=_to_float(timeout_s, 600.0),
            output_format=str(output_format).strip() if isinstance(output_format, str) and output_format.strip() else None,
            composition_mode=str(composition_mode or "auto"),
        )
        self._backend = ElevenLabsMusicBackend(config=cfg)
        return self._backend


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


class _AbstractMusicAceStepCapability(_AbstractMusicCapabilityBase):
    """AbstractCore MusicCapability using the supported ACE-Step path."""

    backend_id = "abstractmusic:acestep"

    def _get_backend(self):
        if self._backend is not None:
            return self._backend

        try:
            return super()._get_backend()
        except NotImplementedError:
            pass

        model_id = _require_model_id(self._owner) or "ACE-Step/acestep-v15-xl-turbo-diffusers"
        device = _owner_cfg(self._owner, "music_device") or _env("ABSTRACTMUSIC_DEVICE", "auto")
        dtype = _owner_cfg(self._owner, "music_torch_dtype") or _env("ABSTRACTMUSIC_TORCH_DTYPE", "auto")

        steps = _owner_cfg_any(self._owner, "music_num_inference_steps") or _env("ABSTRACTMUSIC_NUM_INFERENCE_STEPS")
        duration_s = _owner_cfg_any(self._owner, "music_duration_s") or _env("ABSTRACTMUSIC_DURATION_S")

        def _to_int(v: Any, default: int) -> int:
            try:
                return int(v)
            except Exception:
                return int(default)

        def _to_float(v: Any, default: float) -> float:
            try:
                return float(v)
            except Exception:
                return float(default)

        from ..backends.acestep import AceStepBackend, AceStepBackendConfig

        cfg = AceStepBackendConfig(
            model_id=str(model_id),
            device=str(device or "auto"),
            torch_dtype=str(dtype or "auto"),
            num_inference_steps=_to_int(steps, 8) if steps is not None else None,
            duration_s=_to_float(duration_s, 10.0),
        )
        self._backend = AceStepBackend(config=cfg)
        return self._backend


class _AbstractMusicStableAudio3Capability(_AbstractMusicCapabilityBase):
    """AbstractCore MusicCapability using AbstractMusic's internal Stable Audio 3 runtime."""

    backend_id = "abstractmusic:stable-audio-3"

    def _get_backend(self):
        if self._backend is not None:
            return self._backend

        try:
            return super()._get_backend()
        except NotImplementedError:
            pass

        model_id = _require_model_id(self._owner) or "stabilityai/stable-audio-3-small-music"
        device = _owner_cfg(self._owner, "music_device") or _env("ABSTRACTMUSIC_DEVICE", "auto")
        dtype = _owner_cfg(self._owner, "music_torch_dtype") or _env("ABSTRACTMUSIC_TORCH_DTYPE", "auto")
        steps = _owner_cfg_any(self._owner, "music_num_inference_steps") or _env("ABSTRACTMUSIC_NUM_INFERENCE_STEPS")
        duration_s = _owner_cfg_any(self._owner, "music_duration_s") or _env("ABSTRACTMUSIC_DURATION_S")
        guidance_scale = _owner_cfg_any(self._owner, "music_guidance_scale") or _env("ABSTRACTMUSIC_GUIDANCE_SCALE")

        def _to_int(v: Any, default: int) -> int:
            try:
                return int(v)
            except Exception:
                return int(default)

        def _to_float(v: Any, default: float) -> float:
            try:
                return float(v)
            except Exception:
                return float(default)

        from ..backends.stable_audio_3 import StableAudio3Backend, StableAudio3BackendConfig

        cfg = StableAudio3BackendConfig(
            model_id=str(model_id),
            device=str(device or "auto"),
            torch_dtype=str(dtype or "auto"),
            duration_s=_to_float(duration_s, 30.0),
            num_inference_steps=_to_int(steps, 8),
            guidance_scale=_to_float(guidance_scale, 1.0),
        )
        self._backend = StableAudio3Backend(config=cfg)
        return self._backend


class _AbstractMusicStableAudioCapability(_AbstractMusicCapabilityBase):
    """AbstractCore MusicCapability using Stable Audio Open Small (stable-audio-tools)."""

    backend_id = "abstractmusic:stable-audio"

    def _get_backend(self):
        if self._backend is not None:
            return self._backend

        try:
            return super()._get_backend()
        except NotImplementedError:
            pass

        model_id = _require_model_id(self._owner) or "stabilityai/stable-audio-open-small"
        device = _owner_cfg(self._owner, "music_device") or _env("ABSTRACTMUSIC_DEVICE", "auto")
        steps = _owner_cfg_any(self._owner, "music_num_inference_steps") or _env("ABSTRACTMUSIC_NUM_INFERENCE_STEPS")
        duration_s = _owner_cfg_any(self._owner, "music_duration_s") or _env("ABSTRACTMUSIC_DURATION_S")
        guidance_scale = _owner_cfg_any(self._owner, "music_guidance_scale") or _env("ABSTRACTMUSIC_GUIDANCE_SCALE")

        def _to_int(v: Any, default: int) -> int:
            try:
                return int(v)
            except Exception:
                return int(default)

        def _to_float(v: Any, default: float) -> float:
            try:
                return float(v)
            except Exception:
                return float(default)

        from ..backends.stable_audio import StableAudioBackend, StableAudioBackendConfig

        cfg = StableAudioBackendConfig(
            model_id=str(model_id),
            device=str(device or "auto"),
            duration_s=_to_float(duration_s, 11.0),
            num_inference_steps=_to_int(steps, 8),
            guidance_scale=_to_float(guidance_scale, 1.0),
        )
        self._backend = StableAudioBackend(config=cfg)
        return self._backend


def register(registry: Any) -> None:
    """Register AbstractMusic as an AbstractCore capability plugin."""

    registry.register_music_backend(
        backend_id=_AbstractMusicAceMusicCapability.backend_id,
        factory=lambda owner: _AbstractMusicAceMusicCapability(owner),
        priority=50,
        description="AbstractMusic ACE Music remote API path (lightweight base install; no local model runtime).",
        config_hint="Set music_acemusic_api_key or ACEMUSIC_API_KEY. "
        "Optional: set music_acemusic_base_url or ACEMUSIC_BASE_URL (default: https://api.acemusic.ai), "
        "and music_text_planner / music_text_planner_factory or a narrow host text service for text planning.",
    )

    registry.register_music_backend(
        backend_id=_AbstractMusicElevenLabsMusicCapability.backend_id,
        factory=lambda owner: _AbstractMusicElevenLabsMusicCapability(owner),
        priority=45,
        description="AbstractMusic ElevenLabs Music remote API path (music generation endpoints only).",
        config_hint="Set music_elevenlabs_api_key or ELEVENLABS_API_KEY. "
        "Optional: set music_elevenlabs_base_url or ELEVENLABS_BASE_URL "
        "(default: https://api.elevenlabs.io), music_elevenlabs_model='music_v1', "
        "music_composition_mode='auto'/'prompt'/'plan', and a host text planner for richer composition plans. "
        "Voice/TTS endpoints are intentionally not exposed here; use AbstractVoice for voice.",
    )

    registry.register_music_backend(
        backend_id=_AbstractMusicAceStepCapability.backend_id,
        factory=lambda owner: _AbstractMusicAceStepCapability(owner),
        priority=30,
        description="AbstractMusic ACE-Step path (in-process, package-owned AceStepPipeline adapter).",
        config_hint="Optional: set music_model_id to a HF repo id "
        "(default: 'ACE-Step/acestep-v15-xl-turbo-diffusers'). Local filesystem paths are rejected. "
        "Optionally set music_device='auto'/'cuda'/'mps'/'cpu', music_torch_dtype='auto'/'float32'/'bfloat16', "
        "music_num_inference_steps to override the checkpoint recipe (turbo defaults to 8; base/sft checkpoints should usually use 50), "
        "and music_text_planner / music_text_planner_factory or a narrow host text service for text planning.",
    )

    registry.register_music_backend(
        backend_id=_AbstractMusicStableAudio3Capability.backend_id,
        factory=lambda owner: _AbstractMusicStableAudio3Capability(owner),
        priority=10,
        description="AbstractMusic Stable Audio 3 path (internal package-owned runtime; gated Hugging Face weights).",
        config_hint="Optional: set music_model_id to 'stabilityai/stable-audio-3-small-music' "
        "(default) or 'stabilityai/stable-audio-3-medium'. Local filesystem paths are rejected. "
        "Requires accepted Hugging Face model terms and the stable-audio-3 extra. "
        "Optionally set music_device='auto'/'cuda'/'mps'/'cpu' and music_torch_dtype='auto'/'float32'/'float16'.",
    )

    registry.register_music_backend(
        backend_id=_AbstractMusicStableAudioCapability.backend_id,
        factory=lambda owner: _AbstractMusicStableAudioCapability(owner),
        priority=8,
        description="AbstractMusic Stable Audio Open Small path (stable-audio-tools; gated Hugging Face weights).",
        config_hint="Optional: set music_model_id to 'stabilityai/stable-audio-open-small'. "
        "Local filesystem paths are rejected. Requires accepted Hugging Face model terms "
        "and the stable-audio extra (stable-audio-tools). "
        "Optionally set music_device='auto'/'cuda'/'mps'/'cpu'.",
    )

    registry.register_music_backend(
        backend_id=_AbstractMusicDiffusersCapability.backend_id,
        factory=lambda owner: _AbstractMusicDiffusersCapability(owner),
        priority=0,
        description="AbstractMusic local generation via Diffusers audio pipeline.",
        config_hint="Set music_model_id (or ABSTRACTMUSIC_MODEL_ID) to a Diffusers audio model id "
        "(checkpoint license varies). Local filesystem paths are rejected. "
        "Optionally set music_device='auto'/'cuda'/'mps'/'cpu' and "
        "music_text_planner / music_text_planner_factory or a narrow host text service for text planning.",
    )
