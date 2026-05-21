"""
AbstractCore capability plugin for AbstractMusic.

This registers a `music` capability backend discovered by AbstractCore via the
`abstractcore.capabilities_plugins` entry point group.

Built-in backends:
- Remote ACE Music API backend (default light/base install path).
- Remote ElevenLabs Music API backend (music endpoints only).
- Local ACE-Step Diffusers XL pipeline (in-process, optional extra).
- Local ACE-Step v1.5 pipeline (explicit quality-limited backend).
- Local Stable Audio 3 internal runtime (gated weights; optional extra).
- Local Diffusers audio pipeline (alternative; in-process).
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import re
from typing import Any, Dict, List, Optional

from ..artifacts import RuntimeArtifactStoreAdapter
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
    "abstractmusic:acestep-diffusers": "acestep-diffusers",
    "abstractmusic:acestep-v15": "acestep-v15",
    "abstractmusic:stable-audio": "stable-audio",
    "abstractmusic:stable-audio-3": "stable-audio-3",
    "abstractmusic:diffusers": "diffusers",
}

_BACKEND_KIND_TO_ID = {kind: backend_id for backend_id, kind in _BACKEND_ID_TO_KIND.items()}

_RUNTIME_IMPORTS_BY_EXTRA = {
    "remote": (),
    "acestep": ("torch", "diffusers", "transformers", "accelerate", "safetensors", "huggingface_hub"),
    "acestep-diffusers": ("torch", "diffusers", "transformers", "accelerate", "safetensors", "huggingface_hub"),
    "diffusers": ("torch", "diffusers", "transformers", "accelerate", "safetensors", "huggingface_hub"),
    "musicgen": ("torch", "transformers", "safetensors", "huggingface_hub"),
    "stable-audio": ("torch", "torchaudio", "transformers", "stable_audio_tools"),
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
    known = {
        "ace-music": "ACE Music",
        "elevenlabs": "ElevenLabs",
        "ace-step": "ACE-Step",
        "meta": "Meta",
        "stability-ai": "Stability AI",
        "heartmula": "HeartMuLa",
        "m-a-p": "m-a-p",
        "lh-tech-ai": "LH-Tech-AI",
        "dalision": "Dalision",
        "huggingface": "Hugging Face",
    }
    return known.get(str(provider_id), str(provider_id).replace("-", " ").title())


def _runtime_installed(extra: Optional[str]) -> Optional[bool]:
    imports = _RUNTIME_IMPORTS_BY_EXTRA.get(str(extra or ""))
    if not imports:
        return None
    return all(importlib.util.find_spec(name) is not None for name in imports)


def _spec_matches_provider(spec: MusicModelSpec, provider: Optional[str]) -> bool:
    provider_s = _canonical_provider_id(provider) if provider is not None else ""
    return not provider_s or _canonical_provider_id(spec.provider) == provider_s


def _selected_backend_kind(backend_id: str) -> Optional[str]:
    return _BACKEND_ID_TO_KIND.get(str(backend_id or ""))


def _backend_id_for_spec(spec: MusicModelSpec, *, default_backend_id: str) -> Optional[str]:
    for kind in spec.backend_kinds:
        backend_id = _BACKEND_KIND_TO_ID.get(str(kind))
        if backend_id:
            return backend_id
    return str(default_backend_id).strip() or None


def _model_record_from_spec(spec: MusicModelSpec, *, backend_id: str) -> Dict[str, Any]:
    provider_id = _canonical_provider_id(spec.provider)
    remote = bool(spec.raw.get("remote", False))
    local = bool(spec.raw.get("local", not remote))
    routed_backend_id = _backend_id_for_spec(spec, default_backend_id=backend_id)
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
            "backend_kinds": list(spec.backend_kinds),
            "dependency_extra": spec.dependency_extra,
            "installed": _runtime_installed(spec.dependency_extra),
            "supports_lyrics": bool(spec.supports_lyrics),
            "supports_negative_prompt": bool(spec.supports_negative_prompt),
            "supports_guidance_scale": bool(spec.supports_guidance_scale),
            "max_duration_s": spec.max_duration_s,
            "sample_rate_hz": spec.sample_rate_hz,
            "official_8bit_available": bool(spec.official_8bit_available),
            "preferred_precision": spec.preferred_precision,
        },
    }


def _configured_diffusers_model_record(owner: Any, *, backend_id: str, task: Optional[str]) -> Optional[Dict[str, Any]]:
    model_id = _owner_cfg(owner, "music_model_id") or _env("ABSTRACTMUSIC_MODEL_ID")
    if not model_id:
        return None
    task_s = _normalize_task(task) or "text_to_audio"
    return {
        "model_id": str(model_id),
        "provider_id": "huggingface",
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
            "provider": "Hugging Face",
            "backend_kinds": ["diffusers"],
            "dependency_extra": "diffusers",
            "installed": _runtime_installed("diffusers"),
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
        "and optional lyrics only when requested. For references to named games, films, artists, or songs, "
        "translate the reference into generic musical traits instead of claiming exact imitation. "
        "Favor clear rhythmic structure, instrumentation, section movement, and continuity constraints."
    )


def _planner_user_prompt(request_dict: Dict[str, Any]) -> str:
    return (
        "Create an AbstractMusic prompt plan as JSON for this request.\n"
        "Preserve explicit lyrics unless they are 'auto'. If instrumental is true or the request is clearly "
        "game/action background music, set lyrics to '[Instrumental]'. For long durations, include a concise "
        "section plan in the prompt and avoid empty gaps or long fade-outs.\n\n"
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
                for key in ("prompt", "lyrics", "bpm", "keyscale", "timesignature", "vocal_language")
                if out.get(key) not in (None, "", [])
            ]
        return out


class _AbstractMusicCapabilityBase:
    """Shared capability wrapper (handles ArtifactStore persistence)."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner
        self._backend = None
        self._text_planner = None

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

    def _registry_models(self, *, task: Optional[str] = None, provider: Optional[str] = None) -> List[MusicModelSpec]:
        registry = MusicModelCapabilitiesRegistry()
        models = [
            spec
            for spec in registry.list_models(task=_normalize_task(task))
            if _spec_matches_provider(spec, provider)
        ]
        return list(models)

    def available_providers(self, *, task: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return lightweight music provider availability without loading model runtimes."""

        task_s = _normalize_task(task)
        selected_kind = _selected_backend_kind(str(getattr(self, "backend_id", "")))
        providers: Dict[str, Dict[str, Any]] = {}
        for spec in self._registry_models(task=task_s):
            provider_id = _canonical_provider_id(spec.provider)
            spec_remote = bool(spec.raw.get("remote", False))
            spec_local = bool(spec.raw.get("local", not spec_remote))
            entry = providers.setdefault(
                provider_id,
                {
                    "provider_id": provider_id,
                    "display_name": _display_name(provider_id),
                    "capability": "music",
                    "tasks": [],
                    "local": spec_local,
                    "remote": spec_remote,
                    "status": "available",
                    "backend_id": str(getattr(self, "backend_id", "")),
                    "selected": False,
                    "metadata": {
                        "models": [],
                        "backend_kinds": [],
                        "dependency_extras": [],
                    },
                },
            )
            entry["local"] = bool(entry.get("local")) or spec_local
            entry["remote"] = bool(entry.get("remote")) or spec_remote
            for value in spec.tasks:
                if value not in entry["tasks"]:
                    entry["tasks"].append(value)
            entry["metadata"]["models"].append(spec.id)
            for value in spec.backend_kinds:
                if value not in entry["metadata"]["backend_kinds"]:
                    entry["metadata"]["backend_kinds"].append(value)
                if selected_kind is not None and value == selected_kind:
                    entry["selected"] = True
            if spec.dependency_extra and spec.dependency_extra not in entry["metadata"]["dependency_extras"]:
                entry["metadata"]["dependency_extras"].append(spec.dependency_extra)

        if selected_kind == "diffusers":
            configured = _configured_diffusers_model_record(
                self._owner,
                backend_id=str(getattr(self, "backend_id", "")),
                task=task_s,
            )
            if configured is not None:
                provider_id = str(configured["provider_id"])
                entry = providers.setdefault(
                    provider_id,
                    {
                        "provider_id": provider_id,
                        "display_name": _display_name(provider_id),
                        "capability": "music",
                        "tasks": [],
                        "local": True,
                        "remote": False,
                        "status": "configured",
                        "backend_id": str(getattr(self, "backend_id", "")),
                        "selected": True,
                        "metadata": {"models": [], "backend_kinds": [], "dependency_extras": []},
                    },
                )
                for value in configured.get("tasks", []):
                    if value not in entry["tasks"]:
                        entry["tasks"].append(value)
                if configured["model_id"] not in entry["metadata"]["models"]:
                    entry["metadata"]["models"].append(configured["model_id"])
                if "diffusers" not in entry["metadata"]["backend_kinds"]:
                    entry["metadata"]["backend_kinds"].append("diffusers")
                if "diffusers" not in entry["metadata"]["dependency_extras"]:
                    entry["metadata"]["dependency_extras"].append("diffusers")
                entry["selected"] = True
                entry["installed"] = _runtime_installed("diffusers")

        for entry in providers.values():
            installed_values = [
                _runtime_installed(extra)
                for extra in entry["metadata"].get("dependency_extras", [])
            ]
            installed_known = [value for value in installed_values if value is not None]
            if installed_known:
                entry["installed"] = any(installed_known)
            if not entry["tasks"] and task_s:
                entry["tasks"] = [task_s]

        return sorted(providers.values(), key=lambda item: (not bool(item.get("selected")), str(item["provider_id"])))

    def list_available_providers(self, *, task: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.available_providers(task=task)

    def list_models(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return normalized music model records for AbstractCore discovery."""

        task_s = _normalize_task(task)
        provider_s = provider_id or provider
        backend_id = str(getattr(self, "backend_id", ""))
        records = [
            _model_record_from_spec(spec, backend_id=backend_id)
            for spec in self._registry_models(task=task_s, provider=provider_s)
        ]
        if _selected_backend_kind(backend_id) == "diffusers" and _canonical_provider_id(provider_s) in {"", "huggingface"}:
            configured = _configured_diffusers_model_record(self._owner, backend_id=backend_id, task=task_s)
            if configured is not None:
                known_ids = {str(item.get("model_id")) for item in records}
                if str(configured.get("model_id")) not in known_ids:
                    records.insert(0, configured)
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
        return {
            "capability": "music",
            "backend_id": str(getattr(self, "backend_id", "")),
            "task": _normalize_task(task),
            "providers": self.available_providers(task=task),
            "models": self.list_models(task=task),
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


class _AbstractMusicAceStepDiffusersCapability(_AbstractMusicCapabilityBase):
    """AbstractCore MusicCapability using ACE-Step Diffusers XL Turbo."""

    backend_id = "abstractmusic:acestep-diffusers"

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

        from ..backends.acestep_diffusers import AceStepDiffusersBackend, AceStepDiffusersBackendConfig

        cfg = AceStepDiffusersBackendConfig(
            model_id=str(model_id),
            device=str(device or "auto"),
            torch_dtype=str(dtype or "auto"),
            num_inference_steps=_to_int(steps, 16),
            duration_s=_to_float(duration_s, 10.0),
        )
        self._backend = AceStepDiffusersBackend(config=cfg)
        return self._backend


class _AbstractMusicAceStepV15Capability(_AbstractMusicCapabilityBase):
    """AbstractCore MusicCapability using the standalone ACE-Step v1.5 path."""

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

        from ..backends.acestep_v15 import AceStepV15Backend, AceStepV15BackendConfig

        cfg_kwargs = {
            "repo_id": str(repo_id),
            "device": str(device or "auto"),
            "torch_dtype": str(dtype or "auto"),
            "vae_torch_dtype": str(dtype or "auto"),
        }
        if isinstance(revision, str) and revision.strip():
            cfg_kwargs["revision"] = str(revision).strip()
        cfg = AceStepV15BackendConfig(**cfg_kwargs)
        self._backend = AceStepV15Backend(config=cfg)
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
        backend_id=_AbstractMusicAceStepDiffusersCapability.backend_id,
        factory=lambda owner: _AbstractMusicAceStepDiffusersCapability(owner),
        priority=30,
        description="AbstractMusic ACE-Step Diffusers XL Turbo path (in-process, package-owned adapter).",
        config_hint="Optional: set music_model_id to a HF repo id "
        "(default: 'ACE-Step/acestep-v15-xl-turbo-diffusers'). Local filesystem paths are rejected. "
        "Optionally set music_device='auto'/'cuda'/'mps'/'cpu', music_torch_dtype='auto'/'float32'/'bfloat16', "
        "and music_text_planner / music_text_planner_factory or a narrow host text service for text planning.",
    )

    registry.register_music_backend(
        backend_id=_AbstractMusicAceStepV15Capability.backend_id,
        factory=lambda owner: _AbstractMusicAceStepV15Capability(owner),
        priority=5,
        description="AbstractMusic standalone ACE-Step v1.5 path (explicit quality-limited backend).",
        config_hint="Optional: set music_model_id to a HF repo id (default: 'ACE-Step/Ace-Step1.5'). "
        "Local filesystem paths are rejected. "
        "Optionally set music_device='auto'/'cuda'/'mps'/'cpu', music_torch_dtype='auto'/'float32'/'bfloat16', "
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
