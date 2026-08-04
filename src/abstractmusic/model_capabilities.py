"""Packaged model capability registry for AbstractMusic."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .errors import CapabilityNotSupportedError, UnknownModelError


@dataclass(frozen=True)
class MusicModelSpec:
    """Static metadata for a known music/audio model."""

    id: str
    provider: str
    backend_kinds: Sequence[str]
    license: Optional[str]
    commercial_allowed: Optional[bool]
    recommended: bool
    default_for_backend: bool
    status: str
    tasks: Sequence[str]
    input_modalities: Sequence[str]
    output_formats: Sequence[str]
    supports_lyrics: bool
    supports_negative_prompt: bool
    supports_guidance_scale: bool
    supports_reference_audio: bool
    supports_video: bool
    #: True when long template captions are known to degrade this checkpoint's
    #: output; planners should keep enhancement compact unless explicitly asked.
    caption_sensitive: bool
    max_duration_s: Optional[float]
    sample_rate_hz: Optional[int]
    official_8bit_available: bool
    preferred_precision: str
    dependency_extra: Optional[str]
    source_url: Optional[str]
    notes: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def supports_task(self, task: str) -> bool:
        return str(task) in {str(t) for t in self.tasks}


def _as_str_sequence(value: Any, *, field_name: str) -> Sequence[str]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    out: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} contains an invalid item: {item!r}")
        out.append(item.strip())
    return tuple(out)


def _optional_float(value: Any, *, field_name: str) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception as e:
        raise ValueError(f"{field_name} must be a number or null") from e


def _optional_int(value: Any, *, field_name: str) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception as e:
        raise ValueError(f"{field_name} must be an integer or null") from e


def _parse_model(raw: Dict[str, Any]) -> MusicModelSpec:
    if not isinstance(raw, dict):
        raise ValueError("model entry must be an object")
    model_id = str(raw.get("id") or "").strip()
    if not model_id:
        raise ValueError("model entry missing id")
    provider = str(raw.get("provider") or "").strip()
    if not provider:
        raise ValueError(f"model {model_id!r} missing provider")
    tasks = _as_str_sequence(raw.get("tasks"), field_name=f"{model_id}.tasks")
    if not tasks:
        raise ValueError(f"model {model_id!r} must declare at least one task")
    output_formats = _as_str_sequence(raw.get("output_formats"), field_name=f"{model_id}.output_formats")
    if not output_formats:
        raise ValueError(f"model {model_id!r} must declare at least one output format")
    return MusicModelSpec(
        id=model_id,
        provider=provider,
        backend_kinds=_as_str_sequence(raw.get("backend_kinds"), field_name=f"{model_id}.backend_kinds"),
        license=str(raw["license"]).strip() if raw.get("license") is not None else None,
        commercial_allowed=raw.get("commercial_allowed") if isinstance(raw.get("commercial_allowed"), bool) else None,
        recommended=bool(raw.get("recommended", False)),
        default_for_backend=bool(raw.get("default_for_backend", False)),
        status=str(raw.get("status") or "unknown"),
        tasks=tasks,
        input_modalities=_as_str_sequence(raw.get("input_modalities"), field_name=f"{model_id}.input_modalities"),
        output_formats=output_formats,
        supports_lyrics=bool(raw.get("supports_lyrics", False)),
        supports_negative_prompt=bool(raw.get("supports_negative_prompt", False)),
        supports_guidance_scale=bool(raw.get("supports_guidance_scale", False)),
        supports_reference_audio=bool(raw.get("supports_reference_audio", False)),
        supports_video=bool(raw.get("supports_video", False)),
        caption_sensitive=bool(raw.get("caption_sensitive", False)),
        max_duration_s=_optional_float(raw.get("max_duration_s"), field_name=f"{model_id}.max_duration_s"),
        sample_rate_hz=_optional_int(raw.get("sample_rate_hz"), field_name=f"{model_id}.sample_rate_hz"),
        official_8bit_available=bool(raw.get("official_8bit_available", False)),
        preferred_precision=str(raw.get("preferred_precision") or "").strip(),
        dependency_extra=str(raw["dependency_extra"]).strip() if raw.get("dependency_extra") else None,
        source_url=str(raw["source_url"]).strip() if raw.get("source_url") else None,
        notes=str(raw.get("notes") or "").strip(),
        raw=dict(raw),
    )


class MusicModelCapabilitiesRegistry:
    """Load and query packaged model capability metadata."""

    def __init__(self, *, data: Optional[Dict[str, Any]] = None) -> None:
        self._models: Dict[str, MusicModelSpec] = {}
        self._version = 1
        self._load(data if data is not None else self._load_default_data())

    @property
    def version(self) -> int:
        return int(self._version)

    def _load_default_data(self) -> Dict[str, Any]:
        with resources.files("abstractmusic").joinpath("assets/music_model_capabilities.json").open(
            "r", encoding="utf-8"
        ) as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("music model capabilities asset must contain a JSON object")
        return data

    def _load(self, data: Dict[str, Any]) -> None:
        try:
            self._version = int(data.get("version", 1))
        except Exception as e:
            raise ValueError("registry version must be an integer") from e
        raw_models = data.get("models")
        if not isinstance(raw_models, list):
            raise ValueError("registry models must be a list")
        parsed: Dict[str, MusicModelSpec] = {}
        for raw in raw_models:
            spec = _parse_model(raw)
            if spec.id in parsed:
                raise ValueError(f"duplicate model id in registry: {spec.id}")
            parsed[spec.id] = spec
        self._models = parsed

    def list_models(self, *, task: Optional[str] = None, recommended: Optional[bool] = None) -> Sequence[MusicModelSpec]:
        models: Iterable[MusicModelSpec] = self._models.values()
        if task is not None:
            models = [m for m in models if m.supports_task(str(task))]
        if recommended is not None:
            models = [m for m in models if bool(m.recommended) is bool(recommended)]
        return tuple(sorted(models, key=lambda m: (not m.recommended, m.provider.lower(), m.id.lower())))

    def get(self, model_id: str) -> MusicModelSpec:
        key = str(model_id or "").strip()
        try:
            return self._models[key]
        except KeyError as e:
            raise UnknownModelError(f"Unknown music model id: {model_id!r}") from e

    def require_support(self, model_id: str, task: str) -> MusicModelSpec:
        spec = self.get(model_id)
        if not spec.supports_task(task):
            raise CapabilityNotSupportedError(f"Model {model_id!r} does not support task {task!r}.")
        return spec
