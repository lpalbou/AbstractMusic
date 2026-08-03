"""ElevenLabs Music remote backend.

This backend intentionally uses only the Python standard library so the base
`abstractmusic` package remains lightweight. It only calls ElevenLabs Music
endpoints; text-to-speech and voice APIs belong in AbstractVoice.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..availability import RemoteEndpoint
from ..errors import AbstractMusicError, CapabilityNotSupportedError
from ..prompt_planner import is_instrumental_lyrics
from ..types import (
    AudioGenerationRequest,
    GeneratedAsset,
    MusicBackendCapabilities,
    MusicCompositionPlan,
    MusicSectionPlan,
    ProviderModelInfo,
)


_DEFAULT_BASE_URL = "https://api.elevenlabs.io"
_DEFAULT_MODEL = "music_v1"
_DEFAULT_MP3_OUTPUT_FORMAT = "mp3_44100_128"
_MAX_PROMPT_CHARS = 4100


def _env(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(str(name), "")
        text = str(value).strip()
        if text:
            return text
    return None


def _join_url(base_url: str, path: str) -> str:
    return f"{str(base_url).rstrip('/')}/{str(path).lstrip('/')}"


def _mime_for_format(fmt: str) -> str:
    value = str(fmt or "").strip().lower()
    if value == "wav":
        return "audio/wav"
    if value == "mp3":
        return "audio/mpeg"
    return "application/octet-stream"


def _sniff_mime(data: bytes, fallback: str) -> str:
    raw = bytes(data[:16])
    if raw.startswith(b"RIFF") and raw[8:12] == b"WAVE":
        return "audio/wav"
    if raw.startswith(b"ID3") or (len(raw) >= 2 and raw[0] == 0xFF and raw[1] & 0xE0 == 0xE0):
        return "audio/mpeg"
    return str(fallback or "application/octet-stream")


def _coerce_audio_bytes(raw: bytes) -> bytes:
    """Accept raw audio bytes or the base64 string shape used in API examples."""

    data = bytes(raw or b"")
    stripped = data.strip()
    if not stripped:
        return data
    if (stripped.startswith(b'"') and stripped.endswith(b'"')) or (
        stripped.startswith(b"'") and stripped.endswith(b"'")
    ):
        try:
            decoded = json.loads(stripped.decode("utf-8"))
            if isinstance(decoded, str):
                stripped = decoded.encode("ascii")
        except Exception:
            pass
    if re.fullmatch(rb"[A-Za-z0-9+/=\s]+", stripped) and len(stripped) > 24:
        try:
            maybe = base64.b64decode(stripped, validate=False)
            if maybe.startswith(b"RIFF") or maybe.startswith(b"ID3"):
                return maybe
        except Exception:
            pass
    return data


def _json_payload(raw: bytes) -> Any:
    return json.loads(bytes(raw).decode("utf-8"))


def _extract_error_detail(raw: bytes) -> str:
    text = bytes(raw or b"").decode("utf-8", errors="replace").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except Exception:
        return text
    detail = parsed.get("detail") if isinstance(parsed, dict) else None
    if isinstance(detail, dict):
        status = detail.get("status")
        message = detail.get("message") or detail.get("detail")
        data = detail.get("data")
        pieces = [str(v) for v in (status, message) if v]
        if isinstance(data, dict):
            suggestion = data.get("prompt_suggestion") or data.get("composition_plan_suggestion")
            if suggestion:
                pieces.append(f"suggestion={suggestion!r}")
        return "; ".join(pieces) or json.dumps(detail, ensure_ascii=True)
    if isinstance(detail, str):
        return detail
    return text


def _request(
    url: str,
    *,
    api_key: Optional[str],
    method: str = "POST",
    payload: Optional[Dict[str, Any]] = None,
    accept: str = "*/*",
    timeout_s: float = 600.0,
) -> Tuple[bytes, Dict[str, str]]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=True).encode("utf-8")
    headers = {
        "Accept": accept,
        "User-Agent": "abstractmusic/elevenlabs-music",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["xi-api-key"] = str(api_key)
    request = urllib.request.Request(str(url), data=body, headers=headers, method=str(method).upper())
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:  # nosec B310 - user-selected API.
            raw = response.read()
            response_headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
    except urllib.error.HTTPError as exc:
        raw = b""
        try:
            raw = exc.read()
        except Exception:
            pass
        detail = _extract_error_detail(raw) or str(exc)
        if int(exc.code) == 402 and "limited_access" in detail:
            detail = (
                f"{detail.rstrip('.')}. ElevenLabs Music API access is not available on this account tier; "
                "use a Music-enabled paid plan, ACE Music, or a local backend."
            )
        raise AbstractMusicError(f"ElevenLabs Music API request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise AbstractMusicError(f"ElevenLabs Music API request failed: {exc}") from exc
    return bytes(raw), response_headers


def _duration_ms(duration_s: Optional[float]) -> Optional[int]:
    if duration_s is None:
        return None
    ms = int(round(float(duration_s) * 1000.0))
    return max(3000, min(600000, ms))


def _dedupe(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return out


def _list_str(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _split_negative_prompt(value: Optional[str]) -> List[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return _dedupe(re.split(r"[,;\n]+", value))


def _split_lyrics_sections(lyrics: Optional[str]) -> List[Tuple[str, List[str]]]:
    if not isinstance(lyrics, str) or not lyrics.strip() or is_instrumental_lyrics(lyrics):
        return []
    sections: List[Tuple[str, List[str]]] = []
    current_name = "Verse"
    current_lines: List[str] = []
    for raw_line in lyrics.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        label = re.fullmatch(r"\[([^\]]+)\]", line)
        if label:
            if current_lines:
                sections.append((current_name, current_lines))
                current_lines = []
            current_name = label.group(1).strip() or "Section"
            continue
        current_lines.append(line)
    if current_lines:
        sections.append((current_name, current_lines))
    return sections


def _section_durations(total_ms: Optional[int], count: int) -> List[int]:
    count = max(1, int(count))
    total = int(total_ms or 30000)
    total = max(3000 * count, min(600000, total))
    base = total // count
    durations = [max(3000, base) for _ in range(count)]
    durations[-1] += total - sum(durations)
    return durations


def _default_instrumental_sections(total_ms: Optional[int]) -> List[Tuple[str, List[str]]]:
    total = int(total_ms or 30000)
    if total <= 20000:
        return [("Main Theme", ["primary motif", "clear groove", "complete ending"])]
    if total <= 60000:
        return [
            ("Intro", ["establish motif", "rhythmic pulse"]),
            ("Build", ["add layers", "stronger drums"]),
            ("Climax", ["full arrangement", "highest energy"]),
            ("Outro", ["resolve theme", "clean ending"]),
        ]
    return [
        ("Intro", ["atmospheric setup", "recognizable motif"]),
        ("Theme A", ["main groove", "memorable hook"]),
        ("Development", ["variation", "new counter melody"]),
        ("Bridge", ["contrast", "reduced texture"]),
        ("Climax", ["full energy", "expanded harmony"]),
        ("Outro", ["final motif", "natural ending"]),
    ]


def _fallback_composition_plan(request: AudioGenerationRequest) -> Dict[str, Any]:
    extra = dict(request.extra or {})
    total_ms = _duration_ms(request.duration_s)
    instrumental = bool(extra.get("instrumental", False)) or is_instrumental_lyrics(request.lyrics)

    positive_global = [str(request.prompt or "").strip()]
    if extra.get("bpm") is not None:
        positive_global.append(f"{int(extra['bpm'])} BPM")
    if extra.get("keyscale") is not None:
        positive_global.append(str(extra["keyscale"]))
    if extra.get("timesignature") is not None:
        positive_global.append(f"{extra['timesignature']} time signature")
    if instrumental:
        positive_global.append("instrumental")

    negative_global = _split_negative_prompt(request.negative_prompt)
    if instrumental:
        negative_global.extend(["vocals", "lyrics"])

    lyric_sections = _split_lyrics_sections(request.lyrics)
    section_specs = lyric_sections or _default_instrumental_sections(total_ms)
    durations = _section_durations(total_ms, len(section_specs))
    sections = []
    for idx, (name, lines_or_styles) in enumerate(section_specs):
        if lyric_sections:
            positive_styles = [str(name), "clear vocal phrasing"]
            lines = list(lines_or_styles)
        else:
            positive_styles = list(lines_or_styles)
            lines = []
        sections.append(
            {
                "section_name": str(name),
                "positive_local_styles": _dedupe(positive_styles),
                "negative_local_styles": [],
                "duration_ms": int(durations[idx]),
                "lines": lines,
            }
        )

    return {
        "positive_global_styles": _dedupe(positive_global),
        "negative_global_styles": _dedupe(negative_global),
        "sections": sections,
    }


def _normalize_section(section: Any) -> Dict[str, Any]:
    if isinstance(section, MusicSectionPlan):
        raw = section.to_dict()
    elif isinstance(section, Mapping):
        raw = dict(section)
    else:
        raw = dict(section)

    name = raw.get("section_name") or raw.get("name") or "Section"
    positive = raw.get("positive_local_styles")
    if positive is None:
        positive = raw.get("positive_styles")
    if positive is None:
        positive = raw.get("styles")
    negative = raw.get("negative_local_styles")
    if negative is None:
        negative = raw.get("negative_styles")
    return {
        "section_name": str(name).strip() or "Section",
        "positive_local_styles": _dedupe(positive if isinstance(positive, Sequence) and not isinstance(positive, str) else [positive]),
        "negative_local_styles": _dedupe(negative if isinstance(negative, Sequence) and not isinstance(negative, str) else [negative]),
        "duration_ms": int(raw["duration_ms"]) if raw.get("duration_ms") is not None else 10000,
        "lines": _list_str(raw.get("lines")),
    }


def _normalize_composition_plan(plan: Any) -> Dict[str, Any]:
    if isinstance(plan, MusicCompositionPlan):
        raw = plan.to_dict()
    elif isinstance(plan, Mapping):
        raw = dict(plan)
    else:
        raw = dict(plan)

    # Already in ElevenLabs-native shape.
    if "positive_global_styles" in raw or "negative_global_styles" in raw:
        positive = raw.get("positive_global_styles") or []
        negative = raw.get("negative_global_styles") or []
    else:
        positive = raw.get("positive_styles") or raw.get("styles") or []
        negative = raw.get("negative_styles") or []

    sections = raw.get("sections") or []
    if not isinstance(sections, Sequence) or isinstance(sections, (str, bytes)):
        raise ValueError("composition_plan.sections must be a sequence")
    return {
        "positive_global_styles": _dedupe(positive if isinstance(positive, Sequence) and not isinstance(positive, str) else [positive]),
        "negative_global_styles": _dedupe(negative if isinstance(negative, Sequence) and not isinstance(negative, str) else [negative]),
        "sections": [_normalize_section(section) for section in sections],
    }


@dataclass(frozen=True)
class ElevenLabsMusicBackendConfig:
    """Configuration for ElevenLabs Music API generation."""

    base_url: str = _DEFAULT_BASE_URL
    api_key: Optional[str] = None
    model: str = _DEFAULT_MODEL
    timeout_s: float = 600.0
    output_format: Optional[str] = None
    mp3_output_format: str = _DEFAULT_MP3_OUTPUT_FORMAT
    composition_mode: str = "auto"
    respect_sections_durations: bool = True
    store_for_inpainting: bool = False
    sign_with_c2pa: bool = False

    @classmethod
    def from_env(cls) -> "ElevenLabsMusicBackendConfig":
        return cls(
            base_url=_env("ELEVENLABS_BASE_URL") or _DEFAULT_BASE_URL,
            api_key=_env("ELEVENLABS_API_KEY"),
        )

    def health_endpoint(self) -> Optional[RemoteEndpoint]:
        """Return a cheap read-only endpoint proving this API answers, if configured."""

        if not self.api_key:
            return None
        return RemoteEndpoint(
            url=_join_url(self.base_url, "/v1/models"),
            headers={"xi-api-key": str(self.api_key)},
        )


class ElevenLabsMusicBackend:
    """Remote text-to-music backend for ElevenLabs Music only."""

    backend_id = "abstractmusic:elevenlabs-music"

    def __init__(self, *, config: Optional[ElevenLabsMusicBackendConfig] = None) -> None:
        self._config = config or ElevenLabsMusicBackendConfig.from_env()

    @property
    def config(self) -> ElevenLabsMusicBackendConfig:
        return self._config

    def _api_key(self) -> str:
        key = self._config.api_key or _env("ELEVENLABS_API_KEY")
        if not key:
            raise AbstractMusicError("Missing ElevenLabs API key. Set ELEVENLABS_API_KEY.")
        return str(key)

    def get_capabilities(self) -> MusicBackendCapabilities:
        return MusicBackendCapabilities(
            supported_tasks=("text_to_music",),
            output_formats=("wav", "mp3"),
            supports_lyrics=True,
            supports_negative_prompt=True,
            supports_guidance_scale=False,
            supports_reference_audio=False,
            supports_video=False,
            max_duration_s=600.0,
            sample_rates_hz=None,
            model_id=self._config.model or _DEFAULT_MODEL,
            license=None,
            commercial_allowed=None,
            official_8bit_available=False,
            preferred_precision="remote hosted API",
        )

    def list_provider_models(self, *, task: Optional[str] = None) -> Sequence[ProviderModelInfo]:
        if task is not None and str(task) not in {"text_to_music", "music", "t2m"}:
            return ()
        return (
            ProviderModelInfo(
                id=_DEFAULT_MODEL,
                object="model",
                owned_by="ElevenLabs",
                capabilities=("text_to_music", "composition_plan"),
                raw={"provider": self.backend_id, "music_only": True},
            ),
        )

    def create_composition_plan(
        self,
        prompt: str,
        *,
        duration_s: Optional[float] = None,
        source_composition_plan: Optional[Any] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "prompt": str(prompt or "")[:_MAX_PROMPT_CHARS],
            "model_id": self._config.model or _DEFAULT_MODEL,
        }
        ms = _duration_ms(duration_s)
        if ms is not None:
            payload["music_length_ms"] = ms
        if source_composition_plan is not None:
            payload["source_composition_plan"] = _normalize_composition_plan(source_composition_plan)
        raw, _headers = _request(
            _join_url(self._config.base_url, "/v1/music/plan"),
            api_key=self._api_key(),
            payload=payload,
            accept="application/json",
            timeout_s=min(float(self._config.timeout_s), 120.0),
        )
        parsed = _json_payload(raw)
        if not isinstance(parsed, dict):
            raise AbstractMusicError("ElevenLabs Music plan endpoint returned an unexpected payload.")
        return _normalize_composition_plan(parsed)

    def _output_format_param(self, fmt: str) -> Optional[str]:
        if self._config.output_format:
            return str(self._config.output_format).strip()
        if str(fmt).lower() == "mp3":
            return str(self._config.mp3_output_format or _DEFAULT_MP3_OUTPUT_FORMAT)
        # Leave WAV/default unspecified. The Music API's default example is a RIFF/WAVE payload,
        # while PCM output_format values are raw samples and not a WAV container.
        return None

    def _request_payload(self, request: AudioGenerationRequest) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        extra = dict(request.extra or {})
        mode = str(extra.get("composition_mode") or self._config.composition_mode or "auto").strip().lower()
        composition_plan = request.composition_plan if request.composition_plan is not None else extra.get("composition_plan")
        instrumental = bool(extra.get("instrumental", False)) or is_instrumental_lyrics(request.lyrics)
        warnings: List[str] = []

        use_plan = composition_plan is not None or mode in {"plan", "composition", "composition_plan"}
        if mode == "auto" and (request.negative_prompt or (request.lyrics and not instrumental)):
            use_plan = True

        payload: Dict[str, Any] = {
            "model_id": self._config.model or _DEFAULT_MODEL,
        }
        if use_plan:
            payload["composition_plan"] = (
                _normalize_composition_plan(composition_plan)
                if composition_plan is not None
                else _fallback_composition_plan(request)
            )
            payload["respect_sections_durations"] = bool(
                extra.get("respect_sections_durations", self._config.respect_sections_durations)
            )
            if request.seed is not None:
                payload["seed"] = int(request.seed)
        else:
            if request.negative_prompt:
                raise CapabilityNotSupportedError(
                    "ElevenLabs Music negative_prompt requires composition_mode='auto' or 'plan'."
                )
            if request.lyrics and not instrumental:
                raise CapabilityNotSupportedError(
                    "ElevenLabs Music explicit lyrics require composition_mode='auto' or 'plan'."
                )
            payload["prompt"] = str(request.prompt or "")[:_MAX_PROMPT_CHARS]
            ms = _duration_ms(request.duration_s)
            if ms is not None:
                payload["music_length_ms"] = int(ms)
            if instrumental:
                payload["force_instrumental"] = True
            if request.seed is not None:
                warnings.append("seed_ignored_in_elevenlabs_prompt_mode")

        if bool(extra.get("store_for_inpainting", self._config.store_for_inpainting)):
            payload["store_for_inpainting"] = True
        if bool(extra.get("sign_with_c2pa", self._config.sign_with_c2pa)):
            payload["sign_with_c2pa"] = True
        return payload, {"warnings": tuple(warnings), "composition_mode": "plan" if use_plan else "prompt"}

    def generate_audio(self, request: AudioGenerationRequest) -> GeneratedAsset:
        fmt = str(request.format or "wav").strip().lower() or "wav"
        if fmt not in {"wav", "mp3"}:
            raise CapabilityNotSupportedError("ElevenLabs Music backend supports format='wav' or 'mp3'.")
        if request.guidance_scale is not None:
            raise CapabilityNotSupportedError("ElevenLabs Music backend does not support guidance_scale.")

        payload, payload_meta = self._request_payload(request)
        output_format = self._output_format_param(fmt)
        url = _join_url(self._config.base_url, "/v1/music")
        if output_format:
            url = f"{url}?{urllib.parse.urlencode({'output_format': output_format})}"

        raw, headers = _request(
            url,
            api_key=self._api_key(),
            payload=payload,
            accept="audio/*,application/octet-stream,*/*",
            timeout_s=float(self._config.timeout_s),
        )
        data = _coerce_audio_bytes(raw)
        header_mime = str(headers.get("content-type") or "").split(";", 1)[0].strip()
        mime = _sniff_mime(data, header_mime or _mime_for_format(fmt))
        metadata = {
            "backend": self.backend_id,
            "provider": self.backend_id,
            "remote": True,
            "music_only": True,
            "model": self._config.model or _DEFAULT_MODEL,
            "base_url": self._config.base_url,
            "format": fmt,
            "mime_type": mime,
            "elevenlabs_output_format": output_format,
            "song_id": headers.get("song-id"),
            **payload_meta,
        }
        return GeneratedAsset(data=bytes(data), mime_type=mime, media_type="audio", metadata=metadata)
