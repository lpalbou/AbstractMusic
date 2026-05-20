"""ACE Music remote backend.

This backend intentionally uses only the Python standard library so the base
`abstractmusic` package can remain lightweight and remote-capable.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from ..errors import AbstractMusicError, CapabilityNotSupportedError
from ..types import AudioGenerationRequest, GeneratedAsset, MusicBackendCapabilities, ProviderModelInfo


_DEFAULT_BASE_URL = "https://api.acemusic.ai"


def _env(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(str(name), "")
        text = str(value).strip()
        if text:
            return text
    return None


def _join_url(base_url: str, path: str) -> str:
    return f"{str(base_url).rstrip('/')}/{str(path).lstrip('/')}"


def _json_request(
    url: str,
    *,
    api_key: Optional[str],
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout_s: float = 300.0,
) -> Dict[str, Any]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=True).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "User-Agent": "abstractmusic/remote",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(str(url), data=body, headers=headers, method=str(method).upper())
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:  # nosec B310 - user-selected remote API.
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        raise AbstractMusicError(f"ACE Music API request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise AbstractMusicError(f"ACE Music API request failed: {exc}") from exc

    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise AbstractMusicError("ACE Music API returned non-JSON response.") from exc
    if not isinstance(parsed, dict):
        raise AbstractMusicError("ACE Music API returned an unexpected JSON payload.")
    return parsed


def _decode_data_url(value: str) -> Tuple[bytes, str]:
    text = str(value or "").strip()
    if not text.startswith("data:") or "," not in text:
        raise ValueError("not a data URL")
    header, encoded = text.split(",", 1)
    mime = header[5:].split(";", 1)[0].strip() or "application/octet-stream"
    return base64.b64decode(encoded), mime


def _download_audio_url(url: str, *, timeout_s: float) -> Tuple[bytes, str]:
    request = urllib.request.Request(str(url), headers={"User-Agent": "abstractmusic/remote"}, method="GET")
    with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:  # nosec B310 - URL is API-provided audio output.
        data = response.read()
        mime = response.headers.get("content-type") or "application/octet-stream"
    return bytes(data), str(mime).split(";", 1)[0]


def _mime_for_format(fmt: str) -> str:
    value = str(fmt or "").strip().lower()
    if value == "wav":
        return "audio/wav"
    if value == "mp3":
        return "audio/mpeg"
    if value == "flac":
        return "audio/flac"
    if value == "ogg":
        return "audio/ogg"
    return "application/octet-stream"


@dataclass(frozen=True)
class AceMusicBackendConfig:
    """Configuration for ACE Music's OpenAI-compatible remote music API."""

    base_url: str = _DEFAULT_BASE_URL
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout_s: float = 600.0
    temperature: float = 0.85
    top_p: float = 0.9
    sample_mode: bool = False
    use_format: bool = False
    use_cot_caption: bool = True
    use_cot_language: bool = True
    thinking: bool = False
    batch_size: int = 1

    @classmethod
    def from_env(cls) -> "AceMusicBackendConfig":
        return cls(
            base_url=_env("ACEMUSIC_BASE_URL") or _DEFAULT_BASE_URL,
            api_key=_env("ACEMUSIC_API_KEY"),
        )


class AceMusicBackend:
    """Remote text-to-music backend for ACE Music's hosted ACE-Step API."""

    backend_id = "abstractmusic:acemusic"

    def __init__(self, *, config: Optional[AceMusicBackendConfig] = None) -> None:
        self._config = config or AceMusicBackendConfig.from_env()

    @property
    def config(self) -> AceMusicBackendConfig:
        return self._config

    def _api_key(self) -> str:
        key = self._config.api_key or _env("ACEMUSIC_API_KEY")
        if not key:
            raise AbstractMusicError(
                "Missing ACE Music API key. Set ACEMUSIC_API_KEY. "
                "Keys can be created from https://acemusic.ai/playground/api-key."
            )
        return str(key)

    def get_capabilities(self) -> MusicBackendCapabilities:
        return MusicBackendCapabilities(
            supported_tasks=("text_to_music",),
            output_formats=("wav", "mp3", "flac"),
            supports_lyrics=True,
            supports_negative_prompt=False,
            supports_guidance_scale=True,
            supports_reference_audio=False,
            supports_video=False,
            max_duration_s=300.0,
            sample_rates_hz=None,
            model_id=self._config.model or "acemusic/ace-step-api",
            license=None,
            commercial_allowed=None,
            official_8bit_available=False,
            preferred_precision="remote API",
        )

    def list_provider_models(self, *, task: Optional[str] = None) -> Sequence[ProviderModelInfo]:
        if task is not None and str(task) not in {"text_to_music", "music", "t2m"}:
            return ()
        try:
            payload = _json_request(
                _join_url(self._config.base_url, "/v1/models"),
                api_key=self._api_key(),
                timeout_s=min(float(self._config.timeout_s), 30.0),
            )
        except Exception:
            return ()
        models = payload.get("data")
        if not isinstance(models, list):
            return ()
        out = []
        for item in models:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id") or "").strip()
            if not model_id:
                continue
            out.append(
                ProviderModelInfo(
                    id=model_id,
                    object=str(item.get("object")) if item.get("object") is not None else None,
                    created=int(item["created"]) if isinstance(item.get("created"), int) else None,
                    owned_by=str(item.get("owned_by")) if item.get("owned_by") is not None else "ACE Music",
                    capabilities=("text_to_music",),
                    raw=dict(item),
                )
            )
        return tuple(out)

    def generate_audio(self, request: AudioGenerationRequest) -> GeneratedAsset:
        fmt = str(request.format or "wav").strip().lower() or "wav"
        if fmt not in {"wav", "mp3", "flac"}:
            raise CapabilityNotSupportedError("ACE Music remote backend supports format='wav', 'mp3', or 'flac'.")
        if request.negative_prompt:
            raise CapabilityNotSupportedError("ACE Music remote backend does not support negative_prompt.")

        extra = dict(request.extra or {})
        audio_config: Dict[str, Any] = {"format": fmt}
        if request.duration_s is not None:
            audio_config["duration"] = float(request.duration_s)
        if request.vocal_language:
            audio_config["vocal_language"] = str(request.vocal_language)
        if extra.get("bpm") is not None:
            audio_config["bpm"] = int(extra["bpm"])
        if extra.get("keyscale") is not None:
            audio_config["key_scale"] = str(extra["keyscale"])
        if extra.get("timesignature") is not None:
            audio_config["time_signature"] = str(extra["timesignature"])
        instrumental = bool(extra.get("instrumental", False)) or str(request.lyrics or "").strip() == "[Instrumental]"
        if instrumental:
            audio_config["instrumental"] = True

        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": str(request.prompt or "")}],
            "stream": False,
            "audio_config": audio_config,
            "temperature": float(self._config.temperature),
            "top_p": float(self._config.top_p),
            "sample_mode": bool(extra.get("sample_mode", self._config.sample_mode)),
            "thinking": bool(extra.get("thinking", self._config.thinking)),
            "use_format": bool(extra.get("use_format", self._config.use_format)),
            "use_cot_caption": bool(extra.get("use_cot_caption", self._config.use_cot_caption)),
            "use_cot_language": bool(extra.get("use_cot_language", self._config.use_cot_language)),
            "task_type": str(extra.get("task_type") or "text2music"),
            "batch_size": max(1, int(extra.get("batch_size", self._config.batch_size))),
        }
        if self._config.model:
            payload["model"] = str(self._config.model)
        if request.lyrics and not instrumental:
            payload["lyrics"] = str(request.lyrics)
        if request.seed is not None:
            payload["seed"] = int(request.seed)
        if request.guidance_scale is not None:
            payload["guidance_scale"] = float(request.guidance_scale)

        response = _json_request(
            _join_url(self._config.base_url, "/v1/chat/completions"),
            api_key=self._api_key(),
            method="POST",
            payload=payload,
            timeout_s=float(self._config.timeout_s),
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AbstractMusicError(f"ACE Music API returned no choices: {response!r}")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise AbstractMusicError(f"ACE Music API returned no message: {response!r}")
        audios_raw = message.get("audio")
        if isinstance(audios_raw, dict):
            audios = [audios_raw]
        else:
            audios = audios_raw
        if not isinstance(audios, list) or not audios:
            raise AbstractMusicError(f"ACE Music API returned no audio payload: {response!r}")
        first = audios[0]
        if not isinstance(first, dict):
            raise AbstractMusicError("ACE Music API returned malformed audio payload.")
        audio_url = first.get("audio_url")
        if isinstance(audio_url, dict):
            url = str(audio_url.get("url") or "")
        else:
            url = str(first.get("url") or "")
        if not url:
            raise AbstractMusicError("ACE Music API returned audio without a URL.")

        if url.startswith("data:"):
            data, mime = _decode_data_url(url)
        elif url.startswith(("https://", "http://")):
            data, mime = _download_audio_url(url, timeout_s=float(self._config.timeout_s))
        else:
            raise AbstractMusicError("ACE Music API returned unsupported audio URL format.")

        metadata = {
            "backend": self.backend_id,
            "provider": "ACE Music",
            "remote": True,
            "model": self._config.model or response.get("model") or "acemusic/ace-step-api",
            "base_url": self._config.base_url,
            "format": fmt,
            "mime_type": mime or _mime_for_format(fmt),
            "audio_count": len(audios),
            "api_content": message.get("content"),
        }
        return GeneratedAsset(
            data=bytes(data),
            mime_type=str(mime or _mime_for_format(fmt)),
            media_type="audio",
            metadata=metadata,
        )
