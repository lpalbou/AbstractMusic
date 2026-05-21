import base64
import json
import io
import wave

import pytest


class _Response:
    def __init__(self, data: bytes, *, content_type: str = "application/json"):
        self._data = bytes(data)
        self.headers = {"content-type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._data


@pytest.mark.unit
def test_acemusic_backend_requires_api_key(monkeypatch):
    from abstractmusic.backends.acemusic import AceMusicBackend, AceMusicBackendConfig
    from abstractmusic.errors import AbstractMusicError
    from abstractmusic.types import AudioGenerationRequest

    monkeypatch.delenv("ACEMUSIC_API_KEY", raising=False)

    backend = AceMusicBackend(config=AceMusicBackendConfig(api_key=None))

    with pytest.raises(AbstractMusicError, match="Missing ACE Music API key"):
        backend.generate_audio(AudioGenerationRequest(prompt="ambient music"))


@pytest.mark.unit
def test_acemusic_backend_decodes_data_url_and_sends_music_fields(monkeypatch):
    from abstractmusic.backends.acemusic import AceMusicBackend, AceMusicBackendConfig
    from abstractmusic.types import AudioGenerationRequest

    wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt "
    data_url = "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("ascii")
    response = {
        "model": "ace-music-test",
        "choices": [
            {
                "message": {
                    "content": "created",
                    "audio": [{"audio_url": {"url": data_url}}],
                }
            }
        ],
    }
    requests = []

    def _fake_urlopen(request, timeout):
        requests.append({"request": request, "timeout": timeout})
        return _Response(json.dumps(response).encode("utf-8"))

    monkeypatch.setattr("abstractmusic.backends.acemusic.urllib.request.urlopen", _fake_urlopen)

    backend = AceMusicBackend(
        config=AceMusicBackendConfig(
            base_url="https://api.example.test",
            api_key="secret-token",
            model="ace-music-test",
            timeout_s=42,
        )
    )
    asset = backend.generate_audio(
        AudioGenerationRequest(
            prompt="heroic fantasy instrumental",
            lyrics="[Instrumental]",
            duration_s=30,
            seed=123,
            guidance_scale=1.4,
            format="wav",
            vocal_language="en",
            extra={"bpm": 128, "keyscale": "D minor", "timesignature": "4"},
        )
    )

    assert asset.data == wav_bytes
    assert asset.mime_type == "audio/wav"
    assert asset.metadata["backend"] == "abstractmusic:acemusic"
    assert asset.metadata["remote"] is True
    assert "secret-token" not in repr(asset.metadata)

    assert requests
    request = requests[0]["request"]
    assert request.full_url == "https://api.example.test/v1/chat/completions"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer secret-token"
    payload = json.loads(request.data.decode("utf-8"))
    assert payload["messages"][0]["content"] == "<prompt>heroic fantasy instrumental</prompt>"
    assert payload["audio_config"]["duration"] == 30
    assert payload["audio_config"]["format"] == "wav"
    assert payload["audio_config"]["instrumental"] is True
    assert payload["audio_config"]["bpm"] == 128
    assert payload["audio_config"]["key_scale"] == "D minor"
    assert payload["seed"] == 123
    assert payload["guidance_scale"] == 1.4
    assert "lyrics" not in payload


@pytest.mark.unit
def test_acemusic_backend_enforces_wav_duration_when_valid_wav(monkeypatch):
    from abstractmusic.backends.acemusic import AceMusicBackend, AceMusicBackendConfig
    from abstractmusic.types import AudioGenerationRequest

    def _make_wav(*, duration_s: float, sample_rate: int = 48000) -> bytes:
        frames = int(round(float(duration_s) * float(sample_rate)))
        silence = b"\x00" * (frames * 2 * 2)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as writer:
            writer.setnchannels(2)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.writeframes(silence)
        return buf.getvalue()

    wav_bytes = _make_wav(duration_s=2.0)
    data_url = "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("ascii")
    response = {
        "model": "ace-music-test",
        "choices": [{"message": {"content": "created", "audio": [{"audio_url": {"url": data_url}}]}}],
    }

    def _fake_urlopen(request, timeout):
        return _Response(json.dumps(response).encode("utf-8"))

    monkeypatch.setattr("abstractmusic.backends.acemusic.urllib.request.urlopen", _fake_urlopen)

    backend = AceMusicBackend(
        config=AceMusicBackendConfig(
            base_url="https://api.example.test",
            api_key="secret-token",
            model="ace-music-test",
            timeout_s=42,
        )
    )
    asset = backend.generate_audio(
        AudioGenerationRequest(
            prompt="heroic fantasy instrumental",
            duration_s=1,
            format="wav",
        )
    )

    assert asset.mime_type == "audio/wav"
    report = asset.metadata.get("duration_enforcement")
    assert isinstance(report, dict)
    assert report.get("trimmed") is True
    with wave.open(io.BytesIO(asset.data), "rb") as reader:
        assert reader.getframerate() == 48000
        assert reader.getnframes() == 48000


@pytest.mark.unit
def test_acemusic_backend_does_not_wrap_prompt_in_sample_mode(monkeypatch):
    from abstractmusic.backends.acemusic import AceMusicBackend, AceMusicBackendConfig
    from abstractmusic.types import AudioGenerationRequest

    wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt "
    data_url = "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("ascii")
    response = {
        "model": "ace-music-test",
        "choices": [{"message": {"content": "created", "audio": [{"audio_url": {"url": data_url}}]}}],
    }
    requests = []

    def _fake_urlopen(request, timeout):
        requests.append(request)
        return _Response(json.dumps(response).encode("utf-8"))

    monkeypatch.setattr("abstractmusic.backends.acemusic.urllib.request.urlopen", _fake_urlopen)

    backend = AceMusicBackend(config=AceMusicBackendConfig(base_url="https://api.example.test", api_key="secret-token"))
    backend.generate_audio(
        AudioGenerationRequest(
            prompt="Generate an upbeat pop song about summer and travel",
            extra={"sample_mode": True},
            format="wav",
        )
    )

    payload = json.loads(requests[0].data.decode("utf-8"))
    assert payload["sample_mode"] is True
    assert payload["messages"][0]["content"] == "Generate an upbeat pop song about summer and travel"


@pytest.mark.unit
def test_acemusic_backend_does_not_wrap_prompt_when_lyrics_field_is_set(monkeypatch):
    from abstractmusic.backends.acemusic import AceMusicBackend, AceMusicBackendConfig
    from abstractmusic.types import AudioGenerationRequest

    wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt "
    data_url = "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("ascii")
    response = {
        "model": "ace-music-test",
        "choices": [{"message": {"content": "created", "audio": [{"audio_url": {"url": data_url}}]}}],
    }
    requests = []

    def _fake_urlopen(request, timeout):
        requests.append(request)
        return _Response(json.dumps(response).encode("utf-8"))

    monkeypatch.setattr("abstractmusic.backends.acemusic.urllib.request.urlopen", _fake_urlopen)

    backend = AceMusicBackend(config=AceMusicBackendConfig(base_url="https://api.example.test", api_key="secret-token"))
    backend.generate_audio(
        AudioGenerationRequest(
            prompt="Energetic EDM with heavy bass drops",
            lyrics="[Verse 1]\\nHello world",
            format="wav",
        )
    )

    payload = json.loads(requests[0].data.decode("utf-8"))
    assert payload["messages"][0]["content"] == "<prompt>Energetic EDM with heavy bass drops</prompt>"
    assert payload["lyrics"] == "[Verse 1]\\nHello world"


@pytest.mark.unit
def test_acemusic_backend_capabilities_are_remote_multi_format():
    from abstractmusic.backends.acemusic import AceMusicBackend

    caps = AceMusicBackend().get_capabilities()

    assert caps.model_id == "acemusic/ace-step-api"
    assert set(caps.output_formats or ()) == {"wav", "mp3", "flac"}
    assert caps.supports_lyrics is True
    assert caps.supports_guidance_scale is True
    assert caps.preferred_precision == "remote API"
