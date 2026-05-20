import base64
import json

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
def test_acemusic_backend_requires_api_key():
    from abstractmusic.backends.acemusic import AceMusicBackend, AceMusicBackendConfig
    from abstractmusic.errors import AbstractMusicError
    from abstractmusic.types import AudioGenerationRequest

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
    assert payload["messages"][0]["content"] == "heroic fantasy instrumental"
    assert payload["audio_config"]["duration"] == 30
    assert payload["audio_config"]["format"] == "wav"
    assert payload["audio_config"]["instrumental"] is True
    assert payload["audio_config"]["bpm"] == 128
    assert payload["audio_config"]["key_scale"] == "D minor"
    assert payload["seed"] == 123
    assert payload["guidance_scale"] == 1.4
    assert "lyrics" not in payload


@pytest.mark.unit
def test_acemusic_backend_capabilities_are_remote_multi_format():
    from abstractmusic.backends.acemusic import AceMusicBackend

    caps = AceMusicBackend().get_capabilities()

    assert caps.model_id == "acemusic/ace-step-api"
    assert set(caps.output_formats or ()) == {"wav", "mp3", "flac"}
    assert caps.supports_lyrics is True
    assert caps.supports_guidance_scale is True
    assert caps.preferred_precision == "remote API"
