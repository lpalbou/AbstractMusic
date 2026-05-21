import json
import urllib.error

import pytest


class _Response:
    def __init__(self, data: bytes, *, content_type: str = "audio/wav", headers=None):
        self._data = bytes(data)
        self.headers = {"content-type": content_type, **dict(headers or {})}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._data


@pytest.mark.unit
def test_elevenlabs_music_backend_requires_api_key(monkeypatch):
    from abstractmusic.backends.elevenlabs_music import ElevenLabsMusicBackend, ElevenLabsMusicBackendConfig
    from abstractmusic.errors import AbstractMusicError
    from abstractmusic.types import AudioGenerationRequest

    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    backend = ElevenLabsMusicBackend(config=ElevenLabsMusicBackendConfig(api_key=None))

    with pytest.raises(AbstractMusicError, match="Missing ElevenLabs API key"):
        backend.generate_audio(AudioGenerationRequest(prompt="ambient music"))


@pytest.mark.unit
def test_elevenlabs_music_backend_posts_prompt_music_only_request(monkeypatch):
    from abstractmusic.backends.elevenlabs_music import ElevenLabsMusicBackend, ElevenLabsMusicBackendConfig
    from abstractmusic.types import AudioGenerationRequest

    wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt "
    requests = []

    def _fake_urlopen(request, timeout):
        requests.append({"request": request, "timeout": timeout})
        return _Response(wav_bytes, headers={"song-id": "song-123"})

    monkeypatch.setattr("abstractmusic.backends.elevenlabs_music.urllib.request.urlopen", _fake_urlopen)

    backend = ElevenLabsMusicBackend(
        config=ElevenLabsMusicBackendConfig(
            base_url="https://api.example.test",
            api_key="secret-token",
            timeout_s=42,
        )
    )
    asset = backend.generate_audio(
        AudioGenerationRequest(
            prompt="cinematic instrumental cue",
            lyrics="[Instrumental]",
            duration_s=3,
            format="wav",
            extra={"instrumental": True},
        )
    )

    assert asset.data == wav_bytes
    assert asset.mime_type == "audio/wav"
    assert asset.metadata["backend"] == "abstractmusic:elevenlabs-music"
    assert asset.metadata["music_only"] is True
    assert asset.metadata["song_id"] == "song-123"
    assert "secret-token" not in repr(asset.metadata)

    request = requests[0]["request"]
    assert request.full_url == "https://api.example.test/v1/music"
    assert request.get_method() == "POST"
    assert request.get_header("Xi-api-key") == "secret-token"
    payload = json.loads(request.data.decode("utf-8"))
    assert payload["prompt"] == "cinematic instrumental cue"
    assert payload["music_length_ms"] == 3000
    assert payload["force_instrumental"] is True
    assert payload["model_id"] == "music_v1"
    assert "composition_plan" not in payload


@pytest.mark.unit
def test_elevenlabs_music_backend_uses_composition_plan_for_lyrics_and_negative_prompt(monkeypatch):
    from abstractmusic.backends.elevenlabs_music import ElevenLabsMusicBackend, ElevenLabsMusicBackendConfig
    from abstractmusic.types import AudioGenerationRequest

    requests = []

    def _fake_urlopen(request, timeout):
        requests.append({"request": request, "timeout": timeout})
        return _Response(b"ID3\x03\x00\x00music-bytes", content_type="audio/mpeg")

    monkeypatch.setattr("abstractmusic.backends.elevenlabs_music.urllib.request.urlopen", _fake_urlopen)

    backend = ElevenLabsMusicBackend(
        config=ElevenLabsMusicBackendConfig(
            base_url="https://api.example.test",
            api_key="secret-token",
            output_format="mp3_44100_128",
        )
    )
    asset = backend.generate_audio(
        AudioGenerationRequest(
            prompt="upbeat synth pop",
            lyrics="[Verse]\nMoving through the city lights\n[Chorus]\nWe rise tonight",
            negative_prompt="dark, slow",
            duration_s=12,
            seed=123,
            format="mp3",
            extra={"bpm": 124, "keyscale": "A minor", "timesignature": "4"},
        )
    )

    assert asset.mime_type == "audio/mpeg"
    assert asset.metadata["composition_mode"] == "plan"
    request = requests[0]["request"]
    assert request.full_url == "https://api.example.test/v1/music?output_format=mp3_44100_128"
    payload = json.loads(request.data.decode("utf-8"))
    assert "prompt" not in payload
    assert payload["seed"] == 123
    plan = payload["composition_plan"]
    assert "124 BPM" in plan["positive_global_styles"]
    assert "A minor" in plan["positive_global_styles"]
    assert plan["negative_global_styles"] == ["dark", "slow"]
    assert [s["section_name"] for s in plan["sections"]] == ["Verse", "Chorus"]
    assert plan["sections"][0]["lines"] == ["Moving through the city lights"]


@pytest.mark.unit
def test_elevenlabs_music_backend_capabilities_are_music_only_remote():
    from abstractmusic.backends.elevenlabs_music import ElevenLabsMusicBackend

    caps = ElevenLabsMusicBackend().get_capabilities()
    models = ElevenLabsMusicBackend().list_provider_models(task="text_to_music")

    assert caps.model_id == "music_v1"
    assert set(caps.output_formats or ()) == {"wav", "mp3"}
    assert caps.supports_lyrics is True
    assert caps.supports_negative_prompt is True
    assert caps.supports_guidance_scale is False
    assert caps.preferred_precision == "remote hosted API"
    assert models and models[0].id == "music_v1"
    assert "composition_plan" in models[0].capabilities


@pytest.mark.unit
def test_elevenlabs_music_backend_does_not_silently_drop_lyrics_in_forced_prompt_mode():
    from abstractmusic.backends.elevenlabs_music import ElevenLabsMusicBackend, ElevenLabsMusicBackendConfig
    from abstractmusic.errors import CapabilityNotSupportedError
    from abstractmusic.types import AudioGenerationRequest

    backend = ElevenLabsMusicBackend(
        config=ElevenLabsMusicBackendConfig(api_key="secret-token", composition_mode="prompt")
    )

    with pytest.raises(CapabilityNotSupportedError, match="explicit lyrics"):
        backend.generate_audio(
            AudioGenerationRequest(
                prompt="pop song",
                lyrics="[Verse]\nHello",
                duration_s=3,
            )
        )


@pytest.mark.unit
def test_elevenlabs_music_backend_explains_free_tier_limited_access(monkeypatch):
    from abstractmusic.backends.elevenlabs_music import ElevenLabsMusicBackend, ElevenLabsMusicBackendConfig
    from abstractmusic.errors import AbstractMusicError
    from abstractmusic.types import AudioGenerationRequest

    class _ErrBody:
        def read(self):
            return json.dumps(
                {
                    "detail": {
                        "status": "limited_access",
                        "message": "Music API is not available for free users.",
                    }
                }
            ).encode("utf-8")

    def _fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 402, "Payment Required", {}, _ErrBody())

    monkeypatch.setattr("abstractmusic.backends.elevenlabs_music.urllib.request.urlopen", _fake_urlopen)

    backend = ElevenLabsMusicBackend(config=ElevenLabsMusicBackendConfig(api_key="secret-token"))

    with pytest.raises(AbstractMusicError, match="Music-enabled paid plan"):
        backend.generate_audio(AudioGenerationRequest(prompt="ambient music", duration_s=3))
