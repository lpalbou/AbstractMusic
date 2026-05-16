import io
import sys
import types
import wave

import numpy as np
import pytest


class _FakeBatch(dict):
    def to(self, device):
        self["device"] = device
        return self


@pytest.mark.unit
def test_musicgen_backend_uses_transformers_contract(monkeypatch):
    from abstractmusic.backends.musicgen import MusicGenBackend, MusicGenBackendConfig
    from abstractmusic.types import AudioGenerationRequest

    calls = {}

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id):
            calls["processor_model_id"] = model_id
            return cls()

        def __call__(self, **kwargs):
            calls["processor_call"] = kwargs
            return _FakeBatch({"input_ids": np.asarray([[1, 2, 3]])})

    class FakeAudioEncoder:
        sampling_rate = 32000

    class FakeConfig:
        audio_encoder = FakeAudioEncoder()

    class FakeModel:
        config = FakeConfig()

        @classmethod
        def from_pretrained(cls, model_id, torch_dtype=None):
            calls["model_load"] = (model_id, str(torch_dtype))
            return cls()

        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            calls["eval"] = True

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            t = np.linspace(0, 1.0, 32000, endpoint=False, dtype=np.float32)
            return np.sin(2 * np.pi * 220 * t)[None, None, :]

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoProcessor = FakeProcessor
    transformers_mod.MusicgenForConditionalGeneration = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    backend = MusicGenBackend(config=MusicGenBackendConfig(device="cpu", duration_s=2.0))
    asset = backend.generate_audio(
        AudioGenerationRequest(prompt="lo-fi music", duration_s=2.0, seed=123, guidance_scale=2.5)
    )

    assert asset.mime_type == "audio/wav"
    assert asset.metadata["backend"] == "abstractmusic:musicgen"
    assert asset.metadata["model_id"] == "facebook/musicgen-small"
    assert asset.metadata["sampling_rate"] == 32000
    assert asset.metadata["max_new_tokens"] == 100
    assert calls["processor_model_id"] == "facebook/musicgen-small"
    assert calls["processor_call"]["text"] == ["lo-fi music"]
    assert calls["generate"]["guidance_scale"] == 2.5
    assert calls["generate"]["max_new_tokens"] == 100

    with wave.open(io.BytesIO(asset.data), "rb") as wf:
        assert wf.getframerate() == 32000
        assert wf.getnchannels() == 1


@pytest.mark.unit
def test_musicgen_backend_capabilities_are_noncommercial():
    from abstractmusic.backends.musicgen import MusicGenBackend

    caps = MusicGenBackend().get_capabilities()

    assert caps.model_id == "facebook/musicgen-small"
    assert caps.commercial_allowed is False
    assert caps.supports_guidance_scale is True
