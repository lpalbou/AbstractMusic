import io
import sys
import types
import wave

import numpy as np
import pytest


@pytest.mark.unit
def test_stable_audio_backend_uses_vendored_loader_contract(monkeypatch):
    from abstractmusic.backends.stable_audio import StableAudioBackend, StableAudioBackendConfig
    from abstractmusic.types import AudioGenerationRequest

    calls = {}

    import torch

    class FakeInner:
        def parameters(self):
            yield torch.zeros(1, dtype=torch.float32)

        def __call__(self, x, t, **kwargs):
            calls["model_call"] = {"shape": tuple(x.shape), "kwargs": kwargs}
            return torch.zeros_like(x)

    class FakeModel:
        pretransform = None
        io_channels = 2
        diffusion_objective = "rectified_flow"
        dist_shift = None
        model = FakeInner()

        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            calls["eval"] = True

        def conditioner(self, conditioning, device):
            calls["conditioning"] = (conditioning, device)
            return {"prompt": torch.ones(1)}

        def get_conditioning_inputs(self, conditioning_tensors):
            calls["conditioning_tensors"] = conditioning_tensors
            return {"cond": torch.ones(1, 1)}

    def fake_get_pretrained_model(model_id):
        calls["model_id"] = model_id
        return FakeModel(), {"sample_rate": 44100, "sample_size": 44100}

    stable_mod = types.ModuleType("abstractmusic.vendor.stable_audio_open_min")
    stable_mod.get_pretrained_model = fake_get_pretrained_model
    monkeypatch.setitem(sys.modules, "abstractmusic.vendor.stable_audio_open_min", stable_mod)

    backend = StableAudioBackend(config=StableAudioBackendConfig(device="cpu", duration_s=1.0))
    asset = backend.generate_audio(
        AudioGenerationRequest(prompt="short synth loop", duration_s=1.0, seed=321, guidance_scale=1.5)
    )

    assert asset.mime_type == "audio/wav"
    assert asset.metadata["backend"] == "abstractmusic:stable-audio"
    assert asset.metadata["model_id"] == "stabilityai/stable-audio-open-small"
    assert asset.metadata["sampling_rate"] == 44100
    assert calls["model_id"] == "stabilityai/stable-audio-open-small"
    assert calls["conditioning"][0][0]["prompt"] == "short synth loop"
    assert calls["model_call"]["kwargs"]["cfg_scale"] == 1.5
    assert calls["model_call"]["kwargs"]["batch_cfg"] is True

    with wave.open(io.BytesIO(asset.data), "rb") as wf:
        assert wf.getframerate() == 44100
        assert wf.getnchannels() == 2


@pytest.mark.unit
def test_stable_audio_backend_capabilities_are_gated_noncommercial():
    from abstractmusic.backends.stable_audio import StableAudioBackend

    caps = StableAudioBackend().get_capabilities()

    assert caps.model_id == "stabilityai/stable-audio-open-small"
    assert caps.commercial_allowed is False
    assert caps.max_duration_s == 11.0
