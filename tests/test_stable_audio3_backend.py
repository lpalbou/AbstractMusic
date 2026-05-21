import io
import wave

import pytest


class _FakeRuntime:
    def __init__(self, config):
        self.config = config
        self.device = "cpu"
        self.model_half = False
        self.calls = []

    def load(self):
        self.loaded = True

    def _sample_rate(self):
        return 44100

    def generate(self, **kwargs):
        self.calls.append(dict(kwargs))
        import torch

        duration_s = float(kwargs["duration_s"])
        samples = max(1, int(44100 * duration_s))
        t = torch.linspace(0, duration_s, samples)
        audio = 0.1 * torch.sin(2 * torch.pi * 220.0 * t)
        return audio.repeat(1, 2, 1)


@pytest.mark.unit
def test_stable_audio3_backend_maps_request_to_internal_runtime(monkeypatch):
    import abstractmusic.backends.stable_audio_3 as sa3
    from abstractmusic.types import AudioGenerationRequest

    runtimes = []

    class CapturingRuntime(_FakeRuntime):
        def __init__(self, config):
            super().__init__(config)
            runtimes.append(self)

    monkeypatch.setattr(sa3, "_StableAudio3Runtime", CapturingRuntime)

    backend = sa3.StableAudio3Backend(config=sa3.StableAudio3BackendConfig(device="cpu", duration_s=1.0))
    asset = backend.generate_audio(
        AudioGenerationRequest(
            prompt="fast arcade synth music",
            duration_s=1.0,
            seed=123,
            num_inference_steps=4,
            guidance_scale=1.25,
            extra={"sampler_type": "euler", "chunked_decode": "false"},
        )
    )

    assert asset.mime_type == "audio/wav"
    assert asset.metadata["backend"] == "abstractmusic:stable-audio-3"
    assert asset.metadata["runtime"] == "abstractmusic.vendor.stable_audio3_min"
    assert asset.metadata["model_id"] == "stabilityai/stable-audio-3-small-music"
    assert asset.metadata["seed"] == 123
    assert asset.metadata["num_inference_steps"] == 4
    assert asset.metadata["guidance_scale"] == 1.25
    assert asset.metadata["sampler_type"] == "euler"
    assert asset.metadata["chunked_decode"] is False
    assert runtimes[0].calls[0]["prompt"] == "fast arcade synth music"
    assert runtimes[0].calls[0]["seed"] == 123

    with wave.open(io.BytesIO(asset.data), "rb") as wf:
        assert wf.getframerate() == 44100
        assert wf.getnchannels() == 2


@pytest.mark.unit
def test_stable_audio3_backend_rejects_unsupported_inputs(monkeypatch):
    import abstractmusic.backends.stable_audio_3 as sa3
    from abstractmusic.types import AudioGenerationRequest

    monkeypatch.setattr(sa3, "_StableAudio3Runtime", _FakeRuntime)
    backend = sa3.StableAudio3Backend()

    with pytest.raises(ValueError, match="lyrics"):
        backend.generate_audio(AudioGenerationRequest(prompt="x", lyrics="words"))

    with pytest.raises(ValueError, match="negative_prompt"):
        backend.generate_audio(AudioGenerationRequest(prompt="x", negative_prompt="noise"))

    with pytest.raises(ValueError, match="up to 120"):
        backend.generate_audio(AudioGenerationRequest(prompt="x", duration_s=121))


@pytest.mark.unit
def test_stable_audio3_backend_rejects_nonfinite_audio(monkeypatch):
    import abstractmusic.backends.stable_audio_3 as sa3
    from abstractmusic.types import AudioGenerationRequest

    class BadRuntime(_FakeRuntime):
        def generate(self, **kwargs):
            import torch

            return torch.tensor([[[float("nan")]]])

    monkeypatch.setattr(sa3, "_StableAudio3Runtime", BadRuntime)
    backend = sa3.StableAudio3Backend()

    with pytest.raises(RuntimeError, match="non-finite"):
        backend.generate_audio(AudioGenerationRequest(prompt="x", duration_s=1))


@pytest.mark.unit
def test_stable_audio3_backend_capabilities():
    from abstractmusic.backends.stable_audio_3 import StableAudio3Backend

    caps = StableAudio3Backend().get_capabilities()

    assert caps.model_id == "stabilityai/stable-audio-3-small-music"
    assert caps.max_duration_s == 120.0
    assert caps.sample_rates_hz == (44100,)
    assert caps.supports_guidance_scale is True
    assert caps.supports_lyrics is False
