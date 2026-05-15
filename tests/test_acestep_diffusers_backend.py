import types
import wave

import pytest

from abstractmusic.types import AudioGenerationRequest


@pytest.mark.unit
def test_acestep_diffusers_maps_unified_request(monkeypatch):
    from abstractmusic.backends.acestep_diffusers import AceStepDiffusersBackend, AceStepDiffusersBackendConfig

    calls = {}

    class _FakeGenerator:
        def __init__(self, device=None):
            self.device = device
            self.seed = None

        def manual_seed(self, seed):
            self.seed = seed
            return self

    fake_torch = types.SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        float32="fp32",
        Generator=_FakeGenerator,
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )

    class _FakeVae:
        def __init__(self):
            self.tiling_enabled = False

        def enable_tiling(self):
            self.tiling_enabled = True

    class _FakePipe:
        sample_rate = 48000

        def __init__(self):
            self.vae = _FakeVae()
            self.device = None

        @classmethod
        def from_pretrained(cls, model_id, torch_dtype=None):
            calls["load"] = (model_id, torch_dtype)
            return cls()

        def to(self, device):
            self.device = device
            calls["device"] = device
            return self

        def __call__(
            self,
            *,
            prompt,
            lyrics,
            audio_duration,
            vocal_language,
            num_inference_steps,
            shift,
            generator=None,
        ):
            calls["call"] = {
                "prompt": prompt,
                "lyrics": lyrics,
                "audio_duration": audio_duration,
                "vocal_language": vocal_language,
                "num_inference_steps": num_inference_steps,
                "shift": shift,
                "generator_device": getattr(generator, "device", None),
                "generator_seed": getattr(generator, "seed", None),
                "tiling_enabled": self.vae.tiling_enabled,
            }
            left = [0.2 if (i // 2400) % 2 == 0 else -0.2 for i in range(48000)]
            right = list(left)
            return types.SimpleNamespace(audios=[[left, right]])

    monkeypatch.setattr("abstractmusic.backends.acestep_diffusers._lazy_import_torch", lambda: fake_torch)
    monkeypatch.setattr("abstractmusic.backends.acestep_diffusers._lazy_import_acestep_pipeline", lambda: _FakePipe)

    backend = AceStepDiffusersBackend(
        config=AceStepDiffusersBackendConfig(
            model_id="ACE-Step/acestep-v15-xl-turbo-diffusers",
            device="auto",
            torch_dtype="auto",
            duration_s=5.0,
        )
    )
    asset = backend.generate_audio(
        AudioGenerationRequest(
            prompt="bright synthwave",
            lyrics="[Verse]\nNeon night",
            vocal_language="en",
            duration_s=3.0,
            num_inference_steps=8,
            seed=42,
        )
    )

    assert calls["load"] == ("ACE-Step/acestep-v15-xl-turbo-diffusers", "fp16")
    assert calls["device"] == "mps"
    assert calls["call"]["prompt"] == "bright synthwave"
    assert calls["call"]["lyrics"] == "[Verse]\nNeon night"
    assert calls["call"]["audio_duration"] == 3.0
    assert calls["call"]["vocal_language"] == "en"
    assert calls["call"]["num_inference_steps"] == 8
    assert calls["call"]["shift"] == 3.0
    assert calls["call"]["generator_device"] == "mps"
    assert calls["call"]["generator_seed"] == 42
    assert calls["call"]["tiling_enabled"] is True

    with wave.open(__import__("io").BytesIO(asset.data), "rb") as wf:
        assert wf.getframerate() == 48000
        assert wf.getnchannels() == 2
        assert wf.getnframes() == 48000
    assert asset.metadata["backend"] == "abstractmusic:acestep-diffusers"
    assert asset.metadata["audio_stats"]["probably_noise_or_invalid"] is False


@pytest.mark.unit
def test_acestep_diffusers_rejects_negative_prompt_and_guidance():
    from abstractmusic.backends.acestep_diffusers import AceStepDiffusersBackend, AceStepDiffusersBackendConfig

    backend = AceStepDiffusersBackend(config=AceStepDiffusersBackendConfig())
    with pytest.raises(ValueError, match="negative_prompt"):
        backend.generate_audio(AudioGenerationRequest(prompt="x", negative_prompt="noise"))
    with pytest.raises(ValueError, match="guidance"):
        backend.generate_audio(AudioGenerationRequest(prompt="x", guidance_scale=3.0))
