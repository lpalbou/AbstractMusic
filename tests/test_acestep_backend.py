import types
import wave

import pytest

from abstractmusic.types import AudioGenerationRequest


@pytest.mark.unit
def test_acestep_maps_unified_request(monkeypatch):
    from abstractmusic.backends.acestep import AceStepBackend, AceStepBackendConfig

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
        def from_pretrained(cls, model_id, torch_dtype=None, local_files_only=False):
            calls["load"] = (model_id, torch_dtype, local_files_only)
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
            guidance_scale=7.0,
            shift,
            generator=None,
        ):
            calls["call"] = {
                "prompt": prompt,
                "lyrics": lyrics,
                "audio_duration": audio_duration,
                "vocal_language": vocal_language,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "shift": shift,
                "generator_device": getattr(generator, "device", None),
                "generator_seed": getattr(generator, "seed", None),
                "tiling_enabled": self.vae.tiling_enabled,
            }
            left = [0.2 if (i // 2400) % 2 == 0 else -0.2 for i in range(48000)]
            right = list(left)
            return types.SimpleNamespace(audios=[[left, right]])

    monkeypatch.setattr("abstractmusic.backends.acestep._lazy_import_torch", lambda: fake_torch)
    monkeypatch.setattr("abstractmusic.backends.acestep._lazy_import_acestep_pipeline", lambda: _FakePipe)

    backend = AceStepBackend(
        config=AceStepBackendConfig(
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
            guidance_scale=4.0,
            seed=42,
        )
    )

    assert calls["load"] == ("ACE-Step/acestep-v15-xl-turbo-diffusers", "fp32", True)
    assert calls["device"] == "mps"
    assert calls["call"]["prompt"] == "bright synthwave"
    assert calls["call"]["lyrics"] == "[Verse]\nNeon night"
    assert calls["call"]["audio_duration"] == 3.0
    assert calls["call"]["vocal_language"] == "en"
    assert calls["call"]["num_inference_steps"] == 8
    assert calls["call"]["guidance_scale"] == 4.0
    assert calls["call"]["shift"] == 3.0
    assert calls["call"]["generator_device"] == "mps"
    assert calls["call"]["generator_seed"] == 42
    assert calls["call"]["tiling_enabled"] is True

    with wave.open(__import__("io").BytesIO(asset.data), "rb") as wf:
        assert wf.getframerate() == 48000
        assert wf.getnchannels() == 2
        assert wf.getnframes() == 48000
    assert asset.metadata["backend"] == "abstractmusic:acestep"
    assert asset.metadata["audio_stats"]["probably_noise_or_invalid"] is False
    assert asset.metadata["energy_continuity_stats"]["has_long_low_energy_gap"] is False


@pytest.mark.unit
def test_acestep_rejects_negative_prompt():
    from abstractmusic.backends.acestep import AceStepBackend, AceStepBackendConfig

    backend = AceStepBackend(config=AceStepBackendConfig())
    with pytest.raises(ValueError, match="negative_prompt"):
        backend.generate_audio(AudioGenerationRequest(prompt="x", negative_prompt="noise"))


@pytest.mark.unit
def test_acestep_accepts_batched_pipeline_audio_shape():
    np = pytest.importorskip("numpy")

    from abstractmusic.backends.acestep import _encode_wav_bytes

    audio = np.zeros((1, 2, 48000), dtype=np.float32)
    audio[0, 0, :] = 0.1
    audio[0, 1, :] = -0.1
    data = _encode_wav_bytes(audio, sample_rate=48000)

    with wave.open(__import__("io").BytesIO(data), "rb") as wf:
        assert wf.getframerate() == 48000
        assert wf.getnchannels() == 2
        assert wf.getnframes() == 48000


@pytest.mark.unit
def test_acestep_prefers_mps_bfloat16_when_supported(monkeypatch):
    from abstractmusic.backends.acestep import _resolve_dtype

    class _FakeTensor:
        dtype = "bf16"

        def __add__(self, _other):
            return self

    fake_torch = types.SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        float32="fp32",
        ones=lambda *_a, **_k: _FakeTensor(),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )

    assert _resolve_dtype(fake_torch, "auto", device="mps") == "bf16"


@pytest.mark.unit
def test_acestep_defaults_follow_checkpoint_variant():
    from abstractmusic.backends.acestep import _default_guidance_scale, _default_num_inference_steps

    assert _default_num_inference_steps("ACE-Step/Ace-Step1.5") == 8
    assert _default_guidance_scale("ACE-Step/Ace-Step1.5") == 1.0
    assert _default_num_inference_steps("ACE-Step/acestep-v15-xl-turbo-diffusers") == 8
    assert _default_guidance_scale("ACE-Step/acestep-v15-xl-turbo-diffusers") == 1.0
    assert _default_num_inference_steps("ACE-Step/acestep-v15-base") == 50
    assert _default_guidance_scale("ACE-Step/acestep-v15-base") == 7.0
    assert _default_num_inference_steps("ACE-Step/acestep-v15-sft") == 50
    assert _default_guidance_scale("ACE-Step/acestep-v15-sft") == 7.0
    # The loadable Diffusers-layout XL checkpoints follow the model card:
    # 50 steps and guidance 7.0 because they are not guidance-distilled.
    assert _default_num_inference_steps("ACE-Step/acestep-v15-xl-sft-diffusers") == 50
    assert _default_guidance_scale("ACE-Step/acestep-v15-xl-sft-diffusers") == 7.0
    assert _default_num_inference_steps("ACE-Step/acestep-v15-xl-base-diffusers") == 50
    assert _default_guidance_scale("ACE-Step/acestep-v15-xl-base-diffusers") == 7.0


@pytest.mark.unit
def test_acestep_rejects_native_layout_checkpoints_with_actionable_error():
    """Explicitly selecting a native-layout id must fail fast and name the fix,
    not surface a misleading network error from from_pretrained."""

    from abstractmusic.backends.acestep import AceStepBackend, AceStepBackendConfig
    from abstractmusic.errors import CapabilityNotSupportedError

    backend = AceStepBackend(config=AceStepBackendConfig(model_id="ACE-Step/Ace-Step1.5"))
    with pytest.raises(CapabilityNotSupportedError) as excinfo:
        backend._load_pipe()
    message = str(excinfo.value)
    assert "native ACE-Step repository layout" in message
    assert "acestep-v15-xl-turbo-diffusers" in message  # names a working alternative
    assert "connect" not in message.lower()  # not a network-shaped error


@pytest.mark.unit
def test_generic_diffusers_backend_also_rejects_native_layout_checkpoints():
    """The generic diffusers backend must not become an escape hatch that turns a
    known-incompatible checkpoint back into a misleading 404 at load time."""

    from abstractmusic.backends.diffusers_audio import DiffusersAudioBackend, DiffusersAudioBackendConfig
    from abstractmusic.errors import CapabilityNotSupportedError

    backend = DiffusersAudioBackend(
        config=DiffusersAudioBackendConfig(model_id="ACE-Step/acestep-v15-base", pipeline_class="AceStepPipeline")
    )
    with pytest.raises(CapabilityNotSupportedError) as excinfo:
        backend._load_pipe()
    assert "repository layout" in str(excinfo.value)
    assert "404" not in str(excinfo.value)


@pytest.mark.unit
def test_acestep_retries_explicit_mps_fp16_with_mps_bfloat16(monkeypatch):
    from abstractmusic.backends.acestep import AceStepBackend, AceStepBackendConfig

    loads = []

    class _FakeGenerator:
        def __init__(self, device=None):
            self.device = device

        def manual_seed(self, seed):
            self.seed = seed
            return self

    class _FakeTensor:
        dtype = "bf16"

        def __add__(self, _other):
            return self

    fake_torch = types.SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        float32="fp32",
        Generator=_FakeGenerator,
        ones=lambda *_a, **_k: _FakeTensor(),
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )

    class _FakePipe:
        sample_rate = 48000

        def __init__(self, dtype):
            self.dtype = dtype
            self.device = None
            self.vae = types.SimpleNamespace(enable_tiling=lambda: None)

        @classmethod
        def from_pretrained(cls, model_id, torch_dtype=None, local_files_only=False):
            loads.append((model_id, torch_dtype, local_files_only))
            return cls(torch_dtype)

        def to(self, device):
            self.device = device
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
            _ = prompt, lyrics, audio_duration, vocal_language, num_inference_steps, shift, generator
            if self.dtype == "fp16":
                return types.SimpleNamespace(audios=[[[float("nan")] * 48000, [float("nan")] * 48000]])
            return types.SimpleNamespace(audios=[[[0.0] * 48000, [0.0] * 48000]])

    monkeypatch.setattr("abstractmusic.backends.acestep._lazy_import_torch", lambda: fake_torch)
    monkeypatch.setattr("abstractmusic.backends.acestep._lazy_import_acestep_pipeline", lambda: _FakePipe)

    backend = AceStepBackend(
        config=AceStepBackendConfig(
            model_id="ACE-Step/acestep-v15-xl-turbo-diffusers",
            device="mps",
            torch_dtype="float16",
        )
    )
    asset = backend.generate_audio(AudioGenerationRequest(prompt="x", duration_s=1.0, seed=7))

    assert bytes(asset.data)[:4] == b"RIFF"
    assert loads == [
        ("ACE-Step/acestep-v15-xl-turbo-diffusers", "fp16", True),
        ("ACE-Step/acestep-v15-xl-turbo-diffusers", "bf16", True),
    ]
    assert asset.metadata["device"] == "mps"
    assert asset.metadata["dtype"] == "bf16"
    assert asset.metadata["fallback_events"] == ("mps_nonfinite_audio_mps_bfloat16_retry",)
