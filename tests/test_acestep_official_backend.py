import io
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest


def _wav_info(data: bytes) -> tuple[int, int, int]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate()


def _wav_bytes(duration_s: float = 1.0, sample_rate: int = 48_000) -> bytes:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 220 * t)
    pcm = (audio * 32767).astype("<i2")
    path = Path("/tmp/abstractmusic-official-test.wav")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    data = path.read_bytes()
    path.unlink(missing_ok=True)
    return data


@pytest.mark.unit
def test_acestep_official_backend_uses_upstream_contract(monkeypatch, tmp_path):
    from abstractmusic.backends.acestep_official import AceStepOfficialBackend, AceStepOfficialBackendConfig
    from abstractmusic.types import AudioGenerationRequest

    calls = {}

    class FakeDit:
        def initialize_service(self, **kwargs):
            calls["dit_init"] = kwargs
            return "dit ok", True

    class FakeLlm:
        llm_initialized = True

        def initialize(self, **kwargs):
            calls["llm_init"] = kwargs
            return "llm ok", True

    class FakeParams:
        def __init__(self, **kwargs):
            calls["params"] = kwargs

    class FakeConfig:
        def __init__(self, **kwargs):
            calls["config"] = kwargs

    class FakeResult:
        success = True
        extra_outputs = {"lm_metadata": {"bpm": 120}, "time_costs": {"pipeline_total_time": 1.0}}

        def __init__(self, path):
            self.audios = [{"path": str(path), "sample_rate": 48_000}]
            self.status_message = "ok"
            self.error = None

    def fake_generate_music(_dit, _llm, *, params, config, save_dir):
        _ = params, config
        out = Path(save_dir) / "out.wav"
        out.write_bytes(_wav_bytes())
        return FakeResult(out)

    handler_mod = types.ModuleType("acestep.handler")
    handler_mod.AceStepHandler = FakeDit
    llm_mod = types.ModuleType("acestep.llm_inference")
    llm_mod.LLMHandler = FakeLlm
    inference_mod = types.ModuleType("acestep.inference")
    inference_mod.GenerationParams = FakeParams
    inference_mod.GenerationConfig = FakeConfig
    inference_mod.generate_music = fake_generate_music
    package_mod = types.ModuleType("acestep")

    monkeypatch.setitem(sys.modules, "acestep", package_mod)
    monkeypatch.setitem(sys.modules, "acestep.handler", handler_mod)
    monkeypatch.setitem(sys.modules, "acestep.llm_inference", llm_mod)
    monkeypatch.setitem(sys.modules, "acestep.inference", inference_mod)

    source_dir = tmp_path / "ACE-Step"
    (source_dir / "acestep").mkdir(parents=True)
    (source_dir / "acestep" / "handler.py").write_text("", encoding="utf-8")

    backend = AceStepOfficialBackend(
        config=AceStepOfficialBackendConfig(
            source_dir=str(source_dir),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            lm_backend="mlx",
            device="auto",
        )
    )
    asset = backend.generate_audio(
        AudioGenerationRequest(
            prompt="bright melodic synth loop",
            lyrics="[Instrumental]",
            duration_s=10,
            num_inference_steps=8,
            seed=123,
            extra={"bpm": 128, "keyscale": "F# major", "timesignature": "4"},
        )
    )

    assert asset.mime_type == "audio/wav"
    assert asset.metadata["backend"] == "abstractmusic:acestep-official"
    assert asset.metadata["lm_backend"] == "mlx"
    assert calls["dit_init"]["use_mlx_dit"] is True
    assert calls["llm_init"]["lm_model_path"] == "acestep-5Hz-lm-1.7B"
    assert calls["params"]["caption"] == "bright melodic synth loop"
    assert calls["params"]["bpm"] == 128
    assert calls["params"]["keyscale"] == "F# major"
    assert calls["params"]["timesignature"] == "4"
    assert calls["params"]["guidance_scale"] == 1.0
    assert calls["params"]["shift"] == 3.0
    assert calls["params"]["infer_method"] == "ode"
    assert calls["params"]["sampler_mode"] == "euler"
    assert calls["params"]["lm_temperature"] == 0.85
    assert calls["params"]["lm_cfg_scale"] == 2.0
    assert calls["params"]["audio_cover_strength"] == 1.0
    assert calls["params"]["cover_noise_strength"] == 0.0
    assert calls["config"]["seeds"] == [123]


@pytest.mark.unit
def test_acestep_official_turbo_guidance_capability_is_disabled():
    from abstractmusic.backends.acestep_official import AceStepOfficialBackend, AceStepOfficialBackendConfig

    turbo = AceStepOfficialBackend(config=AceStepOfficialBackendConfig(dit_model="acestep-v15-turbo"))
    sft = AceStepOfficialBackend(config=AceStepOfficialBackendConfig(dit_model="acestep-v15-sft"))

    assert turbo.get_capabilities().supports_guidance_scale is False
    assert sft.get_capabilities().supports_guidance_scale is True


@pytest.mark.unit
def test_acestep_official_audio_result_prefers_tensor_pcm16(tmp_path):
    from abstractmusic.backends.acestep_official import _audio_result_to_wav_bytes

    sample_rate = 48_000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    stereo_channel_first = np.stack(
        [
            0.25 * np.sin(2 * np.pi * 220 * t),
            0.25 * np.sin(2 * np.pi * 330 * t),
        ],
        axis=0,
    ).astype(np.float32)
    upstream_file = tmp_path / "upstream-float.wav"
    upstream_file.write_bytes(b"not the output we want")

    data, source = _audio_result_to_wav_bytes(
        {
            "tensor": stereo_channel_first,
            "sample_rate": sample_rate,
            "path": str(upstream_file),
        }
    )

    assert source == "tensor_pcm16"
    assert _wav_info(data) == (2, 2, sample_rate)
