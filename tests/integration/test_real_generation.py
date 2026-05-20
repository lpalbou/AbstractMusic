import os
from pathlib import Path

import pytest

from abstractmusic.audio_analysis import inspect_music_signal_file
from abstractmusic.types import AudioGenerationRequest


pytestmark = pytest.mark.integration


def _enabled() -> bool:
    return str(os.environ.get("ABSTRACTMUSIC_RUN_REAL_MODEL_TESTS", "")).strip().lower() in {"1", "true", "yes", "on"}


def _artifact_dir() -> Path:
    return Path(os.environ.get("ABSTRACTMUSIC_TEST_ARTIFACT_DIR", "test-artifacts/real-generation")).expanduser()


@pytest.mark.skipif(not _enabled(), reason="set ABSTRACTMUSIC_RUN_REAL_MODEL_TESTS=1 to run real model smoke tests")
def test_real_generation_wav_is_valid_and_music_like():
    backend_kind = str(os.environ.get("ABSTRACTMUSIC_REAL_BACKEND", "acestep")).strip().lower()
    model_id = str(os.environ.get("ABSTRACTMUSIC_REAL_MODEL_ID", "")).strip()
    device = str(os.environ.get("ABSTRACTMUSIC_REAL_DEVICE", "auto")).strip() or "auto"
    dtype = str(os.environ.get("ABSTRACTMUSIC_REAL_DTYPE", "auto")).strip() or "auto"
    duration = float(os.environ.get("ABSTRACTMUSIC_REAL_DURATION_S", "10.0"))
    steps_raw = os.environ.get("ABSTRACTMUSIC_REAL_STEPS")
    steps = int(steps_raw) if steps_raw else None

    if backend_kind in {"acestep", "acestep-v15", "legacy"}:
        from abstractmusic.backends.acestep_v15 import AceStepV15Backend, AceStepV15BackendConfig

        backend = AceStepV15Backend(
            config=AceStepV15BackendConfig(
                repo_id=model_id or "ACE-Step/Ace-Step1.5",
                device=device,
                torch_dtype=dtype,
                vae_torch_dtype=dtype,
                default_duration_s=duration,
                fix_nfe=steps or 8,
            )
        )
    elif backend_kind == "acestep-diffusers":
        from abstractmusic.backends.acestep_diffusers import AceStepDiffusersBackend, AceStepDiffusersBackendConfig

        backend = AceStepDiffusersBackend(
            config=AceStepDiffusersBackendConfig(
                model_id=model_id or "ACE-Step/acestep-v15-xl-turbo-diffusers",
                device=device,
                torch_dtype=dtype,
                duration_s=duration,
                num_inference_steps=steps or 8,
            )
        )
    else:
        from abstractmusic.backends.diffusers_audio import DiffusersAudioBackend, DiffusersAudioBackendConfig

        if not model_id:
            pytest.fail("ABSTRACTMUSIC_REAL_MODEL_ID is required for ABSTRACTMUSIC_REAL_BACKEND=diffusers")
        backend = DiffusersAudioBackend(
            config=DiffusersAudioBackendConfig(
                model_id=model_id,
                device=device,
                torch_dtype=dtype,
                duration_s=duration,
                num_inference_steps=steps or 50,
            )
        )

    asset = backend.generate_audio(
        AudioGenerationRequest(
            prompt="upbeat synthwave instrumental, bright melody, steady drums",
            duration_s=duration,
            num_inference_steps=steps,
            seed=123,
        )
    )

    out_dir = _artifact_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{backend_kind.replace(':', '_')}_smoke.wav"
    out_path.write_bytes(asset.data)

    stats = inspect_music_signal_file(out_path)
    wav = stats.wav
    assert wav.sample_rate_hz > 0
    assert wav.channels in {1, 2}
    assert wav.frames > 0
    assert duration * 0.5 <= wav.duration_s <= max(duration * 1.5, duration + 1.0)
    assert wav.peak > 0.001
    assert wav.rms > 0.0001
    assert abs(wav.dc_offset) < 0.10
    assert wav.clipped_ratio < 0.05
    assert not wav.is_probably_noise_or_invalid
    assert stats.has_harmonic_structure, stats
    assert not stats.is_probably_repetitive_noise, stats
    assert stats.is_probably_music_like, stats
