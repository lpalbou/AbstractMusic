import types

import pytest


@pytest.mark.unit
def test_diffusers_backend_mps_fallback_reloads_cpu_float32(monkeypatch, capsys):
    from abstractmusic.backends.diffusers_audio import DiffusersAudioBackend, DiffusersAudioBackendConfig
    from abstractmusic.types import AudioGenerationRequest

    # --- Fake torch (avoid importing real torch in unit tests) ---
    class _FakeGen:
        def __init__(self, device=None):
            self.device = device
            self.seed = None

        def manual_seed(self, seed: int):
            self.seed = int(seed)
            return self

    fake_torch = types.SimpleNamespace(
        float16="float16",
        float32="float32",
        bfloat16="bfloat16",
        Generator=_FakeGen,
    )

    monkeypatch.setattr("abstractmusic.backends.diffusers_audio._lazy_import_torch", lambda: fake_torch)

    # --- Fake pipes (simulate MPS kernel limitation then CPU success) ---
    class _PipeMPS:
        def __call__(  # signature matters (inspect.signature is used)
            self,
            prompt,
            *,
            num_inference_steps=None,
            guidance_scale=None,
            negative_prompt=None,
            generator=None,
            audio_length_in_s=None,
            audio_end_in_s=None,
        ):
            _ = prompt, num_inference_steps, guidance_scale, negative_prompt, generator, audio_length_in_s, audio_end_in_s
            raise NotImplementedError("Output channels > 65536 not supported at the MPS device.")

    class _Out:
        def __init__(self, audios):
            self.audios = audios

    class _PipeCPU:
        def __call__(
            self,
            prompt,
            *,
            num_inference_steps=None,
            guidance_scale=None,
            negative_prompt=None,
            generator=None,
            audio_length_in_s=None,
            audio_end_in_s=None,
        ):
            _ = prompt, num_inference_steps, guidance_scale, negative_prompt, generator, audio_length_in_s, audio_end_in_s
            # 2 samples of mono audio
            return _Out(audios=[[0.0, 0.0]])

    backend = DiffusersAudioBackend(
        config=DiffusersAudioBackendConfig(model_id="dummy", device="mps", torch_dtype="float16", num_inference_steps=1, duration_s=1.0)
    )

    calls = []

    def _fake_load_pipe():
        # Route based on current config to simulate reload behavior.
        calls.append((backend._config.device, backend._config.torch_dtype))
        if backend._config.device == "mps":
            backend._pipe_device = "mps"
            return _PipeMPS()
        backend._pipe_device = "cpu"
        return _PipeCPU()

    monkeypatch.setattr(backend, "_load_pipe", _fake_load_pipe)

    asset = backend.generate_audio(AudioGenerationRequest(prompt="sci fi music", num_inference_steps=1, duration_s=1.0, seed=0))

    # Fallback should switch config to CPU float32 and succeed.
    assert backend._config.device == "cpu"
    assert backend._config.torch_dtype == "float32"
    assert calls[0] == ("mps", "float16")
    assert calls[-1] == ("cpu", "float32")

    # Output is WAV bytes
    assert asset.mime_type == "audio/wav"
    assert isinstance(asset.data, (bytes, bytearray))
    assert bytes(asset.data)[:4] == b"RIFF"

    captured = capsys.readouterr()
    assert "WARNING #FALLBACK" in (captured.err or "")

