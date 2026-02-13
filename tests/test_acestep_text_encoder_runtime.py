import pytest


@pytest.mark.unit
def test_acestep_resolve_text_encoder_runtime_on_mps_uses_cpu_float32():
    import torch

    from abstractmusic.backends.acestep_v15 import _resolve_text_encoder_runtime

    dev, dtype = _resolve_text_encoder_runtime(torch, model_device="mps", model_dtype=torch.float16)
    assert dev == "cpu"
    assert dtype == torch.float32


@pytest.mark.unit
def test_acestep_resolve_text_encoder_runtime_non_mps_keeps_model_settings():
    import torch

    from abstractmusic.backends.acestep_v15 import _resolve_text_encoder_runtime

    dev, dtype = _resolve_text_encoder_runtime(torch, model_device="cpu", model_dtype=torch.float32)
    assert dev == "cpu"
    assert dtype == torch.float32

