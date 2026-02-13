import pytest


@pytest.mark.unit
def test_acestep_default_dtypes_on_mps_are_fp16():
    import torch

    from abstractmusic.backends.acestep_v15 import _default_model_dtype, _default_vae_dtype

    assert _default_model_dtype(torch, "mps") == torch.float16
    assert _default_vae_dtype(torch, "mps") == torch.float16

