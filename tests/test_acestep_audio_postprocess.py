import pytest


@pytest.mark.unit
def test_remove_dc_offset_tensor_channelwise():
    import torch

    from abstractmusic.backends.acestep_v15 import _remove_dc_offset

    # Shape [B, C, T] with strong per-channel DC bias.
    t = torch.linspace(0, 1, 1000, dtype=torch.float32)
    wav = torch.stack(
        [
            0.6 + 0.01 * torch.sin(2 * torch.pi * 5 * t),
            -0.2 + 0.02 * torch.cos(2 * torch.pi * 3 * t),
        ],
        dim=0,
    ).unsqueeze(0)  # [1, 2, T]

    centered = _remove_dc_offset(wav)
    means = centered.mean(dim=-1)
    assert torch.all(torch.abs(means) < 1e-5)

