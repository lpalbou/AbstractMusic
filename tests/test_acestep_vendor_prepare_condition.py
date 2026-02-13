import pytest


@pytest.mark.unit
def test_vendored_prepare_condition_skips_lm_hints_when_not_cover():
    import torch

    from abstractmusic.vendor.acestep_v15_turbo.modeling_acestep_v15_turbo import (
        AceStepConditionGenerationModel,
    )

    class _DummyModel:
        def __init__(self):
            self.tokenize_called = False
            self.detokenize_called = False

            def _encoder(**_kwargs):
                return torch.zeros((1, 3, 8), dtype=torch.float16), torch.ones((1, 3), dtype=torch.float16)

            self.encoder = _encoder

            def _tokenize(*_args, **_kwargs):
                self.tokenize_called = True
                raise AssertionError("tokenize() must not be called when is_covers is all zeros")

            self.tokenize = _tokenize

            def _detokenize(*_args, **_kwargs):
                self.detokenize_called = True
                raise AssertionError("detokenize() must not be called when is_covers is all zeros")

            self.detokenize = _detokenize

    dummy = _DummyModel()

    src_latents = torch.zeros((1, 25, 64), dtype=torch.float16)
    chunk_masks = torch.ones((1, 25, 64), dtype=torch.float16)
    is_covers = torch.zeros((1,), dtype=torch.long)

    _, _, context_latents = AceStepConditionGenerationModel.prepare_condition(
        dummy,  # invoke method on a lightweight fake instead of constructing the full model
        text_hidden_states=torch.zeros((1, 6, 1024), dtype=torch.float16),
        text_attention_mask=torch.ones((1, 6), dtype=torch.float16),
        lyric_hidden_states=torch.zeros((1, 5, 1024), dtype=torch.float16),
        lyric_attention_mask=torch.ones((1, 5), dtype=torch.float16),
        refer_audio_acoustic_hidden_states_packed=torch.zeros((1, 750, 64), dtype=torch.float16),
        refer_audio_order_mask=torch.tensor([0], dtype=torch.long),
        hidden_states=src_latents,
        attention_mask=torch.ones((1, 25), dtype=torch.float16),
        silence_latent=torch.zeros((1, 25, 64), dtype=torch.float16),
        src_latents=src_latents,
        chunk_masks=chunk_masks,
        is_covers=is_covers,
    )

    assert dummy.tokenize_called is False
    assert dummy.detokenize_called is False
    assert context_latents.shape == (1, 25, 128)
    assert torch.equal(context_latents[..., :64], src_latents)

