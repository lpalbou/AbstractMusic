from contextlib import contextmanager

import pytest


@pytest.mark.unit
def test_acestep_vendored_model_filters_meta_and_init_empty_weights(monkeypatch):
    import torch
    from transformers.modeling_utils import PreTrainedModel

    from abstractmusic.vendor.acestep_v15_turbo.modeling_acestep_v15_turbo import (
        AceStepConditionGenerationModel,
    )

    @contextmanager
    def other_ctx():
        yield

    @contextmanager
    def init_empty_weights():  # matches Transformers v4 naming
        yield

    def _fake_super_get_init_context(*_a, **_k):
        return [other_ctx(), torch.device("meta"), init_empty_weights()]

    monkeypatch.setattr(PreTrainedModel, "get_init_context", classmethod(_fake_super_get_init_context))

    # The vendored override must be compatible with both Transformers v4 and v5 call conventions.
    ctxs_v5 = AceStepConditionGenerationModel.get_init_context(torch.float32, False, False)
    ctxs_v4 = AceStepConditionGenerationModel.get_init_context(False, False)

    for ctxs in (ctxs_v5, ctxs_v4):
        assert all(not (isinstance(c, torch.device) and c.type == "meta") for c in ctxs)
        assert all(getattr(getattr(c, "func", None), "__name__", "") != "init_empty_weights" for c in ctxs)
        assert any(getattr(getattr(c, "func", None), "__name__", "") == "other_ctx" for c in ctxs)

