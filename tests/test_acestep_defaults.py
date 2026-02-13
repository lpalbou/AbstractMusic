import pytest


@pytest.mark.unit
def test_acestep_default_infer_method_is_ode():
    from abstractmusic.backends.acestep_v15 import AceStepV15BackendConfig

    cfg = AceStepV15BackendConfig()
    assert cfg.infer_method == "ode"

