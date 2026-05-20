import pytest


@pytest.mark.unit
def test_acestep_default_infer_method_is_ode():
    from abstractmusic.backends.acestep_v15 import AceStepV15BackendConfig

    cfg = AceStepV15BackendConfig()
    assert cfg.infer_method == "ode"


@pytest.mark.unit
def test_acestep_default_uses_seeded_random_source_latents():
    from abstractmusic.backends.acestep_v15 import AceStepV15BackendConfig

    cfg = AceStepV15BackendConfig()
    assert cfg.use_random_src_latents is True
    assert cfg.use_audio_code_planner is False
