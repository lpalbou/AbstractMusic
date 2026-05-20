import pytest

from abstractmusic.backends.acestep_diffusers import AceStepDiffusersBackendConfig
from abstractmusic.backends.acestep_v15 import AceStepV15BackendConfig
from abstractmusic.backends.diffusers_audio import DiffusersAudioBackendConfig
from abstractmusic.backends.musicgen import MusicGenBackendConfig
from abstractmusic.backends.stable_audio import StableAudioBackendConfig


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda: DiffusersAudioBackendConfig(model_id="./local-model"),
            id="diffusers-audio",
        ),
        pytest.param(
            lambda: AceStepDiffusersBackendConfig(model_id="./local-model"),
            id="acestep-diffusers",
        ),
        pytest.param(
            lambda: MusicGenBackendConfig(model_id="./local-model"),
            id="musicgen",
        ),
        pytest.param(
            lambda: StableAudioBackendConfig(model_id="./local-model"),
            id="stable-audio",
        ),
        pytest.param(
            lambda: AceStepV15BackendConfig(repo_id="./local-model"),
            id="acestep-v15-primary",
        ),
        pytest.param(
            lambda: AceStepV15BackendConfig(lm_fallback_repo_id="./local-model"),
            id="acestep-v15-fallback",
        ),
    ],
)
def test_backend_configs_reject_local_model_paths(factory):
    with pytest.raises(ValueError, match="local filesystem path"):
        factory()


@pytest.mark.unit
def test_model_id_validation_rejects_existing_repo_like_local_directory(monkeypatch, tmp_path):
    model_dir = tmp_path / "facebook" / "musicgen-small"
    model_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="local filesystem path"):
        MusicGenBackendConfig(model_id="facebook/musicgen-small")
