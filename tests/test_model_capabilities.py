import os
import subprocess
import sys

import pytest


@pytest.mark.unit
def test_music_model_registry_contains_reviewed_models():
    from abstractmusic.model_capabilities import MusicModelCapabilitiesRegistry

    reg = MusicModelCapabilitiesRegistry()
    ids = {m.id for m in reg.list_models()}

    assert "ACE-Step/Ace-Step1.5" in ids
    assert "ACE-Step/acestep-v15-xl-turbo-diffusers" in ids
    assert "HeartMuLa/HeartMuLa-oss-3B-happy-new-year" in ids
    assert "m-a-p/YuE-s1-7B-anneal-en-cot" in ids
    assert "LH-Tech-AI/TinyMozart_v2_85M" in ids
    assert "Dalision/Omni2Sound" in ids


@pytest.mark.unit
def test_heartmula_registry_metadata():
    from abstractmusic.model_capabilities import MusicModelCapabilitiesRegistry

    spec = MusicModelCapabilitiesRegistry().get("HeartMuLa/HeartMuLa-oss-3B-happy-new-year")

    assert spec.supports_task("text_to_music")
    assert spec.license == "Apache-2.0"
    assert spec.commercial_allowed is True
    assert spec.supports_lyrics is True
    assert spec.official_8bit_available is False
    assert "HeartCodec" in spec.notes


@pytest.mark.unit
def test_yue_registry_metadata():
    from abstractmusic.model_capabilities import MusicModelCapabilitiesRegistry

    spec = MusicModelCapabilitiesRegistry().get("m-a-p/YuE-s1-7B-anneal-en-cot")

    assert spec.supports_task("text_to_music")
    assert spec.supports_task("lyrics_to_music")
    assert spec.license == "Apache-2.0"
    assert spec.commercial_allowed is True
    assert spec.supports_lyrics is True
    assert spec.official_8bit_available is False
    assert spec.dependency_extra == "yue"
    assert "not an end-to-end audio checkpoint" in spec.notes


@pytest.mark.unit
def test_acestep_v15_registry_recommends_official_backend_not_custom_path():
    from abstractmusic.model_capabilities import MusicModelCapabilitiesRegistry

    spec = MusicModelCapabilitiesRegistry().get("ACE-Step/Ace-Step1.5")

    assert spec.recommended is True
    assert spec.status == "official-backend-implemented-custom-invalid"
    assert "acestep-official" in spec.backend_kinds
    assert "rotor-like" in spec.notes
    assert "custom backend" in spec.notes


@pytest.mark.unit
def test_registry_rejects_unsupported_task():
    from abstractmusic.errors import CapabilityNotSupportedError
    from abstractmusic.model_capabilities import MusicModelCapabilitiesRegistry

    reg = MusicModelCapabilitiesRegistry()
    with pytest.raises(CapabilityNotSupportedError):
        reg.require_support("LH-Tech-AI/TinyMozart_v2_85M", "text_to_music")


@pytest.mark.unit
def test_import_abstractmusic_does_not_import_heavy_stacks():
    env = dict(os.environ)
    src = os.path.abspath("src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys; import abstractmusic; "
        "assert 'torch' not in sys.modules; "
        "assert 'diffusers' not in sys.modules; "
        "assert 'transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.unit
def test_pyproject_keeps_heavy_runtime_deps_out_of_base():
    import tomllib
    from pathlib import Path

    data = tomllib.loads(Path("pyproject.toml").read_text())
    deps = {str(d).split(">=")[0].split("<")[0].split("[")[0].lower() for d in data["project"].get("dependencies", [])}
    assert "torch" not in deps
    assert "diffusers" not in deps
    assert "transformers" not in deps
    assert "numpy" not in deps

    extras = data["project"]["optional-dependencies"]
    for extra in ["acestep", "acestep-official", "acestep-diffusers", "diffusers", "local", "apple", "gpu", "yue"]:
        assert extra in extras
    assert any(str(dep).startswith("torch") for dep in extras["acestep"])
    assert any(str(dep).startswith("mlx-lm") for dep in extras["apple"])
    assert any(str(dep).startswith("diffusers") for dep in extras["acestep-diffusers"])
