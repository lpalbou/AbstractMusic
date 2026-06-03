import os
import subprocess
import sys

import pytest


@pytest.mark.unit
def test_music_model_registry_contains_reviewed_models():
    from abstractmusic.model_capabilities import MusicModelCapabilitiesRegistry

    reg = MusicModelCapabilitiesRegistry()
    ids = {m.id for m in reg.list_models()}

    assert "acemusic/ace-step-api" in ids
    assert "elevenlabs/music_v1" in ids
    assert "ACE-Step/Ace-Step1.5" in ids
    assert "ACE-Step/acestep-v15-base" in ids
    assert "ACE-Step/acestep-v15-sft" in ids
    assert "ACE-Step/acestep-v15-xl-turbo-diffusers" in ids
    assert "facebook/musicgen-small" in ids
    assert "stabilityai/stable-audio-open-small" in ids
    assert "stabilityai/stable-audio-3-small-music" in ids
    assert "stabilityai/stable-audio-3-medium" in ids
    assert "HeartMuLa/HeartMuLa-oss-3B-happy-new-year" in ids
    assert "m-a-p/YuE-s1-7B-anneal-en-cot" in ids
    assert "LH-Tech-AI/TinyMozart_v2_85M" in ids
    assert "Dalision/Omni2Sound" in ids


@pytest.mark.unit
def test_acemusic_registry_metadata_tracks_light_remote_default():
    from abstractmusic.model_capabilities import MusicModelCapabilitiesRegistry

    spec = MusicModelCapabilitiesRegistry().get("acemusic/ace-step-api")

    assert spec.recommended is True
    assert spec.backend_kinds[0] == "acemusic"
    assert spec.dependency_extra == "remote"
    assert spec.supports_lyrics is True
    assert spec.supports_guidance_scale is True
    assert set(spec.output_formats) == {"wav", "mp3", "flac"}
    assert spec.raw["remote"] is True
    assert spec.raw["local"] is False


@pytest.mark.unit
def test_elevenlabs_registry_metadata_tracks_music_only_remote_backend():
    from abstractmusic.model_capabilities import MusicModelCapabilitiesRegistry

    spec = MusicModelCapabilitiesRegistry().get("elevenlabs/music_v1")

    assert spec.recommended is True
    assert spec.backend_kinds[0] == "elevenlabs"
    assert spec.dependency_extra == "remote"
    assert spec.supports_lyrics is True
    assert spec.supports_negative_prompt is True
    assert spec.supports_guidance_scale is False
    assert set(spec.output_formats) == {"wav", "mp3"}
    assert spec.raw["remote"] is True
    assert spec.raw["local"] is False
    assert "text-to-speech" in spec.notes


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
def test_musicgen_small_registry_metadata():
    from abstractmusic.model_capabilities import MusicModelCapabilitiesRegistry

    spec = MusicModelCapabilitiesRegistry().get("facebook/musicgen-small")

    assert spec.supports_task("text_to_music")
    assert spec.license == "CC-BY-NC-4.0"
    assert spec.commercial_allowed is False
    assert spec.sample_rate_hz == 32000
    assert spec.supports_guidance_scale is True
    assert spec.dependency_extra == "musicgen"
    assert "300M" in spec.notes


@pytest.mark.unit
def test_stable_audio_registry_metadata():
    from abstractmusic.model_capabilities import MusicModelCapabilitiesRegistry

    spec = MusicModelCapabilitiesRegistry().get("stabilityai/stable-audio-open-small")

    assert not spec.supports_task("text_to_music")
    assert spec.supports_task("text_to_audio")
    assert spec.commercial_allowed is False
    assert spec.max_duration_s == 11
    assert spec.supports_guidance_scale is True
    assert spec.dependency_extra == "stable-audio"
    assert "Hugging Face access approval" in spec.notes


@pytest.mark.unit
def test_stable_audio3_registry_metadata():
    from abstractmusic.model_capabilities import MusicModelCapabilitiesRegistry

    spec = MusicModelCapabilitiesRegistry().get("stabilityai/stable-audio-3-small-music")

    assert spec.supports_task("text_to_music")
    assert spec.supports_task("text_to_audio")
    assert spec.backend_kinds[0] == "stable-audio-3"
    assert spec.dependency_extra == "stable-audio-3"
    assert spec.sample_rate_hz == 44100
    assert spec.max_duration_s == 120
    assert spec.supports_guidance_scale is True
    assert spec.supports_lyrics is False
    assert "AbstractMusic owns the inference runtime" in spec.notes

    medium = MusicModelCapabilitiesRegistry().get("stabilityai/stable-audio-3-medium")
    assert medium.max_duration_s == 380
    assert medium.status == "configured-unvalidated-gated-heavy"


@pytest.mark.unit
def test_acestep_registry_tracks_supported_route():
    from abstractmusic.model_capabilities import MusicModelCapabilitiesRegistry

    spec = MusicModelCapabilitiesRegistry().get("ACE-Step/acestep-v15-xl-turbo-diffusers")

    assert spec.recommended is True
    assert spec.default_for_backend is True
    assert spec.status == "validated-mps-bf16-cpu-fallback"
    assert spec.backend_kinds[0] == "acestep"
    assert spec.dependency_extra == "acestep"
    assert "Public `acestep` catalog entry" in spec.notes

    turbo = MusicModelCapabilitiesRegistry().get("ACE-Step/Ace-Step1.5")
    assert turbo.backend_kinds[0] == "acestep"
    assert turbo.dependency_extra == "acestep"
    assert turbo.status == "official-pipeline-compatible-unvalidated"

    base = MusicModelCapabilitiesRegistry().get("ACE-Step/acestep-v15-base")
    assert base.backend_kinds[0] == "acestep"
    assert base.supports_guidance_scale is True

    sft = MusicModelCapabilitiesRegistry().get("ACE-Step/acestep-v15-sft")
    assert sft.backend_kinds[0] == "acestep"
    assert sft.supports_guidance_scale is True


@pytest.mark.unit
def test_acestep_catalog_does_not_surface_unreviewed_community_conversions():
    from abstractmusic.model_capabilities import MusicModelCapabilitiesRegistry

    ids = {m.id for m in MusicModelCapabilitiesRegistry().list_models()}
    assert not any(model_id.startswith("Runware/acestep-") for model_id in ids)


@pytest.mark.unit
def test_registry_tracks_default_models_for_cli_engines():
    from abstractmusic.model_capabilities import MusicModelCapabilitiesRegistry

    reg = MusicModelCapabilitiesRegistry()
    defaults: dict[str, list[str]] = {}
    for spec in reg.list_models():
        if not getattr(spec, "default_for_backend", False):
            continue
        for kind in spec.backend_kinds:
            defaults.setdefault(str(kind), []).append(str(spec.id))

    assert defaults.get("acemusic") == ["acemusic/ace-step-api"]
    assert defaults.get("elevenlabs") == ["elevenlabs/music_v1"]
    assert defaults.get("acestep") == ["ACE-Step/acestep-v15-xl-turbo-diffusers"]
    assert defaults.get("musicgen") == ["facebook/musicgen-small"]
    assert defaults.get("stable-audio") == ["stabilityai/stable-audio-open-small"]
    assert defaults.get("stable-audio-3") == ["stabilityai/stable-audio-3-small-music"]


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
    from pathlib import Path

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
        import tomli as tomllib  # type: ignore[no-redef]

    data = tomllib.loads(Path("pyproject.toml").read_text())
    deps = {str(d).split(">=")[0].split("<")[0].split("[")[0].lower() for d in data["project"].get("dependencies", [])}
    assert "torch" not in deps
    assert "diffusers" not in deps
    assert "transformers" not in deps
    assert "numpy" not in deps

    extras = data["project"]["optional-dependencies"]
    for extra in [
        "acestep",
        "diffusers",
        "remote",
        "local",
        "apple",
        "gpu",
        "all-apple",
        "all-gpu",
        "musicgen",
        "stable-audio",
        "stable-audio-3",
        "yue",
    ]:
        assert extra in extras
    assert any(str(dep).startswith("torch") for dep in extras["acestep"])
    assert any(str(dep).startswith("mlx-lm") for dep in extras["apple"])
    assert any(str(dep).startswith("diffusers") for dep in extras["acestep"])
    assert extras["remote"] == []
    assert any(str(dep).startswith("torchaudio") for dep in extras["all-apple"])
    assert any(str(dep).startswith("alias-free-torch") for dep in extras["all-gpu"])
    assert any(str(dep).startswith("vector-quantize-pytorch") for dep in extras["all-gpu"])
    assert any(str(dep).startswith("transformers>=5.8") for dep in extras["stable-audio-3"])
    assert not any("stable-audio-3" in str(dep) for dep in extras["stable-audio-3"])
    assert not any(str(dep).startswith("torchaudio") for dep in extras["stable-audio-3"])
    assert not any(str(dep).startswith("soundfile") for dep in extras["stable-audio-3"])
    assert not any(str(dep).startswith("flash-attn") for dep in extras["stable-audio-3"])
    assert not any(str(dep).startswith("tqdm") for dep in extras["stable-audio-3"])
    assert not any(str(dep).startswith("packaging") for dep in extras["stable-audio-3"])
