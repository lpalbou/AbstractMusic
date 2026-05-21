from pathlib import Path

import pytest


@pytest.mark.unit
def test_stable_audio3_backend_does_not_import_upstream_runtime():
    paths = [
        Path("src/abstractmusic/backends/stable_audio_3.py"),
        *Path("src/abstractmusic/vendor/stable_audio3_min").rglob("*.py"),
    ]

    joined = "\n".join(path.read_text() for path in paths)

    forbidden = [
        "import stable_audio_3",
        "from stable_audio_3",
        "import stable_audio_tools",
        "from stable_audio_tools",
        "git+https://github.com/Stability-AI/stable-audio-3",
        "import flash_attn",
        "from flash_attn",
        "import torchaudio",
        "from torchaudio",
        "import soundfile",
        "from soundfile",
        "import tqdm",
        "from tqdm",
        "import packaging",
        "from packaging",
    ]
    for pattern in forbidden:
        assert pattern not in joined


@pytest.mark.unit
def test_stable_audio3_runtime_package_is_internal():
    import abstractmusic.vendor.stable_audio3_min.factory as factory

    assert factory.__name__.startswith("abstractmusic.vendor.stable_audio3_min")
