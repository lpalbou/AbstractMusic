"""
Pytest bootstrap for AbstractMusic.

This repo uses a `src/` layout. For local test runs (without an editable install),
we add `src/` to `sys.path` so `import abstractmusic` works reliably.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def pytest_configure() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if src.exists() and src.is_dir():
        sys.path.insert(0, str(src))


@pytest.fixture(autouse=True)
def hermetic_availability(request, monkeypatch, tmp_path):
    """Keep provider discovery off the network and out of the developer's model cache.

    Tests that need a cached model or a reachable provider say so explicitly, by
    populating this cache root or patching the probe. Integration tests are left
    alone: they download and load real weights, and must use the real cache.
    """

    from abstractmusic import availability

    if request.node.get_closest_marker("integration"):
        yield None
        return

    cache_root = tmp_path / "hf-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))
    for key in ("HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(
        availability,
        "probe_endpoint",
        lambda endpoint, timeout_s=0.0: availability.RemoteProbe(
            status="unreachable", detail="network disabled in tests"
        ),
    )
    availability.clear_probe_cache()
    yield cache_root
    availability.clear_probe_cache()


@pytest.fixture
def cache_model(hermetic_availability):
    """Materialize a repo in the hermetic cache using the real Hugging Face layout."""

    def _cache(repo_id: str, *, weight_name: str = "model.safetensors") -> Path:
        repo_dir = hermetic_availability / ("models--" + repo_id.replace("/", "--"))
        blob = repo_dir / "blobs" / ("0" * 40)
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(b"\0")
        snapshot = repo_dir / "snapshots" / ("f" * 40)
        snapshot.mkdir(parents=True, exist_ok=True)
        # Snapshot entries are symlinks into blobs/, exactly as huggingface_hub writes them.
        link = snapshot / weight_name
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(blob)
        return repo_dir

    return _cache
