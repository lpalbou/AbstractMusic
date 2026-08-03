"""Tests for the dependency-free availability probes."""

from __future__ import annotations

import json
import threading
import time
import urllib.error

import pytest

from abstractmusic import availability
from abstractmusic.availability import (
    RemoteEndpoint,
    RemoteProbe,
    hf_cache_root,
    is_model_cached,
    cached_model_ids,
    probe_endpoint,
    probe_endpoints,
)


# --------------------------------------------------------------------------- #
# Cache root resolution
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_cache_root_follows_huggingface_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "explicit"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "legacy"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    assert hf_cache_root() == tmp_path / "explicit"

    monkeypatch.delenv("HF_HUB_CACHE")
    assert hf_cache_root() == tmp_path / "legacy"

    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE")
    assert hf_cache_root() == tmp_path / "home" / "hub"

    monkeypatch.delenv("HF_HOME")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert hf_cache_root() == tmp_path / "xdg" / "huggingface" / "hub"


@pytest.mark.unit
def test_cache_root_ignores_blank_environment_values(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", "   ")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    assert hf_cache_root() == tmp_path / "home" / "hub"


# --------------------------------------------------------------------------- #
# Local model presence
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_cached_model_is_detected_from_real_cache_layout(cache_model):
    cache_model("ACE-Step/Ace-Step1.5")
    assert is_model_cached("ACE-Step/Ace-Step1.5") is True
    assert is_model_cached("ACE-Step/not-downloaded") is False


@pytest.mark.unit
def test_weights_in_subdirectories_count(cache_model, hermetic_availability):
    cache_model("stabilityai/stable-audio-3-small-music", weight_name="transformer/model.safetensors")
    assert is_model_cached("stabilityai/stable-audio-3-small-music") is True


@pytest.mark.unit
def test_empty_or_metadata_only_snapshot_is_not_cached(hermetic_availability):
    repo = hermetic_availability / "models--ACE-Step--empty"
    snapshot = repo / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    assert is_model_cached("ACE-Step/empty") is False


@pytest.mark.unit
def test_dangling_weight_symlink_is_not_cached(cache_model, hermetic_availability):
    repo = cache_model("ACE-Step/interrupted")
    next(iter((repo / "blobs").iterdir())).unlink()
    assert is_model_cached("ACE-Step/interrupted") is False


@pytest.mark.unit
def test_leftover_incomplete_blob_does_not_hide_complete_weights(cache_model):
    """huggingface_hub keeps `.incomplete` blobs so downloads can resume.

    They accumulate for files nobody needs; treating them as repo-wide damage
    would make a perfectly usable provider vanish with no explanation.
    """

    repo = cache_model("ACE-Step/usable")
    (repo / "blobs" / "unrelated-file.incomplete").write_bytes(b"partial")
    assert is_model_cached("ACE-Step/usable") is True


@pytest.mark.unit
def test_partially_downloaded_shards_are_not_cached(cache_model, hermetic_availability):
    """One shard on disk must not vouch for a checkpoint that would still download."""

    repo = cache_model("ACE-Step/sharded", weight_name="model-00001-of-00002.safetensors")
    snapshot = next((repo / "snapshots").iterdir())
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    assert is_model_cached("ACE-Step/sharded") is False

    blob = next((repo / "blobs").iterdir())
    (snapshot / "model-00002-of-00002.safetensors").symlink_to(blob)
    assert is_model_cached("ACE-Step/sharded") is True


@pytest.mark.unit
def test_shard_index_in_a_subdirectory_is_resolved_relative_to_itself(cache_model):
    repo = cache_model("ACE-Step/nested", weight_name="transformer/model-00001-of-00002.safetensors")
    snapshot = next((repo / "snapshots").iterdir())
    (snapshot / "transformer" / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00002-of-00002.safetensors"}})
    )
    assert is_model_cached("ACE-Step/nested") is False


@pytest.mark.unit
def test_unreadable_shard_index_falls_back_to_plain_presence(cache_model):
    repo = cache_model("ACE-Step/badindex")
    snapshot = next((repo / "snapshots").iterdir())
    (snapshot / "model.safetensors.index.json").write_text("not json at all")
    assert is_model_cached("ACE-Step/badindex") is True


@pytest.mark.unit
def test_only_the_checked_out_revision_decides_presence(cache_model, hermetic_availability):
    """A leftover snapshot from an older revision must not vouch for the current one."""

    repo = cache_model("ACE-Step/revisioned")
    old_snapshot = next((repo / "snapshots").iterdir())
    new_snapshot = repo / "snapshots" / ("a" * 40)
    new_snapshot.mkdir()
    (new_snapshot / "config.json").write_text("{}")

    refs = repo / "refs"
    refs.mkdir()
    (refs / "main").write_text("a" * 40)
    assert is_model_cached("ACE-Step/revisioned") is False

    (refs / "main").write_text(old_snapshot.name)
    assert is_model_cached("ACE-Step/revisioned") is True


@pytest.mark.unit
def test_non_repo_ids_are_never_cached():
    assert is_model_cached("") is False
    assert is_model_cached("not-a-repo-id") is False
    assert is_model_cached("/etc/passwd") is False


@pytest.mark.unit
def test_cached_model_ids_returns_the_present_subset(cache_model):
    cache_model("ACE-Step/here")
    assert cached_model_ids(["ACE-Step/here", "ACE-Step/absent"]) == frozenset({"ACE-Step/here"})


@pytest.mark.unit
def test_presence_check_never_imports_a_model_runtime(cache_model):
    import sys

    cache_model("ACE-Step/Ace-Step1.5")
    is_model_cached("ACE-Step/Ace-Step1.5")
    assert not {"torch", "diffusers", "transformers"} & set(sys.modules)


# --------------------------------------------------------------------------- #
# Remote probes
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize(
    "status_code,expected",
    [(200, "available"), (204, "available"), (302, "available"), (404, "available"), (429, "available")],
)
@pytest.mark.unit
def test_answering_endpoint_is_available(monkeypatch, status_code, expected):
    monkeypatch.setattr(
        availability.urllib.request, "urlopen", lambda request, timeout=None: _FakeResponse(status_code)
    )
    assert probe_endpoint(RemoteEndpoint(url="https://api.example/v1/models")).status == expected


@pytest.mark.parametrize("status_code", [401, 403])
@pytest.mark.unit
def test_rejected_credentials_are_not_usable(monkeypatch, status_code):
    def _raise(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, status_code, "denied", {}, None)

    monkeypatch.setattr(availability.urllib.request, "urlopen", _raise)
    probe = probe_endpoint(RemoteEndpoint(url="https://api.example/v1/models"))
    assert probe.status == "unauthorized"
    assert probe.usable is False
    assert "credentials" in probe.detail


@pytest.mark.unit
def test_server_error_is_not_usable(monkeypatch):
    def _raise(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 503, "down", {}, None)

    monkeypatch.setattr(availability.urllib.request, "urlopen", _raise)
    probe = probe_endpoint(RemoteEndpoint(url="https://api.example/v1/models"))
    assert probe.status == "unavailable"
    assert probe.usable is False


@pytest.mark.unit
def test_connection_failure_is_unreachable(monkeypatch):
    def _raise(request, timeout=None):
        raise urllib.error.URLError("name or service not known")

    monkeypatch.setattr(availability.urllib.request, "urlopen", _raise)
    probe = probe_endpoint(RemoteEndpoint(url="https://api.example/v1/models"))
    assert probe.status == "unreachable"
    assert probe.usable is False


@pytest.mark.unit
def test_probe_sends_provider_headers(monkeypatch):
    seen = {}

    def _urlopen(request, timeout=None):
        seen["headers"] = dict(request.headers)
        seen["timeout"] = timeout
        return _FakeResponse(200)

    monkeypatch.setattr(availability.urllib.request, "urlopen", _urlopen)
    probe_endpoint(
        RemoteEndpoint(url="https://api.example/v1/models", headers={"Authorization": "Bearer k"}),
        timeout_s=5.0,
    )
    # urllib capitalizes header names.
    assert seen["headers"]["Authorization"] == "Bearer k"
    assert seen["timeout"] == 5.0


@pytest.mark.unit
def test_endpoints_are_probed_in_parallel(monkeypatch):
    monkeypatch.setattr(
        availability,
        "probe_endpoint",
        lambda endpoint, timeout_s=5.0: (time.sleep(0.3), RemoteProbe(status="available"))[1],
    )
    started = time.monotonic()
    results = probe_endpoints(
        {name: RemoteEndpoint(url=f"https://{name}.example/v1/models") for name in ("a", "b", "c", "d")},
        use_cache=False,
    )
    elapsed = time.monotonic() - started
    assert set(results) == {"a", "b", "c", "d"}
    assert all(probe.usable for probe in results.values())
    assert elapsed < 0.9, f"probes did not run concurrently ({elapsed:.2f}s)"


@pytest.mark.unit
def test_a_hung_probe_cannot_outlive_the_deadline(monkeypatch):
    release = threading.Event()

    def _hang(endpoint, timeout_s=5.0):
        release.wait(30)  # simulates a name lookup no socket timeout can interrupt
        return RemoteProbe(status="available")

    monkeypatch.setattr(availability, "probe_endpoint", _hang)
    started = time.monotonic()
    try:
        results = probe_endpoints(
            {"slow": RemoteEndpoint(url="https://slow.example/v1/models")},
            timeout_s=0.5,
            use_cache=False,
        )
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert elapsed < 1.0, f"probe round outlived its deadline ({elapsed:.2f}s)"
    assert results["slow"].status == "unreachable"
    assert results["slow"].usable is False


@pytest.mark.unit
def test_the_five_second_deadline_is_the_one_actually_applied(monkeypatch):
    """Pin the default end to end, not just the constant's literal value."""

    seen = []

    def _record(endpoint, timeout_s):
        seen.append(timeout_s)
        return RemoteProbe(status="available")

    monkeypatch.setattr(availability, "probe_endpoint", _record)
    probe_endpoints({"a": RemoteEndpoint(url="https://a.example/v1/models")}, use_cache=False)
    assert seen == [5.0]


@pytest.mark.unit
def test_an_unresponsive_provider_is_not_re_probed_on_every_call(monkeypatch):
    """The timeout is the only outcome that costs the deadline; paying it twice is the bug."""

    release = threading.Event()
    starts = []

    def _hang(endpoint, timeout_s=5.0):
        starts.append(time.monotonic())
        release.wait(30)
        return RemoteProbe(status="available")

    monkeypatch.setattr(availability, "probe_endpoint", _hang)
    endpoints = {"slow": RemoteEndpoint(url="https://slow.example/v1/models")}
    try:
        first = time.monotonic()
        probe_endpoints(endpoints, timeout_s=0.5)
        second = time.monotonic()
        results = probe_endpoints(endpoints, timeout_s=0.5)
        third = time.monotonic()
    finally:
        release.set()

    assert (second - first) >= 0.4, "the first round should actually wait"
    assert (third - second) < 0.2, "a second round must not wait on the same hung probe again"
    assert len(starts) == 1, f"spawned {len(starts)} probes for one hung endpoint"
    assert results["slow"].usable is False


@pytest.mark.unit
def test_a_late_probe_result_heals_the_cache(monkeypatch):
    """The worker owns the outcome, so a slow-but-alive provider recovers by itself."""

    release = threading.Event()

    def _slow(endpoint, timeout_s=5.0):
        release.wait(5)
        return RemoteProbe(status="available")

    monkeypatch.setattr(availability, "probe_endpoint", _slow)
    endpoints = {"slow": RemoteEndpoint(url="https://slow.example/v1/models")}
    assert probe_endpoints(endpoints, timeout_s=0.3)["slow"].usable is False

    release.set()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if probe_endpoints(endpoints, timeout_s=0.3)["slow"].usable:
            return
        time.sleep(0.05)
    pytest.fail("late probe result never reached the cache")


@pytest.mark.unit
def test_no_endpoints_means_no_threads_and_no_results():
    assert probe_endpoints({}) == {}


@pytest.mark.unit
def test_results_are_reused_within_the_cache_window(monkeypatch):
    calls = []

    def _count(endpoint, timeout_s=5.0):
        calls.append(endpoint.url)
        return RemoteProbe(status="available")

    monkeypatch.setattr(availability, "probe_endpoint", _count)
    endpoints = {"a": RemoteEndpoint(url="https://a.example/v1/models", headers={"k": "secret"})}
    assert probe_endpoints(endpoints)["a"].usable is True
    assert probe_endpoints(endpoints)["a"].usable is True
    assert len(calls) == 1

    availability.clear_probe_cache()
    assert probe_endpoints(endpoints)["a"].usable is True
    assert len(calls) == 2


@pytest.mark.unit
def test_changed_credentials_are_not_served_from_cache(monkeypatch):
    calls = []

    def _count(endpoint, timeout_s=5.0):
        calls.append(dict(endpoint.headers))
        return RemoteProbe(status="available")

    monkeypatch.setattr(availability, "probe_endpoint", _count)
    url = "https://a.example/v1/models"
    probe_endpoints({"a": RemoteEndpoint(url=url, headers={"Authorization": "Bearer old"})})
    probe_endpoints({"a": RemoteEndpoint(url=url, headers={"Authorization": "Bearer new"})})
    assert len(calls) == 2


@pytest.mark.unit
def test_a_programming_error_is_not_reported_as_a_dead_provider(monkeypatch):
    """"Measure, don't assume" must not quietly absorb bugs in our own code."""

    def _boom(request, timeout=None):
        raise AttributeError("refactor left a stale attribute")

    monkeypatch.setattr(availability.urllib.request, "urlopen", _boom)
    with pytest.raises(AttributeError):
        probe_endpoint(RemoteEndpoint(url="https://api.example/v1/models"))
