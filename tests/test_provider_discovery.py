"""Provider discovery must be honest and cheap.

Honest: a provider is listed only when it can actually run right now — local
weights on disk, or a remote API that answers. Cheap: listing never imports a
model runtime and never loads weights.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from abstractmusic import availability
from abstractmusic.availability import RemoteProbe
from abstractmusic.integrations import abstractcore_plugin as plugin


class _Registry:
    def __init__(self) -> None:
        self.factories = {}

    def register_music_backend(self, *, backend_id, factory, **kwargs):
        self.factories[str(backend_id)] = factory


class _Owner:
    def __init__(self, config=None) -> None:
        self.config = dict(config or {})


def _capability(backend_id: str = "abstractmusic:acestep", config=None):
    registry = _Registry()
    plugin.register(registry)
    return registry.factories[backend_id](_Owner(config))


@pytest.fixture
def all_runtimes_installed(monkeypatch):
    monkeypatch.setattr(plugin, "_runtime_installed", lambda extra: True)


# --------------------------------------------------------------------------- #
# Local providers
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_local_provider_without_weights_is_not_available(all_runtimes_installed):
    providers = _capability().available_providers(task="text_to_music")
    assert providers == []


@pytest.mark.unit
def test_local_provider_is_available_once_weights_are_on_disk(all_runtimes_installed, cache_model):
    cache_model("ACE-Step/acestep-v15-xl-turbo-diffusers")
    providers = _capability().available_providers(task="text_to_music")

    assert [item["provider_id"] for item in providers] == ["acestep"]
    entry = providers[0]
    assert entry["local"] is True
    assert entry["remote"] is False
    assert entry["installed"] is True
    assert entry["configured"] is True
    assert entry["status"] == "available"
    assert entry["metadata"]["models"] == ["ACE-Step/acestep-v15-xl-turbo-diffusers"]


@pytest.mark.unit
def test_only_present_models_are_listed(all_runtimes_installed, cache_model):
    cache_model("ACE-Step/acestep-v15-xl-sft-diffusers")
    records = _capability().list_models(task="text_to_music")

    assert [item["model_id"] for item in records] == ["ACE-Step/acestep-v15-xl-sft-diffusers"]
    assert records[0]["metadata"]["cached"] is True
    assert records[0]["metadata"]["installed"] is True


@pytest.mark.unit
def test_incompatible_layout_models_are_never_offered_even_when_cached(all_runtimes_installed, cache_model):
    """A cached checkpoint the backend cannot load must not be reported as runnable.

    ACE-Step publishes official checkpoints in a native transformers layout that
    `AceStepPipeline.from_pretrained` cannot load; having its weights on disk
    does not make it usable.
    """

    cache_model("ACE-Step/Ace-Step1.5")
    capability = _capability()
    assert capability.available_providers(task="text_to_music") == []
    assert capability.list_models(task="text_to_music") == []

    # The moment a loadable-layout checkpoint is present, the provider appears —
    # and still without the incompatible one.
    cache_model("ACE-Step/acestep-v15-xl-sft-diffusers")
    records = capability.list_models(task="text_to_music")
    assert [item["model_id"] for item in records] == ["ACE-Step/acestep-v15-xl-sft-diffusers"]


@pytest.mark.unit
def test_cached_incompatible_checkpoint_is_explained_not_denied(all_runtimes_installed, cache_model):
    """With only an incompatible checkpoint cached, the provider must say why it
    is unusable — not claim no weights exist — and keep the catalog visible."""

    cache_model("ACE-Step/Ace-Step1.5")
    details = {item["provider_id"]: item for item in _capability().provider_details(task="text_to_music")}
    acestep = details["acestep"]

    assert acestep["usable"] is False
    assert acestep["status"] == "no-loadable-weights"
    assert "ACE-Step/Ace-Step1.5" in acestep["metadata"]["reason"]
    assert "ACE-Step/Ace-Step1.5" in acestep["metadata"]["cached_models"]
    # Catalog knowledge survives: both the incompatible and the loadable ids are listed.
    assert "ACE-Step/Ace-Step1.5" in acestep["metadata"]["models"]
    assert "ACE-Step/acestep-v15-xl-sft-diffusers" in acestep["metadata"]["models"]


@pytest.mark.unit
def test_configured_model_id_cannot_resurrect_an_incompatible_checkpoint(
    monkeypatch, all_runtimes_installed, cache_model
):
    """music_model_id must not route an unloadable checkpoint through the
    generic diffusers provider: no Diffusers pipeline can load it either."""

    cache_model("ACE-Step/Ace-Step1.5")
    monkeypatch.setenv("ABSTRACTMUSIC_MODEL_ID", "ACE-Step/Ace-Step1.5")
    capability = _capability("abstractmusic:diffusers")

    assert capability.available_providers(task="text_to_music") == []
    assert capability.list_models(task="text_to_music") == []

    details = {item["provider_id"]: item for item in capability.provider_details(task="text_to_music")}
    diffusers = details["diffusers"]
    assert diffusers["usable"] is False
    assert diffusers["status"] == "incompatible-model"
    assert "layout" in diffusers["metadata"]["reason"]


@pytest.mark.unit
def test_missing_runtime_hides_a_provider_with_weights(monkeypatch, cache_model):
    cache_model("ACE-Step/acestep-v15-xl-turbo-diffusers")
    monkeypatch.setattr(plugin, "_runtime_installed", lambda extra: False)
    assert _capability().available_providers(task="text_to_music") == []


@pytest.mark.unit
def test_configured_diffusers_model_needs_its_weights(monkeypatch, all_runtimes_installed, cache_model):
    monkeypatch.setenv("ABSTRACTMUSIC_MODEL_ID", "someone/custom-audio-model")
    capability = _capability("abstractmusic:diffusers")
    assert capability.available_providers(task="text_to_audio") == []

    cache_model("someone/custom-audio-model")
    providers = capability.available_providers(task="text_to_audio")
    assert [item["provider_id"] for item in providers] == ["diffusers"]
    assert providers[0]["metadata"]["models"] == ["someone/custom-audio-model"]

    records = capability.list_models(task="text_to_audio")
    assert records[0]["model_id"] == "someone/custom-audio-model"


# --------------------------------------------------------------------------- #
# Runtime probe (unstubbed — this is what replaced `import diffusers`)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_package_submodule_probe_answers_without_importing_the_package():
    import importlib.util

    assert importlib.util.find_spec("diffusers") is not None, "test needs diffusers installed"
    before = set(sys.modules)
    assert plugin._package_ships_module("diffusers", "pipelines", "ace_step") is True
    assert plugin._package_ships_module("diffusers", "pipelines", "not_a_real_pipeline") is False
    assert plugin._package_ships_module("no_such_package_at_all", "anything") is False
    assert not {"torch", "diffusers", "transformers"} & (set(sys.modules) - before)


@pytest.mark.unit
def test_acestep_runtime_requires_the_acestep_pipeline(monkeypatch):
    """A diffusers without the ACE-Step pipeline must not report ACE-Step as installed."""

    monkeypatch.setattr(plugin.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(plugin, "_package_ships_module", lambda package, *parts: False)
    assert plugin._runtime_installed("acestep") is False
    # Other extras do not depend on that pipeline.
    assert plugin._runtime_installed("musicgen") is True


@pytest.mark.unit
def test_unknown_extras_report_unknown_rather_than_installed():
    assert plugin._runtime_installed("heartmula") is None
    assert plugin._runtime_installed(None) is None


# --------------------------------------------------------------------------- #
# Remote providers
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_remote_provider_without_credentials_is_never_probed(monkeypatch):
    monkeypatch.delenv("ACEMUSIC_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    probed = []
    monkeypatch.setattr(
        availability,
        "probe_endpoint",
        lambda endpoint, timeout_s=0.0: probed.append(endpoint) or RemoteProbe(status="available"),
    )

    assert _capability("abstractmusic:acemusic").available_providers(task="text_to_music") == []
    assert probed == []


@pytest.mark.unit
def test_remote_provider_is_available_when_its_api_answers(monkeypatch):
    monkeypatch.setenv("ACEMUSIC_API_KEY", "key")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr(
        availability, "probe_endpoint", lambda endpoint, timeout_s=0.0: RemoteProbe(status="available")
    )

    providers = _capability("abstractmusic:acemusic").available_providers(task="text_to_music")
    assert [item["provider_id"] for item in providers] == ["acemusic"]
    assert providers[0]["remote"] is True
    assert providers[0]["installed"] is True
    assert providers[0]["selected"] is True


@pytest.mark.unit
@pytest.mark.parametrize("status", ["unreachable", "unauthorized", "unavailable"])
def test_remote_provider_that_cannot_serve_is_not_available(monkeypatch, status):
    monkeypatch.setenv("ACEMUSIC_API_KEY", "key")
    monkeypatch.setattr(availability, "probe_endpoint", lambda endpoint, timeout_s=0.0: RemoteProbe(status=status))
    assert _capability("abstractmusic:acemusic").available_providers(task="text_to_music") == []


@pytest.mark.unit
def test_remote_probe_uses_the_configured_endpoint_and_credentials(monkeypatch):
    monkeypatch.delenv("ACEMUSIC_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    seen = []
    monkeypatch.setattr(
        availability,
        "probe_endpoint",
        lambda endpoint, timeout_s=0.0: seen.append(endpoint) or RemoteProbe(status="available"),
    )

    capability = _capability(
        "abstractmusic:elevenlabs-music",
        config={
            "music_elevenlabs_base_url": "https://proxy.internal",
            "music_elevenlabs_api_key": "cfg-key",
        },
    )
    capability.available_providers(task="text_to_music")

    assert [endpoint.url for endpoint in seen] == ["https://proxy.internal/v1/models"]
    assert seen[0].headers == {"xi-api-key": "cfg-key"}


@pytest.mark.unit
def test_both_remote_providers_are_probed_in_one_round(monkeypatch):
    monkeypatch.setenv("ACEMUSIC_API_KEY", "a")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "b")
    seen = []
    monkeypatch.setattr(
        availability,
        "probe_endpoint",
        lambda endpoint, timeout_s=0.0: seen.append(endpoint.url) or RemoteProbe(status="available"),
    )

    providers = _capability("abstractmusic:acemusic").available_providers(task="text_to_music")
    assert {item["provider_id"] for item in providers} == {"acemusic", "elevenlabs-music"}
    assert len(seen) == 2


@pytest.mark.unit
def test_default_remote_probe_deadline_is_five_seconds():
    assert availability.REMOTE_PROBE_TIMEOUT_S == 5.0


# --------------------------------------------------------------------------- #
# Diagnosability: an empty provider list must never be a dead end
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_provider_details_explains_every_provider_that_is_missing(monkeypatch, cache_model):
    monkeypatch.setenv("ACEMUSIC_API_KEY", "wrong-key")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr(
        plugin,
        "_runtime_installed",
        lambda extra: {"acestep": True, "stable-audio-3": False}.get(extra, False),
    )
    monkeypatch.setattr(
        availability,
        "probe_endpoint",
        lambda endpoint, timeout_s=0.0: availability.RemoteProbe(
            status="unauthorized", detail="HTTP 401: credentials rejected", latency_ms=12
        ),
    )

    capability = _capability()
    assert capability.available_providers(task="text_to_music") == []

    details = {item["provider_id"]: item for item in capability.provider_details(task="text_to_music")}
    assert all(item["usable"] is False for item in details.values())

    assert details["acemusic"]["status"] == "unauthorized"
    assert details["acemusic"]["metadata"]["reason"] == "HTTP 401: credentials rejected"
    assert details["acemusic"]["metadata"]["latency_ms"] == 12

    assert details["elevenlabs-music"]["status"] == "not-configured"
    assert "API key" in details["elevenlabs-music"]["metadata"]["reason"]

    assert details["acestep"]["status"] == "no-local-weights"
    assert details["acestep"]["installed"] is True
    assert details["acestep"]["metadata"]["cached_models"] == []
    # The catalog stays visible so a user can see what there is to download.
    assert "ACE-Step/acestep-v15-xl-turbo-diffusers" in details["acestep"]["metadata"]["models"]

    assert details["stable-audio-3"]["status"] == "not-installed"
    assert details["stable-audio-3"]["installed"] is False


@pytest.mark.unit
def test_provider_details_marks_the_usable_ones_and_names_their_cached_models(
    all_runtimes_installed, cache_model
):
    cache_model("ACE-Step/acestep-v15-xl-sft-diffusers")
    details = {item["provider_id"]: item for item in _capability().provider_details(task="text_to_music")}

    assert details["acestep"]["usable"] is True
    assert details["acestep"]["metadata"]["cached_models"] == ["ACE-Step/acestep-v15-xl-sft-diffusers"]
    assert len(details["acestep"]["metadata"]["models"]) > 1, "uncached models stay discoverable"
    assert details["stable-audio-3"]["usable"] is False
    # Usable providers sort first.
    assert [item["usable"] for item in _capability().provider_details(task="text_to_music")][0] is True


@pytest.mark.unit
def test_catalog_probes_remote_providers_once(monkeypatch):
    monkeypatch.setenv("ACEMUSIC_API_KEY", "key")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    calls = []
    monkeypatch.setattr(
        availability,
        "probe_endpoint",
        lambda endpoint, timeout_s=0.0: calls.append(endpoint.url)
        or availability.RemoteProbe(status="available"),
    )

    catalog = _capability("abstractmusic:acemusic").capability_catalog(task="text_to_music")
    assert len(calls) == 1, f"catalog fired {len(calls)} probe rounds"
    assert [item["provider_id"] for item in catalog["providers"]] == ["acemusic"]
    assert catalog["provider_details"]
    assert catalog["models"]


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_discovery_never_imports_a_model_runtime(tmp_path):
    """The regression this whole path exists for: listing must not import torch."""

    script = """
import json, sys
from abstractmusic.integrations import abstractcore_plugin as plugin

class Registry:
    def __init__(self): self.factories = {}
    def register_music_backend(self, *, backend_id, factory, **kw): self.factories[backend_id] = factory

class Owner:
    config = {"music_model_id": "someone/custom-audio-model"}

registry = Registry()
plugin.register(registry)
capability = registry.factories["abstractmusic:acestep"](Owner())
capability.available_providers(task="text_to_music")
capability.list_models(task="text_to_music")
capability.provider_details(task="text_to_music")
capability.list_operations(task="text_to_music")
capability.capability_catalog(task="text_to_music")
print(json.dumps(sorted({"torch", "diffusers", "transformers", "torchaudio"} & set(sys.modules))))
"""
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "HF_HUB_CACHE": str(tmp_path / "empty-cache"),
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, f"discovery crashed:\n{result.stderr}"
    assert json.loads(result.stdout.strip()) == [], "discovery imported a model runtime"
