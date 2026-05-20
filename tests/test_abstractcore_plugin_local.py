import pytest

from abstractmusic.types import GeneratedAsset
from abstractmusic.errors import AbstractMusicError, CapabilityNotSupportedError


class _StubBackend:
    backend_id = "stub"

    def generate_audio(self, request):
        return GeneratedAsset(data=b"wav-bytes", mime_type="audio/wav", metadata={"k": "v"})


class _DummyOwner:
    def __init__(self, config):
        self.config = dict(config)


class _Registry:
    def __init__(self):
        self.registrations = []

    def register_music_backend(self, **kwargs):
        self.registrations.append(dict(kwargs))


class _ArtifactStore:
    def __init__(self):
        self.calls = []

    def store(self, content: bytes, *, content_type: str = "application/octet-stream", run_id=None, tags=None, artifact_id=None):
        self.calls.append(
            {
                "content": bytes(content),
                "content_type": str(content_type),
                "run_id": run_id,
                "tags": dict(tags) if isinstance(tags, dict) else tags,
                "artifact_id": artifact_id,
            }
        )
        return {"artifact_id": "a1"}


@pytest.mark.unit
def test_plugin_registers_backend_factory():
    from abstractmusic.integrations.abstractcore_plugin import register

    reg = _Registry()
    register(reg)
    assert len(reg.registrations) == 3
    backend_ids = {r["backend_id"] for r in reg.registrations}
    assert backend_ids == {
        "abstractmusic:acestep-v15",
        "abstractmusic:acestep-diffusers",
        "abstractmusic:diffusers",
    }
    priorities = {r["backend_id"]: r["priority"] for r in reg.registrations}
    assert priorities["abstractmusic:acestep-diffusers"] > priorities["abstractmusic:acestep-v15"]
    assert all(callable(r["factory"]) for r in reg.registrations)


def _get_factory(reg: _Registry, backend_id: str):
    for r in reg.registrations:
        if r.get("backend_id") == backend_id:
            return r["factory"]
    raise KeyError(backend_id)


@pytest.mark.unit
def test_capability_t2m_returns_bytes_without_artifact_store():
    from abstractmusic.integrations.abstractcore_plugin import register

    reg = _Registry()
    register(reg)
    factory = _get_factory(reg, "abstractmusic:acestep-v15")

    owner = _DummyOwner({"music_backend_instance": _StubBackend()})
    cap = factory(owner)
    out = cap.t2m("hello", format="wav")
    assert out == b"wav-bytes"


@pytest.mark.unit
def test_capability_t2m_stores_when_artifact_store_provided():
    from abstractmusic.integrations.abstractcore_plugin import register

    reg = _Registry()
    register(reg)
    factory = _get_factory(reg, "abstractmusic:acestep-v15")

    owner = _DummyOwner({"music_backend_instance": _StubBackend()})
    cap = factory(owner)

    store = _ArtifactStore()
    ref = cap.t2m("hello", format="wav", artifact_store=store, run_id="r1", tags={"foo": "bar"}, metadata={"m": 1})
    assert isinstance(ref, dict)
    assert ref.get("$artifact") == "a1"
    assert store.calls and store.calls[0]["content"] == b"wav-bytes"
    assert store.calls[0]["content_type"] == "audio/wav"
    assert store.calls[0]["run_id"] == "r1"
    assert store.calls[0]["tags"]["foo"] == "bar"


@pytest.mark.unit
def test_capability_rejects_non_wav_format():
    from abstractmusic.integrations.abstractcore_plugin import register

    reg = _Registry()
    register(reg)
    factory = _get_factory(reg, "abstractmusic:acestep-v15")

    owner = _DummyOwner({"music_backend_instance": _StubBackend()})
    cap = factory(owner)
    with pytest.raises(CapabilityNotSupportedError):
        cap.t2m("hello", format="mp3")


@pytest.mark.unit
def test_capability_requires_model_id_when_not_injected():
    from abstractmusic.integrations.abstractcore_plugin import register

    reg = _Registry()
    register(reg)
    factory = _get_factory(reg, "abstractmusic:diffusers")

    owner = _DummyOwner({"music_backend": "diffusers"})
    cap = factory(owner)
    with pytest.raises(AbstractMusicError):
        cap.t2m("hello", format="wav")


@pytest.mark.unit
def test_capability_rejects_local_model_path():
    from abstractmusic.integrations.abstractcore_plugin import register

    reg = _Registry()
    register(reg)
    factory = _get_factory(reg, "abstractmusic:diffusers")

    owner = _DummyOwner({"music_backend": "diffusers", "music_model_id": "./local-model"})
    cap = factory(owner)
    with pytest.raises(AbstractMusicError, match="local filesystem path"):
        cap.t2m("hello", format="wav")
