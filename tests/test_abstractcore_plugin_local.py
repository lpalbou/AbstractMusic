import pytest

from abstractmusic.types import GeneratedAsset, MusicBackendCapabilities
from abstractmusic.errors import AbstractMusicError, CapabilityNotSupportedError


class _StubBackend:
    backend_id = "stub"

    def generate_audio(self, request):
        return GeneratedAsset(data=b"wav-bytes", mime_type="audio/wav", metadata={"k": "v"})


class _CaptureBackend:
    backend_id = "capture"

    def __init__(self):
        self.last_request = None

    def get_capabilities(self):
        return MusicBackendCapabilities(
            supported_tasks=("text_to_music",),
            output_formats=("wav",),
            supports_lyrics=True,
            model_id="fake-model",
        )

    def generate_audio(self, request):
        self.last_request = request
        return GeneratedAsset(data=b"planned-wav", mime_type="audio/wav", metadata={"backend": "capture"})


class _DummyOwner:
    def __init__(self, config):
        self.config = dict(config)


class _HostContext:
    def __init__(self, service):
        self.text_generation = service


class _CoreLikeHostContext:
    def __init__(self, service):
        self._service = service

    @property
    def text(self):
        return self._service

    def service(self, name):
        if str(name) in {"text", "text_generation", "core_text", "abstractcore.text"}:
            return self._service
        raise KeyError(name)


class _OwnerWithHostTextService(_DummyOwner):
    def __init__(self, config, service):
        super().__init__(config)
        self.capability_host_context = _HostContext(service)


class _OwnerWithCoreLikeHostTextService(_DummyOwner):
    def __init__(self, config, service):
        super().__init__(config)
        self.capability_host_context = _CoreLikeHostContext(service)


class _FakeCoreTextService:
    def __init__(self):
        self.structured_calls = []

    def generate_structured(self, prompt, *, json_schema=None, system_prompt=None, purpose=None, metadata=None):
        self.structured_calls.append(
            {
                "prompt": prompt,
                "json_schema": json_schema,
                "system_prompt": system_prompt,
                "purpose": purpose,
                "metadata": metadata,
            }
        )
        return {
            "prompt": (
                "Original retro arcade space shooter battle cue with rapid synth bass, "
                "punchy drums, minor chord stabs, laser-like arpeggios, and no vocal melody."
            ),
            "lyrics": "[Instrumental]",
            "vocal_language": "en",
            "bpm": 150,
            "keyscale": "C minor",
            "timesignature": "4",
            "instrumental": True,
            "generated_fields": ["prompt", "lyrics", "bpm", "keyscale", "timesignature"],
            "confidence": 0.82,
        }


class _FakeCoreTextOnlyResult:
    text = '{"prompt": "text-only host planned synth battle cue", "lyrics": "[Instrumental]", "bpm": 148}'
    model = "fake-text-model"


class _FakeCoreTextOnlyService:
    def __init__(self):
        self.text_calls = []

    def generate_text(self, prompt, *, system_prompt=None, max_output_tokens=None, temperature=None, purpose=None, metadata=None):
        self.text_calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "purpose": purpose,
                "metadata": metadata,
            }
        )
        return _FakeCoreTextOnlyResult()


class _FakeCoreTextServiceWithoutJsonSchema:
    def __init__(self):
        self.structured_calls = []
        self.text_calls = []

    def generate_structured(self, prompt, *, json_schema=None, system_prompt=None, purpose=None, metadata=None):
        self.structured_calls.append(
            {
                "prompt": prompt,
                "json_schema": json_schema,
                "system_prompt": system_prompt,
                "purpose": purpose,
                "metadata": metadata,
            }
        )
        if json_schema is not None:
            raise ValueError("json_schema structured host text generation is not exposed by this service yet")
        return _FakeCoreTextOnlyResult()

    def generate_text(self, prompt, *, system_prompt=None, max_output_tokens=None, temperature=None, purpose=None, metadata=None):
        self.text_calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "purpose": purpose,
                "metadata": metadata,
            }
        )
        return _FakeCoreTextOnlyResult()


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


@pytest.mark.unit
def test_capability_uses_injected_text_planner_without_abstractcore_import():
    from abstractmusic.integrations.abstractcore_plugin import register

    reg = _Registry()
    register(reg)
    factory = _get_factory(reg, "abstractmusic:acestep-diffusers")
    backend = _CaptureBackend()
    calls = []

    def planner(request):
        calls.append(dict(request))
        return {
            "prompt": "planned fantasy cue with brass theme",
            "lyrics": "[Instrumental]",
            "vocal_language": "en",
            "bpm": 96,
            "keyscale": "D minor",
            "timesignature": "4",
            "planner_backend": "host-text-planner",
            "generated_fields": ["prompt", "lyrics", "bpm"],
        }

    owner = _DummyOwner({"music_backend_instance": backend, "music_text_planner": planner})
    cap = factory(owner)
    out = cap.t2m("heroic fantasy", duration_s=30)

    assert out == b"planned-wav"
    assert calls and calls[0]["prompt"] == "heroic fantasy"
    assert backend.last_request.prompt == "planned fantasy cue with brass theme"
    assert backend.last_request.lyrics == "[Instrumental]"
    assert backend.last_request.vocal_language == "en"
    assert backend.last_request.extra["bpm"] == 96
    assert backend.last_request.extra["keyscale"] == "D minor"


@pytest.mark.unit
def test_capability_text_planner_mode_off_does_not_call_injected_planner():
    from abstractmusic.integrations.abstractcore_plugin import register

    reg = _Registry()
    register(reg)
    factory = _get_factory(reg, "abstractmusic:acestep-diffusers")
    backend = _CaptureBackend()
    calls = []

    def planner(request):
        calls.append(dict(request))
        return {"prompt": "should not be used"}

    owner = _DummyOwner(
        {
            "music_backend_instance": backend,
            "music_text_planner": planner,
            "music_text_planner_mode": "off",
        }
    )
    cap = factory(owner)
    out = cap.t2m("raw user prompt", duration_s=30)

    assert out == b"planned-wav"
    assert calls == []
    assert backend.last_request.prompt == "raw user prompt"


@pytest.mark.unit
def test_capability_text_planner_required_mode_needs_provider():
    from abstractmusic.integrations.abstractcore_plugin import register

    reg = _Registry()
    register(reg)
    factory = _get_factory(reg, "abstractmusic:acestep-diffusers")
    owner = _DummyOwner({"music_backend_instance": _CaptureBackend(), "music_text_planner_mode": "required"})
    cap = factory(owner)

    with pytest.raises(TypeError, match="required"):
        cap.t2m("raw user prompt", duration_s=30)


@pytest.mark.unit
def test_capability_uses_abstractcore_host_text_service_for_planning():
    from abstractmusic.integrations.abstractcore_plugin import register

    reg = _Registry()
    register(reg)
    factory = _get_factory(reg, "abstractmusic:acestep-diffusers")
    backend = _CaptureBackend()
    service = _FakeCoreTextService()
    owner = _OwnerWithHostTextService({"music_backend_instance": backend}, service)
    cap = factory(owner)
    store = _ArtifactStore()

    out = cap.t2m("game music of the space shooter rtype", duration_s=120, artifact_store=store)

    assert out["$artifact"] == "a1"
    assert out["metadata"]["planner_backend"] == "abstractcore-host-text-service"
    assert out["metadata"]["planner_confidence"] == 0.82
    assert store.calls and store.calls[0]["content"] == b"planned-wav"
    assert service.structured_calls
    call = service.structured_calls[0]
    assert call["purpose"] == "abstractmusic.text_planning"
    assert call["json_schema"]["required"] == ["prompt"]
    assert call["metadata"]["capability"] == "music"
    assert "space shooter" in call["prompt"]
    assert backend.last_request.prompt.startswith("Original retro arcade space shooter")
    assert backend.last_request.lyrics == "[Instrumental]"
    assert backend.last_request.vocal_language == "en"
    assert backend.last_request.extra["bpm"] == 150
    assert backend.last_request.extra["keyscale"] == "C minor"


@pytest.mark.unit
def test_capability_text_planner_mode_off_ignores_host_text_service():
    from abstractmusic.integrations.abstractcore_plugin import register

    reg = _Registry()
    register(reg)
    factory = _get_factory(reg, "abstractmusic:acestep-diffusers")
    backend = _CaptureBackend()
    service = _FakeCoreTextService()
    owner = _OwnerWithHostTextService(
        {"music_backend_instance": backend, "music_text_planner_mode": "off"},
        service,
    )
    cap = factory(owner)

    out = cap.t2m("raw user prompt", duration_s=30)

    assert out == b"planned-wav"
    assert service.structured_calls == []
    assert backend.last_request.prompt == "raw user prompt"


@pytest.mark.unit
def test_capability_host_text_service_can_fall_back_to_json_text():
    from abstractmusic.integrations.abstractcore_plugin import register

    reg = _Registry()
    register(reg)
    factory = _get_factory(reg, "abstractmusic:acestep-diffusers")
    backend = _CaptureBackend()
    service = _FakeCoreTextOnlyService()
    owner = _DummyOwner({"music_backend_instance": backend, "music_host_text_service": service})
    cap = factory(owner)
    store = _ArtifactStore()

    out = cap.t2m("space shooter boss stage", duration_s=60, artifact_store=store)

    assert out["$artifact"] == "a1"
    assert out["metadata"]["planner_backend"] == "abstractcore-host-text-service"
    assert out["metadata"]["planner_model"] == "fake-text-model"
    assert service.text_calls
    assert service.text_calls[0]["purpose"] == "abstractmusic.text_planning"
    assert backend.last_request.prompt == "text-only host planned synth battle cue"
    assert backend.last_request.lyrics == "[Instrumental]"
    assert backend.last_request.extra["bpm"] == 148


@pytest.mark.unit
def test_capability_uses_core_like_host_context_text_service():
    from abstractmusic.integrations.abstractcore_plugin import register

    reg = _Registry()
    register(reg)
    factory = _get_factory(reg, "abstractmusic:acestep-diffusers")
    backend = _CaptureBackend()
    service = _FakeCoreTextServiceWithoutJsonSchema()
    owner = _OwnerWithCoreLikeHostTextService({"music_backend_instance": backend}, service)
    cap = factory(owner)

    out = cap.t2m("space shooter boss stage", duration_s=60)

    assert out == b"planned-wav"
    assert service.structured_calls
    assert service.text_calls
    assert service.text_calls[0]["purpose"] == "abstractmusic.text_planning"
    assert backend.last_request.prompt == "text-only host planned synth battle cue"
    assert backend.last_request.lyrics == "[Instrumental]"


@pytest.mark.unit
def test_capability_exposes_generic_music_discovery_without_loading_runtime():
    from abstractmusic.integrations.abstractcore_plugin import register

    reg = _Registry()
    register(reg)
    factory = _get_factory(reg, "abstractmusic:acestep-diffusers")
    cap = factory(_DummyOwner({}))

    providers = cap.available_providers(task="t2m")
    provider_ids = {item["provider_id"] for item in providers}
    assert "ace-step" in provider_ids
    assert all(item["capability"] == "music" for item in providers)

    models = cap.list_models(task="text_to_music", provider="ACE-Step")
    model_ids = {item["model_id"] for item in models}
    assert "ACE-Step/acestep-v15-xl-turbo-diffusers" in model_ids
    diffusers_record = next(item for item in models if item["model_id"] == "ACE-Step/acestep-v15-xl-turbo-diffusers")
    assert diffusers_record["provider_id"] == "ace-step"
    assert diffusers_record["backend_id"] == "abstractmusic:acestep-diffusers"
    assert diffusers_record["formats"] == ["wav"]
    assert diffusers_record["metadata"]["supports_lyrics"] is True

    operations = cap.list_operations(task="text2music")
    assert operations and operations[0]["task"] == "text_to_music"
    assert operations[0]["artifact_output"] is True
    assert operations[0]["parameter_schema"]["required"] == ["prompt"]

    catalog = cap.capability_catalog(task="text_to_music")
    assert catalog["capability"] == "music"
    assert catalog["providers"]
    assert catalog["models"]
    assert catalog["operations"]
