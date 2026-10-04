"""R10.1 (2026-10-04): a sound effect request runs the SFX checkpoint it names, for the length
it asks (`seconds`), with its prompt as written.

Before: the output.sound route (stable-audio-3-small-sfx) ran the small MUSIC checkpoint for
30 s ("laser gunshot" came back as a 30 s music-like clip), because the plugin ignored the
request's `model` and defaulted every SA3 request to 30 s.
"""

import pytest


class _FakeRuntime:
    """Stands in for the SA3 runtime: records the engine call, loads no weights."""

    instances = []

    def __init__(self, config):
        self.config = config
        self.device = "cpu"
        self.model_half = False
        self.calls = []
        self.unloaded = 0
        _FakeRuntime.instances.append(self)

    def load(self):
        pass

    def unload(self):
        self.unloaded += 1

    def _sample_rate(self):
        return 44100

    def generate(self, **kwargs):
        import torch

        self.calls.append(dict(kwargs))
        samples = int(44100 * float(kwargs["duration_s"]))
        return torch.zeros(1, 2, samples)


class _Owner:
    def __init__(self, config=None):
        self.config = dict(config or {})


class _Registry:
    def __init__(self):
        self.backends = {}

    def register_music_backend(self, *, backend_id, factory, **_kw):
        self.backends[backend_id] = factory


@pytest.fixture()
def sa3_capability(monkeypatch):
    import abstractmusic.backends.stable_audio_3 as sa3
    from abstractmusic.integrations.abstractcore_plugin import register

    _FakeRuntime.instances = []
    monkeypatch.setattr(sa3, "_StableAudio3Runtime", _FakeRuntime)
    reg = _Registry()
    register(reg)

    def make(config=None):
        return reg.backends["abstractmusic:stable-audio-3"](_Owner(config))

    return make


def _wav_seconds(data: bytes) -> float:
    import io
    import wave

    with wave.open(io.BytesIO(data), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


@pytest.mark.unit
def test_sfx_route_runs_the_named_sfx_checkpoint_for_the_requested_seconds(sa3_capability):
    cap = sa3_capability()
    out = cap.generate(
        "laser gunshot", task="text_to_audio", model="stabilityai/stable-audio-3-small-sfx", seconds=3
    )
    runtime = _FakeRuntime.instances[-1]
    assert runtime.config.model_id == "stabilityai/stable-audio-3-small-sfx"
    call = runtime.calls[-1]
    assert call["duration_s"] == 3.0  # the engine's seconds_total conditioning and trim length
    assert call["prompt"] == "laser gunshot"
    # Model card (small-sfx): 8 steps, cfg 1.0, pingpong.
    assert call["steps"] == 8
    assert call["cfg_scale"] == 1.0
    assert call["sampler_type"] == "pingpong"
    assert _wav_seconds(out) == pytest.approx(3.0, abs=0.01)


@pytest.mark.unit
def test_sfx_without_seconds_defaults_to_five_seconds(sa3_capability):
    cap = sa3_capability()
    cap.generate("laser gunshot", task="text_to_audio", model="stabilityai/stable-audio-3-small-sfx")
    assert _FakeRuntime.instances[-1].calls[-1]["duration_s"] == 5.0


@pytest.mark.unit
def test_music_without_seconds_keeps_thirty_seconds(sa3_capability):
    cap = sa3_capability()
    cap.generate("calm piano", task="text_to_music", model="stabilityai/stable-audio-3-small-music")
    runtime = _FakeRuntime.instances[-1]
    assert runtime.config.model_id == "stabilityai/stable-audio-3-small-music"
    assert runtime.calls[-1]["duration_s"] == 30.0


@pytest.mark.unit
def test_duration_s_is_the_same_argument_as_seconds(sa3_capability):
    cap = sa3_capability()
    cap.generate("door slam", task="text_to_audio", model="small-sfx", duration_s=2.5)
    assert _FakeRuntime.instances[-1].calls[-1]["duration_s"] == 2.5
    with pytest.raises(ValueError, match="disagree"):
        cap.generate("door slam", task="text_to_audio", seconds=2, duration_s=3)
    for bad in (0, -1, "three", float("nan")):
        with pytest.raises(ValueError, match="seconds"):
            cap.generate("door slam", task="text_to_audio", seconds=bad)


@pytest.mark.unit
def test_switching_checkpoint_unloads_the_previous_one(sa3_capability):
    cap = sa3_capability()
    cap.generate("calm piano", task="text_to_music", model="stabilityai/stable-audio-3-small-music", seconds=1)
    music_runtime = _FakeRuntime.instances[-1]
    cap.generate("laser gunshot", task="text_to_audio", model="stabilityai/stable-audio-3-small-sfx", seconds=1)
    sfx_runtime = _FakeRuntime.instances[-1]
    assert sfx_runtime is not music_runtime
    assert music_runtime.unloaded == 1
    # Same checkpoint again: no rebuild.
    cap.generate("laser gunshot", task="text_to_audio", model="small-sfx", seconds=1)
    assert _FakeRuntime.instances[-1] is sfx_runtime


@pytest.mark.unit
def test_unknown_sa3_model_is_refused_not_replaced(sa3_capability):
    cap = sa3_capability()
    with pytest.raises(ValueError, match="StableAudio3Backend supports model_id"):
        cap.generate("laser gunshot", task="text_to_audio", model="stabilityai/stable-audio-open-small")


@pytest.mark.unit
def test_sound_effect_prompt_skips_the_music_text_planner(sa3_capability):
    calls = []

    class Planner:
        def create_plan(self, request):
            calls.append(request)
            return {"caption": "epic orchestral battle music", "lyrics": "[Instrumental]"}

    cap = sa3_capability({"music_text_planner": Planner()})
    cap.generate("laser gunshot", task="text_to_audio", model="small-sfx", seconds=1)
    assert calls == []
    assert _FakeRuntime.instances[-1].calls[-1]["prompt"] == "laser gunshot"
    # Music still goes through the planner.
    cap.generate("battle", task="text_to_music", model="small-music", seconds=1)
    assert len(calls) == 1


@pytest.mark.unit
def test_sfx_negative_prompt_is_refused_with_a_reason(sa3_capability):
    from abstractmusic.errors import CapabilityNotSupportedError

    cap = sa3_capability()
    with pytest.raises((CapabilityNotSupportedError, ValueError), match="negative_prompt"):
        cap.generate("laser gunshot", task="text_to_audio", model="small-sfx", negative_prompt="music")


@pytest.mark.unit
def test_library_backend_default_length_follows_the_checkpoint(monkeypatch):
    import abstractmusic.backends.stable_audio_3 as sa3
    from abstractmusic.types import AudioGenerationRequest

    monkeypatch.setattr(sa3, "_StableAudio3Runtime", _FakeRuntime)
    sfx = sa3.StableAudio3Backend(config=sa3.StableAudio3BackendConfig(model_id="small-sfx"))
    sfx.generate_audio(AudioGenerationRequest(prompt="gunshot"))
    assert _FakeRuntime.instances[-1].calls[-1]["duration_s"] == 5.0
    music = sa3.StableAudio3Backend(config=sa3.StableAudio3BackendConfig(model_id="small-music"))
    music.generate_audio(AudioGenerationRequest(prompt="piano"))
    assert _FakeRuntime.instances[-1].calls[-1]["duration_s"] == 30.0


@pytest.mark.unit
def test_sound_effect_task_defaults_to_five_seconds_on_any_backend(sa3_capability):
    """The 5 s default belongs to the TASK (text_to_audio), not only to the SFX checkpoint:
    a sound effect routed to any backend without a length asks it for 5 s; music asks nothing."""

    from abstractmusic.types import GeneratedAsset

    class Capture:
        backend_id = "capture"
        requests = []

        def generate_audio(self, request):
            Capture.requests.append(request)
            return GeneratedAsset(data=b"wav", mime_type="audio/wav", metadata={})

    cap = sa3_capability({"music_backend_instance": Capture()})
    cap.generate("laser gunshot", task="text_to_audio")
    assert Capture.requests[-1].duration_s == 5.0
    cap.generate("calm piano", task="text_to_music")
    assert Capture.requests[-1].duration_s is None
