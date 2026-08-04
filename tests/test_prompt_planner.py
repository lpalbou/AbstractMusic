import argparse

import pytest


@pytest.mark.unit
def test_prompt_planner_enhances_heroic_fantasy_prompt():
    from abstractmusic.prompt_planner import create_prompt_plan

    plan = create_prompt_plan(
        "an heroic fantasy epic music in the same style as conan the barbarian",
        duration_s=30,
        enhance_prompt=True,
    )

    assert plan.enhanced_prompt is True
    assert "French horns" in plan.prompt
    assert "timpani" in plan.prompt
    assert "Target duration is about 30 seconds" in plan.prompt
    assert plan.bpm == 96
    assert plan.keyscale == "D minor"
    assert plan.timesignature == "4"
    assert plan.lyrics is None


@pytest.mark.unit
def test_prompt_planner_adds_long_form_structure_by_default():
    from abstractmusic.prompt_planner import create_prompt_plan

    plan = create_prompt_plan(
        "an heroic fantasy epic music in the same style as conan the barbarian",
        duration_s=120,
    )

    assert plan.enhanced_prompt is True
    assert plan.structured_prompt is True
    assert "Long-form structure:" in plan.prompt
    assert "0:00-" in plan.prompt
    assert "full brass and wide drums reach the climax" in plan.prompt
    assert "more than 8 bars" in plan.prompt
    assert plan.bpm == 96
    assert plan.keyscale == "D minor"


@pytest.mark.unit
def test_prompt_planner_can_disable_long_form_structure():
    from abstractmusic.prompt_planner import create_prompt_plan

    plan = create_prompt_plan("heroic fantasy music", duration_s=120, structure_prompt=False)

    assert plan.enhanced_prompt is False
    assert plan.structured_prompt is False
    assert "Long-form structure:" not in plan.prompt
    assert plan.bpm is None


@pytest.mark.unit
def test_prompt_planner_auto_lyrics_generates_structured_sections():
    from abstractmusic.prompt_planner import create_prompt_plan

    plan = create_prompt_plan("heroic fantasy battle song", lyrics="auto", vocal_language="en")

    assert plan.generated_lyrics is True
    assert plan.enhanced_prompt is True
    assert plan.vocal_language == "en"
    assert plan.lyrics is not None
    assert "[Verse 1]" in plan.lyrics
    assert "[Chorus]" in plan.lyrics
    assert "Raise the" in plan.lyrics


@pytest.mark.unit
def test_prompt_planner_instrumental_overrides_auto_lyrics():
    from abstractmusic.prompt_planner import create_prompt_plan

    plan = create_prompt_plan("instrumental cinematic fantasy", lyrics="auto", auto_lyrics=True)

    assert plan.instrumental is True
    assert plan.generated_lyrics is False
    assert plan.lyrics == "[Instrumental]"


@pytest.mark.unit
def test_prompt_planner_expands_space_shooter_game_music_as_instrumental_action():
    from abstractmusic.prompt_planner import create_prompt_plan

    plan = create_prompt_plan("game music of the space shooter rtype", duration_s=120)

    assert plan.instrumental is True
    assert plan.lyrics == "[Instrumental]"
    assert plan.enhanced_prompt is True
    assert plan.structured_prompt is True
    assert plan.bpm == 150
    assert plan.keyscale == "C minor"
    assert "retro arcade space-shooter" in plan.prompt
    assert "constant forward momentum" in plan.prompt
    assert "continuous gameplay loop" in plan.prompt
    assert "no empty gaps" in plan.prompt
    assert "space-battle hook" in plan.prompt


@pytest.mark.unit
def test_prompt_planner_never_forces_caption_expansion_on_short_prompts():
    """Style profiles must not silently replace a short prompt with a template
    caption: the forced arcade expansion reproducibly collapsed guided ACE-Step
    XL checkpoints into unpitched noise-sweep output at 30s, while the same
    seed with the raw prompt generated music."""

    from abstractmusic.prompt_planner import create_prompt_plan

    raw = "rhythmic space shooter game music, fast drums, evolving synth bass, arcade lead theme"
    plan = create_prompt_plan(raw, duration_s=30)

    assert plan.prompt == raw
    assert plan.enhanced_prompt is False
    # Profile-derived instrumental defaults still apply; only the caption is untouched.
    assert plan.instrumental is True


@pytest.mark.unit
def test_prompt_planner_expands_short_prompts_only_on_explicit_request():
    from abstractmusic.prompt_planner import create_prompt_plan

    raw = "rhythmic space shooter game music, fast drums, evolving synth bass, arcade lead theme"
    plan = create_prompt_plan(raw, duration_s=30, enhance_prompt=True)

    assert plan.enhanced_prompt is True
    assert "retro arcade space-shooter" in plan.prompt


@pytest.mark.unit
def test_prompt_planner_expands_platform_game_music_as_loopable_instrumental():
    from abstractmusic.prompt_planner import create_prompt_plan

    plan = create_prompt_plan("game music of mario bross", duration_s=120)

    assert plan.instrumental is True
    assert plan.lyrics == "[Instrumental]"
    assert plan.bpm == 132
    assert plan.keyscale == "C major"
    assert "retro platform-game music" in plan.prompt
    assert "loopable" in plan.prompt
    assert "bonus-room bridge" in plan.prompt


@pytest.mark.unit
def test_injected_music_text_planner_is_used_in_auto_mode():
    from abstractmusic.prompt_planner import MusicPlanningRequest, compile_music_prompt_plan, create_music_prompt_plan

    calls = []

    class Planner:
        def plan_music_text(self, request):
            calls.append(dict(request))
            return {
                "prompt": "planned heroic orchestral cue with clear section changes",
                "lyrics": "[Verse 1]\nWe ride into dawn",
                "vocal_language": "en",
                "bpm": "111",
                "keyscale": "A minor",
                "timesignature": "6/8",
                "planner_backend": "fake-llm",
                "planner_model": "fake-model",
                "generated_fields": ["prompt", "lyrics", "bpm"],
                "confidence": 0.91,
                "extra_note": "kept in raw",
            }

    request = MusicPlanningRequest(prompt="heroic fantasy", duration_s=30, backend="acestep")
    plan = create_music_prompt_plan(request, provider=Planner(), mode="auto")
    compiled = compile_music_prompt_plan(plan, backend="diffusers", native_lyrics_supported=False)

    assert calls and calls[0]["prompt"] == "heroic fantasy"
    assert plan.planner_backend == "fake-llm"
    assert plan.planner_model == "fake-model"
    assert plan.bpm == 111
    assert plan.raw["extra_note"] == "kept in raw"
    assert compiled.lyrics is None
    assert "\n\nLyrics:\n[Verse 1]" in compiled.prompt
    assert "lyrics_folded_into_prompt" in compiled.metadata["planner_warnings"]


@pytest.mark.unit
def test_injected_music_text_planner_failure_falls_back_unless_required():
    from abstractmusic.prompt_planner import MusicPlanningRequest, create_music_prompt_plan

    class FailingPlanner:
        def create_plan(self, request):
            raise RuntimeError("planner unavailable")

    request = MusicPlanningRequest(prompt="ambient lofi", duration_s=60)
    plan = create_music_prompt_plan(request, provider=FailingPlanner(), mode="auto")

    assert plan.planner_backend == "deterministic-fallback"
    assert "Long-form structure:" in plan.prompt
    assert "text_planner_failed:RuntimeError" in plan.warnings

    with pytest.raises(RuntimeError):
        create_music_prompt_plan(request, provider=FailingPlanner(), mode="required")

    with pytest.raises(TypeError, match="required"):
        create_music_prompt_plan(request, mode="required")


@pytest.mark.unit
def test_music_text_planner_off_preserves_user_text_and_metadata():
    from abstractmusic.prompt_planner import MusicPlanningRequest, create_music_prompt_plan

    request = MusicPlanningRequest(
        prompt="raw prompt",
        lyrics="auto",
        duration_s=90,
        bpm=77,
        keyscale="G minor",
        auto_lyrics=True,
        enhance_prompt=True,
    )
    plan = create_music_prompt_plan(request, mode="off")

    assert plan.prompt == "raw prompt"
    assert plan.lyrics is None
    assert plan.bpm == 77
    assert plan.keyscale == "G minor"
    assert plan.structured_prompt is False
    assert "auto_lyrics_ignored_with_planner_off" in plan.warnings


@pytest.mark.unit
def test_cli_resolves_generation_text_for_acestep_and_generic_backend(capsys):
    from abstractmusic.cli import _resolve_generation_text

    ace_args = argparse.Namespace(
        backend="acestep",
        duration=30.0,
        vocal_language="en",
        bpm=None,
        keyscale=None,
        timesignature=None,
        instrumental=False,
        enhance_prompt=True,
        structure_prompt=True,
        auto_lyrics=True,
        text_planner="deterministic",
        print_plan=True,
    )
    prompt, lyrics, meta, composition_plan = _resolve_generation_text(ace_args, "heroic fantasy song", None)

    assert "French horns" in prompt
    assert lyrics is not None and "[Chorus]" in lyrics
    assert composition_plan is None
    assert meta["bpm"] == 96
    assert meta["structured_prompt"] is False
    assert meta["generated_lyrics"] is True
    assert "Effective music plan:" in capsys.readouterr().err

    generic_args = argparse.Namespace(**{**vars(ace_args), "backend": "musicgen", "print_plan": False})
    prompt, lyrics, meta, composition_plan = _resolve_generation_text(generic_args, "heroic fantasy song", "auto")

    assert lyrics is None
    assert composition_plan is None
    assert "\n\nLyrics:\n" in prompt
    assert meta["generated_lyrics"] is True


@pytest.mark.unit
def test_caption_sensitive_models_get_compact_structure_captions():
    """Long-form structure on a caption-sensitive checkpoint must render the
    section map only — none of the template prose that degrades guided XL
    output — while non-sensitive checkpoints keep the full bundle."""

    from abstractmusic.prompt_planner import MusicPlanningRequest, create_music_prompt_plan

    raw = "rhythmic space shooter game music, fast drums, evolving synth bass, arcade lead theme"
    sensitive = create_music_prompt_plan(
        MusicPlanningRequest(
            prompt=raw, duration_s=60.0, instrumental=True,
            model_id="ACE-Step/acestep-v15-xl-sft-diffusers",
        ),
        mode="deterministic",
    )
    assert sensitive.prompt.startswith(raw)
    assert "Long-form structure:" in sensitive.prompt
    assert "retro arcade space-shooter" not in sensitive.prompt
    assert "constant forward momentum" not in sensitive.prompt
    assert "continuous gameplay loop" not in sensitive.prompt
    assert "compact_caption_for_caption_sensitive_model" in sensitive.warnings

    full = create_music_prompt_plan(
        MusicPlanningRequest(
            prompt=raw, duration_s=60.0, instrumental=True,
            model_id="ACE-Step/acestep-v15-xl-turbo-diffusers",
        ),
        mode="deterministic",
    )
    assert "retro arcade space-shooter" in full.prompt
    assert "Long-form structure:" in full.prompt


@pytest.mark.unit
def test_explicit_enhance_overrides_caption_sensitivity_with_warning():
    from abstractmusic.prompt_planner import MusicPlanningRequest, create_music_prompt_plan

    plan = create_music_prompt_plan(
        MusicPlanningRequest(
            prompt="rhythmic space shooter game music", duration_s=30.0, instrumental=True,
            enhance_prompt=True, model_id="ACE-Step/acestep-v15-xl-sft-diffusers",
        ),
        mode="deterministic",
    )
    assert "retro arcade space-shooter" in plan.prompt  # the user's explicit call wins
    assert "caption_expansion_on_caption_sensitive_model" in plan.warnings


@pytest.mark.unit
def test_auto_lyrics_on_caption_sensitive_model_keeps_prompt_raw():
    from abstractmusic.prompt_planner import MusicPlanningRequest, create_music_prompt_plan

    raw = "rhythmic space shooter game music"
    plan = create_music_prompt_plan(
        MusicPlanningRequest(
            prompt=raw, duration_s=30.0, auto_lyrics=True,
            model_id="ACE-Step/acestep-v15-xl-base-diffusers",
        ),
        mode="deterministic",
    )
    assert plan.prompt == raw  # lyrics generated, caption untouched
    assert plan.lyrics


@pytest.mark.unit
def test_injected_planner_long_caption_on_sensitive_model_is_recorded():
    """LLM planners own their captions, but a long expansion aimed at a
    caption-sensitive checkpoint must land in provenance, not pass silently."""

    from abstractmusic.prompt_planner import MusicPlanningRequest, create_music_prompt_plan

    class TemplateHeavyPlanner:
        def plan_music_text(self, request):
            return {
                "prompt": "arcade music. " + "dense template instruction prose. " * 40,
                "planner_backend": "fake-llm",
            }

    plan = create_music_prompt_plan(
        MusicPlanningRequest(
            prompt="arcade music", duration_s=30.0,
            model_id="ACE-Step/acestep-v15-xl-sft-diffusers",
        ),
        provider=TemplateHeavyPlanner(),
        mode="auto",
    )
    assert "long_planner_caption_on_caption_sensitive_model" in plan.warnings

    # Same planner, non-sensitive checkpoint: no warning.
    plan2 = create_music_prompt_plan(
        MusicPlanningRequest(
            prompt="arcade music", duration_s=30.0,
            model_id="ACE-Step/acestep-v15-xl-turbo-diffusers",
        ),
        provider=TemplateHeavyPlanner(),
        mode="auto",
    )
    assert "long_planner_caption_on_caption_sensitive_model" not in plan2.warnings


@pytest.mark.unit
def test_manager_model_id_backstops_backends_that_report_none():
    """MusicManager(model_id=...) must reach the planner even when the backend's
    capabilities omit a model id — third-party backends often do."""

    from abstractmusic.music_manager import MusicManager
    from abstractmusic.types import GeneratedAsset, MusicBackendCapabilities

    class BareBackend:
        backend_id = "third-party"

        def get_capabilities(self):
            return MusicBackendCapabilities(supported_tasks=("text_to_music",), model_id=None)

        def generate_audio(self, request):
            self.last_request = request
            return GeneratedAsset(data=b"x", mime_type="audio/wav", metadata={})

    backend = BareBackend()
    mm = MusicManager(backend=backend, model_id="ACE-Step/acestep-v15-xl-sft-diffusers")
    mm.generate_audio(
        "rhythmic space shooter game music", planning=True, duration_s=60.0, instrumental=True
    )
    prompt = backend.last_request.prompt
    assert "retro arcade space-shooter" not in prompt  # compact fired via manager model_id
    assert "Long-form structure:" in prompt


@pytest.mark.unit
def test_cli_planner_sees_backend_default_model_when_none_given():
    import argparse

    from abstractmusic import cli as cli_mod

    seen = {}
    real = cli_mod.create_music_prompt_plan

    def capture(request, **kwargs):
        seen["model_id"] = request.model_id
        return real(request, **kwargs)

    args = argparse.Namespace(
        backend="acestep", model_id=None, duration=30.0, vocal_language=None, bpm=None,
        keyscale=None, timesignature=None, positive_styles=None, negative_styles=None,
        instrumental=True, enhance_prompt=False, structure_prompt=True, auto_lyrics=False,
        text_planner="deterministic", print_plan=False,
    )
    import unittest.mock as mock
    with mock.patch.object(cli_mod, "create_music_prompt_plan", side_effect=capture):
        cli_mod._resolve_generation_text(args, "short synth loop", None)
    assert seen["model_id"] == "ACE-Step/acestep-v15-xl-turbo-diffusers"
