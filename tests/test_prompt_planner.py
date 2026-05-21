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

    request = MusicPlanningRequest(prompt="heroic fantasy", duration_s=30, backend="acestep-diffusers")
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
        backend="acestep-diffusers",
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
