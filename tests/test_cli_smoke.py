import pytest


@pytest.mark.unit
def test_cli_help_does_not_error():
    from abstractmusic.cli import main

    # Should exit cleanly and not attempt model downloads/import heavy stacks.
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0


@pytest.mark.unit
def test_cli_prints_clean_error_without_traceback(monkeypatch, capsys):
    from abstractmusic.cli import main
    from abstractmusic.errors import AbstractMusicError

    def _boom(args):
        raise AbstractMusicError("provider unavailable")

    monkeypatch.setattr("abstractmusic.cli._cmd_t2m", _boom)

    code = main(["t2m", "hello"])

    assert code == 1
    captured = capsys.readouterr()
    assert captured.err.strip() == "ERROR: provider unavailable"
    assert "Traceback" not in captured.err


@pytest.mark.unit
def test_cli_allows_common_flags_after_subcommand():
    # Argparse subparsers normally reject top-level options placed after the subcommand.
    # We intentionally support the natural ordering used in docs:
    #   abstractmusic --backend acestep t2m "..." --duration 10 --out out.wav
    from abstractmusic.cli import build_parser

    args = build_parser().parse_args(
        ["--backend", "acestep", "t2m", "sci fi music", "--duration", "10", "--out", "out.wav"]
    )
    assert args.backend == "acestep-diffusers"
    assert args.cmd == "t2m"
    assert float(args.duration) == 10.0


@pytest.mark.unit
def test_cli_default_backend_is_light_remote_acemusic(monkeypatch):
    from abstractmusic.cli import build_parser

    monkeypatch.delenv("ABSTRACTMUSIC_BACKEND", raising=False)
    monkeypatch.delenv("ACEMUSIC_BASE_URL", raising=False)

    args = build_parser().parse_args(["t2m", "sci fi music", "--duration", "10"])

    assert args.backend == "acemusic"
    assert args.format == "wav"
    assert args.acemusic_base_url == "https://api.acemusic.ai"


@pytest.mark.unit
def test_cli_accepts_acemusic_remote_aliases_and_formats():
    from abstractmusic.cli import build_parser

    args = build_parser().parse_args(
        [
            "--engine",
            "ace-music",
            "t2m",
            "sci fi music",
            "--format",
            "mp3",
            "--acemusic-base-url",
            "https://api.example.test",
        ]
    )

    assert args.backend == "acemusic"
    assert args.format == "mp3"
    assert args.acemusic_base_url == "https://api.example.test"


@pytest.mark.unit
def test_cli_accepts_elevenlabs_music_aliases_and_config():
    from abstractmusic.cli import build_parser

    args = build_parser().parse_args(
        [
            "--engine",
            "eleven",
            "t2m",
            "cinematic instrumental",
            "--format",
            "mp3",
            "--elevenlabs-base-url",
            "https://api.example.test",
            "--elevenlabs-model",
            "music_v1",
            "--elevenlabs-output-format",
            "mp3_44100_128",
            "--composition-mode",
            "plan",
        ]
    )

    assert args.backend == "elevenlabs"
    assert args.format == "mp3"
    assert args.elevenlabs_base_url == "https://api.example.test"
    assert args.elevenlabs_model == "music_v1"
    assert args.elevenlabs_output_format == "mp3_44100_128"
    assert args.composition_mode == "plan"


@pytest.mark.unit
def test_cli_accepts_explicit_standalone_acestep_v15_backend():
    from abstractmusic.cli import build_parser

    args = build_parser().parse_args(
        ["--backend", "acestep-v15", "t2m", "sci fi music", "--duration", "5", "--out", "out.wav"]
    )
    assert args.backend == "acestep-v15"
    assert args.cmd == "t2m"


@pytest.mark.unit
def test_cli_rejects_removed_acestep_source_arg():
    from abstractmusic.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--backend",
                "acestep",
                "--acestep" + "-source-dir",
                "/tmp/ACE-Step-1.5",
                "t2m",
                "sci fi music",
            ]
        )


@pytest.mark.unit
def test_cli_rejects_local_model_path():
    from abstractmusic.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--model-id", "./local-model", "repl"])


@pytest.mark.unit
def test_cli_rejects_removed_cache_dir():
    from abstractmusic.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--cache-dir", "/tmp/hf-cache", "repl"])


@pytest.mark.unit
def test_cli_accepts_vocal_language_hint():
    from abstractmusic.cli import build_parser

    args = build_parser().parse_args(
        [
            "--backend",
            "acestep",
            "t2m",
            "sci fi music",
            "--vocal-language",
            "en",
        ]
    )

    assert args.vocal_language == "en"


@pytest.mark.unit
def test_cli_accepts_prompt_enhancement_and_lyrics_flags():
    from abstractmusic.cli import build_parser

    args = build_parser().parse_args(
        [
            "--backend",
            "acestep",
            "t2m",
            "heroic fantasy music",
            "--enhance-prompt",
            "--no-structure-prompt",
            "--auto-lyrics",
            "--text-planner",
            "auto",
            "--print-plan",
        ]
    )

    assert args.enhance_prompt is True
    assert args.structure_prompt is False
    assert args.auto_lyrics is True
    assert args.text_planner == "auto"
    assert args.instrumental is False
    assert args.print_plan is True


@pytest.mark.unit
def test_cli_accepts_quality_retry_control():
    from abstractmusic.cli import build_parser

    args = build_parser().parse_args(
        [
            "--backend",
            "acestep",
            "t2m",
            "sci fi music",
            "--quality-retries",
            "3",
        ]
    )

    assert args.quality_retries == 3


@pytest.mark.unit
def test_cli_accepts_acestep_diffusers_backend():
    from abstractmusic.cli import build_parser

    args = build_parser().parse_args(
        ["--backend", "acestep-diffusers", "t2m", "sci fi music", "--duration", "5", "--out", "out.wav"]
    )
    assert args.backend == "acestep-diffusers"
    assert args.cmd == "t2m"


@pytest.mark.unit
def test_cli_accepts_engine_alias_for_backend_flag():
    from abstractmusic.cli import build_parser

    args = build_parser().parse_args(["--engine", "xl", "repl"])

    assert args.backend == "acestep-diffusers"
    assert args.cmd == "repl"


@pytest.mark.unit
def test_cli_accepts_generic_ace_alias_as_standalone_backend():
    from abstractmusic.cli import build_parser

    args = build_parser().parse_args(["repl", "--engine", "ace"])

    assert args.backend == "acestep-diffusers"


@pytest.mark.unit
def test_cli_accepts_legacy_aliases_for_custom_acestep_backend():
    from abstractmusic.cli import build_parser

    args = build_parser().parse_args(["repl", "--engine", "v15"])
    assert args.backend == "acestep-v15"

    args = build_parser().parse_args(["repl", "--engine", "legacy"])
    assert args.backend == "acestep-v15"


@pytest.mark.unit
def test_cli_accepts_musicgen_and_stable_audio_engines():
    from abstractmusic.cli import build_parser

    args = build_parser().parse_args(["--engine", "musicgen-small", "t2m", "lo-fi music", "--duration", "5"])
    assert args.backend == "musicgen"
    assert args.cmd == "t2m"

    args = build_parser().parse_args(["repl", "--engine", "stable-audio-open-small", "--duration", "5"])
    assert args.backend == "stable-audio"
    assert args.cmd == "repl"

    args = build_parser().parse_args(["--engine", "sa3", "t2m", "arcade music", "--duration", "30"])
    assert args.backend == "stable-audio-3"
    assert args.cmd == "t2m"

    args = build_parser().parse_args(["repl", "--engine", "stable-audio-3-small-music", "--duration", "30"])
    assert args.backend == "stable-audio-3"
    assert args.cmd == "repl"


@pytest.mark.unit
def test_music_repl_switches_engine_and_parameters_without_loading_backend(capsys):
    from abstractmusic.cli import MusicREPL, build_parser

    args = build_parser().parse_args(["repl", "--duration", "5"])
    repl = MusicREPL(args)

    repl.onecmd("/engine xl")
    repl.onecmd("/duration 12")
    repl.onecmd("/bpm 128")
    repl.onecmd("/keyscale F# major")
    repl.onecmd("/timesignature 4")
    repl.onecmd("/steps 16")
    repl.onecmd("/seed 42")
    repl.onecmd("/lm-backend cpu")
    repl.onecmd("/verbose on")
    repl.onecmd("/lyrics [Instrumental]")
    repl.onecmd("/enhance-prompt on")
    repl.onecmd("/structure-prompt off")
    repl.onecmd("/auto-lyrics on")
    repl.onecmd("/instrumental on")
    repl.onecmd("/print-plan on")
    repl.onecmd("/text-planner off")
    repl.onecmd("/format flac")
    repl.onecmd("/composition-mode plan")
    repl.onecmd("/params")

    assert repl.args.backend == "acestep-diffusers"
    assert repl.args.duration == 12.0
    assert repl.args.bpm == 128
    assert repl.args.keyscale == "F# major"
    assert repl.args.timesignature == "4"
    assert repl.args.steps == 16
    assert repl.args.seed == 42
    assert repl.args.lm_backend == "cpu"
    assert repl.args.verbose is True
    assert repl.args.lyrics == "[Instrumental]"
    assert repl.args.enhance_prompt is True
    assert repl.args.structure_prompt is False
    assert repl.args.auto_lyrics is True
    assert repl.args.instrumental is True
    assert repl.args.print_plan is True
    assert repl.args.text_planner == "off"
    assert repl.args.format == "flac"
    assert repl.args.composition_mode == "plan"
    assert repl._manager is None
    out = capsys.readouterr().out
    assert "engine: acestep-diffusers" in out
    assert "duration: 12" in out
    assert "verbose: on" in out
    assert "lyrics: [Instrumental]" in out
    assert "enhance-prompt: on" in out
    assert "structure_prompt: off" in out
    assert "auto-lyrics: on" in out
    assert "instrumental: on" in out
    assert "print-plan: on" in out
    assert "text_planner: off" in out
    assert "format: flac" in out
    assert "composition-mode: plan" in out


@pytest.mark.unit
def test_music_repl_bare_prompt_dispatches_generate(monkeypatch):
    from abstractmusic.cli import MusicREPL, build_parser

    args = build_parser().parse_args(["repl"])
    repl = MusicREPL(args)
    calls = []
    monkeypatch.setattr(repl, "_generate", lambda prompt: calls.append(prompt))

    repl.onecmd("bright melodic synth loop")

    assert calls == ["bright melodic synth loop"]


@pytest.mark.unit
def test_music_repl_prompt_state_runs_current_prompt(monkeypatch):
    from abstractmusic.cli import MusicREPL, build_parser

    args = build_parser().parse_args(["repl"])
    repl = MusicREPL(args)
    calls = []
    monkeypatch.setattr(repl, "_generate", lambda prompt=None: calls.append(prompt or repl.current_prompt))

    repl.onecmd("/prompt bright melodic synth loop")
    repl.onecmd("/run")

    assert repl.current_prompt == "bright melodic synth loop"
    assert calls == ["bright melodic synth loop"]


@pytest.mark.unit
def test_music_repl_normalizes_hyphenated_slash_commands(monkeypatch):
    from abstractmusic.cli import MusicREPL, build_parser

    args = build_parser().parse_args(["repl"])
    repl = MusicREPL(args)
    calls = []
    monkeypatch.setattr(repl, "_generate", lambda prompt=None: calls.append(prompt))

    repl.onecmd("/guidance-scale 4")
    repl.onecmd("/shift 3")
    repl.onecmd("/infer-method sde")
    repl.onecmd("/lm-temperature 0.95")
    repl.onecmd("/lm-cfg-scale 2.5")
    repl.onecmd("/out-dir /tmp/abstractmusic-repl")

    assert repl.args.guidance_scale == 4.0
    assert repl.args.shift == 3.0
    assert repl.args.infer_method == "sde"
    assert repl.args.lm_temperature == 0.95
    assert repl.args.lm_cfg_scale == 2.5
    assert str(repl.out_dir) == "/tmp/abstractmusic-repl"
    assert calls == []


@pytest.mark.unit
def test_music_repl_request_parameters_do_not_reload_manager():
    from abstractmusic.cli import MusicREPL, build_parser

    args = build_parser().parse_args(["repl"])
    repl = MusicREPL(args)
    repl._manager = object()
    repl._manager_dirty = False

    repl.onecmd("/duration 30")
    repl.onecmd("/bpm 128")
    repl.onecmd("/steps 8")
    repl.onecmd("/seed 123")
    repl.onecmd("/guidance 1")
    repl.onecmd("/shift 3")
    repl.onecmd("/infer-method ode")
    repl.onecmd("/lm-temperature 0.85")

    assert repl._manager_dirty is False
    repl.onecmd("/engine xl")
    assert repl._manager_dirty is True
