# Completed: REPL UX And Default Model Routing Hardening

## Metadata
- Created: 2026-05-21
- Status: Completed
- Completed: 2026-05-21
- Priority: P1

## Context

AbstractMusic exposes a CLI REPL intended for interactive prompt iteration across multiple music
engines and model ids. Real user sessions showed recurring UX failures:

- `/model ...` did not reliably change the engine, so `/run` could generate with an unexpected
  remote backend after selecting a local model.
- `/models` output was hard to scan and did not lead with the engine/backend information.
- There was no `/engines` overview and `/help` did not clearly explain defaults or switching
  rules.
- Local Hugging Face backends failed with `local_files_only=True` errors without an easy way to
  toggle cache-only vs download behavior.
- The intended REPL entry point for users is `abstractmusic cli`, but the CLI only exposed `repl`.

Separately, remote ACE Music duration requests were reported as unreliable depending on prompt
format.

## What We Want To Do

- Make REPL engine/model selection explicit, discoverable, and hard to misuse.
- Make defaults data-driven from the packaged model registry, not hardcoded in the CLI.
- Provide a clear downloads/offline toggle for local Hugging Face engines.
- Make `abstractmusic cli` a first-class entry point for starting the REPL.
- Ensure ACE Music duration requests are robust with tagged prompts.

## Completion report

2026-05-21:

- Improved REPL discoverability with `/status` and `/engines`, plus a clearer `/help`.
- Reformatted `/models` as an aligned table that starts with the engine column and groups by
  engine.
- Made `/model <id>` auto-switch the engine when the model is known in the packaged registry, and
  auto-clamp duration when the model declares a `max_duration_s`.
- Added a REPL `/download on|off` toggle plus a CLI `--download/--no-download` flag that controls
  Hugging Face `local_files_only` behavior for local engines.
- Added the `abstractmusic cli` alias for the REPL subcommand.
- Removed hardcoded default-model mappings and sourced defaults from
  `src/abstractmusic/assets/music_model_capabilities.json` via a new `default_for_backend` flag.
- Hardened ACE Music prompt formatting by wrapping prompts in `<prompt>...</prompt>` by default
  (unless `sample_mode` is enabled), matching the tagged-mode expectation for duration-sensitive
  requests.

Touched:

- `src/abstractmusic/cli.py`
- `src/abstractmusic/model_capabilities.py`
- `src/abstractmusic/assets/music_model_capabilities.json`
- `docs/getting-started.md`
- `docs/repl.md`
- `docs/README.md`
- `llms.txt`
- `llms-full.txt`
- `mkdocs.yml`
- `tests/test_model_capabilities.py`
- `docs/backlog/overview.md`
- `docs/backlog/completed/0086_repl_ux_and_default_model_routing_hardening.md`

Validation:

- `python -m py_compile src/abstractmusic/cli.py src/abstractmusic/model_capabilities.py`
- `python -m pytest -q`
