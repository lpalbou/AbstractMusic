# Changelog

All notable changes to AbstractMusic will be documented in this file.

## [Unreleased]

No unreleased changes.

## [0.1.9] - 2026-05-21

### Added

- Added provider-neutral style tags: CLI `--style/--negative-style` and REPL `/style` + `/negative-style`.
- Added `stabilityai/stable-audio-3-small-sfx` to the Stable Audio 3 backend and packaged model registry.

### Changed

- AbstractCore host text-planner schema now allows `positive_styles` / `negative_styles` in structured plans.

### Docs

- Documented style tags, instruments, and vocal-trait prompting in `docs/repl.md`, `docs/api.md`, and `docs/faq.md`.
- Added Stable Audio 3 Small SFX to `docs/models.md`.

## [0.1.8] - 2026-05-21

### Added

- Added `abstractmusic cli` as an alias for launching the interactive REPL.
- Added REPL discovery commands: `/status` and `/engines`.
- Added a download/offline toggle: CLI `--download/--no-download` and REPL `/download on|off`.

### Changed

- REPL `/model <id>` now auto-switches the engine when the model is known in the packaged registry,
  and auto-clamps duration when a model has a declared max duration.
- REPL `/models` output is now an aligned table grouped by engine for readability.
- CLI default models are now sourced from the packaged model registry (via `default_for_backend`)
  instead of hardcoded mappings.

### Docs

- Added a dedicated REPL guide (`docs/repl.md`) and linked it from the docs index and MkDocs nav.
- Refreshed backlog lifecycle state (moved completed/deprecated items) and expanded `llms-full.txt`
  to include ADR and backlog overview content for tool-friendly ingest.

## [0.1.7] - 2026-05-21

### Added

- Exposed a Core-friendly music residency surface (`load_resident_model`, `list_loaded_models`,
  `list_resident_models`, `unload_resident_model`) for local backends, with best-effort backend
  `preload()` / `unload()` hooks.

### Changed

- Expanded the ACE-Step Diffusers model registry with additional community Diffusers conversions
  (Runware) so users can test smaller/alternate DiT variants via `--model-id`.
- ACE-Step Diffusers now advertises and forwards `guidance_scale` when provided.

### Docs

- Documented the residency surface and community ACE-Step Diffusers conversions.

## [0.1.6] - 2026-05-21

### Added

- Added an AbstractCore plugin backend for Stable Audio Open Small (`abstractmusic:stable-audio`)
  so discovery/catalog routing matches the advertised optional backend.

### Changed

- ACE Music requests use `<prompt>...</prompt>` tagged mode by default for more reliable duration
  behavior.

### Fixed

- ACE Music WAV responses are trimmed/padded to the requested duration when the remote API returns
  a mismatched length.

## [0.1.5] - 2026-05-21

### Added

- Added a stdlib-only `elevenlabs` remote backend scoped to ElevenLabs Music endpoints, with
  composition-plan request support and AbstractCore plugin registration.
- Added an explicit `stable-audio-3` local backend for `stabilityai/stable-audio-3-small-music`
  using AbstractMusic-owned runtime code and Hugging Face weights/configs only.

### Changed

- Added a provider-neutral `MusicCompositionPlan` request path so host text planners can hand
  structured music intent to compatible backends without coupling planning to transport code.
- Kept Stable Audio 3 behind its own optional extra and excluded upstream Stable Audio runtime
  packages, Flash-Attn, audio codec libraries, UI, training, LoRA, and local-checkout paths from
  that minimal dependency profile.

## [0.1.4] - 2026-05-21

### Added

- Added a stdlib-only `acemusic` remote backend for ACE Music-compatible hosted text-to-music APIs.
- Added `remote` as a no-op optional extra and expanded `all-apple` / `all-gpu` profiles to include
  supported local runtime dependency groups.

### Changed

- Changed the CLI and AbstractCore plugin default backend to lightweight remote generation
  (`acemusic`) so `pip install abstractmusic` remains remote-capable without local ML libraries.
- Standardized ACE Music remote configuration on `ACEMUSIC_API_KEY` and `ACEMUSIC_BASE_URL`.
- Documented that Suno is not included as a default backend until an official stable API contract is
  available; OpenAI and Anthropic are not treated as music-generation providers.

## [0.1.3] - 2026-05-21

### Added

- Added a MkDocs documentation build and GitHub Pages deployment path to the release workflow.
- Added release rehearsal support for manually dispatched releases, including explicit publish
  confirmation before PyPI publication.
- Added AI-readable documentation indexes with `llms.txt` and `llms-full.txt`.
- Added troubleshooting and release-process documentation for installation, local runtime,
  Apple/MPS, Stable Audio, and publishing workflows.
- Added Dependabot configuration for GitHub Actions and Python dependency maintenance.

### Changed

- Documented the AbstractMusic text-planning boundary, including deterministic fallback planning
  and host-injected planning services.
- Documented the default ACE-Step Diffusers backend as the package-owned `acestep` route, with
  Hugging Face model weights allowed and external ACE-Step source/runtime code excluded.
- Documented Apple MPS dtype behavior for ACE-Step Diffusers, preferring stable MPS dtypes before
  CPU fallback.

### Fixed

- Made Python 3.10 test metadata compatible by using `tomli` where `tomllib` is unavailable.
- Hardened release validation to refuse duplicate PyPI versions before publishing.

## [0.1.2] - 2026-05-08

### Added

- Added GitHub Actions CI for Python 3.10 through 3.12 with pytest and package
  build checks.
- Added a trusted-publishing release workflow for tagged or manually dispatched
  releases, including version/changelog validation, distribution artifacts,
  PyPI publication, and GitHub Release creation.
- Added an AbstractMusic GitHub bug report template.

### Changed

- Moved the package version source to `abstractmusic._version` so release
  validation and packaging metadata use the same lightweight version module.

## [0.1.1] - 2026-05-08

### Added

- Added framework-wide install-profile aliases:
  `abstractmusic[apple]`, `abstractmusic[gpu]`,
  `abstractmusic[all-apple]`, and `abstractmusic[all-gpu]`.

### Notes

- AbstractMusic is currently local-first and already installs its ACE-Step
  runtime stack in the base package. The new extras are no-op compatibility
  aliases so Core, Gateway, and root aggregate profiles can cascade cleanly.

## [0.1.0] - 2026-02-12

### Added

- Initial local music/audio generation package with native AbstractCore
  capability plugin integration.
