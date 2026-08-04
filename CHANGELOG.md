# Changelog

All notable changes to AbstractMusic will be documented in this file.

## [Unreleased]

No unreleased changes.

## [0.1.15] - 2026-08-04

### Fixed

- The deterministic prompt planner no longer forces template caption expansion for
  arcade/game-music style prompts. Expansion now happens only when requested
  (`--enhance-prompt`) or required by an opted-in feature (long-form structure maps,
  auto-lyrics). Controlled A/B generations showed the long template captions reliably
  degrade the guided ACE-Step XL checkpoints — sometimes into unpitched noise-sweep
  output — while the same seeds with the raw prompt generate music; the profile's
  bpm/keyscale/timesignature hints were tested separately and are harmless. The
  guidance-distilled turbo checkpoint tolerates the expanded captions and can benefit
  from them, so pass `--enhance-prompt` explicitly to keep the previous behavior there.
- Caption rendering is now checkpoint-aware. Registry entries can declare
  `caption_sensitive`; for those checkpoints the planner renders feature-driven captions
  (long-form structure maps, auto-lyrics) compactly — the user's prompt plus the section
  map, none of the template prose — and records a
  `compact_caption_for_caption_sensitive_model` provenance warning. An explicit
  `--enhance-prompt` still applies the full expansion, with a warning. The two XL
  Diffusers checkpoints are marked sensitive; at 60 s the full-bundle, compact, and raw
  configurations all passed the automated quality-gate set on XL SFT, but the full-bundle
  control later failed a listening review with a double-time opening the gates cannot
  measure — reinforcing the compact default. Injected LLM planners
  keep authority over their captions; a long planner caption aimed at a caption-sensitive
  checkpoint is recorded as `long_planner_caption_on_caption_sensitive_model` in plan
  provenance. Caption-policy warnings print to stderr even without `--print-plan`.
- Corrected the catalog status of `ACE-Step/Ace-Step1.5`, `ACE-Step/acestep-v15-base`, and
  `ACE-Step/acestep-v15-sft`. These official checkpoints are published in the native ACE-Step
  transformers repository layout (no Diffusers `model_index.json`) and cannot be loaded by the
  `acestep` backend's `AceStepPipeline`; they were previously labelled pipeline-compatible.
  They remain in the catalog as `incompatible-native-runtime-layout` and are no longer offered
  as runnable by discovery, even when their weights are cached locally — including when one of
  them is set as `music_model_id` for the generic diffusers route.
- Selecting an incompatible checkpoint explicitly now fails fast with an error naming the
  loadable Diffusers-layout alternatives, instead of surfacing a misleading network error from
  `from_pretrained`. When the only cached ACE-Step weights are an incompatible checkpoint,
  `provider_details(...)` reports `no-loadable-weights` and names it, rather than claiming no
  weights exist.

### Added

- `abstractmusic.audio_analysis.is_probably_noise_texture(...)`: a warning-grade screen for
  the wind/whoosh collapse signature — non-silent audio with no percussive transients, a
  static spectral envelope, and little harmonic variety — which the existing validity and
  repetition checks do not catch. It flagged every listening-confirmed noise-sweep artifact
  in the validation corpus while sparing all musical files; thresholds are fitted to that
  corpus, so treat a flag as a signal to review rather than proof. The `acestep` backend now
  reports it as `audio_stats.probably_noise_texture` in generation metadata.
- `abstractmusic.audio_analysis.evaluate_music_quality_gates(candidate, reference)`: the
  canonical validation gate set (validity, noise texture, collapse, pitch variety, reference
  floor, repetition) that the packaged model registry's validation notes refer to. The gate set
  does not measure tempo stability: a listening review found a 60-second generation with a
  double-time opening that passed every gate.
- `abstractmusic.audio_analysis.inspect_tempo_trajectory_file/bytes(...)`: a diagnostic
  windowed beat-period trajectory (opening vs steady tempo, plateau detection) for inspecting
  rhythm defects. Its `has_probably_double_time_opening` flag is experimental — fitted to a
  single listening-confirmed failure — and is deliberately not part of the canonical gate set.
- Registered the official Diffusers-layout XL checkpoints `ACE-Step/acestep-v15-xl-sft-diffusers`
  and `ACE-Step/acestep-v15-xl-base-diffusers` under the `acestep` backend. They are not
  guidance-distilled: AbstractMusic applies the model card's recommended
  `num_inference_steps=50` and `guidance_scale=7.0` by default for non-turbo variants. The
  8-step XL Turbo checkpoint remains the default and recommended model.

## [0.1.14] - 2026-08-03

### Changed

- Made AbstractCore music discovery cold-start cheap. `available_providers`, `list_models`,
  `list_provider_models`, and `capability_catalog` no longer import `torch`, `transformers`, or
  `diffusers`: a full `capability_catalog(task="text_to_music")` went from ~6-30s and ~3800 loaded
  modules to ~0.03s and ~190 modules.
- Local provider availability is now answered from the filesystem: a provider is reported when its
  optional runtime is installed **and** at least one of its models already has weights in the
  Hugging Face cache. `list_models` reports the models that can run now, each with
  `metadata.cached`.
- Remote provider availability is now measured instead of assumed. Providers with an API key
  configured are probed concurrently, in a single round bounded by a 5s deadline; a provider whose
  API rejects the key or does not answer is no longer reported as available. An unresponsive
  provider costs that deadline once, not once per discovery call, and never accumulates threads.
- ACE-Step runtime detection checks whether the installed `diffusers` ships the ACE-Step pipeline by
  inspecting the package on disk. This is correct for pre-releases, editable installs, and source
  checkouts.

### Compatibility

- Discovery results are narrower than in 0.1.13. `available_providers(...)` and `list_models(...)`
  now omit local providers whose weights are not downloaded and remote providers that do not answer,
  where previously they were listed as available. If you relied on those methods for the full
  catalog, use `provider_details(...)`, or read the packaged registry through
  `MusicModelCapabilitiesRegistry`, which is unchanged and still lists every known model.

### Added

- `provider_details(task=...)` on the capability, also included in `capability_catalog(...)`. It
  reports every known provider with `usable` plus the reason it is not — rejected credentials, an
  uninstalled extra, or weights that have not been downloaded — along with its full model catalog
  and which of those models are cached. `available_providers(...)` stays limited to runnable
  providers, so an empty list is no longer a dead end. (Addresses backlog 0087.)
- `abstractmusic.availability`: a stdlib-only module for Hugging Face cache-presence checks
  (`hf_cache_root`, `is_model_cached`, `cached_model_ids`) and bounded parallel endpoint probes
  (`RemoteEndpoint`, `probe_endpoints`). It mirrors `huggingface_hub`'s own cache-root precedence so
  discovery looks exactly where the loader will look, without adding a dependency. Presence honours
  the checked-out revision and shard index files, so a half-downloaded checkpoint is not reported as
  ready; leftover `.incomplete` blobs, which `huggingface_hub` keeps for resuming, do not hide an
  otherwise complete model.
- `health_endpoint()` on `AceMusicBackendConfig` and `ElevenLabsMusicBackendConfig`, so each remote
  backend owns the endpoint and auth header used to check that it is answering.

## [0.1.13] - 2026-06-03

### Changed

- Added route-specific audio generation registry coverage so music output and sound-effects output can be selected independently by hosts.
- Tightened Stable Audio model capability metadata and plugin discovery tests for the separated `output.music` and `output.sound` routes.

## [0.1.12] - 2026-05-23

### Changed

- Collapsed the public ACE-Step surface to a single `acestep` backend backed by the package-owned
  `AceStepPipeline` adapter and removed the old `acestep-v15` / `acestep-diffusers` backend split.
- Removed vendored ACE-Step v1.5 model/runtime code from the package and kept local ACE-Step
  execution on Hugging Face weights plus AbstractMusic-owned orchestration only.
- Reduced the packaged ACE-Step discovery catalog to reviewed official checkpoints under the public
  `acestep` backend and stopped surfacing unreviewed community conversions in the default model
  registry.
- Made AbstractCore music discovery report backend-oriented provider ids and only return
  providers/models whose runtime is actually usable in the current environment.

### Docs

- Updated README, getting-started, API, models, troubleshooting, backlog overview, and
  `llms-full.txt` for the single-backend ACE-Step contract and truthful discovery behavior.
- Repaired the AI-readable docs index link to `ACKNOWLEDGMENTS.md`.

## [0.1.11] - 2026-05-21

### Changed

- `stable-audio` no longer requires installing `stable-audio-tools`; AbstractMusic now vendors the minimal `stable-audio-tools==0.0.19` model code for Stable Audio Open Small.

### Docs

- Added Stable Audio 3 Small SFX and Stable Audio Open Small SFX examples to `README.md` and `docs/getting-started.md`.

## [0.1.10] - 2026-05-21

### Added

- Exposed `positive_styles` / `negative_styles` in the AbstractCore plugin operation schema so hosts can discover and pass style tags.

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
