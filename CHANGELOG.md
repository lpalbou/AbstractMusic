# Changelog

All notable changes to AbstractMusic will be documented in this file.

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
