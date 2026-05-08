# Changelog

All notable changes to AbstractMusic will be documented in this file.

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
