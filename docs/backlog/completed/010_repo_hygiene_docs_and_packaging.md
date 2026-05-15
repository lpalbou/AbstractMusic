# Planned: Repo Hygiene Docs And Packaging

## Metadata
- Created: 2026-05-15
- Status: Completed
- Completed: 2026-05-15
- Priority: P0

## Context

AbstractMusic needs to look and behave like the other AbstractFramework packages before model work
can be trusted. AbstractVision and AbstractVoice both have root project policy files and a core
documentation set. AbstractMusic currently has a README and changelog, but not the full baseline.

## Current code reality

- Root `README.md` exists but is mostly ACE-Step focused.
- Missing root files include `LICENSE`, `.gitignore`, `ACKNOWLEDGMENTS.md`, `CONTRIBUTING.md`,
  and `SECURITY.md`.
- Missing core docs include `docs/README.md`, `docs/getting-started.md`,
  `docs/architecture.md`, `docs/api.md`, and `docs/faq.md`.
- `pyproject.toml` uses legacy license metadata and base dependencies include Torch, Diffusers,
  Transformers, Accelerate, Safetensors, Hugging Face Hub, Einops, and NumPy.
- The working tree contains generated WAV outputs and Python bytecode/cache artifacts.
- CI exists in `.github/workflows/ci.yml`, but it only validates the current fast test path and
  package build.

## Problem

The repo is not clean enough for collaborative implementation or release. Generated artifacts and
missing policy/docs files make it hard to distinguish source truth from experiment residue.

## What we want to do

Set up the repository baseline: policy files, docs baseline, ignore rules, license metadata, and
packaging sanity checks.

## Why

Music model work is expensive and failure-prone. The repository needs strong source hygiene so
agents can iterate without accidentally treating local outputs or caches as implementation.

## Requirements

- Add `LICENSE` for the project license declared in package metadata.
- Add `.gitignore` entries for Python bytecode, build outputs, egg-info, pytest cache, model
  caches, generated WAV/MP3/FLAC outputs, and OS/editor artifacts.
- Add `ACKNOWLEDGMENTS.md` covering Diffusers, PyTorch, Transformers, Hugging Face Hub,
  Safetensors, Einops, NumPy, vendored ACE-Step code, ACE-Step weights, Qwen text encoder, and
  optional/future model caveats.
- Add `CONTRIBUTING.md` with editable install, test commands, lazy import rules, model artifact
  rules, docs rules, and public API stability expectations.
- Add `SECURITY.md` with private reporting to `contact@abstractcore.ai`.
- Create the core docs set: `docs/README.md`, `docs/getting-started.md`,
  `docs/architecture.md`, `docs/api.md`, `docs/faq.md`.
- Make docs truthful about what currently works, what is mocked, and what requires real model
  weights.
- Do not commit generated media or model weights.

## Suggested implementation

Borrow structure from AbstractVision and AbstractVoice, but write music-specific content. Update
package metadata to modern setuptools style only if it can be done without expanding the scope.

## Scope

- Root repo files and docs baseline.
- Ignore rules and metadata corrections.
- Documentation of current limitations.

## Non-goals

- Do not implement provider changes.
- Do not add a docs publishing system unless a small `mkdocs.yml` is clearly needed and follows
  sibling package conventions.

## Dependencies and related tasks

- `docs/backlog/planned/000_critical_assessment_and_roadmap.md`
- `docs/backlog/recurrent/dependency_and_artifact_hygiene.md`

## Expected outcomes

- A fresh clone has a clear README, docs entry points, contribution/security guidance, and no
  source-confusing generated artifacts.
- Future provider work has a reliable place to document behavior and limitations.

## Validation

- `python -m pytest -q -m "not integration"`
- `python -m build` and `python -m twine check dist/*` if build tooling is available.
- `git status --short` should not show generated media, bytecode, or cache files after cleanup.
- Manually inspect docs links.

## Progress checklist

- [ ] Add missing root policy files.
- [ ] Add docs baseline.
- [ ] Add or update ignore rules.
- [ ] Clean generated artifacts from source tracking/staging only when safe and intentional.
- [ ] Validate tests and packaging.

## Guidance for the implementing agent

Do not delete user-created files blindly. If generated outputs are tracked, remove them from source
control intentionally while preserving local files only if the user wants them kept.

## Completion report

### Summary

Implemented the repository hygiene baseline. Removed tracked generated artifacts from source
control, added ignore rules, added root project policy files, added the docs baseline, and fixed
packaging metadata so builds pass with modern setuptools license handling.

### Files and symbols touched

- Added `.gitignore`, `LICENSE`, `ACKNOWLEDGMENTS.md`, `CONTRIBUTING.md`, and `SECURITY.md`
- Added `docs/README.md`, `docs/getting-started.md`, `docs/architecture.md`, `docs/api.md`,
  `docs/faq.md`, and `docs/models.md`
- Updated `pyproject.toml` to use modern MIT license metadata and include packaged assets
- Removed tracked `.DS_Store`, generated WAV outputs, `src/abstractmusic.egg-info`, and tracked
  `__pycache__`/`.pyc` files

### Validation

- `python -m pytest -q -m "not integration"` passed.
- `python -m build && python -m twine check dist/*` passed.
- `git ls-files | rg '(\\.DS_Store$|\\.wav$|\\.pyc$|__pycache__|egg-info)'` returned no tracked
  generated artifacts.

### Residual risks

The base dependency split remains planned in `050_dependency_profiles_and_optional_providers.md`.
No git history rewrite was performed because the removed artifacts are not secrets and do not
require purging from published history.
