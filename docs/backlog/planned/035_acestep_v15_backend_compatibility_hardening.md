# Planned: ACE-Step V1.5 Backend Compatibility Hardening

## Metadata
- Created: 2026-05-15
- Status: Planned
- Completed: N/A
- Priority: P1

## Context

The current custom ACE-Step v1.5 backend is the only implemented serious local music provider, but
it is also the most fragile part of the repo. It vendors custom Transformers model code and applies
runtime patches around several upstream behaviors.

## Current code reality

- `src/abstractmusic/backends/acestep_v15.py` patches Transformers RoPE validation,
  `PretrainedConfig.torch_dtype`, Diffusers Oobleck weight norm loading, and Hugging Face Hub
  download arguments.
- The vendored model code under `src/abstractmusic/vendor/acestep_v15_turbo/` imports modern
  Transformers internals, but `pyproject.toml` currently allows a very broad
  `transformers>=4.0,<6.0.0` range.
- The backend loads multiple subcomponents manually: DiT, Qwen3 tokenizer/text encoder, Oobleck
  VAE, and `silence_latent.pt`.
- The default `auto` device can select MPS, where the backend has accumulated dtype, CPU text
  encoder, memory cap, VAE tiling, and fallback logic.
- Fast tests mock the model, tokenizer, text encoder, VAE, and HF download path, so they do not
  validate real compatibility with supported library versions.

## Problem

The backend metadata permits dependency combinations that the implementation is unlikely to support.
The runtime patches may be necessary, but without version gates and real smoke tests they are
indistinguishable from brittle monkeypatching.

## What we want to do

Decide whether the custom ACE-Step v1.5 backend remains a supported provider, and if it does,
harden it with exact dependency policy, version-gated patches, and real validation.

## Why

The project cannot be called working if its default provider depends on unbounded upstream internals.
At the same time, the smaller ACE-Step v1.5 path may remain useful on consumer hardware even after
the XL Diffusers provider lands.

## Requirements

- Identify the exact tested versions or version ranges for:
  - `torch`
  - `diffusers`
  - `transformers`
  - `huggingface_hub`
  - `safetensors`
  - `einops`
- Tighten dependency bounds for any extra that installs the custom backend.
- Add startup/version checks that fail with actionable errors when a known-incompatible version is
  installed.
- Convert runtime monkeypatches into version-gated compatibility shims with comments explaining
  the upstream behavior they address.
- Add tests proving shims activate only when intended.
- Add one real generation smoke path for this backend under the integration-test matrix.
- Add a documented retirement path if the Diffusers ACE-Step provider fully replaces this backend.

## Suggested implementation

Start by creating a compatibility matrix in docs, then update dependency metadata and code checks.
Do not guess ranges from memory; inspect the vendored code imports and test in at least one clean
environment.

## Scope

- Dependency bounds and optional dependency messages for the custom ACE-Step backend.
- Version-gated compatibility shims.
- Real smoke validation for the backend.
- Docs describing whether it is default, legacy, or experimental.

## Non-goals

- Do not rewrite the whole ACE-Step inference loop unless the smoke tests prove it is necessary.
- Do not add new ACE-Step model variants here; XL Turbo/SFT are covered separately.
- Do not keep the custom backend as default if it cannot pass real generation validation.

## Dependencies and related tasks

- `docs/backlog/planned/030_acestep_diffusers_xl_provider.md`
- `docs/backlog/completed/040_real_generation_validation_matrix.md`
- `docs/backlog/planned/050_dependency_profiles_and_optional_providers.md`

## Expected outcomes

- The custom ACE-Step v1.5 backend is either clearly supported with tested versions or clearly
  marked legacy/experimental.
- Supported installs no longer break under dependency versions allowed by the package metadata.
- Runtime compatibility patches are understandable, tested, and bounded.

## Validation

- `python -m pytest -q -m "not integration"`
- Clean-environment install test for the relevant extra.
- Real short generation smoke test when model cache/hardware permits.
- Manual check that incompatible dependency versions produce actionable errors.

## Progress checklist

- [ ] Build a compatibility matrix from code imports and real installs.
- [ ] Tighten dependency metadata for the backend extra.
- [ ] Add version checks and version-gated shims.
- [ ] Add tests for compatibility behavior.
- [ ] Run real smoke validation or record why it could not run.

## Progress notes

2026-05-15: A 3-second Apple MPS smoke produced a valid WAV after the
`ACE-Step/Ace-Step1.5` snapshot finished downloading, but user listening feedback and
harmonic/envelope analysis rejected it as fast rotor-like audio rather than music. The custom path
is therefore not validated and should probably be retired or rewritten around the official 5Hz LM
semantic-code phase before broad default claims.

## Guidance for the implementing agent

Do not paper over import failures with broader fallbacks. A smaller set of known-good versions is
better than metadata that claims support for versions the backend cannot actually run.
