# Completed: Stable Audio 3 Local Provider Spike

## Metadata
- Created: 2026-05-21
- Status: Completed
- Completed: 2026-05-21
- Priority: P1

## ADR status
- Governing ADRs: `docs/adr/0001_music_provider_abstraction_and_dependency_policy.md`
- ADR impact: None

## Context

Remote music endpoints are useful, but they do not solve AbstractMusic's strategic problem:
locally served, prompt-conditioned music generation that can run behind the same manager,
planning, quality-gate, and AbstractCore plugin contracts.

MiniMax has an official `music-2.6-free` API model, but its access and pricing docs are framed
around token plans, pay-as-you-go account balances, and limited-free quotas. That is not a clean
reason to add a third remote backend while ACE Music already works and ElevenLabs is implemented
but account-gated.

The next local engine candidate should be Stable Audio 3.0, starting with Stable Audio 3.0 Small
and only then testing Medium. Stability AI announced open-weight Small and Medium models for local
generation, with Small intended for full music composition on-device and up to two minutes of
audio, and Medium intended for stronger musicality and longer tracks. The runtime implementation
must be AbstractMusic-owned; the upstream repository may be inspected as reference material but
must not be imported, installed, wrapped, downloaded, or required at generation time.

Sources reviewed:

- MiniMax Music Generation API: https://platform.minimax.io/docs/api-reference/music-generation
- MiniMax pricing overview and pay-as-you-go docs: https://platform.minimax.io/docs/pricing/overview
- Stable Audio 3.0 announcement: https://stability.ai/news-updates/meet-stable-audio-3-the-model-family-built-for-artistic-experimentation-with-open-weight-models
- Stable Audio 3 research note: https://stability.ai/research/stable-audio-3
- Stable Audio 3 GitHub repository, reference only: https://github.com/Stability-AI/stable-audio-3

## Current code reality

- `src/abstractmusic/backends/stable_audio.py` still targets the older
  `stabilityai/stable-audio-open-small` model through `stable-audio-tools` and a short-clip
  rectified-flow path. `stable-audio` and `stable-audio-open-small` remain aliases for that legacy
  provider.
- `src/abstractmusic/backends/stable_audio_3.py` now exposes `abstractmusic:stable-audio-3` and
  CLI aliases such as `stable-audio-3`, `stable-audio-3-small`, and
  `stable-audio-3-small-music`.
- `src/abstractmusic/vendor/stable_audio3_min/` contains the package-owned minimum inference
  subset used by the Stable Audio 3 path. It is adapted from MIT-licensed reference code and
  excludes the upstream CLI, UI, training, LoRA, local checkout, and Git/package runtime paths.
- `pyproject.toml` has a dedicated `stable-audio-3` optional extra with only the top-level runtime
  libraries needed by the internal text-to-music path: Torch, Transformers, Safetensors, Hugging
  Face Hub, NumPy, and Einops.
- The model capability registry includes `stabilityai/stable-audio-3-small-music` and tracks
  `stabilityai/stable-audio-3-medium` behind the same backend, but Medium is not promoted.
- `abstractmusic.audio_analysis` harmonic-diversity and spectrotemporal artifact gates are used
  for real generation validation.

## Problem

AbstractMusic needs a second serious local/open-weight music engine beside ACE-Step. MusicGen Small
is non-commercial and useful mostly as a comparison baseline. The existing Stable Audio Open Small
path is short-clip oriented. HeartMuLa and YuE are promising but heavier and more complex to
integrate correctly.

## What we want to do

Implement and validate a Stable Audio 3.0 local provider spike, beginning with Stable Audio 3.0
Small. If Small loads, generates, and passes quality checks through AbstractMusic, decide whether
to promote it to a supported optional provider and whether Medium should become a second model
variant behind the same backend.

## Why

Stable Audio 3.0 appears to match the local-generation priority better than another remote wrapper:
open weights, long enough duration for actual music, continuation and inpainting controls that
could later map to AbstractMusic composition workflows, and local hardware targets including
laptops. The implementation cost is higher because AbstractMusic must own the runtime boundary
rather than depending on the upstream package as executable code.

## Requirements

- Keep the base package dependency-free.
- Put any unavoidable runtime dependencies behind a dedicated optional extra, most likely
  `abstractmusic[stable-audio-3]` or a broader platform extra that includes it.
- Do not use a local source checkout, PyPI package, Git package install, wrapper, bootstrapper, or
  source download as the runtime implementation.
- Use Hugging Face model weights only, subject to license acceptance. Implement model loading,
  conditioning, sampling, and decoding in package-owned AbstractMusic code.
- Reimplement only the minimum inference surface needed for text-to-music first; defer LoRA, UI,
  training, standalone CLI parity, and broad audio editing unless they become necessary.
- Keep `stable-audio-open-small` as a legacy/short-clip backend unless there is a clean migration
  path.
- Add capability metadata for Stable Audio 3.0 Small before exposing it through the CLI.
- Preserve the generic request flow: prompt, duration, seed, lyrics/vocal intent when supported,
  model choice, output metadata, fallback metadata, and quality metadata.
- Validate on at least 30-second and 120-second prompts before marking it recommended.
- Run repetition, harmonic-diversity, non-finite, clipping, and spectrotemporal artifact gates.
- Record Apple behavior separately from CUDA/GPU behavior.

## Suggested implementation

1. Inspect the official Stable Audio 3 model architecture, configs, checkpoint layout, and license
   boundaries as reference material only.
2. Add package-owned Stable Audio 3 runtime modules with lazy imports and no base import-time
   dependency changes.
3. Expose the explicit backend id `abstractmusic:stable-audio-3` and CLI aliases such as
   `stable-audio-3` and `stable-audio-3-small`, without changing the existing `stable-audio`
   alias until validation is complete.
4. Add registry entries for Stable Audio 3.0 Small and tracked Medium.
5. Add tests with fake model components/checkpoints so CI does not need model weights.
6. Run real local smokes with Hugging Face access accepted, then compare objective metrics and
   listening impressions against ACE-Step and known bad repeated-tone artifacts.

## Scope

- `pyproject.toml` optional extras
- `src/abstractmusic/backends/`
- `src/abstractmusic/cli.py`
- `src/abstractmusic/assets/music_model_capabilities.json`
- `docs/models.md`, `README.md`, and related coredoc pages if the provider validates
- backend unit tests, CLI smoke tests, capability tests, and real-generation notes

## Non-goals

- Do not add Stability remote API support as part of this item.
- Do not add or rely on the upstream `stable_audio_3` package as runtime implementation.
- Do not replace ACE-Step as the default local engine before real audio validation.
- Do not promise Stable Audio 3.0 Medium until Small proves the backend contract and runtime path.
- Do not bypass Hugging Face or Stability license gates.
- Do not hide provider-specific limitations behind generic success metadata.

## Dependencies and related tasks

- `docs/adr/0001_music_provider_abstraction_and_dependency_policy.md`
- `docs/backlog/completed/045_audio_artifact_screening_and_quality_metadata.md`
- `docs/backlog/completed/065_acestep_repetition_quality_gate.md`
- `docs/backlog/deprecated/0075_stable_audio_open_small_validation.md` (legacy Open Small validation item)
- `docs/backlog/proposed/0082_local_engine_priority_after_remote_baseline.md`

## Expected outcomes

- AbstractMusic has a documented decision on whether Stable Audio 3.0 Small is viable as the next
  local music engine under an AbstractMusic-owned runtime boundary.
- If viable, the provider is implemented behind an optional extra with import-light behavior and
  quality metadata.
- If not viable, the exact blocker is documented: package dependency issue, license/access gate,
  Apple runtime failure, unacceptable audio quality, or prompt-adherence failure.
- Medium is either queued behind the same backend after Small validates, or deferred with a clear
  hardware/runtime reason.

## Validation

Fast checks:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  tests/test_model_capabilities.py tests/test_cli_smoke.py \
  tests/test_stable_audio_backend.py tests/test_stable_audio3_backend.py \
  tests/test_stable_audio3_runtime_boundary.py
```

Real smoke, after installing the optional extra and accepting model access:

```bash
.venv/bin/abstractmusic --backend stable-audio-3 t2m \
  "rhythmic space shooter game music, urgent arpeggiated synth bass, fast drums, no vocals" \
  --duration 30 \
  --out smoke-artifacts/stable-audio-3-small-30s.wav

.venv/bin/abstractmusic --backend stable-audio-3 t2m \
  "heroic fantasy orchestral adventure, brass fanfare, driving percussion, cinematic strings, no vocals" \
  --duration 120 \
  --out smoke-artifacts/stable-audio-3-small-120s.wav
```

Quality checks:

- WAV structure and sample-rate validation
- non-finite, clipping, silence, and long-pause checks
- harmonic diversity and repeated-pulse detection
- reference-floor comparison against accepted ACE-Step reference and known bad artifacts
- human listening review before changing recommendation status

## Progress checklist

- [x] Confirm exact model identifiers, checkpoint tensor layout, architecture config, and license
      constraints.
- [x] Decide what minimum Stable Audio 3 runtime code belongs under AbstractMusic ownership.
- [x] Decide whether the extra should be `stable-audio-3` or folded into the platform extras only.
- [x] Add backend config, lazy imports, and fake-runtime unit tests.
- [x] Add capability registry metadata and CLI aliases.
- [x] Run local 30-second smoke.
- [x] Run local 120-second smoke.
- [x] Record Apple runtime behavior for 30-second and 120-second Small Music generation.
- [ ] Record GPU runtime behavior when available.
- [x] Update docs for the implemented Small Music path and current limits.

## Recent validation

2026-05-21 local Small Music validation used the fixed prompt
`rhythmic space shooter game music, urgent arpeggiated synth bass, fast drums, bright lead theme`,
seed `123`, and `--text-planner off`.

- R1 Stable Audio 3 Small Music at 30 seconds and 8 steps wrote
  `test-artifacts/sa3-validation/2026-05-21-r1/stable-audio-3-small/synthwave-30s-seed123.wav`
  but failed the reference-floor gate.
- R2 changed one global generation parameter, raising the default Stable Audio 3 step count to 16.
  The R2 artifact
  `test-artifacts/sa3-validation/2026-05-21-r2/stable-audio-3-small/synthwave-30s-seed123.wav`
  passed duration, WAV validity, harmonic-diversity, spectrotemporal artifact, continuity, and
  reference-floor gates.
- The comparison ACE-Step artifact
  `test-artifacts/sa3-validation/2026-05-21-r1/acestep-diffusers/synthwave-30s-seed123.wav`
  failed spectrotemporal and reference-floor gates for this prompt.
- R3 used the RType-style 120-second prompt
  `game music of the space shooter rtype, rhythmic arcade shooter, urgent arpeggiated synth bass,
  fast drums, bright lead theme, no vocals`, seed `123`, and `--text-planner off`.
  `test-artifacts/sa3-validation/2026-05-21-r3/stable-audio-3-small/rtype-120s-seed123.wav`
  passed all gates; the matching ACE-Step artifact
  `test-artifacts/sa3-validation/2026-05-21-r3/acestep-diffusers/rtype-120s-seed123.wav`
  failed the reference-floor gate.
- The implementation is not yet marked recommended because broader prompt/seed matrix and GPU
  runtime behavior are still open.

## Release report for 0.1.5

This section preserves the implementation summary because chat transcripts are not a durable
handoff artifact.

- Release target: `0.1.5`, because PyPI and GitHub already contain `0.1.4`.
- Implemented backend: `abstractmusic:stable-audio-3`, exposed as `--backend stable-audio-3`,
  `sa3`, `stable-audio-3-small`, and `stable-audio-3-small-music`.
- Model used by default: `stabilityai/stable-audio-3-small-music`; Medium is listed but not
  promoted.
- Runtime boundary: generation uses `src/abstractmusic/backends/stable_audio_3.py` plus the
  internal `src/abstractmusic/vendor/stable_audio3_min/` subset and Hugging Face weights/configs
  only. It does not import or install `stable_audio_3`, `stable_audio_tools`, Flash-Attn,
  `torchaudio`, `soundfile`, `tqdm`, or a local Stable Audio checkout.
- Dependency boundary: the base package remains dependency-free. `abstractmusic[stable-audio-3]`
  contains only `numpy`, `torch`, `transformers`, `safetensors`, `huggingface_hub`, and `einops`.
- Validation artifacts:
  - Stable Audio 3 120-second RType smoke:
    `test-artifacts/sa3-validation/2026-05-21-r3/stable-audio-3-small/rtype-120s-seed123.wav`
  - ACE-Step comparison with the same prompt/seed/duration:
    `test-artifacts/sa3-validation/2026-05-21-r3/acestep-diffusers/rtype-120s-seed123.wav`
  - Metrics:
    `test-artifacts/sa3-validation/2026-05-21-r3/analysis/summary.tsv`
- Objective result: Stable Audio 3 passed all gates in the R3 120-second comparison; ACE-Step
  failed the reference-floor gate for the same prompt.
- Local clean-room check: after uninstalling `stable-audio-tools`, `find_spec("stable_audio_3")`,
  `find_spec("stable_audio_tools")`, and `find_spec("flash_attn")` all returned false, and a short
  `--backend stable-audio-3` smoke still generated `/tmp/abstractmusic-sa3-no-upstream-tools.wav`.
- Remaining caveats: do not mark Stable Audio 3 recommended until broader prompt/seed validation
  and GPU behavior are recorded.

## Guidance for the implementing agent

Treat Stable Audio 3.0 as the next local-engine spike, not as a guaranteed replacement. Start with
Small because it validates the integration boundary at lower cost. Move to Medium only after the
backend, quality gates, and duration handling are proven through AbstractMusic.

## Completion report

2026-05-21:

- Stable Audio 3 Small Music is implemented as a package-owned runtime backend (`stable-audio-3`)
  with Hugging Face weights only (no upstream `stable_audio_3` runtime dependency).
- Real 30s and 120s smokes for Small Music were recorded and passed the existing quality gates
  (WAV validity, non-finite/clipping/silence checks, harmonic-diversity floor, and
  spectrotemporal artifact screening); see `test-artifacts/sa3-validation/`.
- Medium remains listed but is not promoted as a default/recommended path because it is CUDA
  oriented and GPU runtime behavior is not yet recorded in this repo.

Follow-ups:

- If GPU runtime behavior becomes a release-blocking question, add a new planned item that records
  a minimal CUDA smoke and documents hardware constraints. This item is closed because the local
  Small Music spike and validation evidence landed.
