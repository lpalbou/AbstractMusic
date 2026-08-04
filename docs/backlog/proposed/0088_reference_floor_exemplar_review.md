# Proposed: Review the accepted reference-floor exemplar and entropy ratio gate

## Metadata
- Created: 2026-08-04
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: `docs/adr/0001_music_provider_abstraction_and_dependency_policy.md`
- ADR impact: None

## Context

The harmonic-diversity reference floor compares candidates against
`smoke-artifacts/user-repl/official-1p7b-30s-hq.wav`. During the ACE-Step XL sft/base validation
matrix (2026-08-04, 8 gated 30-second runs), an adversarial audit of the gate evidence surfaced
two patterns worth a deliberate review:

- Every reference-floor failure in the matrix (7 of 7, across sft, base, and a turbo control)
  failed the same ratio: `spectral_entropy_cv_ratio`, once by 1% (0.3466 vs the 0.35 cutoff).
  No other floor ratio dominated.
- The accepted reference itself trips `has_long_low_energy_gap` and `has_long_trailing_fade`
  (a ~6.3-second near-silent tail), which none of the failing candidates do. Entropy-CV rewards
  such dynamic-range swings, so the exemplar may set an entropy-variance bar that steady,
  artifact-free arrangements legitimately do not clear.
- The validated default (`acestep-v15-xl-turbo-diffusers`) also failed the floor and tripped the
  static-spectral-loop flag on one control prompt/seed, so the floor is stricter than the
  default's own typical output on at least some prompts.

## Problem or opportunity

The floor may be encoding "resembles this specific HQ track's dynamics" rather than "clears a
minimum musical-diversity bar". If so, it under-reports usable checkpoints (XL sft failed 4 of 4
runs on this single ratio family) while remaining fully passable (XL base passed 5/5 gates at
seed 7).

## Proposed direction

- Decide whether the entropy-CV ratio should stay reference-relative or become an absolute
  threshold (the absolute `has_spectral_motion` cutoff of 0.08 was met by every failing run).
- Consider selecting or adding a reference exemplar without long near-silent segments, or gating
  reference eligibility on the energy-continuity checks the candidates are implicitly held to.
- Re-run the recorded XL sft/base matrix against any revised floor before changing any model's
  registry status; keep the current statuses truthful until then.

## Non-goals

- Do not loosen the collapse, low-pitch-variety, or spectrotemporal artifact gates; they
  discriminated correctly throughout the matrix (including catching a turbo degeneration).
