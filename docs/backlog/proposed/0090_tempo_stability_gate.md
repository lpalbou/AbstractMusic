# Proposed: Tempo-stability quality gate needs a labeled rhythm corpus

## Metadata
- Created: 2026-08-04
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: `docs/adr/0001_music_provider_abstraction_and_dependency_policy.md`
- ADR impact: None

## Context

A maintainer listening review (2026-08-04) found a 60-second XL SFT generation
(`smoke-artifacts/xl-validation/A_sft_60s_fullbundle_seed123.wav`) the owner described as "much
too fast until maybe 20s, then it's ok" — and it passed every gate in
`evaluate_music_quality_gates`. The windowed tracker localizes a flat 2:1 onset-rate plateau over
~0-12 s; the 12-20 s span the owner heard as fast reads ~100 BPM, an unresolved
human/tracker disagreement worth keeping. None of the canonical gates measure tempo over time.

Windowed beat-period analysis (`abstractmusic.audio_analysis.inspect_tempo_trajectory_*`,
added as a diagnostic) quantifies the defect: a flat ~201 BPM opening plateau, then ~100 BPM,
an exact 2:1 relation. But the obvious gate — opening/steady tempo ratio — is falsified by the
existing labels: the listening-confirmed *good* track `sft_30s_seed123.wav` has a *higher*
opening ratio (3.15) because its intro is a musical build that glides down (230→216→144→73)
rather than sitting on a wrong-tempo plateau.

The current experimental flag (`has_probably_double_time_opening`) therefore requires a flat
opening plateau at a near-integer multiple of a majority-consistent steady tempo. It matches
the one labeled failure and spares all labeled good files, but it is fitted to a single
positive example and must not join the canonical gate set in this state.

## Problem or opportunity

Rhythm defects are audible, gate-invisible, and now demonstrated in a shipped validation
artifact. A trustworthy gate needs data, not thresholds fitted to n=1.

## Proposed direction

- Collect listening labels for the open cases: `B_sft_60s_compact_seed123.wav` (fast opening,
  non-integer 2.68 ratio, not flagged by the listener when delivered alongside A) and
  `C_sft_60s_raw_seed123.wav` (steady 201 BPM trajectory — steady-fast or tracker octave error?).
- Grow the labeled corpus across checkpoints, durations, and genres, including legitimate
  tempo-change music (intro builds, breakdowns, rubato) as negative examples — and specifically
  same-tempo arrangement changes (straight intro into backbeat or half-time groove), which the
  current tracker reads as an integer onset-rate halving and falsely flags.
- Regenerate the A configuration (60 s full bundle, XL sft) at several seeds to establish whether
  the double-time opening is seed-driven or configuration-driven.
- Promote `has_probably_double_time_opening` into `evaluate_music_quality_gates` only after it
  holds on the grown corpus; until then it stays a diagnostic.
- Octave-robust beat tracking (harmonic lag scoring, e.g. comparing ac[lag] with ac[2*lag] and
  ac[lag//2]) is required before trusting the raw trajectory at all, not just before gating: a
  synthesized defect at a 1.85 ratio currently reads as ratio 0.92 — invisible, not merely
  unflagged.

## Non-goals

- No general-purpose tempo estimation claims; the scope is generative failure screening.
- Do not add MIR dependencies (librosa/madmom) to the base package for this.
