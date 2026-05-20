# Planned: YuE Optional Provider

## Metadata
- Created: 2026-05-15
- Status: Planned
- Completed: N/A
- Priority: P2

## Context

`m-a-p/YuE-s1-7B-anneal-en-cot` is an Apache-2.0 YuE Stage-1 music model. It is
music-relevant and commercially usable under the reviewed metadata, but it is not an end-to-end
audio generator by itself.

## Current code reality

- AbstractMusic now has a model capability registry entry for
  `m-a-p/YuE-s1-7B-anneal-en-cot`.
- There is no YuE backend.
- A usable YuE path appears to require the Stage-1 model, a Stage-2 model such as
  `m-a-p/YuE-s2-1B-general`, and the official codec/upsampler path.

## Problem

Adding only the Stage-1 model would not satisfy the project goal of generating audible music.
YuE should not become a default or recommended provider until the full pipeline produces valid WAV
audio through the unified `MusicManager` abstraction.

## What we want to do

Evaluate YuE as an optional provider behind the `yue` extra, using official model sources and the
smallest required runtime surface.

## Why

YuE may be useful for lyrics-conditioned full-song generation, but it is a multi-stage system. It
needs careful dependency containment and real validation before it can help users.

## Requirements

- Use official YuE sources only.
- Prefer official 8-bit artifacts if the maintainers publish them; otherwise use official 16-bit
  artifacts where supported.
- Keep the provider optional behind `abstractmusic[yue]`.
- Do not add YuE dependencies to the base install.
- Implement through the unified backend protocol and `MusicManager` request fields.
- Confirm that Stage-1, Stage-2, codec, and upsampler pieces are all present and licensed.
- Add real smoke validation that writes WAV under an ignored artifact directory.
- Inspect generated WAV stats for duration, channels, sample rate, non-silence, clipping, DC
  offset, and obvious noise signals.
- Mark the registry status as validated only after a generated output passes the real smoke.

## Scope

- YuE research and provider design.
- Optional dependency profile.
- Capability metadata and docs.
- Real audio smoke test.

## Non-goals

- Do not make YuE a default provider in this task.
- Do not ship partial semantic-token output as “music.”
- Do not add non-official quantized forks as the preferred path.

## Dependencies and related tasks

- `docs/backlog/completed/040_real_generation_validation_matrix.md`
- `docs/backlog/completed/050_dependency_profiles_and_optional_providers.md`

## Expected outcomes

- A clear go/no-go decision for YuE as a full audio provider.
- If viable, users can generate music via `MusicManager.t2m(...)` with a YuE backend.
- If not viable, the registry remains honest: YuE stays research/planned, not recommended.

## Validation

- `python -m pytest -q -m "not integration"`
- A gated real YuE integration smoke test when hardware/model cache permits.
- Manual WAV inspection report showing the output is not silent, clipped, DC-biased, or obvious
  broadband noise.

## Progress checklist

- [ ] Confirm full YuE pipeline components and licenses.
- [ ] Design the optional provider dependency boundary.
- [ ] Implement backend if the pipeline can be kept maintainable.
- [ ] Add unit tests with fakes.
- [ ] Run one real generation smoke and inspect the audio output.

## Guidance for the implementing agent

Do not confuse “music language model” with “working audio generator.” The accepted output is a
valid generated audio file through the AbstractMusic abstraction.
