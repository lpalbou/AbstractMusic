# Getting Started

## Install

```bash
pip install abstractmusic
```

Install a local runtime extra before generation:

```bash
pip install "abstractmusic[acestep]"  # default ACE-Step Diffusers path
pip install "abstractmusic[acestep-v15]"  # explicit quality-limited ACE-Step v1.5 path
pip install "abstractmusic[acestep-diffusers]"
pip install "abstractmusic[musicgen]"
pip install "abstractmusic[stable-audio]"
pip install --no-deps stable-audio-tools==0.0.19
```

The extra `stable-audio` intentionally avoids the full `stable-audio-tools` dependency graph
because the upstream package pulls UI/training dependencies and pins packages that do not install
cleanly on Python 3.12. Install `stable-audio-tools` with `--no-deps`; AbstractMusic provides the
minimal inference loop it needs.

## Generate Music

```python
from abstractmusic import MusicManager
from abstractmusic.backends import AceStepDiffusersBackend, AceStepDiffusersBackendConfig

backend = AceStepDiffusersBackend(config=AceStepDiffusersBackendConfig())
music = MusicManager(backend=backend)
wav_bytes = music.t2m("uplifting synthwave with punchy drums", duration_s=10.0)
open("out.wav", "wb").write(wav_bytes)
```

Generated WAV files should be treated as artifacts, not source files.

## CLI

```bash
abstractmusic --backend acestep t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend acestep-v15 t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend acestep-diffusers t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend musicgen t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend stable-audio t2m "short ambient synth loop" --out out.wav --duration 10
abstractmusic --backend acestep t2m "heroic fantasy epic music" --enhance-prompt --auto-lyrics --print-plan --out out.wav --duration 30
abstractmusic --backend acestep t2m "heroic fantasy epic instrumental music" --duration 120 --instrumental --print-plan --out out.wav
abstractmusic --backend acestep t2m "raw prompt only" --text-planner off --out out.wav --duration 30
```

Use `--verbose` when you need backend logs and progress bars. By default the CLI keeps
ACE-Step startup/generation logs quiet and prints the output path.
For generations of 45 seconds or more, `--structure-prompt` is enabled by default and adds a compact
section map to the caption. Use `--no-structure-prompt` to pass long prompts through unchanged.
`--text-planner deterministic` uses the dependency-free fallback planner. `--text-planner off`
preserves raw user text and explicit metadata. Library and AbstractCore plugin callers can use
`text_planner_mode="auto"` with an injected planner to replace only the intelligence layer.

## Interactive REPL

Use the REPL to try prompts, engines, and generation parameters without restarting:

```bash
abstractmusic repl --engine acestep --duration 10 --out-dir smoke-artifacts/repl
abstractmusic repl --engine xl --duration 10 --out-dir smoke-artifacts/repl
abstractmusic repl --engine musicgen --duration 10 --out-dir smoke-artifacts/repl
```

Inside the REPL:

```text
/engine acestep
/duration 12
/bpm 128
/steps 8
/shift 3
/seed 123
/verbose off
/lyrics [Instrumental]
/enhance-prompt on
/structure-prompt on
/text-planner deterministic
/auto-lyrics on
/prompt bright melodic synth pop loop with steady drums
/run
bright melodic synth pop loop with steady drums
/params
/models
/exit
```

Engines currently exposed through the unified CLI are `acestep`, `acestep-diffusers`,
`acestep-v15`, `diffusers`, `musicgen`, and `stable-audio`. `acestep`
and `ace` are aliases for the validated `acestep-diffusers` backend. `musicgen` is a small
non-commercial validation backend. `stable-audio` is gated on Hugging Face and supports short
clips up to 11 seconds.

The default ACE-Step backend is package-owned: it uses Diffusers AceStepPipeline, Hugging Face
weights, and AbstractMusic orchestration without an external ACE-Step source tree. The explicit
`acestep-v15` backend uses vendored model code but is quality-limited after repeated-loop
validation failures.

For ACE-Step turbo checkpoints, keep `/shift 3` with `/steps 8` unless deliberately testing a
quality issue. The turbo schedule is tuned around `shift=3.0`; `shift=1.0` with 8 steps
can produce collapsed or overly repetitive output.

Duration can be set when starting the REPL (`abstractmusic repl --duration 30`) or during a session
with `/duration 30`. ACE-Step v1.5 constrains generation to 10-600 seconds; values below 10 seconds
are not a reliable smoke target for that backend.

## Real Model Caveat

The fast unit test suite uses fakes for most model components. Before changing defaults or claiming
a provider works on a platform, run an opt-in real smoke test. The smoke checks WAV validity plus
harmonic structure and fast repetitive-envelope failure modes, so valid-but-rotor-like audio should
fail:

```bash
ABSTRACTMUSIC_RUN_REAL_MODEL_TESTS=1 \
ABSTRACTMUSIC_REAL_BACKEND=acestep \
ABSTRACTMUSIC_REAL_DEVICE=auto \
ABSTRACTMUSIC_REAL_DURATION_S=10 \
python -m pytest -q tests/integration/test_real_generation.py
```

Generated smoke artifacts are written under `test-artifacts/` by default and are ignored by git.

On Apple hardware, the default `acestep` path uses PyTorch MPS first. Its automatic dtype prefers
MPS bfloat16 when available, MPS float32 otherwise, and CPU float32 only if MPS still returns
non-finite audio.
