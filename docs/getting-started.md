# Getting Started

## Install

```bash
pip install abstractmusic
```

Install a local runtime extra before generation:

```bash
pip install "abstractmusic[acestep]"
pip install "abstractmusic[acestep-official]"
pip install "abstractmusic[acestep-diffusers]"
```

## Generate Music

```python
from abstractmusic import MusicManager
from abstractmusic.backends import AceStepOfficialBackend, AceStepOfficialBackendConfig

backend = AceStepOfficialBackend(config=AceStepOfficialBackendConfig())
music = MusicManager(backend=backend)
wav_bytes = music.t2m("uplifting synthwave with punchy drums", duration_s=10.0)
open("out.wav", "wb").write(wav_bytes)
```

Generated WAV files should be treated as artifacts, not source files.

## CLI

```bash
abstractmusic --backend acestep-official t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend acestep t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend acestep-diffusers t2m "ambient lo-fi study music" --out out.wav --duration 10
```

Use `--verbose` when you need upstream backend logs and progress bars. By default the CLI keeps
ACE-Step startup/generation logs quiet and prints the output path.

## Interactive REPL

Use the REPL to try prompts, engines, and generation parameters without restarting:

```bash
abstractmusic repl --engine official --duration 10 --out-dir smoke-artifacts/repl
abstractmusic repl --engine xl --duration 10 --out-dir smoke-artifacts/repl
```

Inside the REPL:

```text
/engine official
/lm-backend mlx
/lm acestep-5Hz-lm-1.7B
/duration 12
/steps 8
/shift 3
/audio-cover-strength 1
/seed 123
/verbose off
/lyrics [Instrumental]
/prompt bright melodic synth pop loop with steady drums
/run
bright melodic synth pop loop with steady drums
/params
/models
/exit
```

Engines currently exposed through the unified CLI are `acestep-official`, `acestep`,
`acestep-diffusers`, and `diffusers`. `acestep` is the older custom path and remains useful for
compatibility work, but `acestep-official` is the recommended ACE-Step path.

The official backend defaults to the bundled `acestep-5Hz-lm-1.7B` model. The smaller
`acestep-5Hz-lm-0.6B` model can still be selected with `/lm acestep-5Hz-lm-0.6B` for faster
experiments, but it is lower quality and is not the default.

For ACE-Step turbo checkpoints, keep `/shift 3` with `/steps 8` unless deliberately testing a
quality issue. The upstream turbo schedule is tuned around `shift=3.0`; `shift=1.0` with 8 steps
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
ABSTRACTMUSIC_REAL_BACKEND=acestep-official \
ABSTRACTMUSIC_ACESTEP_SOURCE_DIR=/path/to/ACE-Step-1.5-main \
ABSTRACTMUSIC_REAL_DEVICE=auto \
ABSTRACTMUSIC_REAL_DURATION_S=10 \
python -m pytest -q tests/integration/test_real_generation.py
```

Generated smoke artifacts are written under `test-artifacts/` by default and are ignored by git.

On Apple hardware, use `acestep-official` so the upstream MLX path is preferred when MLX/MLX-LM is
available. CPU is an explicit fallback path.
