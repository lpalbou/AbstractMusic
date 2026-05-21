# Getting Started

## Install

```bash
pip install abstractmusic
```

The base package includes stdlib-only ACE Music and ElevenLabs Music remote backends and no local
ML runtime stack. Set a remote API key before using the default CLI backend:

```bash
export ACEMUSIC_API_KEY=...
abstractmusic t2m "ambient lo-fi study music" --out out.wav --duration 30
```

ElevenLabs Music is explicit and music-only:

```bash
export ELEVENLABS_API_KEY=...
abstractmusic --backend elevenlabs t2m "cinematic instrumental synth cue" --format mp3 --out out.mp3 --duration 30
```

Install a local runtime profile when you want in-process generation:

```bash
pip install "abstractmusic[remote]"  # no-op alias; base install already supports remote clients
pip install "abstractmusic[acestep]"  # local ACE-Step Diffusers path
pip install "abstractmusic[acestep-v15]"  # explicit quality-limited ACE-Step v1.5 path
pip install "abstractmusic[acestep-diffusers]"
pip install "abstractmusic[all-apple]"
pip install "abstractmusic[all-gpu]"
pip install "abstractmusic[musicgen]"
pip install "abstractmusic[stable-audio]"
pip install "abstractmusic[stable-audio-3]"
```

The extra `stable-audio` targets the gated `stabilityai/stable-audio-open-small` checkpoint and
vendors the minimal `stable-audio-tools==0.0.19` model code inside AbstractMusic, so you do **not**
need to install the upstream `stable-audio-tools` package.

The `stable-audio-3` extra installs only the top-level runtime libraries needed by
AbstractMusic's internal Stable Audio 3 text-to-music path: Torch, Transformers, Safetensors,
Hugging Face Hub, NumPy, Einops, and Packaging. It does not install or import the upstream
`stable_audio_3` package, `stable-audio-tools`, UI, training, LoRA, CoreML/TFLite, or
Flash-Attention dependencies.

## Generate Music

```python
from abstractmusic import MusicManager
from abstractmusic.backends import AceMusicBackend, AceMusicBackendConfig

backend = AceMusicBackend(config=AceMusicBackendConfig(api_key="..."))
music = MusicManager(backend=backend)
wav_bytes = music.t2m("uplifting synthwave with punchy drums", duration_s=30.0)
open("out.wav", "wb").write(wav_bytes)
```

For ElevenLabs Music:

```python
from abstractmusic import MusicManager
from abstractmusic.backends import ElevenLabsMusicBackend, ElevenLabsMusicBackendConfig

backend = ElevenLabsMusicBackend(config=ElevenLabsMusicBackendConfig(api_key="..."))
music = MusicManager(backend=backend)
mp3_bytes = music.t2m("cinematic instrumental synth cue", duration_s=30.0, format="mp3")
open("out.mp3", "wb").write(mp3_bytes)
```

Generated WAV files should be treated as artifacts, not source files.

## CLI

```bash
abstractmusic t2m "ambient lo-fi study music" --out out.wav --duration 30
abstractmusic --backend acemusic t2m "ambient lo-fi study music" --format mp3 --out out.mp3 --duration 30
abstractmusic --backend elevenlabs t2m "cinematic instrumental synth cue" --format mp3 --out out.mp3 --duration 30
abstractmusic --backend elevenlabs t2m "upbeat pop song" --lyrics auto --composition-mode plan --format mp3 --out out.mp3 --duration 30
abstractmusic --backend acestep t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend acestep-v15 t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend acestep-diffusers t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend musicgen t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend stable-audio t2m "short ambient synth loop" --out out.wav --duration 10
abstractmusic --backend stable-audio-3 t2m "rhythmic space shooter game music" --out out.wav --duration 30 --steps 16
abstractmusic --backend stable-audio-3 t2m "A short sci-fi laser zap, crisp transient, no reverb tail" --model-id stabilityai/stable-audio-3-small-sfx --duration 3 --out sfx.wav
abstractmusic --backend stable-audio t2m "A heavy wooden door slam, close-mic, natural room tail" --duration 11 --out slam.wav
abstractmusic --backend acestep t2m "heroic fantasy epic music" --enhance-prompt --auto-lyrics --print-plan --out out.wav --duration 30
abstractmusic --backend acestep t2m "heroic fantasy epic instrumental music" --duration 120 --instrumental --print-plan --out out.wav
abstractmusic --backend acestep t2m "raw prompt only" --text-planner off --out out.wav --duration 30
abstractmusic --backend acemusic t2m "A soulful pop song with a big uplifting chorus and a catchy hook" --lyrics auto --style "soulful pop, female vocalist, warm bass, tight drums, big chorus, modern radio mix" --format mp3 --duration 25 --out soulful-pop.mp3
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
abstractmusic cli --engine acemusic --duration 30 --out-dir smoke-artifacts/repl
abstractmusic cli --engine acestep --duration 10 --out-dir smoke-artifacts/repl
abstractmusic cli --engine xl --duration 10 --out-dir smoke-artifacts/repl
abstractmusic cli --engine musicgen --duration 10 --out-dir smoke-artifacts/repl
```

Inside the REPL:

```text
/status
/engines
/models
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
/exit
```

Engines currently exposed through the unified CLI are `acemusic`, `elevenlabs`, `acestep`,
`acestep-diffusers`, `acestep-v15`, `diffusers`, `musicgen`, `stable-audio`, and
`stable-audio-3`. `acemusic` is the default lightweight remote backend and accepts aliases such as `remote` and `ace-music`.
`elevenlabs` accepts aliases such as `eleven` and `11labs` and only uses ElevenLabs Music APIs.
`acestep` and `ace` are aliases for the validated `acestep-diffusers` backend. `musicgen` is a small
non-commercial validation backend. `stable-audio` is gated on Hugging Face and supports short
clips up to 11 seconds. `stable-audio-3` targets `stabilityai/stable-audio-3-small-music` through
AbstractMusic-owned internal runtime code and currently supports text-to-music only.

The local ACE-Step backend is package-owned: it uses Diffusers AceStepPipeline, Hugging Face
weights, and AbstractMusic orchestration without an external ACE-Step source tree. The explicit
`acestep-v15` backend uses vendored model code but is quality-limited after repeated-loop
validation failures.

For ACE-Step turbo checkpoints, keep `/shift 3` with `/steps 8` unless deliberately testing a
quality issue. The turbo schedule is tuned around `shift=3.0`; `shift=1.0` with 8 steps
can produce collapsed or overly repetitive output.

Duration can be set when starting the REPL (`abstractmusic cli --duration 30`) or during a session
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

On Apple hardware, the local `acestep` path uses PyTorch MPS first. Its automatic dtype prefers
MPS bfloat16 when available, MPS float32 otherwise, and CPU float32 only if MPS still returns
non-finite audio.
