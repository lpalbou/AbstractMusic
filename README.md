# AbstractMusic

`abstractmusic` is a local-first **text-to-music / text-to-audio** library designed to plug into **AbstractCore** as an optional capability plugin.

## Install

```bash
pip install abstractmusic
```

The base package is import-light: contracts, manager, CLI shell, plugin wiring, docs, and model
metadata. Install a local runtime extra before generating:

```bash
pip install "abstractmusic[acestep]"  # default ACE-Step Diffusers path
pip install "abstractmusic[acestep-v15]"  # explicit quality-limited ACE-Step v1.5 path
pip install "abstractmusic[acestep-diffusers]"
pip install "abstractmusic[apple]"
pip install "abstractmusic[gpu]"
pip install "abstractmusic[all-apple]"
pip install "abstractmusic[all-gpu]"
```

The `acestep` profile installs the package-owned ACE-Step route. On Apple MPS, AbstractMusic
falls back to CPU float32 if the Diffusers ACE-Step pipeline returns non-finite audio.

## Quickstart (local generation)

```python
from abstractmusic import MusicManager
from abstractmusic.backends import AceStepDiffusersBackend, AceStepDiffusersBackendConfig

backend = AceStepDiffusersBackend(config=AceStepDiffusersBackendConfig())

mm = MusicManager(backend=backend)
wav_bytes = mm.t2m("uplifting synthwave with punchy drums", duration_s=10.0)
open("out.wav", "wb").write(wav_bytes)
```

The explicit quality-limited ACE-Step v1.5 backend can also be selected through the same public abstraction:

```python
from abstractmusic import MusicManager
from abstractmusic.backends import AceStepV15Backend, AceStepV15BackendConfig

backend = AceStepV15Backend(config=AceStepV15BackendConfig())
mm = MusicManager(backend=backend)
wav_bytes = mm.t2m("upbeat synthwave instrumental", duration_s=10.0)
open("out.wav", "wb").write(wav_bytes)
```

## Quickstart (AbstractCore integration)

```python
from abstractcore import create_llm

llm = create_llm(
    # Any provider/model works here. The LLM does *not* synthesize audio.
    "ollama",
    model="qwen3:4b-instruct",
    music_backend="acestep",
    music_model_id="ACE-Step/Ace-Step1.5",
)

wav_bytes = llm.music.t2m("ambient lo-fi study music", format="wav", duration_s=10.0)
open("out.wav", "wb").write(wav_bytes)
```

## Notes

- Audio output baseline is **WAV** (no external codecs required).
- Model weights are downloaded on first use via the Hugging Face cache (same workflow as Diffusers-based vision).
- The default ACE-Step path is `acestep` / `acestep-diffusers`, which uses package-owned orchestration around Diffusers AceStepPipeline and Hugging Face checkpoint files rather than an external ACE-Step source tree.
- `acestep-v15` remains explicit and quality-limited after repeated-loop validation failures.
- `musicgen` and `stable-audio` are optional small-model comparison backends; both are non-commercial and not default providers.
- For Stable Audio Open Small, install `stable-audio-tools` with `--no-deps` after `abstractmusic[stable-audio]`; AbstractMusic avoids the upstream package's UI/training dependency chain and owns the minimal inference loop.
- The standalone `acestep-v15` backend vendors the checkpoint’s custom Transformers model code into `abstractmusic` so we do **not** use `trust_remote_code` there.
- Known model/provider metadata is packaged in `src/abstractmusic/assets/music_model_capabilities.json`.
  See `docs/models.md` for the reviewed model list and precision policy.

## CLI / REPL

After installation, `abstractmusic` provides a small CLI:

```bash
# One-shot generation
abstractmusic --backend acestep t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend acestep-v15 t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend acestep-diffusers t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend musicgen t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend stable-audio t2m "short ambient synth loop" --out out.wav --duration 10

# Interactive REPL
abstractmusic --engine xl repl
abstractmusic --engine musicgen repl
```

The REPL accepts bare prompts, a reusable `/prompt` + `/run` flow, and slash commands for engine/parameter changes:

```text
/engine xl
/duration 12
/steps 8
/seed 123
/verbose off
/lyrics [Instrumental]
/prompt bright melodic synth pop loop with steady drums
/run
bright melodic synth pop loop with steady drums
```

Set duration either at startup (`abstractmusic repl --duration 30`) or inside the REPL
(`/duration 30`). ACE-Step v1.5 expects 10-600 seconds. Add `--verbose` or use `/verbose on` only
when you want backend logs and progress bars.

## Licensing note

- The default backend example uses **ACE-Step Diffusers XL Turbo** (`ACE-Step/acestep-v15-xl-turbo-diffusers`), tagged `license:mit` on Hugging Face, through the package-owned adapter.
- The vendored standalone ACE-Step model code files carry **Apache-2.0** headers (both permissive).
- The ACE-Step Diffusers XL example uses `ACE-Step/acestep-v15-xl-turbo-diffusers`, tagged `license:mit` on Hugging Face.
- `facebook/musicgen-small` is exposed through `--backend musicgen`; its model weights are **CC BY-NC 4.0**, so it is a non-commercial validation backend.
- `stabilityai/stable-audio-open-small` is exposed through `--backend stable-audio`; it is gated on Hugging Face and uses the **Stability AI Community License**.
- If you switch to `--backend diffusers`, **model licenses vary** by checkpoint. Choose a model compatible with your intended usage.

### macOS / Apple Silicon note (MLX/MPS)

On Apple systems, the default `acestep` / `acestep-diffusers` path tries PyTorch MPS first and
falls back to CPU float32 when the pipeline returns non-finite audio.

Some Diffusers audio pipelines can fail on the `mps` device due to PyTorch backend limitations (typically during vocoder inference).
`abstractmusic` will **retry on CPU** with a clear warning (`#FALLBACK`) when it detects the known MPS channel-limit error.
To force CPU directly, use `--device cpu`.

For the explicit `acestep-v15` PyTorch/MPS path, `abstractmusic` defaults to **fp16** (bf16 disabled) to keep memory usage reasonable on typical unified-memory Macs.
If you run into numerical issues, you can override with `--dtype float32` (at the cost of significantly higher memory use).
The standalone path caps MPS memory to ~16 GiB by setting `PYTORCH_MPS_HIGH_WATERMARK_RATIO` (configurable via `--mps-max-memory-gb` or `--mps-high-watermark-ratio`).
In addition, standalone ACE-Step text-encoder conditioning is executed on **CPU float32** on MPS builds as a compatibility fallback (`#FALLBACK`) to avoid known mixed-dtype MPSGraph kernel aborts; conditioning tensors are cast back to the model dtype/device before diffusion.
The standalone ACE-Step backend keeps turbo controls at `infer_method=ode`, `steps=8`, `shift=3.0`, but uses seeded random source latents for direct text-to-music to avoid silence-conditioned tone collapse.
The experimental 5Hz LM audio-code planner is off by default because using coarse code hints as full cover conditioning can imprint repetitive artifacts.
If a standalone run returns non-finite latents, `abstractmusic` retries once with the alternate infer method using an incremented seed (`#FALLBACK`) instead of writing a silent/invalid WAV.

For instrumental standalone ACE-Step runs, pass lyrics as `[Instrumental]`.
Standalone decoded waveforms are DC-centered before normalization to avoid one-sided/noisy artifacts from amplifying tiny decoder bias.

Upstream references:
- PyTorch MPS env var `PYTORCH_ENABLE_MPS_FALLBACK=1` (fallback to CPU when an op is unsupported): `https://docs.pytorch.org/docs/stable/mps_environment_variables.html`
- Example upstream issue tracking the specific MPS channel-limit error: `https://github.com/pytorch/pytorch/issues/144445`
