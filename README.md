# AbstractMusic

`abstractmusic` is a local-first **text-to-music / text-to-audio** library designed to plug into **AbstractCore** as an optional capability plugin.

## Install

```bash
pip install abstractmusic
```

The base package is import-light: contracts, manager, CLI shell, plugin wiring, docs, and model
metadata. Install a local runtime extra before generating:

```bash
pip install "abstractmusic[acestep]"
pip install "abstractmusic[acestep-official]"
pip install "abstractmusic[acestep-diffusers]"
pip install "abstractmusic[apple]"
pip install "abstractmusic[gpu]"
pip install "abstractmusic[all-apple]"
pip install "abstractmusic[all-gpu]"
```

The `apple` profile includes MLX/MLX-LM for the official ACE-Step path. CPU is a fallback path, not
the default acceleration strategy.

## Quickstart (local generation)

```python
from abstractmusic import MusicManager
from abstractmusic.backends import AceStepOfficialBackend, AceStepOfficialBackendConfig

backend = AceStepOfficialBackend(config=AceStepOfficialBackendConfig())

mm = MusicManager(backend=backend)
wav_bytes = mm.t2m("uplifting synthwave with punchy drums", duration_s=10.0)
open("out.wav", "wb").write(wav_bytes)
```

Set `ABSTRACTMUSIC_ACESTEP_SOURCE_DIR` if the upstream ACE-Step source tree is not installed as a
package. ACE-Step Diffusers XL Turbo can be selected through the same public abstraction:

```python
from abstractmusic import MusicManager
from abstractmusic.backends import AceStepDiffusersBackend, AceStepDiffusersBackendConfig

backend = AceStepDiffusersBackend(config=AceStepDiffusersBackendConfig())
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
    music_backend="acestep-official",
    music_model_id="ACE-Step/Ace-Step1.5",
)

wav_bytes = llm.music.t2m("ambient lo-fi study music", format="wav", duration_s=10.0)
open("out.wav", "wb").write(wav_bytes)
```

## Notes

- Audio output baseline is **WAV** (no external codecs required).
- Model weights are downloaded on first use via the Hugging Face cache (same workflow as Diffusers-based vision).
- The recommended ACE-Step v1.5 path is `acestep-official`, which wraps the upstream handler and 5Hz LM planner. On Apple Silicon it prefers MLX when available.
- The older packaged custom ACE-Step v1.5 path is **not recommended**: a valid 3-second MPS WAV failed music-quality review as fast rotor-like audio.
- The ACE-Step backend vendors the checkpoint’s custom Transformers model code into `abstractmusic` so we do **not** use `trust_remote_code`.
- Known model/provider metadata is packaged in `src/abstractmusic/assets/music_model_capabilities.json`.
  See `docs/models.md` for the reviewed model list and precision policy.

## CLI / REPL

After installation, `abstractmusic` provides a small CLI:

```bash
# One-shot generation
abstractmusic --backend acestep t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend acestep-official t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend acestep-diffusers t2m "ambient lo-fi study music" --out out.wav --duration 10

# Interactive REPL
abstractmusic --backend acestep-official repl
abstractmusic --engine xl repl
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
when you want upstream backend logs and progress bars.

## Licensing note

- The default backend example uses **ACE-Step v1.5** (`ACE-Step/Ace-Step1.5`), tagged `license:mit` on Hugging Face. The official adapter expects the upstream ACE-Step runtime as an optional dependency/source tree.
- The vendored custom ACE-Step model code files carry **Apache-2.0** headers (both permissive), but that backend is not currently recommended.
- The ACE-Step Diffusers XL example uses `ACE-Step/acestep-v15-xl-turbo-diffusers`, tagged `license:mit` on Hugging Face.
- If you switch to `--backend diffusers`, **model licenses vary** by checkpoint. Choose a model compatible with your intended usage.

### macOS / Apple Silicon note (MLX/MPS)

On Apple systems, `acestep-official` prefers the upstream MLX path for the 5Hz LM and enables the
upstream MLX DiT/VAE path when available. CPU is used only when explicitly requested or when a
marked fallback is needed for a known backend limitation.

Some Diffusers audio pipelines can fail on the `mps` device due to PyTorch backend limitations (typically during vocoder inference).
`abstractmusic` will **retry on CPU** with a clear warning (`#FALLBACK`) when it detects the known MPS channel-limit error.
To force CPU directly, use `--device cpu`.

For ACE‑Step v1.5 on MPS, `abstractmusic` defaults to **fp16** (bf16 disabled) to keep memory usage reasonable on typical unified‑memory Macs.
If you run into numerical issues, you can override with `--dtype float32` (at the cost of significantly higher memory use).
By default, ACE‑Step caps MPS memory to ~16 GiB by setting `PYTORCH_MPS_HIGH_WATERMARK_RATIO` (configurable via `--mps-max-memory-gb` or `--mps-high-watermark-ratio`).
In addition, ACE-Step text-encoder conditioning is executed on **CPU float32** on MPS builds as a compatibility fallback (`#FALLBACK`) to avoid known mixed-dtype MPSGraph kernel aborts; conditioning tensors are cast back to the model dtype/device before diffusion.
The ACE-Step backend defaults to `infer_method=ode` (turbo `fix_nfe=8`, upstream default) with `shift=3.0` (upstream schedule). For text2music conditioning, source latents are initialized from seeded random noise (instead of silence) to better match expected model behavior, and chunk masks default to zeros to avoid injecting constant features.
This custom path does not yet use the official ACE-Step 5Hz LM semantic-code phase, which is the leading suspected cause of the rotor-like smoke output.
By default, the prompt is passed as-is (no SFT wrapper); set `use_sft_prompt=True` in config if you need the previous instruction/metas format.
If a run returns non-finite latents, `abstractmusic` retries once with the alternate infer method using an incremented seed (`#FALLBACK`) instead of writing a silent/invalid WAV.

For instrumental prompts (no explicit lyrics), ACE-Step uses a proper **null lyric condition** (mask=0) rather than synthetic placeholder lyric text.
Decoded waveforms are DC-centered before normalization to avoid one-sided/noisy artifacts from amplifying tiny decoder bias.

Upstream references:
- PyTorch MPS env var `PYTORCH_ENABLE_MPS_FALLBACK=1` (fallback to CPU when an op is unsupported): `https://docs.pytorch.org/docs/stable/mps_environment_variables.html`
- Example upstream issue tracking the specific MPS channel-limit error: `https://github.com/pytorch/pytorch/issues/144445`
