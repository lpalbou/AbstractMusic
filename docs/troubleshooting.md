# Troubleshooting

Use this page for user-visible setup, runtime, and release failures. For model support and license
boundaries, see [models.md](models.md). For the public API, see [api.md](api.md).

## Default remote generation fails with a missing API key

The base install uses the ACE Music remote backend by default. Set `ACEMUSIC_API_KEY`, or pass the
key through host configuration:

```bash
export ACEMUSIC_API_KEY=...
abstractmusic t2m "ambient lo-fi study music" --out out.wav --duration 30
```

If you use a compatible proxy or self-hosted endpoint, set `ACEMUSIC_BASE_URL` or pass
`--acemusic-base-url`.

## Local generation fails with missing packages

The base package is import-light and does not install heavy model runtimes. Install the backend
extra you plan to use:

```bash
pip install "abstractmusic[acestep]"
```

For local development:

```bash
python -m pip install -e ".[acestep,test]"
```

Verify the CLI sees the backend:

```bash
abstractmusic --backend acestep t2m "short ambient synth loop" --out out.wav --duration 10
```

## ACE-Step downloads or cache access fails

The local `acestep` backend uses Hugging Face model weights. It does not accept local ACE-Step
source-tree paths or checkpoint-directory overrides. Confirm the model is accessible from the
standard Hugging Face cache:

```bash
python - <<'PY'
from huggingface_hub import model_info
print(model_info("ACE-Step/acestep-v15-xl-turbo-diffusers").id)
PY
```

If the machine is offline, populate the Hugging Face cache before running generation.

## Apple MPS returns non-finite audio or falls back to CPU

On Apple hardware, the local ACE-Step Diffusers backend tries PyTorch MPS first. The automatic
dtype policy avoids fp16 denoising on MPS because it can overflow during transformer inference on
some local stacks. AbstractMusic prefers MPS bfloat16 when supported, then MPS float32, and only
falls back to CPU float32 if MPS still returns invalid audio.

To force CPU directly:

```bash
abstractmusic --backend acestep --device cpu t2m "ambient music" --out out.wav --duration 10
```

## Long generations sound static or repetitive

For generations of 45 seconds or more, the CLI enables `--structure-prompt` by default. This adds a
compact section map to the caption. For more control, pass explicit structure and use a seed:

```bash
abstractmusic --backend acestep t2m \
  "rhythmic space shooter game music, fast drums, evolving synth bass, arcade lead theme" \
  --duration 120 \
  --seed 123 \
  --enhance-prompt \
  --print-plan \
  --out out.wav
```

ACE-Step is still a local model with limits. If a candidate contains clock-like repeated tones,
single-note collapse, or long near-silent pauses, treat it as a failed generation and try a new
seed or a more explicit prompt.

## Stable Audio Open Small cannot download weights

`stabilityai/stable-audio-open-small` is gated on Hugging Face. Accept the model terms with the
Hugging Face account used by the machine. The runtime also expects `stable-audio-tools` to be
installed without its full dependency chain:

```bash
pip install "abstractmusic[stable-audio]"
pip install --no-deps stable-audio-tools==0.0.19
```

## Release workflow refuses to publish

The release workflow validates the package version, changelog entry, tag, and PyPI availability.
Manual publishing requires:

- `version` matching `src/abstractmusic/_version.py`
- `publish=true`
- `publish_confirmation=publish-abstractmusic-<version>`

If PyPI already has the target version, bump the package version and changelog instead of trying to
republish the same artifact.
