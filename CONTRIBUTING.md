# Contributing To AbstractMusic

AbstractMusic is a model-agnostic music generation package for the AbstractFramework ecosystem.
Keep changes small, tested, and honest about provider limitations.

## Development setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Run tests

```bash
python -m pytest -q -m "not integration"
```

Real model tests must stay opt-in because music checkpoints are large and hardware-sensitive.

## Ground rules

- Keep `MusicManager` thin and model-agnostic.
- Keep heavy model/runtime imports lazy.
- Do not commit model weights, generated audio, bytecode, cache directories, or build artifacts.
- Update the model capability registry when adding or changing provider/model behavior.
- Prefer official 8-bit model artifacts when they exist. If no official 8-bit artifact exists,
  prefer the smallest official 16-bit path before using larger precision formats.
- Do not silently ignore request fields such as lyrics, duration, format, seed, guidance, or
  negative prompts.
- Keep docs and README examples truthful about what is tested with real model weights.

## Adding a backend

1. Add or update provider/model metadata in `src/abstractmusic/assets/music_model_capabilities.json`.
2. Implement the backend behind the unified request/result types.
3. Raise clear errors for unsupported inputs instead of ignoring them.
4. Add fast fake-backed unit tests.
5. Add or update an opt-in real smoke test if the backend can run locally.
6. Document dependencies, hardware expectations, license, and precision/quantization status.

## Backlog process

Read `docs/backlog/overview.md` before selecting work. Planned items should be standalone and
moved to `docs/backlog/completed/` only after implementation, docs, and validation are done.

