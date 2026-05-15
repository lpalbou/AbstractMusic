# AbstractMusic Backlog

This folder is the engineering memory for AbstractMusic.

The backlog is intentionally code-first: before implementing or changing an item,
inspect the current code and docs because this project is moving quickly and older
notes may be stale.

## Layout

- `overview.md`: status, priority order, and work ledger.
- `planned/`: committed work that should be implemented.
- `proposed/`: useful ideas that need more evidence before implementation.
- `completed/`: finished work with completion reports.
- `deprecated/`: retired work with deprecation reports.
- `recurrent/`: checklist tasks that should run after relevant changes.

## Rules

- Keep planned items standalone.
- Preserve the model/provider abstraction; do not hard-code one model into the public API.
- Keep heavy local inference dependencies optional where possible.
- Do not commit model weights, generated WAVs, Python bytecode, or cache artifacts.
- Prefer permissive, commercially usable model licenses for default providers.
- Never silently ignore generation inputs such as lyrics, duration, seed, model choice, fallback, or truncation.
- Validate with fast unit tests and at least one real model smoke path before calling provider work complete.

