# Recurrent: Dependency And Artifact Hygiene

## Metadata
- Created: 2026-05-15
- Status: Recurrent

## Run conditions

Run before release, after provider changes, after packaging changes, and after real model smoke
tests that create local output artifacts.

## Checklist

- [ ] Verify base install does not pull heavy inference stacks unless intentionally documented.
- [ ] Verify heavy provider imports remain lazy.
- [ ] Verify generated WAVs, model weights, cache directories, egg-info, and Python bytecode are
      ignored and not staged.
- [ ] Verify licenses for default providers and bundled/vendored code are documented.
- [ ] Run `python -m pytest -q -m "not integration"`.
- [ ] Run the current real-provider smoke test command if the needed model weights are available.

