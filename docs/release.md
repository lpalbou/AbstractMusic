# Release Process

AbstractMusic releases are published through GitHub Actions. The workflow validates tests, package
builds, documentation, changelog content, and package version metadata before publishing.

## Version Source

The canonical package version is:

```text
src/abstractmusic/_version.py
```

`pyproject.toml` reads that value dynamically for packaging metadata. A release tag must use the
matching `v<version>` form.

## Local Preflight

Before triggering a release, run:

```bash
python -m pytest -q -m unit
python -m build
python -m twine check dist/*
mkdocs build --strict
```

When workflow files change, also run a workflow syntax checker such as `actionlint` if available.

## Rehearsal

Manual dispatch of `.github/workflows/release.yml` defaults to `publish=false`. This validates the
version, changelog, package build, and documentation build without creating a tag, publishing to
PyPI, creating a GitHub Release, or deploying docs.

## Publish

Publishing can be triggered by pushing a `v*.*.*` tag or by manual workflow dispatch with:

```text
version=<version>
publish=true
publish_confirmation=publish-abstractmusic-<version>
```

Publishing uses PyPI trusted publishing with the `pypi` environment. Documentation deployment uses
GitHub Pages with the `github-pages` environment.

## Repository Setup

The GitHub repository must have:

- PyPI trusted publisher configured for package `abstractmusic`, workflow
  `.github/workflows/release.yml`, and environment `pypi`
- GitHub Pages source set to GitHub Actions
- Optional environment protection rules for `pypi` and `github-pages`
