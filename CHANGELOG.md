# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches 1.0.

## [Unreleased]

## [0.1.3] - 2026-09-11

### Added
- `RELEASING.md`, documenting the actual release process end-to-end (one-time
  infrastructure reference, versioning, step-by-step tag/release/approve/verify,
  gotchas hit in practice, pre-release checklist).
- `release.yml`: a new `update-citation` job that runs after a successful
  PyPI publish, bumping `CITATION.cff`'s `version:`/`date-released:` to
  match the release tag and committing it straight to `main`. This was
  previously a manual step on the pre-release checklist — easy to forget,
  since it's the one release artifact `setuptools-scm` doesn't touch.

### Fixed
- `docs.yml` never rebuilt/redeployed the docs site on a GitHub Release —
  only on push to `main`, PRs, and manual dispatch. Added `release: {types:
  [published]}` as a trigger, pinned to check out `main` explicitly (rather
  than the tag's commit) so a release cut from an older/hotfix commit can't
  regress the (unversioned, "latest") docs site to stale content.
- `docs/index.md`'s Installation section still said `parx` "is not yet
  published on PyPI" and linked to `road2publish.md`; updated to match
  README's actual `pip install parx` instructions.
- `pyproject.toml`'s `description` and `src/parx/__init__.py`'s module
  docstring still expanded the `parx` acronym as "POLyhedral Activation
  Region Xplorer"; corrected to the canonical "Polyhedral Affine Region
  eXplorer" used everywhere else (README, `CITATION.cff`).
- README's "Future directions" section still listed "Publish to PyPI once
  the API stabilizes" as unfinished work, contradicting the PyPI/DOI badges
  in the same file; removed now that `parx` has been on PyPI since `v0.1.0`.

### Changed
- `CONTRIBUTING.md`'s project-structure tree updated to include
  `analysis.py`, `io_partition.py`, `diagnostics.py`, `precompile.py`,
  `methods/exact_julia_fast.py`, `julia/self_test.jl`, and
  `tests/test_julia_init.py`, which existed in the repo but not in the tree.
- `RELEASING.md`'s pre-release checklist now includes updating this file.

## [0.1.2] - 2026-09-10

### Added
- Zenodo DOI: concept DOI `10.5281/zenodo.22696450` filled into
  `CITATION.cff`'s `identifiers:` field, plus a DOI badge and expanded
  "Citation" section (with bibtex) in README.

### Removed
- Stale `TODO.md`; dangling references to it in README and
  `docs/usage.md` cleaned up too.

## [0.1.1] - 2026-09-10

### Changed
- README's Installation section now leads with `pip install parx` now that
  the package is actually published, instead of the source-install-only
  instructions from before `v0.1.0`; fixed the PyPI version badge.

## [0.1.0] - 2026-09-10

First published release — on PyPI and tagged on GitHub.

### Fixed
- `pyproject.toml` referenced a nonexistent `LICENSE` file while the tracked
  `LICENCE` file was empty; replaced with a real MIT `LICENSE` and the modern
  SPDX `license`/`license-files` metadata form.
- The built sdist/wheel silently omitted all Julia source files
  (`src/parx/julia/src/*.jl`) because `package-data`'s `julia/*` glob does not
  recurse into the nested `julia/src/` directory. Anyone installing from a
  built artifact (rather than an editable/source checkout) would have hit a
  hard failure on first import. Fixed by adding `julia/src/*` to
  `package-data`.
- README referenced a `parx[analysis]` extra that did not exist in
  `pyproject.toml`; added it (`scikit-learn`, used by `plot_partition_pca`).
- Stale/placeholder content in `CONTRIBUTING.md` (clone URL, test file
  listing) and a broken `LICENCE` link in `README.md`.

### Added
- `LICENSE` (MIT).
- `CITATION.cff` for GitHub's citation button and future Zenodo archival.
- CI/docs/license/Python-version badges in `README.md`.
- `road2publish.md`, tracking the path to a PyPI release and Zenodo DOI.
- `release.yml`: builds sdist+wheel and publishes to PyPI via Trusted
  Publishing (OIDC) on every GitHub Release.

### Changed
- Untracked marimo/layout tool artifacts (`.marimo_agent_state.json`,
  `__marimo__/session/*.json`, `layouts/*.json`, `project_overview.py`) that
  don't belong in the package; they remain on disk but are now gitignored.
