# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches 1.0.

## [Unreleased]

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

### Changed
- Untracked marimo/layout tool artifacts (`.marimo_agent_state.json`,
  `__marimo__/session/*.json`, `layouts/*.json`, `project_overview.py`) that
  don't belong in the package; they remain on disk but are now gitignored.
