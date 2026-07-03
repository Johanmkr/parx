# Road to Publish

Goal: get `parx` to a state where (1) colleagues can clone the repo and install/test it locally without friction, (2) it can be distributed as a real package on PyPI, (3) it has a citable Zenodo DOI, and (4) the docs at `docs/` (GitHub Pages) are complete enough to stand on their own. This file is a survey + ordered plan — nothing here has been implemented yet.

Cross-reference: `TODO.md` already tracks feature/engineering work (§8.4 mentions PyPI as a long-term item gated on the Julia sysimage). This document is narrower and focused specifically on **packaging, licensing, citation, and distribution**, which is a smaller and more immediate lift than the "make Julia startup fast" work in TODO.md — colleagues cloning the repo today don't need a sysimage, they just need the current install path to actually work.

---

## Current state (as of this survey)

**Good news — the bones are mostly there:**
- `pyproject.toml` is a real PEP 621 project file (setuptools backend), with proper `[project.optional-dependencies]` (`h5`, `animate`, `embed`, `docs`, `dev`), classifiers, keywords, URLs.
- `package-data` already includes `julia/*` and `juliapkg.json`, so the Julia source ships inside the sdist/wheel (untested — see Phase 2).
- CI (`.github/workflows/ci.yml`) already installs via `uv pip install -e ".[dev]"` + `julia --project=... instantiate` across Python 3.10–3.12 on every push/PR — this is effectively a continuous "fresh clone install" test, which is most of what "testable by colleagues" requires.
- A docs site exists (`mkdocs.yml`, Material theme) with a working GitHub Pages deploy workflow (`.github/workflows/docs.yml`), covering `index.md`, `concepts.md`, `usage.md`, and rendered notebooks.
- `CONTRIBUTING.md` and `OVERVIEW.md` are substantial and reasonably complete.

**Broken / missing — blocking items:**
1. **License is broken.** `pyproject.toml` declares `license = { file = "LICENSE" }`, but no file named `LICENSE` exists in the repo. The tracked file is `LICENCE` (British spelling) — and it is **0 bytes**. So today: the package build will fail to find the referenced license file, and even the file that does exist has no text in it. README claims "MIT. See [LICENCE](LICENCE)." but there is no MIT text anywhere in the repo.
2. **No `CITATION.cff`** (or `.zenodo.json`). Nothing for GitHub's native "Cite this repository" button, and nothing for Zenodo to read metadata from when archiving a release.
3. **No release/publish workflow.** There is no GitHub Actions job that builds an sdist/wheel and pushes to PyPI (or TestPyPI). Versioning is a hardcoded string in `pyproject.toml` (`0.1.0`) with no tagging convention.
4. **Doc/metadata inconsistencies** that will confuse a colleague following the README:
   - README says `pip install "parx[analysis]"` — there is no `analysis` extra in `pyproject.toml` (there's `embed`, which pulls `umap-learn`, not `scikit-learn`). `docs/index.md` correctly lists `h5`/`animate`/`embed`, so README is the stale one.
   - `CONTRIBUTING.md`'s clone instructions use a placeholder `git clone https://github.com/yourname/parx` instead of the real repo URL.
   - README's Installation section says `pip install parx` unqualified, implying it's already on PyPI (it isn't yet) — should be caveated until Phase 3 is done.
5. **No API reference in the docs.** `mkdocs.yml` has no `plugins:` section and no `mkdocstrings`; the `docs` extra in `pyproject.toml` only pulls `mkdocs` + `mkdocs-material`. So `usage.md`/`concepts.md` are hand-written narrative docs only — there's no auto-generated per-function/class reference page for the public API (`compute_partition`, `Partition`, `Region`, `analysis.py`, `viz.py`, etc.). For colleagues actually trying to *use* the library, that's the biggest documentation gap.
6. **No CHANGELOG.md** — nothing tracks what changed release-to-release, which matters once there's a PyPI history and a DOI per version.
7. **Repo hygiene / clutter** tracked in git that has nothing to do with the package and will look odd to an outside colleague or a PyPI project page reviewer: `.marimo_agent_state.json`, `__marimo__/session/project_overview.py.json`, `layouts/*.grid.json`/`*.slides.json`, `project_overview.py` at repo root. Not blocking, but worth a pass before a public release.
8. **No badges** on the README (CI status, docs build status, license, PyPI version, DOI) — small, but this is usually the first thing a colleague or external user checks to gauge whether a repo is alive/trustworthy.

---

## Phase 0 — Fix what's actively broken (do first, cheap)

- [ ] Decide MIT vs. another license, then create a real `LICENSE` file (standard spelling — this is what `pyproject.toml`, PyPI, and GitHub's license detector all expect) with correct copyright holder/year. Delete or redirect the empty `LICENCE` file (keep only one, update every reference to it: `pyproject.toml`, README, CONTRIBUTING's file-tree listing).
- [ ] Fix README's `pip install "parx[analysis]"` → either add a real `analysis` extra to `pyproject.toml` (if `scikit-learn`/PCA support is meant to be pip-installable separately from `dev`) or fix the README to reference the correct extra (`embed`, or move PCA support under a named extra).
- [ ] Fix `CONTRIBUTING.md`'s placeholder clone URL to the real repo.
- [ ] Caveat or remove the bare `pip install parx` line in README/docs until Phase 3 ships (e.g. "not yet on PyPI — install from source until then", with a link to the dev setup instructions).

## Phase 1 — Confirm "clone and test" actually works for a colleague

- [ ] Do a literal clean-clone dry run (fresh tmp dir, no `.venv`, no pre-instantiated Julia depot) following only what's in the README/CONTRIBUTING, to catch anything CI's warm runners might be masking (e.g. Julia General registry download on first instantiate, juliacall's own Julia install path, disk/time cost).
- [ ] Make sure `CONTRIBUTING.md` and README don't disagree on setup steps (currently CONTRIBUTING is the more complete of the two — consider having README link to it rather than duplicating a shorter/staler version).
- [ ] Decide what to do with the tracked non-package clutter (marimo state files, `layouts/`, root-level `project_overview.py`) — either `.gitignore` them properly (some are already gitignored patterns like `__marimo__/` but got force-added, since they show up in `git ls-files`) or move them under something like `notebooks/` / `internal/` so the repo root reads cleanly as "this is a Python package."
- [ ] Add badges to README once CI/docs URLs are confirmed stable (build status, license, Python versions).

## Phase 2 — CITATION and packaging metadata

- [ ] Add `CITATION.cff` at repo root (GitHub renders this automatically as "Cite this repository"; Zenodo also reads it for archived-release metadata). Include: title, authors (with ORCID if available), repository URL, license, and a `version`/`date-released` pair to update per release.
- [ ] Verify the built artifact actually contains the Julia source: run `python -m build` locally and inspect the resulting wheel/sdist contents (`unzip -l`, `tar tzf`) to confirm `julia/*` and `juliapkg.json` are present per the `[tool.setuptools.package-data]` config — this has apparently never been tested since there's no build/publish workflow yet.
- [ ] Add `CHANGELOG.md` (even a minimal Keep-a-Changelog-style file) and start logging entries from here forward.
- [ ] Decide on a versioning approach: keep manually bumping `version = "0.1.0"` in `pyproject.toml`, or switch to `setuptools-scm`/tag-derived versioning so the version always matches the git tag used for the PyPI release and Zenodo archive.

## Phase 3 — PyPI distribution

- [ ] Check `parx` is not already taken on PyPI (and on TestPyPI) — claim the name early if available.
- [ ] Register the project on PyPI and configure **Trusted Publishing** (OIDC from GitHub Actions — no long-lived API tokens to manage/rotate).
- [ ] Add a `release.yml` GitHub Actions workflow: triggered on GitHub Release (tag `vX.Y.Z`), builds sdist+wheel (`python -m build`), and publishes via `pypa/gh-action-pypi-publish` using trusted publishing.
- [ ] Do at least one dry run against **TestPyPI** before the real thing, specifically to verify a `pip install` from TestPyPI on a clean machine can actually resolve `juliacall`/other deps and that the Julia source files land correctly (this is the actual "does packaging work" test — nothing today exercises it).
- [ ] Document the release process itself somewhere (CONTRIBUTING.md or a new `RELEASING.md`): how to bump version, update CHANGELOG, tag, and what the automation does from there.
- [ ] Note the platform-specific angle already flagged in `TODO.md` §4.1/§8.4: today's install relies on juliacall bootstrapping Julia + Julia packages at first-use, which is slow but not what's blocking PyPI per se — that's a UX/startup-time issue, not a packaging blocker. Don't let it gate the initial PyPI release; just document the first-run cost (e.g. in README's Installation section: "first import will download and precompile the Julia environment, ~X minutes").

## Phase 4 — Zenodo DOI

- [ ] Log into [zenodo.org](https://zenodo.org) with GitHub, enable archiving for the `parx` repo (Zenodo → GitHub integration toggle).
- [ ] Make sure `CITATION.cff` (Phase 2) is in good shape *before* the first tagged release, since Zenodo/GitHub both use it for the archived record's metadata.
- [ ] Cut the first GitHub Release (this should be the same tag that triggers the PyPI publish in Phase 3, so PyPI version and Zenodo-archived version stay in lockstep) — Zenodo will automatically mint a DOI for it.
- [ ] After the DOI exists, add it back into `CITATION.cff` (`identifiers:` field) and into the README (DOI badge, "How to cite" section pointing to `CITATION.cff`).
- [ ] Understand Zenodo's concept-DOI vs. version-DOI distinction: the concept DOI always resolves to the latest version and is what should go on the README; each release also gets its own version-specific DOI.

## Phase 5 — Documentation completeness (`docs/` → GitHub Pages)

- [ ] Add an auto-generated API reference: add `mkdocstrings[python]` to the `docs` extra in `pyproject.toml`, add a `plugins:` section to `mkdocs.yml`, and add a `docs/reference.md` (or a `reference/` section) with `::: parx.<module>` blocks for the public surface (`compute_partition`, `load_network`, `Partition`, `Region`, `analysis`, `verify`, `viz`, `io`/`io_partition`, `diagnostics`, `precompile`). This is the single biggest documentation gap right now — `usage.md`/`concepts.md` are good narrative docs but there's no per-function signature/docstring reference.
- [ ] Add the new reference page(s) to `mkdocs.yml`'s `nav:`.
- [ ] Consider running `mkdocs build --strict` on **pull requests** too, not just on push to `main` (current `docs.yml` only triggers on push to `main` with path filters) — otherwise a broken docs build is only caught after merge.
- [ ] Once Phase 3/4 land, update `docs/index.md` and README installation sections to reflect the real PyPI install command and add the DOI/citation info, keeping the two in sync (they've already drifted once — see Phase 0).
- [ ] Add a short "How to cite" page or section once `CITATION.cff` + DOI exist.

## Phase 6 — Nice-to-haves once the above is stable

- [ ] `CHANGELOG.md` entries automated or at least checklist-enforced per release.
- [ ] Consider a `pyproject.toml` classifier bump from `Development Status :: 3 - Alpha` to `4 - Beta` once the first PyPI release is out and the API in `TODO.md`'s Tier 1–3 items has stabilized.
- [ ] Revisit `TODO.md` §4.1 (Julia precompiled sysimage) to cut first-import latency — orthogonal to publishing but the biggest UX complaint a new colleague will hit right after `pip install`.

---

## Suggested ordering for immediate colleague-testing (not the full PyPI/Zenodo path)

If the near-term goal is just "colleagues can clone and try it out this week," the minimum useful slice is **Phase 0 + Phase 1**: fix the license file, fix the two doc inconsistencies (extras name, clone URL), and confirm a truly clean clone installs and runs the test suite. Phases 2–5 (CITATION, PyPI, Zenodo, API docs) matter for the public/citable-package goal but aren't blockers for internal colleague testing via `git clone` + editable install.
