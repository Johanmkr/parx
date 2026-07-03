# Road to Publish

Goal: get `parx` to a state where (1) colleagues can clone the repo and install/test it locally without friction, (2) it can be distributed as a real package on PyPI, (3) it has a citable Zenodo DOI, and (4) the docs at `docs/` (GitHub Pages) are complete enough to stand on their own.

Cross-reference: `TODO.md` already tracks feature/engineering work (§8.4 mentions PyPI as a long-term item gated on the Julia sysimage). This document is narrower and focused specifically on **packaging, licensing, citation, and distribution**.

Status: the repo is now public on GitHub. Phases 0–2 below are done; Phase 3 (PyPI) and Phase 4 (Zenodo DOI) are the remaining work for the citable-package goal.

---

## Current state (last verified 2026-07-03)

**Done:**
- Real `LICENSE` file (MIT, correct copyright holder/year) at repo root; `pyproject.toml` uses the modern `license = "MIT"` / `license-files = ["LICENSE"]` form. The stale empty `LICENCE` file is gone.
- `CITATION.cff` exists at repo root with title, abstract, authors, license, repository URL, keywords, and a `version`/`date-released` pair. The `identifiers:` (DOI) field is present but commented out, pending Phase 4.
- `pyproject.toml` has a real `analysis` extra (`scikit-learn`, used by `plot_partition_pca`) — README and `docs/index.md` now agree.
- `CONTRIBUTING.md`'s clone instructions use the real repo URL (`https://github.com/Johanmkr/parx`), not a placeholder.
- README's Installation section is caveated ("Not yet on PyPI — install from source") and links to this file.
- README has CI, docs, license, and Python-version badges.
- `CHANGELOG.md` exists (Keep-a-Changelog style) and is being updated as work lands.
- Repo hygiene: the marimo/layout tool artifacts (`.marimo_agent_state.json`, `__marimo__/session/*.json`, `layouts/*.json`, `project_overview.py`) are gitignored rather than tracked; `git ls-files` is clean of them.
- API reference docs: `mkdocstrings[python]` is in the `docs` extra, `mkdocs.yml` has a `plugins:` section and a `Reference` nav entry backed by `docs/reference.md`.
- `.github/workflows/docs.yml` runs `mkdocs build --strict` on both push-to-main and pull requests (not just push), so a broken docs build is caught before merge.
- CI (`.github/workflows/ci.yml`) installs via `uv pip install -e ".[dev]"` + `julia --project=... instantiate` across Python 3.10–3.12 on every push/PR. CI and docs workflows are both green on `main`.
- A `python -m build` + wheel/sdist content check confirmed `julia/*` and `julia/src/*` actually ship (the original `package-data` glob didn't recurse into `julia/src/`; fixed and documented in `CHANGELOG.md`).

**Not done — remaining for Phases 3–4:**
1. **No release/publish workflow.** No GitHub Actions job builds an sdist/wheel and pushes to PyPI (or TestPyPI). Versioning is still a hardcoded string in `pyproject.toml` (`0.1.0`) with no tagging convention.
2. **Not on PyPI.** `parx` availability on PyPI/TestPyPI hasn't been checked/claimed.
3. **No Zenodo DOI.** Zenodo archiving isn't enabled for the repo, and `CITATION.cff`'s `identifiers:` field is still commented out.
4. **`Development Status :: 3 - Alpha`** classifier hasn't been revisited — appropriate until the first PyPI release ships.

---

## Phase 0 — Fix what's actively broken (do first, cheap)

- [x] Real `LICENSE` file (MIT), correct copyright holder/year; stale `LICENCE` removed; every reference (`pyproject.toml`, README, CONTRIBUTING) points at it.
- [x] `analysis` extra added to `pyproject.toml`; README matches.
- [x] `CONTRIBUTING.md`'s clone URL fixed to the real repo.
- [x] README's bare `pip install parx` caveated until Phase 3 ships, with a link to source-install instructions.

## Phase 1 — Confirm "clone and test" actually works for a colleague

- [ ] Do a literal clean-clone dry run (fresh tmp dir, no `.venv`, no pre-instantiated Julia depot) following only what's in the README/CONTRIBUTING, to catch anything CI's warm runners might be masking (e.g. Julia General registry download on first instantiate, juliacall's own Julia install path, disk/time cost). *(CI does this on every push/PR across Python 3.10–3.12, which covers most of this, but hasn't been done as a genuinely cold local run.)*
- [x] `CONTRIBUTING.md` and README no longer disagree on setup steps.
- [x] Tracked non-package clutter (marimo state files, `layouts/`, root-level `project_overview.py`) is gitignored.
- [x] Badges added to README (CI, docs, license, Python versions).

## Phase 2 — CITATION and packaging metadata

- [x] `CITATION.cff` added at repo root.
- [x] Verified the built artifact contains the Julia source (`python -m build`, inspected wheel/sdist contents); fixed a `package-data` glob bug that was silently dropping `julia/src/*.jl`.
- [x] `CHANGELOG.md` added and being kept current.
- [ ] Decide on a versioning approach: keep manually bumping `version = "0.1.0"`, or switch to `setuptools-scm`/tag-derived versioning so the version always matches the git tag used for the PyPI release and Zenodo archive. Needed before Phase 3's first tag.

## Phase 3 — PyPI distribution

- [ ] Check `parx` is not already taken on PyPI (and on TestPyPI) — claim the name early if available.
- [ ] Register the project on PyPI and configure **Trusted Publishing** (OIDC from GitHub Actions — no long-lived API tokens to manage/rotate).
- [ ] Add a `release.yml` GitHub Actions workflow: triggered on GitHub Release (tag `vX.Y.Z`), builds sdist+wheel (`python -m build`), and publishes via `pypa/gh-action-pypi-publish` using trusted publishing.
- [ ] Do at least one dry run against **TestPyPI** before the real thing, specifically to verify a `pip install` from TestPyPI on a clean machine can actually resolve `juliacall`/other deps and that the Julia source files land correctly.
- [ ] Document the release process itself somewhere (CONTRIBUTING.md or a new `RELEASING.md`): how to bump version, update CHANGELOG, tag, and what the automation does from there.
- [ ] Note the platform-specific angle already flagged in `TODO.md` §4.1/§8.4: today's install relies on juliacall bootstrapping Julia + Julia packages at first-use, which is slow but not what's blocking PyPI per se — that's a UX/startup-time issue, not a packaging blocker. Don't let it gate the initial PyPI release; just document the first-run cost (e.g. in README's Installation section: "first import will download and precompile the Julia environment, ~X minutes").

## Phase 4 — Zenodo DOI

- [ ] Log into [zenodo.org](https://zenodo.org) with GitHub, enable archiving for the `parx` repo (Zenodo → GitHub integration toggle).
- [ ] `CITATION.cff` (Phase 2) already exists and is in good shape; nothing further needed here before the first tagged release.
- [ ] Cut the first GitHub Release (same tag that triggers the PyPI publish in Phase 3, so PyPI version and Zenodo-archived version stay in lockstep) — Zenodo will automatically mint a DOI for it.
- [ ] After the DOI exists, uncomment/fill in `CITATION.cff`'s `identifiers:` field and add a DOI badge + "How to cite" section to the README.
- [ ] Understand Zenodo's concept-DOI vs. version-DOI distinction: the concept DOI always resolves to the latest version and is what should go on the README; each release also gets its own version-specific DOI.

## Phase 5 — Documentation completeness (`docs/` → GitHub Pages)

- [x] Auto-generated API reference: `mkdocstrings[python]` in the `docs` extra, `plugins:` section in `mkdocs.yml`, `docs/reference.md` with `::: parx.<module>` blocks, and a `Reference` entry in `mkdocs.yml`'s `nav:`.
- [x] `mkdocs build --strict` now runs on pull requests as well as push to `main`.
- [ ] Once Phase 3/4 land, update `docs/index.md` and README installation sections to reflect the real PyPI install command and add the DOI/citation info, keeping the two in sync.
- [ ] Add a short "How to cite" page or section once the DOI exists.

## Phase 6 — Nice-to-haves once the above is stable

- [ ] `CHANGELOG.md` entries automated or at least checklist-enforced per release.
- [ ] Consider a `pyproject.toml` classifier bump from `Development Status :: 3 - Alpha` to `4 - Beta` once the first PyPI release is out and the API in `TODO.md`'s Tier 1–3 items has stabilized.
- [ ] Revisit `TODO.md` §4.1 (Julia precompiled sysimage) to cut first-import latency — orthogonal to publishing but the biggest UX complaint a new colleague will hit right after `pip install`.

---

## Suggested ordering for what's left

Phases 0–2 and the bulk of Phase 5 are done, and the repo is already public with green CI/docs. What remains is entirely the citable-package goal: **Phase 3 (PyPI) → Phase 4 (Zenodo DOI)**, gated on picking a versioning approach (end of Phase 2). Phase 1's clean-clone dry run is a cheap, low-risk sanity check worth doing once before the first PyPI release, since it's the one item CI doesn't fully substitute for.
