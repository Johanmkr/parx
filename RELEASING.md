# Releasing parx

This documents the actual release process — how to cut a new version and what the automation does from there. Written up after actually doing it three times (`v0.1.0`, `v0.1.1`, `v0.1.2`); see `git log` on those tags for the real history if anything here goes stale.

## One-time infrastructure (already set up, reference only)

You shouldn't need to touch any of this for a normal release — it's here so it can be recreated if the repo ever moves, or debugged if something breaks.

- **PyPI Trusted Publisher** (OIDC, no stored token): registered at [pypi.org/manage/account/publishing](https://pypi.org/manage/account/publishing/) with `owner=Johanmkr`, `repo=parx`, `workflow=release.yml`, `environment=pypi`. Must match `release.yml` exactly, or the publish step fails auth with a non-obvious error.
- **GitHub Environment `pypi`**: has a required-reviewer protection rule (self-approval allowed), so every real PyPI publish pauses for a manual click before it runs. Created via `gh api --method PUT repos/Johanmkr/parx/environments/pypi ...`; adjust reviewers in repo Settings → Environments if maintainers change.
- **Zenodo GitHub integration**: enabled at [zenodo.org/account/settings/github](https://zenodo.org/account/settings/github/) for `Johanmkr/parx`. Archives every GitHub Release published *after* the toggle was flipped on (2026-09-10) — it does not backfill older releases (`v0.1.0` predates this and has no DOI, which is fine and expected).
- **Concept DOI**: `10.5281/zenodo.22696450` — this is the one that goes in `README.md`/`CITATION.cff`. It always resolves to the latest archived version, regardless of how many releases come after it.

## Versioning

Nothing to bump by hand. `pyproject.toml` uses `setuptools-scm` (`dynamic = ["version"]`) — the version is derived entirely from git tags:

- Checked out exactly on tag `vX.Y.Z` → version is exactly `X.Y.Z`.
- Any commit after a tag → `X.Y.Z.postN.devM` (`version_scheme = "no-guess-dev"` — it never guesses the *next* version).

The only manual bump is cosmetic: `CITATION.cff`'s `version:` and `date-released:` fields aren't derived from anything and should be updated to match, if you want "Cite this repository" to show the current version.

## Cutting a release

1. **Merge everything you want in the release to `main` first.** The tag will point at whatever `main`'s tip is when you cut it.

   ```bash
   git checkout main
   git pull origin main
   ```

2. **Tag and push:**

   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z: <one-line summary>"
   git push origin vX.Y.Z
   ```

   Always cut the tag fresh from current `main` — never reuse or retarget an old tag (see the gotcha below).

3. **Cut the GitHub Release from that tag:**

   ```bash
   gh release create vX.Y.Z --title "vX.Y.Z" --generate-notes
   ```

   This is the actual trigger. It fires `release.yml`, which:
   - builds sdist + wheel with `uv build` (checking out the exact tag commit, so the version is clean — no dev suffix),
   - **pauses**, waiting for a manual approval on the `pypi` environment,
   - once approved, runs `uv publish --trusted-publishing always` — no API token anywhere, OIDC handles auth.

4. **Watch it and approve:**

   ```bash
   gh run list --workflow=release.yml --limit=1
   gh run watch <run-id>
   ```

   Approve via the URL it prints, or `gh run view --web` → **Review deployments** → check `pypi` → **Approve and deploy**. `gh` has no CLI command for this specific approval step — it's web-UI only.

5. **Verify:**

   ```bash
   curl -s "https://pypi.org/pypi/parx/vX.Y.Z/json" | python3 -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"
   ```

   Use the **version-specific** endpoint (`/pypi/parx/X.Y.Z/json`), not the aggregate `/pypi/parx/json` — the latter's "latest version" field can lag the real upload by several minutes on some CDN edges right after a fresh publish.

6. Zenodo archives the release automatically within a few minutes (no action needed) — check [zenodo.org/account/settings/github/parx](https://zenodo.org/account/settings/github/parx) or search `https://zenodo.org/api/records?q=conceptrecid:22696450&sort=mostrecent` for the new version DOI.

## Gotchas hit in practice

- **A release's workflow is resolved from the workflow file *at the commit the tag points to*, not just "present on the default branch."** `v0.1.0` was tagged before `release.yml` existed; `gh release create` published the release fine but silently queued zero workflow runs (no error anywhere — `gh api .../actions/workflows/<id>/runs` showed `total_count: 0`). Fixed by moving the tag to a commit that included the workflow and recreating the release. Not an issue for any tag cut after `release.yml` already exists on `main` — which is every release going forward, hence step 1's "merge first" order above.

- **The DOI can't exist before a release is archived, so it can never appear in the README baked into that same release's PyPI upload.** Getting the DOI badge to actually show up on the *displayed* PyPI project page (PyPI only shows the latest version's README) takes two rounds: one release to mint the DOI (`v0.1.1`), then a follow-up (`v0.1.2`) that adds the DOI/badges and republishes. Not needed again — the concept DOI, once minted, stays valid for all future versions without any further action.

- **shields.io badges (the PyPI version badge, etc.) are served with `Cache-Control: max-age=10800` (3 hours).** GitHub's image proxy, PyPI's image proxy, and your own browser each cache independently against that header. A badge can show a stale version for up to ~3 hours after a fresh release even though shields.io's underlying data (`img.shields.io/pypi/v/parx.json`) is already correct — this is expected and self-resolves; it's not a sign anything's broken.

## Pre-release checklist

- [ ] Everything intended for this release is merged to `main`
- [ ] `pytest` and `ruff check` are clean on `main` (CI already gates this on every PR, so normally already true)
- [ ] `CITATION.cff`'s `version:`/`date-released:` updated if you want them accurate
- [ ] Tag message says something meaningful (it becomes part of the permanent git history and the Zenodo record's metadata)
