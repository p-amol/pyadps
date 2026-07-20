# Release Process

This is the checklist for cutting a `pyadps` release: version bump through
`dev`, merging to `main`, tagging, the GitHub Release, and publishing to
PyPI. The CI-wait steps are checkpoints — look at the result before
continuing, don't skip past them.

## Phase 0 — Pre-flight, on `dev`

Code for the release should already be merged into `dev` and working.

```bash
cd /path/to/pyadps
git status --short                    # must be clean
python3 -m pytest -q --no-cov         # full suite green
ruff check src/pyadps --exclude src/pyadps/utils
cd docs && rm -rf build && make html SPHINXOPTS="-W --keep-going" && cd ..
```

## Phase 1 — Version bump + CHANGELOG, on `dev`

- Edit `pyproject.toml`: `version = "X.Y.Z"`
- Edit `CHANGELOG.md`:
  - Turn the `## [Unreleased]` section's content into a new dated
    `## [X.Y.Z] - YYYY-MM-DD` entry.
  - Leave a fresh, empty `## [Unreleased]` above it for whatever comes next.

```bash
poetry lock                           # only if a dependency changed too
cd docs && rm -rf build && make html SPHINXOPTS="-W --keep-going" && cd ..
python3 -m pytest -q --no-cov         # re-verify after the bump
```

## Phase 2 — Commit + push to `dev`

```bash
git add pyproject.toml CHANGELOG.md poetry.lock
git commit -m "chore(release): bump version to X.Y.Z and finalize CHANGELOG"
git push origin dev
```

## Phase 3 — Wait for CI green on `dev` ⏸ CHECKPOINT

```bash
gh run list --branch dev --limit 1
```

Status must be `completed`, conclusion `success`. If it failed:
`gh run view <run-id> --log-failed`, fix, recommit, repeat Phase 2–3.

## Phase 4 — Merge `dev` into `main`

```bash
git fetch origin
git checkout main
git pull origin main --quiet
git merge dev                         # expect "Fast-forward"
git push origin main
```

If the merge isn't a fast-forward, stop and investigate — `main` has
diverged from what's expected (e.g. a hotfix landed directly on `main`).

## Phase 5 — Wait for CI green on `main` ⏸ CHECKPOINT

```bash
gh run list --branch main --limit 1
```

Same check as Phase 3, on `main` this time.

## Phase 6 — Tag the release

```bash
git tag -a vX.Y.Z -m "vX.Y.Z - <one-line summary>"
git push origin vX.Y.Z
```

## Phase 7 — GitHub Release

Draft release notes to a file, pulling from the CHANGELOG's new section and
reformatting to match past releases (`## Overview` / `## Key Improvements`
with emoji-headed subsections / `## Detailed Changes` table).

```bash
gh release create vX.Y.Z \
  --title "Software Release: Version vX.Y.Z" \
  --notes-file /path/to/release_notes.md \
  --target main
```

## Phase 8 — Build the package

```bash
git checkout dev --quiet              # back to the normal working branch
rm -rf dist
poetry build
twine check dist/*                    # both wheel and sdist must PASS
```

## Phase 9 — Fresh-install smoke test (strongly recommended)

This is what caught the pyarrow crash in v1.0.1 — don't skip it.

```bash
python3 -m venv /tmp/release_check_venv
/tmp/release_check_venv/bin/pip install dist/*.whl
/tmp/release_check_venv/bin/python -c "import pyadps; print(pyadps.__version__)"
```

Ideally also launch `pyadps-gui` from that venv and upload a real file
through it — a clean `import` isn't enough to catch a runtime crash in a
transitive dependency.

## Phase 10 — Publish to PyPI

```bash
twine upload dist/*                   # credentials read from ~/.pypirc
```

## Phase 11 — Verify live + cleanup

```bash
curl -s https://pypi.org/pypi/pyadps/X.Y.Z/json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d['info']['version'])
print([f['filename'] for f in d['urls']])
"
rm -rf dist /tmp/release_check_venv
```
