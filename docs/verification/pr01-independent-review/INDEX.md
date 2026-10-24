# Evidence index and reproduction

Primary source is frozen in [REPORT.md](REPORT.md); machine-readable findings are
in [findings.json](findings.json). All new files are review evidence, not fixes.

| Evidence | Purpose |
| --- | --- |
| `logs/exact-checks.json`, `logs/exact-*.txt` | Clean detached-source rerun: commands, exits, versions, install/tree/static/unit/build/smoke and empty final Git status |
| `logs/checks.json`, other matching command `.txt` files | First independent run on the review checkout; lock SHA before/after |
| `logs/independent-baseline.txt` | Reproduced original esbuild/Vite resolver conflict in archived main |
| `logs/supplemental.json`, `logs/dev.txt` | Metadata/compatibility, registry/integrity, safe credential heuristic, ignore probes, dev and clean twice |
| `logs/negative-controls.json`, `logs/negative-*.txt` | Disposable mutations; all intended failure controls work except F01 |
| `logs/exception-*.txt` | Exact one-line lint exception and its future-omission blind spot |
| `logs/css-observation.json` | Healthy vs missing required stylesheet: MIME, CSS rules, computed styles, page errors |
| `logs/service-comparison.json` | Real base/target dictionary and synthetic detector-boundary engine comparison |
| `logs/browser-install.txt` | Local matched-browser download HTTP 403 limitation |
| `logs/hosted-source.json`, `logs/hosted-source-excerpts.txt` | Original source CI API status/steps and safe downloaded job-log excerpts |

Run from a fresh clone checked out to the **review evidence branch** with Python 3,
Git, Node 24.19.0, npm 11 and `/usr/bin/chromium` available. The scripts create
disposable archived copies under `/tmp` and use their linked locked dependencies;
they never edit the Author branch. `supplemental.py` intentionally runs the clean
command after its checks, so run it last or rebuild afterward.

```sh
REVIEW=docs/verification/pr01-independent-review
git worktree add --detach /tmp/pr01-review-reproduce f20474be156522ee6141cde9c2b3a958db8d2457
npm_config_cache=/tmp/pr01-review-npm-cache npm ci
python3 "$REVIEW/probes/run-checks.py" --source-root /tmp/pr01-review-reproduce --prefix exact- --smoke
PR01_REVIEW_DIST=/tmp/pr01-review-reproduce/dist python3 "$REVIEW/probes/negative-controls.py"
PR01_REVIEW_DIST=/tmp/pr01-review-reproduce/dist python3 "$REVIEW/probes/css-observation.py"
python3 "$REVIEW/probes/service-comparison.py"
python3 "$REVIEW/probes/supplemental.py"
```

`negative-controls.py` **currently exits 1 intentionally** because the expected
CSS-loss rejection is missing. Its log preserves actual exit 0 of the Author's
unchanged smoke; this is F01, not a failing ordinary build/test run. Probes recording
inherited defects and lint blind spots characterize observations only; they are
not product tests declaring defective behavior desirable.

The matched local browser attempt used:

```sh
PLAYWRIGHT_BROWSERS_PATH=/tmp/pr01-review-browsers npm_config_cache=/tmp/pr01-review-npm-cache npx playwright install chromium
```

Hosted original CI used its matched installed Chromium without that local override.
API queries were read-only: PR #1, all-state PRs, paginated branches, reviews,
threads/comments, original workflow run/jobs/logs and official pinned action
metadata. `git ls-remote` reconfirmed main and source PR bindings before publication.
Final evidence-PR checks are linked separately in that PR's body and final response.

Captured text logs have trailing whitespace and redundant final blank lines normalized. The App UI failure log is an assertion excerpt; its redundant rendered DOM was omitted.
