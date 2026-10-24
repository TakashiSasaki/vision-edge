# Independent re-review of PR-01 F01 CSS smoke correction

**Verdict: PASS for the correction at the frozen source below. F01 / P2 is
closed on that source.** No new P0/P1/P2/P3 finding in the scoped correction.
The corrected foundation is ready for separately authorized PR-02/03 work when
that work includes this correction. Current main still lacks it; this review
does not merge anything or authorize the later phases.

This is a fresh non-Author agent review, dated 2026-10-24 UTC. It independently
observes browser behavior and mutates disposable production builds, rather than
treating the Author's controls or report as independent proof. Only this new
evidence directory is proposed; Author branches and historical evidence stay intact.

## Frozen sources and live preflight

Repository: [TakashiSasaki/vision-edge](https://github.com/TakashiSasaki/vision-edge).

| Role | Branch / commit | Tree |
| --- | --- | --- |
| Reviewed fix [PR #3](https://github.com/TakashiSasaki/vision-edge/pull/3) | `fix/pr01-css-smoke-validation` / `accfb1e895f16b4aec32db5dd5025a1bf2782898` | `027d88998218c98966f95db7bbe1362e6703009c` |
| Fix's sole parent and current main | `main` / `7b813beeebfa3ded90aacda3d69dab76a5ea991d` | `85c621486eafb7eb2bdf95189fc89c059ba7dc83` |
| Original product used for reproduction | `chore/build-ci-test-foundation` / `f20474be156522ee6141cde9c2b3a958db8d2457` | `85c621486eafb7eb2bdf95189fc89c059ba7dc83` |
| Original independent review [PR #2](https://github.com/TakashiSasaki/vision-edge/pull/2) | `review/pr01-build-ci-test-foundation` / `67e0c15e652d88c773a82b06f615b2befa27a695` | `328519ed7bcba68ca303173f5e7bb608c4371365` |

Git fetch, remote refs, GitHub PR/branch/review/check APIs all agreed with the
requested target; no target movement required rebinding. PR #2/#3 were open
Drafts with no submitted reviews, and PR #3 had no comments. The full PR search
returned #1, #2 and #3 only; no duplicate CSS re-review existed. Main and #2's
exact-commit `verify` checks also report completed/success (see `checks-main.json`
and `checks-pr2.json`). This does not close F01 on their old trees.

The reviewer cloned into `/workspace/vision-edge-css-review`, then used a separate
untouched detached checkout `/workspace/vision-edge-css-source` for final execution.
The latter remained at the exact fix HEAD/tree with empty `git status --porcelain`.
No ancestor/checkout `AGENTS.md` or checkout `.agents/skills` exists;
`/workspace/.agents` is empty. `logs/source-bindings.json` records these checks,
source ancestry, file/tree blob IDs and the complete changed-file scope.

## Independent F01 reproduction and correction acceptance

`probes/independent-css.py` archives the frozen original product and builds it
using the same locked dependencies. It independently discovers generated local
stylesheet URLs via HTML parsing. It uses Vite preview and real built HTML/JS/CSS,
first measuring HTTP responses, CSSOM, computed styles, real header/model state,
external loader requests and page errors with its own browser observer. It then
runs the frozen source's unchanged ordinary smoke in a fresh preview/browser.
No Author control implementation is imported.

The exact PR #2 mutation changes only the generated CSS href to
`/assets/review-missing.css`. Both old and corrected builds really return
**200 text/html**, zero parsed rules, body margin **8px**, root overflow
**visible**, header display **block**, and no uncaught page errors. The original
smoke exits **0 / one passed**: F01 reproduced. Corrected smoke exits **1 / one
failed** at **required stylesheet MIME**: the original defect is rejected.

Healthy old and corrected observations both show HTTP 200 `text/css`, **48**
top-level rules, margin **0px**, root overflow **hidden**, header display **flex**,
real local JS HTTP 200, the actual denied-camera UI and blocked MediaPipe loader.
Their complete production build file hashes are identical, including HTML/CSS/JS.
No generated asset filename is hardcoded by the mutations.

| Independent corrected-build case | Smoke exit | Required diagnostic / independent observation |
| --- | --- | --- |
| Healthy before faults | 0 | One actual smoke passed; all critical styles applied |
| Original missing-href mutation | 1 | CSS MIME; actual HTML 200 fallback |
| Delete the actual CSS file | 1 | CSS MIME; unchanged href also returns HTML 200 |
| Remove the local stylesheet link | 1 | Required local stylesheet links |
| Empty CSS, 200 CSS MIME | 1 | Parsed rules; zero rules measured |
| Invalid CSS text, 200 CSS MIME | 1 | Parsed rules; zero rules measured |
| Valid CSS served from `.txt`, 200 text/plain | 1 | CSS MIME; zero rules measured |
| Valid nonempty unrelated selector | 1 | Applied body reset; nonzero rules measured |
| Append body margin 13px override | 1 | Applied body reset; root/header styles still correct |
| Append root overflow visible override | 1 | Applied overflow; body/header styles still correct |
| Append header display block override | 1 | Applied header layout; body/root styles still correct |
| Inject uncaught independent fatal sentinel | 1 | Existing final `pageErrors` assertion, sentinel received |
| Healthy after all faults | 0 | One actual smoke passed; original build hash manifest unchanged |

All **15 independent trials** (two original and thirteen corrected) matched the
expected exit, pass/fail count and assertion diagnostic. Expected negative exits
1 demonstrate sensitivity; they are not failed verification. See
`logs/independent-css-results.json`, each `*-observation.json`, and each
`*-smoke.txt` for the measurements and unabridged test reports.

## Author controls, isolation and CI failure propagation

The Author runner's seven child outcomes were independently rerun: intact 0,
missing CSS 1, absent link 1, empty CSS 1, unrelated CSS 1, fatal JS 1, intact after
controls 0. Its own exit was **0**. A subsequent ordinary smoke passed. It copies
the build/tests/config into a fresh temporary directory, links installed
dependencies, starts each child with retries 0, requires exactly one unexpected
test plus the intended diagnostic for faults, and deletes the temporary tree in
`finally`. It does not mutate the source build or fulfill any app routes.

`probes/control-runner-sensitivity.py` also tested the runner itself in disposable
copies: (a) remove all CSS checks from the child smoke, making missing CSS pass;
(b) preserve the MIME assertion but change its diagnostic to the wrong reason.
The intact child passed in both, but the runner exited **1** respectively at
`expected smoke assertion failure` and `missing diagnostic required stylesheet MIME`.
Both temporary trees were removed after failure; the original build manifest
stayed unchanged. Thus neither an unexpectedly passing fault nor an arbitrary
failing child satisfies the control. See `logs/runner-sensitivity.json` and logs.

The ordinary smoke runs the real application and comparison/model UI interactions.
Same-origin routes only call `continue`; external routes are aborted, and camera
acquisition rejects. There is no fake local HTML/JS, module replacement, fake
inference output, or successful CDN response. Unit tests retain their disclosed
external MediaPipe loader mock/null canvas boundary; this does not leak into e2e.

CI invokes normal smoke, then controls as separate required shell steps. There
is no `continue-on-error`, exit suppression, expected-failure marking or retry
that converts a broken ordinary smoke into success. The upload step runs after
failures but cannot override the job's failure. Test selection remains one
production browser smoke and three unit/component test files, eight tests.

## Clean verification and hosted receipts

Final-source environment: Linux x86_64, Node **24.19.0**, npm **11.9.0**,
Playwright **1.64.1**, system Chromium **154.0.8037.57**. After clean locked install,
these actual commands all exited **0**:

```sh
# In the untouched detached checkout at accfb1e...
npm ci --cache /tmp/vision-edge-review-npm-cache
npm ls --all
npm run typecheck
npm run lint
npm test
npm run build
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run test:e2e
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run test:e2e:controls
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run test:e2e
git diff --check
```

Installation added 265 packages; dependency tree validation passed without
resolver overrides. Tests: **3 files / 8 passed**. `logs/npm-ci-exact-source.txt`,
`logs/checks.json`, and `logs/check-00.txt` through `check-12.txt` preserve output,
versions and exits. `checks.py` can rerun these with `PR01_SOURCE_ROOT` set.

Independent reproduction, with the evidence checkout path substituted as needed:

```sh
PR01_SOURCE_ROOT=/workspace/vision-edge-css-source python3 /workspace/vision-edge-css-review/docs/verification/pr01-f01-independent-review/probes/independent-css.py
PR01_SOURCE_ROOT=/workspace/vision-edge-css-source python3 /workspace/vision-edge-css-review/docs/verification/pr01-f01-independent-review/probes/control-runner-sensitivity.py
```

Run these sequentially because each uses port 4173. The source checkout must have
its frozen Git objects, installed dependencies and built `dist`. Evidence stays
in the evidence checkout; mutations stay in temporary directories.

Exact-fix [CI run 40050196050](https://github.com/TakashiSasaki/vision-edge/actions/runs/40050196050),
[verify job 120740948261](https://github.com/TakashiSasaki/vision-edge/actions/runs/40050196050/job/120740948261):
**completed/success**, exact HEAD `accfb1e895f16b4aec32db5dd5025a1bf2782898`,
tree `027d88998218c98966f95db7bbe1362e6703009c`, updated
**2026-10-24 01:33:03 UTC**. The reviewer fetched run metadata, exact-commit check
runs, all job steps and complete job logs successfully. They independently confirm
exact checkout, Node 24.19.0/npm 11.17.0, all static/build checks, eight tests,
matched Chromium/headless shell **v1248 / Chrome 156.0.8078.4**, one ordinary smoke
pass and all seven intended control outcomes, including five expected child exits 1.
`logs/fix-ci-job.txt`, `fix-ci-0.json`, `fix-ci-2.json`, and `checks-fix.json`
are the downloaded receipts. Text logs only trim trailing line whitespace.
No inaccessible hosted log is being inferred.

The new Draft review PR's final remote HEAD/tree and its terminal CI receipt are
recorded in that PR body, avoiding a self-referential evidence commit.

## Scope, findings and limits

The fix changes the smoke, adds one controls runner/npm script/CI step, and updates
README/evidence. `src/`, product HTML, dependencies, lockfile, Playwright/Vitest
configuration and lint exceptions are unchanged. Complete clean original/fixed
build manifests also match. README and Author evidence accurately describe the
validated contract and local/hosted browser distinction.

New scoped findings: **none**. F01 / P2: **closed at the reviewed fix commit**.
F02 / P3 lint-exception blind spot and F03/F04/F05 / P2 dictionary, runtime/backend
truthfulness and camera lifecycle remain the recorded inherited deferrals from
PR #2, unchanged and outside this focused acceptance; they are not certified or
silently corrected here. `findings.json` makes that scope explicit.

Local matched-browser download was attempted and exited 1 with HTTP **403 Domain
forbidden** (`logs/browser-install.txt`); local browser claims use Chromium 154
only. Exact-fix hosted matched-browser success is separately verified above.
`logs/setup-history.txt` discloses preliminary reviewer harness/setup errors and
the subsequent complete successful final-source runs.

Unverified: real camera/stream lifecycle, real model/GPU inference and accuracy,
geometry/tracking/FPS, audio/full TTS, broad accessibility, Windows/macOS, and
production deployment/source equivalence. Passing the denied-camera offline
shell does not certify them. Main remains the uncorrected parent until a separate
authorized integration includes this fix. No merge, auto-merge, deployment,
release, settings change or PR-02/03 implementation was performed.
