# PR-01 F01 stylesheet smoke correction

Author implementation evidence, 2026-10-24 UTC. This proposes repair of **F01 / P2**
from [independent review PR #2](https://github.com/TakashiSasaki/vision-edge/pull/2).
It is not independent acceptance; a separate non-Author re-review remains required.

## Source and scope

Preflight fetched current main and both relevant remote branches, and inspected
the complete open-PR/branch collections. Main was
`7b813beeebfa3ded90aacda3d69dab76a5ea991d`, tree
`85c621486eafb7eb2bdf95189fc89c059ba7dc83`, identical to reviewed product
`f20474be156522ee6141cde9c2b3a958db8d2457`. Only evidence PR #2 was open;
there was no duplicate fix or competing fix branch. The new branch
`fix/pr01-css-smoke-validation` starts from that main. No `AGENTS.md` or checkout
`.agents/skills` was present, and `/workspace/.agents` was empty.

Read the report, structured finding register, index, and exact negative/CSS
observation probes at review commit `67e0c15e652d88c773a82b06f615b2befa27a695`,
tree `328519ed7bcba68ca303173f5e7bb608c4371365`. Review evidence was extracted
under `/tmp` for execution; its branch and historical evidence are untouched.

Changed infrastructure: production smoke, one disposable-control runner, its npm
script/CI step, and README/evidence. `src/`, application entrypoint, dependencies,
lockfile, camera, inference, models, tracking, TTS, UI and lint exceptions are
unchanged. F02–F05 remain outside this repair. No review branch merge was used.

## Independent reproduction of the original failure

Built the clean current-main product tree before edits with `npm run build`
(exit 0). Executed the exact original `css-observation.py` and
`negative-controls.py` from PR #2 against that build, with system Chromium.
The original probe's source archive is frozen to reviewed product SHA `f20474b`.

`logs/before-css-observation.json` records intact CSS as HTTP 200 `text/css`,
48 top-level rules, body margin `0px`, and body background `rgb(7, 11, 20)`.
Changing only the generated local stylesheet href to `/assets/review-missing.css`
returns HTTP 200 `text/html`, zero rules, margin `8px`, transparent background,
and no page errors. `logs/before-missing-css.txt` records the **unchanged original
smoke exiting 0 / one passed**, reproducing F01. The original negative-control
harness consequently exits 1 because CSS loss should fail; that is the reproduced
coverage defect, not a failed normal CI run. Its other injected controls produced
their expected outcomes, including the existing fatal JavaScript/UI checks.

Actual probe invocation, from `/workspace/vision-edge` after extracting the review
tree to `/tmp/pr01-f01-before` and linking its locked `node_modules`:

```sh
npm run build
# Give the extracted probe access to the frozen source object it archives.
git init /tmp/pr01-f01-before
git -C /tmp/pr01-f01-before fetch /workspace/vision-edge f20474be156522ee6141cde9c2b3a958db8d2457
PR01_REVIEW_DIST=/workspace/vision-edge/dist python3 /tmp/pr01-f01-before/docs/verification/pr01-independent-review/probes/css-observation.py
PR01_REVIEW_DIST=/workspace/vision-edge/dist python3 /tmp/pr01-f01-before/docs/verification/pr01-independent-review/probes/negative-controls.py
```

The original scripts set `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium`.
A first extraction-only attempt to run the negative harness lacked Git metadata;
initializing/fetching the frozen object corrected that setup failure. An early
build started before installation completed also exited 127; the completed
post-install baseline build and subsequent clean verification are the evidence.

## Correction and deterministic controls

The smoke discovers same-origin stylesheet links from the loaded production DOM,
then checks their actual browser responses for status 200 and CSS MIME. It checks
enabled CSSOM sheets with parsed rules, plus the real body's zero-margin reset,
root overflow, and header flex layout. There are no hardcoded hashed filenames,
pixel screenshots or external-resource dependencies. The original app execution,
camera denial, blocked external loader observation, model/comparison interaction,
local request failures and final uncaught-JavaScript assertions remain active.

`scripts/smoke-css-controls.mjs` copies the real build, smoke and preview config
to a temporary directory with the installed dependencies linked in. Every case
runs the same smoke in a fresh Chromium/preview process. It never replaces the
app or edits the original build. Negative cases must exit 1 with exactly one
failed smoke and the intended assertion diagnostic; an unrelated browser/server
failure cannot satisfy a control. The runner exits 0 only if all controls agree.

| Build/control | Required child result | Diagnostic |
| --- | --- | --- |
| Intact | 0 / one pass | All original UI, network, fatal-JS and new CSS checks |
| Missing generated CSS URL | 1 / one failure | Required stylesheet MIME; actual Vite HTML fallback |
| Removed local stylesheet link | 1 / one failure | Required local stylesheet links |
| Empty CSS file, still CSS MIME | 1 / one failure | Required stylesheet parsed rules |
| Nonempty unrelated valid CSS | 1 / one failure | Required stylesheet applied body reset |
| Injected uncaught exception | 1 / one failure | F01 fatal JavaScript control at existing pageErrors check |
| Intact after mutations | 0 / one pass | Build preservation and recovery |

The unrelated-CSS case contains a valid custom-property rule, so MIME and nonzero
rule count alone cannot pass it. CI now runs these controls after the normal smoke;
expected child failures are assertions of sensitivity, not failed CI jobs.

## Local verification and reproduction

Linux x86_64, Node `v24.19.0`, npm `11.9.0`, Playwright `1.64.1`, system Chromium
`154.0.8037.57`. Full output and exits are in `logs/checks.json` and numbered logs.
`logs/source-bindings.json` records file blob IDs and unchanged product/lockfile
bindings. The final publication HEAD/tree, exact-commit clean rerun and terminal
hosted CI receipts are recorded in the Draft PR body to avoid a self-referential
evidence commit. Hosted and local browser results must be read separately.

Run on a fresh final-source checkout:

```sh
npm ci --cache /tmp/vision-edge-npm-cache
npm run typecheck
npm run lint
npm test
npm run build
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run test:e2e
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run test:e2e:controls
git diff --check
```

All commands must exit 0; the control runner prints expected child exits 1 for
the five fault cases and child exits 0 for two intact cases. It uses no resolver
overrides and does not hide smoke failures. On hosted CI omit the system-browser
override after `npx playwright install --with-deps chromium`.

The local matched-browser install command below exited 1 with HTTP 403
`Domain forbidden`; `logs/browser-install.txt` preserves the output. This is an
environment download limit, not a successful matched-browser local run.

```sh
PLAYWRIGHT_BROWSERS_PATH=/tmp/vision-edge-browsers npm_config_cache=/tmp/vision-edge-npm-cache npx playwright install chromium
```

## Independent handoff and limits

Re-review the exact final Draft PR HEAD/tree, verify current base/ancestry and
scope, rerun clean checks and the automated controls, and independently confirm
that the original missing-href probe now exits nonzero specifically for CSS while
the intact real offline application still passes. Check terminal hosted CI belongs
to that exact head and includes the new control step. Only that separate reviewer
can decide whether F01's foundation condition is satisfied. The Author has not
launched or declared independent acceptance.

Unverified: Windows/macOS, real camera/stream lifecycle, GPU/model loading and
inference, accuracy, geometry/tracking/FPS, audible/full TTS, broad accessibility,
and deployment/source equivalence. The denied-camera shell does not certify those
areas. No main merge, auto-merge, deployment, release, branch protection changes,
PR #1/#2 updates or PR-02/03 implementation are part of this work.
