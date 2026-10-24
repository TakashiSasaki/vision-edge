# Independent non-Author review of PR-01

**Outcome: PASS WITH CONDITIONS.** The dependency/build/static/unit foundation
works, but F01 must be corrected and independently reverified before PR-01 final
acceptance or PR-02 start. No P0/P1 finding was identified. One new P2 coverage
defect remains open; three inherited P2 defects and one P3 scoped exception are
recorded separately. A foundation pass is not whole-app or real-device acceptance.

Review performed independently in a fresh selected Linux cloud environment on
2026-10-24 UTC. The reviewer did not implement PR-01, change its branch, submit a
review to PR #1, or repair its implementation. This evidence PR adds only the new
`docs/verification/pr01-independent-review/` directory directly above the Author
branch. Existing source, tests, workflow and accepted evidence are preserved.

## Frozen source and preflight

| Binding | Value |
| --- | --- |
| Repository / target | `TakashiSasaki/vision-edge`, Draft PR #1 |
| Reviewed branch | `chore/build-ci-test-foundation` |
| Reviewed HEAD | `f20474be156522ee6141cde9c2b3a958db8d2457` |
| Reviewed tree | `85c621486eafb7eb2bdf95189fc89c059ba7dc83` |
| Base main HEAD | `90323c00a6cd9a8157bf3eec853dbe5a857f07a1` |
| Base tree | `93d2b21a415fba1ba14189c336d38082fd65f505` |
| Lockfile SHA-256 | `1501945de52bc26a7411bd139358604ab1a189b51608c011a93d3164baa6de5b` |
| Review branch | `review/pr01-build-ci-test-foundation` |

Cloned into `/workspace/pr01-independent-review`, fetched all branches and PR head,
and checked out the expected SHA. GitHub PR metadata, paginated branch search,
all-state PR collection, review submissions, review threads and comments were
checked before review. Only main and the Author branch existed; only PR #1 existed;
there were no reviews, threads or comments. Immediately before publication the
remote main, Author branch and PR head still matched the table and no duplicate
review PR existed. The review branch did not already exist.

No `AGENTS.md` or `.agents/skills` exists in either source tree; `/workspace/.agents`
was empty. No repository skill instructions could therefore be read. The supplied
scope and environment instructions govern this review. Local `gh` authentication
was invalid, so remote metadata and publication use the connected GitHub app; Git
fetch/clone work. No credential values were copied into this evidence.

The exact-source rerun used a second detached worktree at `/tmp/pr01-exact-source`,
with a fresh `node_modules`, no review probes/logs in the source directory, and an
empty final `git status --porcelain`. This matters because Tailwind automatically
discovers text sources: build results from an evidence branch need not have the
same CSS hash as a clean source checkout. Exact-source results are primary.

## Findings register

The structured register in [findings.json](findings.json) includes contracts,
reproduction, observations, expected behavior, impact and correction acceptance
conditions for every item. Locations below refer to the frozen reviewed HEAD.

| ID / severity | Location | Observation and disposition |
| --- | --- | --- |
| **F01 / P2 medium / open condition** | `e2e/smoke.spec.ts:8`, `:53`; `README.md:91` | Smoke passes when the required generated stylesheet is replaced by a missing URL. Correct before foundation acceptance / PR-02. |
| F02 / P3 low / scoped deferral | `src/components/VisionCanvas.tsx:250` | One effect-level dependency diagnostic is suppressed; it also hides new missing dependencies at that same effect. Preserve the narrow exception only until lifecycle/overlay repair. |
| F03 / P2 medium / inherited | `src/services/dictionary.ts:130` | `carpet` translates as car; `constructor` returns a function. Independently observed identically at main and target. Defer to PR-06. |
| F04 / P2 medium / inherited | `src/services/visionEngine.ts:267`, `:121`, `:5` | YOLO/SegFormer share ObjectDetector and generated polygons; backend choices change labels while initialization requests GPU. Source and synthetic-boundary observations; reconcile in PR-02. |
| F05 / P2 medium / inherited | `src/components/VisionCanvas.tsx:127`, `:99`; `src/App.tsx:112` | Fresh inline callback can restart camera effect on parent renders; late acquisition has no cancellation guard. Source-confirmed, without device reproduction. Defer to PR-03. |

**F01 reproduction:** build the reviewed HEAD, archive its unchanged implementation
into a disposable directory, copy that production build, and replace only the
generated stylesheet `href` in `dist/index.html` with `/assets/review-missing.css`.
Run the unchanged smoke. It exits **0, one passed**, even though the stylesheet is
empty and styling is lost. The independent observation records HTTP **200
`text/html`**, zero CSS rules, transparent body background and default **8px** body
margin. The healthy build records `text/css`, 48 rules, `rgb(7, 11, 20)` background
and `0px` margin. Vite's SPA fallback supplies HTML for the missing CSS. Neither a
status >=400 nor `requestfailed` is produced; `pageerror` is empty. This is a
coverage defect, not evidence that the unmodified product currently lacks CSS.

Accept F01 correction when the production smoke validates required CSS content/MIME
or a stable applied-style invariant, this missing-required-CSS control fails, and
the healthy offline production app remains green. Run full hosted CI and focused
independent verification against the corrected exact SHA. Merely pointing to the
existing green CI does not resolve this negative control. The reviewer did not
change the implementation or narrow README claims on the Author's behalf.

## Dependencies, scripts and product diff

Independent archived-main resolver execution reproduced `ERESOLVE`: direct
esbuild 0.25.12 conflicts with Vite 8.3.5's optional peer range
`^0.27.0 || ^0.28.0`. No source/config import of esbuild was found. Removing that
unused direct dependency is justified; locked esbuild 0.28.2 satisfies the tool
tree. `npm ls --all` completes with no invalid peers. Missing optional packages for
other platforms are expected. No force/legacy resolver settings were used:
`force=false`, `legacy-peer-deps=false`, `ignore-scripts=false`.

Lockfile v3 has 333 package entries; all non-root entries have integrity values and
every resolved URL is under public `registry.npmjs.org`. `npm ci` succeeds with 265
installed packages and leaves the lock unchanged. Installed package metadata
confirms TypeScript 6.0.3 lies in typescript-eslint 8.72.0's
`>=4.8.4 <6.1.0` range; the original TypeScript 7 range does not. Vite 8.3.5,
React plugin 6.1.2, Vitest 5.0.5, RTL 16.3.3, React 19 and ESLint 10 peer/engine
requirements are compatible with the tested installation. jsdom 30.1.3 accounts
for the Node 24.15+ minimum. Package engines allow npm 11; `packageManager`
documents 11.9.0 but does not force setup-node's bundled npm version.

MediaPipe SDK 0.10.14 matches the pre-existing 0.10.14 WASM URL. Typechecking and
bundling establish API/build compatibility only; this pin is not a runtime model
or GPU compatibility certification. The Selfie model's `latest` URL remains mutable.

The complete base-to-target product diff was inspected: unused state bindings,
props/imports and the unused RAF timestamp parameter are removed; setters/effects
remain. All paths assign `finalObjects` before return after its initial empty array
is removed. A real-engine comparison with synthetic detector inputs exercises
populated, empty and retention-expiry cases across all four model selections;
base and target object outputs are identical. The probe substitutes detector
boundary results, not real inference. The Vite alias switches from `__dirname`
to supported ESM `import.meta.dirname`. No additional camera, model, inference,
geometry, scheduling or product UI implementation was introduced.

Dev returns HTTP 200 with `/src/main.tsx`. Production build and preview/smoke work.
Clean removes `dist` and disposable `server.js`, and succeeds twice with absent
outputs. Its Node filesystem implementation is plausibly portable; **only Linux
was executed**. Windows/macOS, watch mode and alternate Node/npm versions were not
executed. No deployment workflow/path was established or exercised; README's
published URL is not verified as a deployment of this source or a deploy guarantee.

Ignore probes cover dependency/build/report/cache/editor and local `.env*` files;
`.env.example` stays trackable. Five already-tracked `.aistudio` artifacts remain
tracked despite the new ignore rule; all are inherited, with no new accidental
generated files in PR-01. The placeholder env file is unchanged. Tracked text was
reviewed and scanned for common private-key, GitHub, AWS and Google credential
patterns without printing values; no matches were found. This heuristic is not
an exhaustive secret audit of history or binary images. No credentials were needed.

## ESLint, tests and browser boundaries

ESLint really runs over TS/TSX using typescript-eslint recommended rules and React
Rules of Hooks/exhaustive-deps; it is distinct from `tsc --noEmit`. Independent
stdin fixtures fail for explicit `any`, unused TS bindings, conditional Hooks and
missing effect dependencies. A wrong-type source fixture fails with TS2322.

Removing only the one inline suppression produces exactly one exhaustive-deps
diagnostic for `drawOverlays`. Its capture of `clickRipple` and the dependency
array match main; adding dependencies naively would alter loop restart behavior.
The exception is specific to one rule/line and documents the deferred bug, with no
global rule disable. A separate probe shows that adding a new `facingMode` read to
this same effect also passes under the directive. Future work must revisit this
line rather than assuming exhaustive coverage there (F02). Real click feedback
was not camera/device-tested.

The eight tests call real dictionary known/unknown/ADE fallback and color code,
real TTS initial/disabled/unavailable and subscription code, and real App/Header/
comparison UI. Only external MediaPipe loading is mocked in App tests; camera is
denied and canvas context is null. No test asserts the known dictionary defects or
fake model implementations as correct. Separate mutations of known translation,
unknown fallback, colors, unavailable/disabled TTS return value and header text all
fail existing tests. These controls establish useful assertion sensitivity, not
complete behavioral coverage. Audible speech and inference are untested.

Playwright runs **production preview after build**, not dev or a replacement page.
It exercises real HTML/CSS/JS, camera denial, header/model/comparison interaction,
and blocks all non-local network requests with service workers blocked. The test
waits until the real loader attempts jsDelivr. Offline/CDN-independent shell
behavior passes; offline inference is not claimed. A separate injected uncaught
exception fails specifically at the final `pageErrors` assertion, proving fatal
JavaScript detection. Required CSS loss is the distinct missed control in F01.
Console errors from intentionally blocked external resources are expected; broad
console-error rejection would require filtering those intentional failures.

## Executed checks and hosted CI

Linux x86_64; Node **v24.19.0**, npm **11.9.0**, system Chromium
**154.0.8037.57**. An isolated `/tmp` npm cache avoided the unwritable home cache.

| Command / check | Independent outcome |
| --- | --- |
| `npm ci` in clean detached source | 0; 265 packages; no resolver override |
| `npm ls --all` | 0; compatible dependency tree |
| `npm run typecheck` | 0 |
| `npm run lint` | 0; no warnings |
| `npm test` | 0; 3 files / 8 tests |
| `npm run build` | 0; 1,668 transformed modules |
| `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run test:e2e` | 0; 1 production smoke |
| `npm run dev -- --host 127.0.0.1 --strictPort` | HTTP 200; real source entrypoint |
| `npm run clean`, twice | 0 twice; both targets absent |
| `git diff --check`, exact-source final status | 0; clean |
| Negative controls | Expected failures for types/lint/services/UI/uncaught exception; required CSS loss unexpectedly passes (F01) |
| `npx playwright install chromium` | 1; local HTTP 403 `Domain forbidden` |

The exact clean build produced `index-Bohi0K68.css` (26.75 kB) and
`index-BGqVEvfJ.js` (401.52 kB), matching the first local run. Local download failure
is an environment limitation, not a product defect; local smoke uses the explicit
system-browser override, not the matched Playwright browser. Exact-source primary
logs are indexed in [INDEX.md](INDEX.md).

The connected GitHub app independently returned the original
[run 40047953579](https://github.com/TakashiSasaki/vision-edge/actions/runs/40047953579)
as `pull_request`, `completed/success`, HEAD exactly the reviewed SHA; updated
2026-10-24 01:03:32 UTC. The
[verify job 120733858384](https://github.com/TakashiSasaki/vision-edge/actions/runs/40047953579/job/120733858384)
has successful install, typecheck, lint, unit, build, matched Chromium install,
smoke and upload steps. Downloaded job logs confirm checkout of that SHA, Node
24.19.0 / npm 11.17.0, 8 tests, build, Chromium v1248 / Chrome 156.0.8078.4 download
and 1 smoke pass. No action-runtime deprecation warning was found. Selected safe
excerpts and API metadata are preserved, without checkout authentication material.

Workflow inspection confirms all PRs and main pushes trigger it, with explicit
PR head checkout (not merge-result testing), pinned action commit SHAs, `.nvmrc`,
read-only contents permission, npm package caching, fresh `npm ci`, sequential
failure propagation, zero smoke retries, fresh strict-port preview, 30-second
server startup limit and 15-minute job limit. Upload runs unless cancelled and
retains reports for 14 days. Official action tags resolve to the three pinned SHAs
and their action metadata specifies node24. No new application secret, ruleset,
branch-protection, release dispatch or deployment was introduced. Main-push and
deliberately failing hosted-job executions were not triggered by this review;
their behavior is established from workflow structure and local nonzero controls.

The new evidence PR's CI is separate from this original source CI. Its final
head/tree, terminal run status and URL are recorded in the review PR body and
delivery response after publication, avoiding a self-referential evidence commit.

## Gate and unrun areas

Resolve F01 and reverify the corrected source binding before PR-02 starts. Then
PR-02 can safely use this dependency/build foundation to address F04; it gains no
acceptance for model/device behavior from this review. Carry F02/F03/F05 into their
recorded phases. Keep both PRs Draft; no merge, release, deployment, ruleset change
or PR-02/03 implementation was performed.

Unrun: real camera/stream/device/GPU/model loading and inference; network model
availability/accuracy; timing/FPS correctness; tracking and geometry; actual audio,
full TTS behavior and voices; broad accessibility; Windows/macOS; deployment and
published-site source equivalence. Synthetic service probes and a denied-camera
shell must not be described as those validations.
