# PR-05 independent acceptance review

**Verdict: scoped PASS WITH CONDITIONS.** The two user-visible pointer defects
are independently reproduced on main and corrected at the frozen PR #11 head.
There are no P0/P1/P2 findings. F01 is a nonblocking, P3 floating-point boundary
inconsistency: the strict rational numeric oracle exits 1 on 36 exact fractional
edge vectors. It is not reported as a passing test or as an observed native
pointer failure. All standard checks, independent browser/image scenarios and
three additional component tests pass. Existing PR04 conditions remain below.

This is a separate non-author review, performed only in the selected cloud
environment on 2026-10-24. No implementation correction, existing test change,
workflow change, historical evidence edit, merge or deployment was made. The
new evidence branch is directly stacked on PR11's verified head branch. Its
own HEAD/tree and Draft PR URL are recorded in the PR description and handoff,
avoiding a self-referential commit hash inside this commit.

## Frozen target and provenance

| Item | Verified value |
| --- | --- |
| Product PR | [Draft #11](https://github.com/TakashiSasaki/vision-edge/pull/11) |
| Product branch | `fix/overlay-coordinate-mapping` |
| Reviewed HEAD | `be1402c793616788e81a746b6a21b03a25a5a3ec` |
| Reviewed tree | `a6c4f4eaa0d3db02a85861653d27bfcfdb53005d` |
| Actual main/base | `f97e19ac5f9f07eca23732003844b5082cd27d37` |
| Base tree | `e357f512a371df5c86950913cf3a43208f0321ce` |
| Review branch | `review/pr05-coordinate-independent` |
| Product final-head CI | [run 40091330111, verify job 120864445166](https://github.com/TakashiSasaki/vision-edge/actions/runs/40091330111/job/120864445166), completed/success |

Fetched all remote heads and `refs/pull/11/head` before checkout; the reported
HEAD/tree/base matched. Read PR metadata, comments and review submissions: no
existing review or comments, and no coordinate-review branch, so this does not
duplicate an existing review. Re-fetched main and the product branch after
testing: both remained at the values above. See [initial API snapshot](frozen-api.json)
and [remote branch inventory](remote-heads.txt).

Searched the repository, its parent workspace and mounted instruction directories.
No `AGENTS.md` or `.agents/skills/*/SKILL.md` was present. Read the available
`coordinate-codex-review-cycle` cloud skill. The implementation author was not
used as the expected-value oracle. Read the entire executable diff and author
report/evidence, including the recorded failed experiments and corrected fixture.

[source-binding.json](source-binding.json) hashes all 59 tracked non-document
files against the frozen Git objects. The detached baseline's 337 tracked files
also matched their base Git objects. Only five executable files differ between
base and product: the helper, VisionCanvas click handler, helper/component tests
and two new E2E cases. All other product, test, build and CI files are identical.
The only adjustment to an old component case supplies the newly-read video
rectangle; its original assertions remain. Original smoke and controls are intact.

## Finding F01 — P3 / low, nonblocking

**Exact fractional left edges can be rejected after arithmetic rounding.**
Affected code: [`src/services/videoCoordinates.ts:20-28`](https://github.com/TakashiSasaki/vision-edge/blob/be1402c793616788e81a746b6a21b03a25a5a3ec/src/services/videoCoordinates.ts#L20).

Reproducer, independent of React and the production helper's own calculation:

```js
clientPointToVideo(
  3779 / 24, 82.5,
  { left: 37.25, top: 82.5, width: 960.75, height: 540.25 },
  640, 480,
)
// Expected by the documented inclusive-edge contract: { x: 0, y: 0 }
// Actual: null
```

Exact rational content width is `2161/3`; exact client left is `3779/24`.
Rounding the expected point once gives `157.45833333333334`. The implementation
computes left `157.45833333333337`, then a negative normalized x and returns
null. The difference is about `2.84e-14` CSS pixels. At other offsets, 36 of
1,616 rational/invalid checks reject these fractional left corners/midpoints;
the other 1,580 checks pass, including points 0.001 CSS pixels inside/outside
each edge and every tested invalid input. [All failing vectors](numeric-failures.json),
[strict command output](logs/numeric-rational.txt), and complete vectors/results
in [raw-evidence.zip](raw-evidence.zip) preserve the failure.

This does not reproduce a visible native click failure: native MouseEvent
coordinates in this Chromium are integral, and all 24 real browser cases pass.
The discrepancy is many orders of magnitude below a physical pixel. It does
limit the author's unqualified claim of inclusive *every exact fractional edge*.
Suggested author action: add the rational boundary regression and stabilize the
comparison within representational precision, retaining rejection at 0.001 CSS
pixels outside. Alternatively explicitly document this numerical boundary limit.
This is optional numerical hardening, not a blocking pointer defect. A change
requires a new author checkpoint and re-review; no author code was altered here.

## Independent numeric, browser and image matrix

| Scope | Independent oracle / observation | Result |
| --- | --- | --- |
| 640×480 in 960×540 | Fixed content 720×540, inset (120,0) | PASS |
| 1280×720 in 800×600 | Fixed content 800×450, inset (0,75) | PASS |
| Equal 16:9 | Fixed 960×540, zero inset | PASS |
| 600×600 square source in wide box | Fixed 540×540, inset (210,0) | PASS |
| 1280×720 in square box | Fixed 600×337.5, inset (0,131.25) | PASS |
| Fractional 960.75×540.25 box | Python Fraction oracle, content 2161/3 × 2161/4 | Interior/margins PASS; strict left boundary FAIL F01 |
| All corners, all edge midpoints, center and interior | 1,584 exact-rational vectors, three viewport offsets, four DPR values | 1,548 PASS / 36 F01 |
| 0.001 CSS pixels inside/outside each edge; outside all four corners | Exact rational points, not helper-derived values | PASS |
| Zero, negative, NaN and infinite dimensions/pointers/origins | Additional 32 explicit invalid checks | PASS |
| Real browser layout | Six scenarios × DPR 1/1.25/2/3; native video, no rectangle mocks | 24/24 PASS |
| Page offsets / real scrolling | Expanded page, actual scrollY=400, client rect (37.25,82.5) | 24/24 PASS |
| CSS resize immediately before click | Actual style change and dispatch in one JS task | 24/24 PASS |
| Native intrinsic resize before click / before drawing | Native stream 640×480 → 1280×720; backing still 640×480 during click | PASS |
| Unready/zero/invalid native DOM inputs | Test-only own-property shadowing; no hit/speech on invalid input | 24/24 PASS |
| BBox/pixel-mask/polygon coordinates | Separate component fixture at .25,.25; fixed pixel/vertex/arc expectations | 3/3 PASS |
| Rendered source, BBox and ripple | Pillow screenshot pixel oracle; known colors, fixed geometric coordinates | 26/26 image cases PASS |

The numeric generator uses Python `Fraction` to calculate client points exactly
from fixed reference rectangles, then rounds only once to a JS number. It never
calls the production helper. The browser and image harnesses do not import that
helper at all. DPR is varied as an independent input; no expected coordinates
are multiplied by DPR. The two browser axes remain unmirrored.

The real app uses its existing **test-only** SDK double to report source box
(64,48,128,96). A retained canvas generates native video with background RGB
(12,34,56) and object RGB (240,180,40). Actual source samples and rendered pixels
match. Tests change actual parent CSS size/position to fixed rectangles; neither
video/Canvas `object-fit` nor `getBoundingClientRect` is replaced. Both elements
measure the same rect, contain, centered, transform none, padding/border zero.
Canvas backing follows native video dimensions. Passive Canvas observers record
roundRects/arcs without changing drawing. Speech is intercepted, not audible.

The image oracle checks the source interior/background, the unchanged BBox at
the fixed source edge, selected white BBox and cyan ripple at the known click.
Subpixel strokes are compared against alpha mixtures with the known adjacent
source colors. Original screenshots are retained, not recolored. It separates
drawing from pointer behavior: the base drawing is aligned, while its ripple
appears at the wrong point. See [image results](image-results.json) and samples
[before](before/), [after](after/), [fresh baseline](before-visible-fresh/).

### Raw before/after outcomes

| Case (DPR 2, actual client rect starts 37.25,82.5) | Unmodified base | Frozen repaired product |
| --- | --- | --- |
| 4:3 in 960×540; margin (155.25,195) | Selects dog, speaks once, ripple source (78.5,100) | No selection, speech or ripple |
| Same case, visible source (180,130), client (359.75,228.75) | Fresh scene: no selection, speech or ripple | Selects dog, speaks once; arc (179.3333,129.3333) |
| 16:9 in 800×600; margin (99.75,155.5) | Selects dog, speaks once, ripple source (98.8,87) | No selection, speech or ripple |
| Same case, visible source (180,130), client (149.75,238.75) | Fresh scene: no selection, speech or ripple | Selects dog, speaks once; arc (178.8,128.8) |

Both baseline runs exit 1 by design: [margin-then-visible](logs/browser-before.txt)
and [visible click in a fresh scene](logs/browser-before-visible-fresh.txt).
The latter avoids mistaking a selection retained from a prior false margin hit
for a successful visible click. The repaired 24-scenario sweep exits 0, accepts
at most one CSS pixel of native event quantization, and keeps one acquisition
and unchanged model creation counts. [Measured JSON](after/results.json) records
all geometry, selection, speech, box and ripple values.

The additional native resize test holds only RAF delivery, resizes the retained
canvas stream and dispatches a click from the actual video `resize` event.
During input, video is 1280×720 and Canvas backing remains 640×480; after release,
backing becomes 1280×720 and arc is (319.6666666667,179.3333333333), exactly the
fixed normalized expectation. No stream/task recreation occurs. See
[intrinsic-results.json](intrinsic-results.json). Separate component tests cover
the same race and unchanged mask coordinates without depending on native timing.

## PR04 and preserved behavior

Reviewed frameScheduler's capture/consumed-image identity, bootstrap, owned
VideoFrame/pixel lifetime, JobLock token ownership and generation checks; the
camera hook's acquisition/playback/cleanup generations; VisionEngine's task and
output generations/reentrancy/recovery; VisionCanvas's draw/adopt/HUD lifecycles;
and PerformanceMetrics's successful-event accounting and aging. None changes.
The click handler creates no scheduler, camera or model dependency. Ripple
remains normalized and is multiplied exactly once by backing width/height when
drawn. It does not introduce mirroring or a DPR transform.

Re-ran the **unmodified baseline** suites: 116 unit tests, 14 browser tests and
all 7 controls pass. Re-ran the **frozen product**: 162 unit tests, 16 browser
tests and all 7 controls pass. This measures current counts instead of forcing
historical counts. Native frame identity/initial-frame, compatibility fallbacks,
exception recovery, immutable ownership and lifecycle assertions remain present
and execute. Existing mask alpha/polygon/BBox fallback semantics are unchanged;
no transparent-pixel policy, tracking or speech-policy expansion was reviewed.

Default production bundle contains none of `visionSdkProbe`, `visionSdkGate` or
`Test-only SDK fault`; all are present in the explicitly separate e2e build.
Vite aliases the SDK only for mode `e2e`; the production CI step invokes ordinary
`npm run build`. Normal production builds therefore retain the real SDK.

## Commands, CI and environment

| Command / location | Exit / observed result |
| --- | --- |
| `npm ci` | 254, default `/home/agent/.npm` unavailable; retained failure |
| `npm ci --cache /workspace/npm-cache` | 0, 265 locked packages installed |
| Product `npm run typecheck`, `npm run lint`, `npm test`, `npm run build` | All 0; 162 tests |
| Product `npm run test:e2e`, `npm run test:e2e:controls` | Both 0; 16 browser tests; two intact successes and five specific fault detections |
| Product/base `git diff --check` | 0 |
| Detached base `npm test`, build, E2E, controls | All 0; 116 / 14 / 7 |
| `npx --no-install vitest run --config docs/verification/pr05-independent/vitest.config.mjs` | 0; three independent component cases |
| `node docs/verification/pr05-independent/browser-oracle.mjs` | 0; 24 cases |
| `node docs/verification/pr05-independent/intrinsic-oracle.mjs` | 0; native intrinsic race |
| `python3 docs/verification/pr05-independent/image-oracle.py` | 0; 26 image cases |
| `python3 .../generate-vectors.py`; `node --import tsx .../numeric-oracle.mjs` | Generator 0; strict oracle 1; 1,580 pass / 36 F01 |
| Evidence-inclusive typecheck and lint | Both 0 |

[Standard command/exit/duration ledger](logs/standard.json),
[baseline ledger](logs/baseline-standard.json), readable command logs and raw
logs are retained. Node is 24.19.0, npm 11.9.0, Playwright 1.64.1, local system
Chromium 154.0.8037.57. Local browser commands explicitly set
`PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium`. The supported matched
browser install into a writable workspace path failed HTTP 403 `Domain forbidden`;
no bypass was attempted. This is a denied browser-download action, not a passing
matched-browser local check. Independent tests use synthetic video/test SDK and
intercepted speech, not physical camera, model, GPU or audible audio evidence.

Independently fetched the exact product run/job metadata, artifact metadata and
decoded job log. Checkout ref, checkout output and `git log -1` all name the
reviewed SHA. Every verification step completes successfully; log counts are
162/16/7. Hosted install uses Chromium 156.0.8078.4/v1248, Node 24.19.0 and npm
11.17.0, without the system-browser override. The artifact's run/head binding and
digest `b7e9778fed656cb61c0891d00ad7239b8bef75d7c4c668bc59014607d392252b`
match. See [CI API receipts](ci-receipt.json) and [decoded log](logs/product-ci.txt).
The independently fetched tool-decoded log's original bytes are in the raw ZIP,
SHA-256 `bf7396a9dc715667a47a2b64302275018fd2fd79cc42b93ea08c31660f8aa940`.
This receipt has its own transport decoding and is not asserted byte-identical
to the author's differently fetched log. Product CI is the acceptance evidence;
any subsequent review-branch CI tests the evidence branch only.

## Replay and evidence integrity

Install dependencies with the workspace cache; run `run-standard.py` for the
unchanged package scripts. For the independent browser harness, build with
`npm run build -- --mode e2e --outDir .cache/vision-edge-e2e` and serve it with
`npm run preview -- --outDir .cache/vision-edge-e2e --host 127.0.0.1 --port 4185 --strictPort`.
Then run browser-oracle, intrinsic-oracle and image-oracle. Default browser output
is this directory's `after/`. Baseline used a new detached worktree at the base
SHA, symlinked identical locked dependencies, build mode e2e to
`.cache/independent-e2e`, and preview port 4186. Invoke the same harness with
`REVIEW_URL=http://127.0.0.1:4186 REVIEW_BASELINE=1 REVIEW_OUTPUT=.../before`;
add `REVIEW_NO_MARGIN=1` and a separate output directory for fresh visible-click
reproduction. Expected baseline command exit is 1, not a harness crash.

Run `generate-vectors.py` before the numeric oracle. Its full once-rounded inputs
and outputs are also archived. `raw-evidence.zip` holds 96 original files
(screenshots, command outputs, numeric inputs/results, CI tool envelope/log).
[raw-inventory.json](raw-inventory.json) lists exact SHA-256 values.
[standard-browser-evidence.zip](standard-browser-evidence.zip) preserves the full
ordinary product and baseline Playwright reports/results. Readable text logs
have terminal blank lines/trailing spaces/ANSI color removed where needed;
[normalization records](log-normalization.json) retain original/readable hashes.

Earlier independent harness failures are disclosed: component assertions passed
but teardown tried `mockClear` on drawing style properties; cleanup now checks
for a mock function. The first screenshot oracle sampled under the baseline's
misplaced feedback and assumed fully covered strokes at subpixel scales. It now
uses a fixed unobscured source point and explicit known-color alpha mixtures.
The initial JS numeric sweep double-rounded some reference boundaries; the final
Fraction generator removes that oracle ambiguity and still reproduces F01.
Original failed output remains in the archive/logs. No product or existing test
was changed to obtain a pass.

## Conditions and remaining scope

- Centered contain/no-transform/no-padding CSS is the reviewed geometry contract.
- F01 remains a disclosed low-priority numerical boundary limitation. No blocking
  author correction is required for the reproduced native pointer repair; before
  claiming all exact fractional boundaries, harden it or explicitly accept/document
  that precision limit and return a new checkpoint if implementation changes.
- PR04 identical-content coalescing does not prove physical camera stop. Full RGBA
  compatibility comparison remains O(pixels), on the main thread, outside the
  processing-duration HUD. The prior future async lock-wakeup concern remains
  nonblocking; this PR does not worsen those limitations.
- Physical camera, real SDK/model/GPU, audible audio, mobile orientation/background
  behavior and equality with the public deployment are unexecuted. Synthetic
  evidence does not close these conditions or supply performance measurements.
- Matched local browser download is denied (HTTP 403); local system-browser checks
  and exact product-head hosted matched-browser CI are separately identified.

The review is complete at this frozen checkpoint. No permission to merge PR11
or to start a later phase is implied.
