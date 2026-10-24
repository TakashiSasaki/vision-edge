# PR05-F01 author correction checkpoint

**F01 reproduced and corrected by the author; independent closure is pending.**
This is a response to the separate [review PR #12](https://github.com/TakashiSasaki/vision-edge/pull/12),
not a replacement for its independent verdict. The review was scoped PASS WITH
CONDITIONS, with this P3/nonblocking finding and no P0/P1/P2. No merge, deployment,
later-phase work, PR #12 edits or historical evidence changes occurred.

| Binding | Value |
| --- | --- |
| Author branch / PR | `fix/overlay-coordinate-mapping`, [Draft #11](https://github.com/TakashiSasaki/vision-edge/pull/11) |
| Main / base | `f97e19ac5f9f07eca23732003844b5082cd27d37` |
| Reviewed product HEAD / tree | `be1402c793616788e81a746b6a21b03a25a5a3ec` / `a6c4f4eaa0d3db02a85861653d27bfcfdb53005d` |
| Frozen independent evidence HEAD / tree | `57ee68bc503cfc1633039d948d1b39fcf341caa5` / `a73fd0734d0358de0bbf1a70653f780a3637c659` |
| Corrective executable HEAD / tree | `c585a09a7642fa21d68967dfe3a5a00be2038652` / `bb93e151d9316d4e4022a05773acf83b19f366d6` |
| Executable-head CI, intermediate only | [run 40093874718](https://github.com/TakashiSasaki/vision-edge/actions/runs/40093874718) |

The final evidence-bearing HEAD/tree and its own completed CI run/job/log receipt
are recorded in PR #11's correction checkpoint and author handoff. That avoids
self-referential commit hashes. Executable source/test/configuration in the
evidence commit is identical to c585a09. Intermediate CI is not used as final CI.

## Finding-by-finding response

| Finding | Reproduction / cause | Author response / verification | Closure |
| --- | --- | --- | --- |
| PR05-F01, P3 | Exact rational client left `3779/24` is 157.45833333333334; old computed left is 157.45833333333337. Normalizing first produces a small negative x and returns null. All 36 supplied failures are nine distinct fractional left-edge positions repeated at four DPRs. | Compare client coordinates directly against contained edges formed as combined quotients. Keep strict outside rejection; no epsilon or widened boundary band. Original oracle: 1,616/1,616 after versus 1,580/1,616 before. New regressions and adjacent-float controls pass. | Author correction complete; separate reviewer must determine closure. |

Fetched the frozen review object and read its README, F01 records, exact rational
generator, numeric oracle, original inputs and supporting browser/component/image
probes. The regenerated 1,584 rational inputs equal the review's raw ZIP inputs
as JSON, whose original SHA-256 is
`09d208b26aead94a35a360e059d496c7986d7ee05c9472622bd7d4b784cc10ba`.
Scripts were extracted to a separate directory, with `src` and the locked
`node_modules` linked to the author checkout. Their assertions, import paths
and output code were unchanged. Generated output modifies only that extracted
copy, not the review branch. [Source bindings](source-bindings.json) give exact
script and before/after helper hashes.

## Numerical change and boundary contract

Old computation rounded `sourceWidth * (boxHeight/sourceHeight)`, subtracted
that rounded size from the CSS box, and then centered it. Its several rounding
steps moved some exact Fraction reference edges one float inward.

For a height-constrained image, horizontal edges are now:

```
center = originX * sourceHeight + boxWidth * sourceHeight / 2
halfContent = boxHeight * sourceWidth / 2
left  = (center - halfContent) / sourceHeight
right = (center + halfContent) / sourceHeight
```

The width-constrained case transposes the formula for top/bottom. The fitted
axis uses the actual CSS origin/extent. This avoids centering an already-rounded
scaled size. For the reported rational inputs, the numerator operations are exact
in binary64 and each final boundary quotient rounds once to the supplied oracle
value. Coordinates are compared to those edges **before** normalization. Width
and height are their differences; exact endpoints map to 0/1. A point outside
the computed rectangle is rejected, even if only the next representable float
outside. No fixed pixel epsilon, ULP tolerance band, DPR factor or outside-point
clamping was added.

This is ordinary binary64 arithmetic, not an arbitrary-precision geometry API
or a proof about every possible finite JavaScript input. Nonfinite inputs,
nonpositive dimensions, overflowing calculations or collapsed/unrepresentable
content rectangles return null. The actual centered contain/no-transform CSS
contract remains. The fixed Fraction references and immediately adjacent floats
define the tested edge behavior precisely; native integral pointer behavior and
existing mask/BBox/ripple interfaces are preserved.

Only three existing executable files change: the helper, its unit tests and
VisionCanvas component tests. The click handler, drawing code, speech service,
camera/model lifecycle, scheduler/identity/ownership/generations, metrics, build
configuration, lockfile and all old tests/assertions remain unchanged.

## Before/after evidence and counts

| Check | Before reviewed helper | Corrected helper |
| --- | --- | --- |
| Unchanged frozen reviewer oracle | Exit 1: 1,580 pass / 36 fail | Exit 0: 1,616 pass / zero fail |
| Ordinary focused tests including new regressions | Exit 1: 63 pass / 25 fail | Exit 0: 88 pass |
| Additional exact-rational / adjacent-float oracle | New supplemental coverage | Exit 0: 2,400 pass |
| Frozen reviewer component vectors, author replay | Original review retained | Exit 0: 3 pass |
| Frozen reviewer actual-app browser sweep, author replay | Original review retained | Exit 0: 24 pass |
| Frozen reviewer native intrinsic-resize race, author replay | Original review retained | Exit 0: 1 pass |
| Frozen image oracle, author replay | Uses original preserved baseline images | Exit 0: 26 image cases, zero unexpected failures; corrected after images freshly rendered |

The 25 new ordinary cases cover the exact 36 original vectors through offset,
DPR and edge loops; transpose them for vertical margins; check all exact
edges/corners, next-float-inside and next-float-outside, 1e-12 and 0.001 CSS-pixel
outside controls. A real component case verifies selection at both rational
horizontal edges, exact backing-coordinate ripples, silent adjacent margins,
and no camera/model recreation. The old helper was temporarily restored for
the recorded before run and then restored to the correction before all after
checks. No old assertion was weakened.

The supplemental Python Fraction generator independently derives exact content
rectangles, rounds their edges once, and uses Python `math.nextafter` for adjacent
controls. Ten source/display combinations, three page offsets (including negative
scrolled positions), and DPR 1/1.25/2/3 yield 480 cases each for exact boundaries,
next-float-outside, next-float-inside, 1e-12 outside and 0.001 outside. All 1,440
outside controls reject; all 480 boundaries and 480 inside controls accept. The
JS observer imports the real helper but never computes its own expected edges.

`before-rational.zip` / `after-rational.zip` retain unchanged scripts, exact inputs
and complete outputs; `before-failures.json` retains all original failures.
`near-boundaries.zip` retains all supplemental inputs/results. `author-replay.zip`
retains frozen replay scripts, fresh after screenshots/observations, native resize
and image results, and the original baseline images used for image comparison.
These are **author replays**, not new independent review results.

## Full validation and environment

`python3 docs/verification/pr05-f01/verify.py` runs the actual locked scripts.
[checks.json](logs/checks.json) records ten command exits, all zero, against
[validated-source.json](logs/validated-source.json)'s exact c585a09 executable
HEAD/tree and per-file hashes:

- Node 24.19.0 / npm 11.9.0; `npm ci`, 265 locked packages.
- `npm run typecheck`, `npm run lint`, `npm test`: **187/187** tests (all prior
  162 plus 25 added cases); `npm run build` succeeds.
- `npm run test:e2e`: **16/16**; `npm run test:e2e:controls`: **7/7** (two intact
  successes and five specific fault detections).
- `git diff --check` passes; staged/full-base checks are repeated at publication.

Local browser commands use explicit system Chromium 154.0.8037.57 / Playwright
1.64.1, with the writable npm cache. The known matched-browser CDN HTTP 403 was
not retried or bypassed in this correction. Final hosted CI uses its standard
matched browser independently. No local physical camera, external model/GPU,
audible speech, mobile orientation/background or public-deployment equality is
claimed. Synthetic source, actual DOM/video/Canvas and intercepted speech are
distinguished from real model/audio hardware coverage.

Reproduce the unchanged rational oracle by extracting `before-rational.zip` or
the frozen review's generator/oracle into `docs/verification/pr05-f01/`, running
the generator then `node --import tsx .../numeric-oracle.mjs`; its same relative
import resolves the real author helper. To replay the supplemental oracle:

```
python3 docs/verification/pr05-f01/generate-near-boundaries.py /tmp/pr05-f01-near.json
node --import tsx docs/verification/pr05-f01/near-boundaries.mjs /tmp/pr05-f01-near.json /tmp/pr05-f01-near-results.json
```

For the frozen browser/native/image/component scripts, follow PR #12's original
README with the corrected E2E build on port 4185. `standard-browser-evidence.zip`
retains the ordinary full Playwright report. Command logs, exact raw-byte ZIP and
whitespace-normalization receipts are preserved; `artifacts.json` hashes stored
evidence. Readable TXT logs only trim captured trailing whitespace/terminal blank
lines. Original output hashes in checks.json refer to archived raw bytes.

## Handoff and preserved limits

Re-review F01 against the final public SHA/tree and its own successful CI
checkout, then verify the rational and immediately outside controls. This author
checkpoint does not declare independent closure. Main, PR #12 and all historical
evidence heads are preserved. The only branch advanced is PR #11's author branch.

Original PR-04 conditions remain: identical-content coalescing does not prove
physical camera stop; full RGBA comparison is O(pixels) on the main thread,
outside the existing processing-duration HUD; the future asynchronous lock-wakeup
risk remains nonblocking. This numerical correction does not optimize away image
identity, change mask fallback semantics, or expand into a later roadmap phase.
