# PR05 independent F01 re-review

**Verdict: PASS WITH CONDITIONS; F01 CLOSED. No new product findings.**
The unchanged original rational oracle reproduces 36 failures at the old head
and passes all 1,616 checks at the correction. New reviewer controls pass all
25,020 representable-boundary cases plus 36 invalid/overflow/collapse cases.
The correction does not introduce an epsilon band or accept the adjacent
representable outside points in those controls. The remaining conditions concern
the stated geometry/numeric domain, observed test timing sensitivity and the
previously disclosed PR04/device limitations, not an open coordinate defect.

This is independent re-review, not an author replay. Work ran only in the selected
cloud environment. No product, existing tests, workflow, PR12, historical evidence,
main, deployment or later phase was edited. A new evidence branch is directly
stacked on PR11's frozen correction. Its evidence HEAD/tree and Draft PR URL are
recorded in its PR description and final handoff, without a self-reference here.

## Frozen bindings

| Item | Verified binding |
| --- | --- |
| Product | [Draft PR11](https://github.com/TakashiSasaki/vision-edge/pull/11), `fix/overlay-coordinate-mapping` |
| Reviewed correction HEAD | `c9d68eeb5923e14eb7043f64fda3e1edf10d629f` |
| Reviewed correction tree | `f8e78b24e0bf27393ca9266c4d64bf039d232516` |
| Previous reviewed product HEAD/tree | `be1402c793616788e81a746b6a21b03a25a5a3ec` / `a6c4f4eaa0d3db02a85861653d27bfcfdb53005d` |
| Preserved [PR12](https://github.com/TakashiSasaki/vision-edge/pull/12) evidence HEAD/tree | `57ee68bc503cfc1633039d948d1b39fcf341caa5` / `a73fd0734d0358de0bbf1a70653f780a3637c659` |
| Actual main HEAD/tree | `f97e19ac5f9f07eca23732003844b5082cd27d37` / `e357f512a371df5c86950913cf3a43208f0321ce` |
| New evidence branch | `review/pr05-coordinate-rereview` |

Re-fetched all branches, PR11's head ref, actual main and review history before
work. The reported correction matched. PR11 still had no submitted reviews, and
PR12 is the preserved earlier review. Re-fetched the named branches after tests:
all bindings above remain unchanged. No AGENTS.md or local skill file appeared
in the repository or mounted instruction directory. The previously read review
cycle skill continues to apply; no new delegation was needed.

[source-binding.json](source-binding.json) binds all 59 non-document product files
to the correction's Git objects. [probe-bindings.json](probe-bindings.json) proves
seven original reviewer probe/configuration files match PR12 byte-for-byte, both
in their archived snapshots and in the isolated execution directory. Their
original assertions were not edited. [Remote heads](remote-heads.txt) and
[API/CI receipts](api-receipts.json) preserve the checked public state.

## F01 closure and complete correction review

F01 was P3/nonblocking at the previous target. Its minimal reproducer remains:

```js
clientPointToVideo(3779 / 24, 82.5,
  { left: 37.25, top: 82.5, width: 960.75, height: 540.25 }, 640, 480)
```

Expected inclusive boundary: `{x:0,y:0}`. Previous helper: `null`; corrected
helper: `{x:0,y:0}`. The original exact-Fraction generator and numeric probe ran
unchanged against a detached old-head worktree and then the correction: old
exit 1, 1,580 pass / 36 fail; new exit 0, 1,616 pass / zero fail. Complete original
inputs and before/after outputs are in `raw-evidence.zip`; the 36 original
failures remain at [before/numeric-failures.json](before/numeric-failures.json).

Read the new author README, response, generators/observer and verification
script, then every source/test change. Only these existing executable files
change relative to the previous reviewed product:

- `videoCoordinates.ts`: contained edges use combined quotients; compare client
  positions before normalization; reject nonfinite/collapsed geometry; preserve
  exact endpoint 1 at right/bottom.
- `videoCoordinates.test.ts`: 24 added parameterized cases cover rational edges,
  transpose, DPR and adjacent floats.
- `VisionCanvas.test.tsx`: one added component case checks rational edge hits,
  silent neighboring margins, ripple coordinates and resource counts.

No old assertion was removed or weakened. The click handler, CSS, drawing,
engine/masks, speech service, camera/model lifecycle, scheduler/input identity,
ownership, JobLock, generations, exception recovery, metrics, SDK alias,
dependencies and CI workflow are identical to the prior reviewed product.

The helper now compares against explicit left/right/top/bottom values and only
then normalizes. It contains no epsilon constant or tolerance branch. The exact
right/bottom equality branch maps an already accepted endpoint to 1; it does not
clamp outside points. Code inspection and the adjacent outside controls agree.
No additional product correction is requested for F01 at this checkpoint.

## Representable contract and independent controls

The contract tested here is **inclusive once-rounded binary64 reference edges**
from finite positive input geometry, within the stated centered contain CSS
layout. It is not a promise that all mathematical real positions or all finite
JavaScript magnitudes are representable. A rounded reference edge can lie on
either side of its ideal rational position. The next float outward from that
rounded edge must reject; a displacement that rounds back onto the same edge
is not an outside representable point.

`generate-controls.py` is new reviewer code. It uses Python Fraction, with no
production or author generator imports, to derive exact contained rectangles,
then rounds each reference edge once. It computes expected normalized values
from the actual representable point and reference endpoints. This avoids
mistaking an ideal .5 for a representable midpoint in a tiny box at a large
viewport offset. `math.nextafter` generates independent neighboring floats.

| New control | Count | Outcome |
| --- | ---: | --- |
| Exact corners | 3,456 | PASS |
| Exact edge midpoints | 3,456 | PASS |
| Centers | 864 | PASS |
| Adjacent outside edge floats | 3,456 | Reject, PASS |
| Adjacent inside edge floats | 3,456 | Accept, PASS |
| Outside displacement 1e-9 CSS pixels | 3,420 | Reject, PASS |
| Outside displacement 0.001 CSS pixels | 3,456 | Reject, PASS |
| Adjacent outward corner pairs | 3,456 | Reject, PASS |
| Invalid dimension/coordinate and overflow/collapse | 36 | Return null without exception, PASS |

The 25,020 boundary vectors cover nine native aspect/size pairs up to 8192×4320,
six CSS boxes including fractional 1/64-pixel layout and tiny boxes, four page
origins including negative scrolled positions and offsets near ±2^24, and DPR
1/1.25/2/3. Thirty-six 1e-9 perturbations rounded back to the edge and were
explicitly omitted as non-outside inputs; adjacent-float coverage still applies.
There is no arbitrary widened hitbox in these cases. The additional 2,400 author
controls were separately replayed and pass, but do not replace these new tests.

Overflow/collapse checks deliberately include huge finite intermediates, lost
extents and subnormal underflow. The helper returns null as its documented
fail-closed behavior. This does not establish arbitrary-precision support for
every finite number; it finds no unexpected overflow or cancellation within
the practical CSS/native-size vectors above. [Independent results](independent-results.json)
and raw vectors are preserved, including exact representable reference edges.

## React, browser, rendering and regression matrix

| Check | Independent outcome |
| --- | --- |
| Original reviewer component probes, unchanged | 3/3 PASS: BBox, pixel mask, polygon, intrinsic change before drawing, silent invalid input |
| New edge component probes | 2/2 PASS: horizontal and transposed vertical fractional geometry, all four exact corner hits, exact ripple endpoints and all four nearest outside margins silent |
| Original real-app browser sweep, unchanged, serial | 24/24 PASS: both letterbox directions, equal/square aspects, fractional layout, DPRs, scrolling, immediate CSS resize, invalid input, selected dog, ripple and no margin speech |
| Original native intrinsic-resize probe, unchanged, serial | PASS: video 1280×720 during click while Canvas backing is still 640×480; after drawing backing/ripple match the new size |
| Original screenshot pixel oracle, unchanged, serial | 26/26 PASS: preserved original baseline images plus newly rendered correction images |
| Full original package unit suite | 187/187 PASS, all prior 162 plus 25 author correction tests |
| Full original browser suite, serial | 16/16 PASS, including the original PR04 frame identity, initial-frame and compatibility cases |
| Original smoke controls | 7/7 PASS: two intact successes and five specific fault detections |
| Install/typecheck/lint/build/whitespace | PASS, with environment details below |

No probe expected coordinate comes from the product helper except the observer
calling it to obtain the actual value. Browser video/Canvas rectangles, source
pixels, object-fit/position, transforms and intrinsic/backing sizes are measured
from the real React app. Source video/SDK are synthetic and speech is intercepted.
Fixed pixel colors, known source box and rendered cyan/white strokes bind pointer
selection to the actual displayed content. BBox and both existing mask interfaces
are preserved. No mirror or DPR double transform was introduced.

The new component tests independently use a full-image known BBox and fixed
Fraction endpoints. Every exact corner selects/speaks once and draws at the
correct normalized backing endpoint. Adjacent outside margins never reach hit
testing or trigger additional selection/speech/ripple. Camera acquisition and
task creation counts remain one. No existing test or historical artifact changed.

The 59-file binding, source diff, full regression and original component/native
probes support no new PR04 regression: immutable input ownership, initial-frame
bootstrap, JobLock/generation invalidation, exceptions/recovery, lifecycle and
truthful metric aging remain unchanged. The helper is not called in the
frame-identity test that initially encountered the timing observation below.

## T01 — P3/nonblocking test timing observation, not a product finding

Initial browser suites overlapped. The standard suite exited 1: 15/16 passed;
the fallback frame-identity case had already consumed exactly one newer immutable
image, but its final duration-display assertion saw `フレーム処理: 未計測` rather
than a retained millisecond duration. The metric intentionally ages after two
seconds. A fixed timing expectation under scheduling delay is consistent with
the failure; this observation alone does not prove concurrency as its cause.
The trace, screenshot and report are in [standard-concurrent.zip](standard-concurrent.zip).

The initial original reviewer browser sweep exited 1: 23/24 passed. Equal-aspect
DPR 2 selected and spoke correctly, but the observer had no arc after its fixed
80ms wait. [replay-concurrent.zip](replay-concurrent.zip) preserves that snapshot
and screenshots. These are real failed observations, not reported as passes.

Re-ran both complete suites **serially with the same assertions/configuration**:
standard 16/16, original browser 24/24, then original native-resize and image
checks passed. No timeouts, waits, retries or product behavior were changed.
The initial raw logs remain alongside serial results. T01 limits repeatability
of fixed-time observations under this cloud scheduling; it is not attributed
to the coordinate correction and does not reopen F01. No new product defect
was reproduced. This review does not claim load/performance guarantees.

## Commands, exact product CI and evidence

[commands.json](commands.json), [initial standard ledger](logs/standard.json) and
[serial ledger](logs/serial.json) distinguish every exit/count. `npm ci --cache
/workspace/npm-cache`, typecheck, lint, 187 unit tests, build, seven controls and
whitespace checks exit 0. The initial full browser exit 1 is retained; its full
serial rerun exits 0. Additional evidence-inclusive typecheck and lint pass.

Local versions: Node 24.19.0, npm 11.9.0, Playwright 1.64.1 and explicit system
Chromium 154.0.8037.57. The known matched-browser HTTP 403 was **not retried or
bypassed**. Local matched-browser coverage remains unexecuted. Standard local
browser commands set `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium`.

Independently fetched [product CI run 40094223325 / job 120873001838](https://github.com/TakashiSasaki/vision-edge/actions/runs/40094223325/job/120873001838):
completed/success. Downloaded logs explicitly check out `c9d68eeb...`, confirm
187 unit / 16 browser / 7 controls and all successful workflow steps. Hosted
versions are Node 24.19.0, npm 11.17.0 and matched Chromium 156.0.8078.4/v1248;
no system-browser override is used. Artifact 12560595409 independently binds
that same HEAD, digest
`13bcbf242e56aff971efa664b814f9fbe79d92d6ca181ed5791d5afcecb0c623`.
The independently downloaded original decoded log SHA-256 is
`5debce1ffc9415953d0cb9e7436f5651979d44284344b5006571585136190103`, matching
the author receipt. [Readable log](logs/product-ci.txt) and original bytes in the
raw ZIP are retained. The earlier PR12 evidence CI
[40093113503](https://github.com/TakashiSasaki/vision-edge/actions/runs/40093113503)
is now completed/success. Neither that run nor this new evidence branch's CI
substitutes for exact correction-head CI.

Default production bundles contain none of `visionSdkProbe`, `visionSdkGate` or
`Test-only SDK fault`; explicit e2e bundles contain all three. The existing mode
alias and normal production workflow are unchanged, and reviewer test code has
no production entrypoint import. Test doubles remain test-only.

`raw-evidence.zip` preserves raw command bytes, original numeric before/after
inputs/results, independent vectors, CI receipt and full browser observations.
`frozen-replay-serial.zip` preserves fresh correction screenshots/results and
the original baseline images used by the unchanged image oracle. The two
ordinary full-browser report ZIPs retain both failed and successful attempts.
Readable text trailing whitespace/terminal blank lines/ANSI colors are normalized
only after raw archiving; inventory/hashes document the stored bytes.

## Replay and remaining conditions

Use `setup-frozen-replay.py NEW_DIRECTORY` to extract PR12's original probes and
link this product's `src` and locked dependencies. It refuses an existing target.
Run the frozen generator and numeric probe there; assertions remain unchanged.
For before reproduction, create a detached worktree at `be1402c...` and point
only the isolated replay directory's `src` link at it, then restore the link to
the corrected source. The reviewed product worktree itself is never modified.

Run `generate-controls.py`, then `node --import tsx
docs/verification/pr05-rereview/boundary-controls.mjs`; use the separate
`vitest.config.mjs` for new component edge tests. Build/serve the normal explicit
e2e artifact on port 4185 for frozen browser/native/image probes. `run-standard.py`
captures unchanged package scripts; `run-serial.py` runs the browser checks
sequentially using `/workspace/pr05-rereview-probes`. Preserve distinct logs for
each execution instead of overwriting a historical result.

Remaining conditions are precise and nonblocking:

- Centered contain, no-transform/no-padding CSS and the binary64 representable
  boundary domain above; no arbitrary finite-number or exact-real arithmetic proof.
- T01 cloud timing sensitivity is disclosed; serial and exact-head hosted checks
  pass without test changes. No physical latency/performance conclusion is drawn.
- PR04 identical-content coalescing still does not prove physical camera stop;
  full RGBA compatibility comparison is O(pixels), on the main thread and outside
  processing-duration HUD. The prior future async lock-wakeup concern is unchanged.
- Physical camera, real SDK/model/GPU, audible audio, mobile orientation/background
  and public-deployment equality remain unexecuted. Synthetic evidence is not
  real-device evidence; local matched Chromium remains unavailable after the prior
  HTTP 403, with successful hosted matched-browser CI separately verified.

No blocking author fix remains at this checkpoint. A subsequent product change
requires a new frozen target and appropriate re-review. The parent decides the
next operation; this checkpoint does not authorize a merge or deployment.
