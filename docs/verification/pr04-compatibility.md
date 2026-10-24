# PR-04 residual F01 correction after independent PR #9

PR04-F01 is accepted and repaired in ordinary commit
[`316f149d68e08e50705e00869455835a28ea483d`](https://github.com/TakashiSasaki/vision-edge/commit/316f149d68e08e50705e00869455835a28ea483d)
(tree `e0a49c4a0dea6b740c2c9f6a3f26be1a5f3cf2d3`). **Independent F01 closure
is pending the same separate reviewer.** PR04-F02 was independently closed by
[PR #9](https://github.com/TakashiSasaki/vision-edge/pull/9); its acceptance tests
remain enabled. This report is Author implementation/replay evidence.

| Binding | Frozen value |
| --- | --- |
| Reviewed PR #7 HEAD | `f81ce6102ec4f459ef7a0c3fb7e55d126b157be5` |
| Reviewed tree | `d15536393c00c55c16c8654612d13d8e7e5692a7` |
| PR #9 evidence HEAD | `232b2104a2564d96a01e354bf7aeee7a380733f2` |
| PR #9 evidence tree | `b5d86b085fb679ed618ca73659b97521ac2acf06` |
| Actual base | main `f1a0d0bbe343b666c75eae053340ceefc6730d5d` |
| Implementation branch | `fix/inference-scheduler-metrics`, [PR #7](https://github.com/TakashiSasaki/vision-edge/pull/7) |

The final evidence-bearing HEAD/tree and completed exact-HEAD hosted Run/Job/log
receipt are recorded in PR #7's description, avoiding a self-referential source
commit. PR #7 was returned to Draft for this correction. Ready status requires
successful final-HEAD CI; it does not mean independent closure or merge approval.
PR #8/#9 and all earlier verification records are preserved.

## Reproduction on the reviewed source

Read PR #9's actual REPORT, findings, reproduction instructions, boundary/app/native
vectors and raw observations. Fetched its exact evidence commit and extracted a
read-only copy. A detached checkout `/workspace/vision-edge-pr04-r2` at f81ce610
received those vectors, using the current locked installation by symlink. A
second independent npm ci in that replay checkout is not claimed.

Commands/exits are bound in `pr04-compatibility/logs/before-checks.json`:

- The 11 capture/ownership vectors passed, confirming they do not expose this
  residual capability-absence defect.
- The native six-case capability matrix and forced-delay matrix each exited 1.
- Actual App/Canvas/Engine/SDK-entry tests: immutable and quality controls each
  inferred once. With both capabilities absent, the static source incurred
  **136 SDK calls in 2.3 seconds**, HUD **58.6 FPS**, status `計測中`. This local
  reproduction differs numerically from the reviewer's 140 calls/60.6 FPS but
  demonstrates the same repeated identical image and failure to age input.
- New ordinary browser regressions, applied as tests to frozen product source:
  **4 pass / 2 fail**, both failures the RAF missing/zero-quality variants.
  First setup attempt had no normal production dist and hit the existing
  webServer timeout. Running the normal build fixed setup; the second run is
  the product regression receipt. No timeout or assertion was weakened.

Raw receipts are before-00..04.txt, before-new-browser.txt (setup failure),
before-new-build.txt and before-new-browser-ready.txt (actual assertions).

## Cause and narrow repair

`currentTime` is playback position, not evidence of a newly presented image.
When VideoFrame and usable playback-quality totals were absent, moving playback
position was treated as both a fresh inference candidate and fresh input. A
still MediaStream therefore repeatedly inferred its same pixels and continually
renewed the input-age clock. The rate collector accurately counted those calls,
but the calls were the wrong unit of useful work.

The immutable VideoFrame path and usable positive quality-counter path retain
their existing behavior. Without either capability, production now uses a
private full-resolution 2D sampler and exact RGBA byte/dimension comparison.
There is no hash collision, sparse sample, timestamp proxy or invented FPS cap.
Equal sampled content does not reserve another inference or refresh input age.
The existing clock-driven collector therefore expires its samples and reports
`映像入力停滞`; a single success remains `推論: 未計測` instead of fabricated 0/FPS.

Immediately before execution, after acquiring the owning job token, it samples
again. The exact sampled bytes are copied into a private execution canvas;
that same canvas travels through production VisionCanvas and VisionEngine to
the selected SDK task. Identity and inference therefore refer to the same
image even if the live video changes between reservation and execution. The
attempt consumes its image before work, so an exception does not retry unchanged
pixels indefinitely. Later changed pixels can recover through the existing
inference-error repair. This adds HTMLCanvasElement to the existing inference
input union; the installed SDK accepts TexImageSource, including canvas.

One sampler/context is reused (including browser readback mode). There is at
most one pending opportunity, one active owned image, and bounded pixel arrays
for consumed/observed/current samples (at most three distinct arrays across a
sampling transition), with no retained frame queue/history. Disposal zeros
sampler/owned canvas dimensions and releases references; an active asynchronous
test job retains only its own image until its own completion. Old cleanup/finally
cannot release a newer job's lock. Capture/2D/readback errors deliberately stop
acquisition and expose the existing bounded explicit retry flow.

The compatibility estimator's scope is **image content changes**, not exact
physical frame arrivals: distinct frames with identical RGBA content coalesce.
There is no exact source timestamp available in this path, and none is invented.
Full-image readback/comparison costs O(width × height) CPU work and bounded
O(width × height) storage. It runs on the main thread and can be expensive at
large camera resolutions. No Worker, dependency, configured inference cap or
PR-05 work is introduced. Existing VideoFrame/quality paths avoid this pixel
fallback. Capture cost is outside the existing metric timed immediately around
the complete detectVideoFrame call; that metric still includes SDK and engine
post-processing and is not labeled total pipeline/SDK-only/GPU duration.

F02 bootstrap uses the same capture/deduplication path. A ready first image is
processed without another notification; a late notification of that image
coalesces. New sessions reset identity, accepting the first same-content/time
image again. Camera/model readiness, generations, ownership, drawing/TTS faults,
HUD rate bounds and smoothing-error repair retain their prior contracts.

## Added acceptance coverage and replay disclosures

Seven added unit/engine cases bring the suite from 109 to **116**. They cover
moving clock with fixed pixels, genuine changed pixels, delayed capture, failure
and recovery without same-image retry, every-byte comparison (including swapped
pixels with equal histograms), dimensions, capture failure after lock acquisition,
owned canvas release, and both actual selected SDK tasks with canvas dimensions.
Existing abstract Canvas fixtures now explicitly provide their logical mocked
playback-quality capability; all prior assertions remain. They are not presented
as native capability-absence tests.

Six new ordinary-CI browser cases use native canvas.captureStream/video progression
and the actual production App/Canvas/Engine/scheduler. Immutable and quality paths
are controls; missing and zero quality are tested with both rVFC and RAF. The
four pixel cases require one red SDK input after 2.3 seconds, insufficient-sample
HUD and stalled input; a genuinely new blue image must then reach the SDK with
no extra camera acquisition/model initialization. Stop closes tracks. Existing
native first-frame/restart/delayed-execution tests remain unchanged in substance.
Test injection is confined to the explicit test SDK build, never normal output.

PR #9 replay after repair: **11 unit boundaries, six native capability cases,
three actual-app cases, six forced-delay cases all pass** (after-checks.json and
after-00..03.txt). Its observer originally accepts VideoFrame/live video only;
canvas observation requires `new VideoFrame(canvas, {timestamp: 0})`. The local
replay adapts only that constructor's options for canvas. Assertions are unchanged;
original/adapted SHA-256 receipts are in observer-adaptation.json, and the replay
script is saved. Observer timestamp 0 is an API-required synthetic observation
value, never product identity, SDK timing or claimed source PTS. Evidence PR #9
itself is untouched. These are Author replays; independent acceptance remains
with the reviewer.

The original PR #8 abstract vectors also lack readable canvas pixel fixtures.
Their raw replay records **8 pass / 21 fail** (original-unit-raw.txt), chiefly
because that fixture cannot perform the newly required pixel readback. A separate
explicit fixture adapter exposes the old logical frame markers as a mocked
quality capability and delegates to the real production scheduler; it yields
**29/29** (original-unit-adapted.txt). Original vector files/assertions are
byte-for-byte preserved. This adapted result exercises prior logical lifecycle
contracts, **not** the absence path. It does not replace the unadapted native
PR #9/new-browser capability-absence tests above, and no raw failure is concealed.
The adapter and config are saved under probes/ for reproducibility.

## Full validation and environment

`pr04-compatibility/logs/checks.json` binds all commands to exits, all **0**:

| Check | Result |
| --- | --- |
| node --version / npm --version | 24.19.0 / 11.9.0 |
| npm ci | locked clean installation, 265 packages |
| npm run typecheck / npm run lint | pass |
| npm test | 116/116, 9 files |
| npm run build | pass |
| npm run test:e2e | 14/14 |
| npm run test:e2e:controls | all 7 expected outcomes |
| git diff --check / git diff origin/main --check | pass |
| Original main source regression replay | 61/61 |

The controls retain their two intact successes and five diagnostic-specific
expected child failures; the overall control runner exits 0. No check, assertion,
lint rule, CI job or supported fallback is disabled. Product/test source hashes
are saved in verified-source.json. Evidence-log whitespace normalization, if any,
is separately recorded with original bytes/hashes, without changing outcomes.

Local browser: explicit system Chromium 154.0.8037.57, Playwright 1.64.1. The
matched local browser download was blocked earlier with HTTP 403. Exact final-HEAD
hosted matched-browser CI is a separate required receipt in the PR description.
Native MediaStreams contain synthetic images and SDK results are explicit doubles;
this is not physical-camera, real-model/GPU or audible-TTS certification. GPU
requested remains distinct from utilization unmeasured. Prior nonblocking R01
future asynchronous lock-wakeup limitation remains disclosed.

## Factual execution record and re-review handoff

Available public/test records establish the correction's active work, without
asserting an unavailable exact message-delivery timestamp:

- 2026-10-24 08:31:44 UTC (17:31:44 JST): GitHub Draft conversion response for
  this correction. PR #9 evidence acquisition/reproduction followed.
- 08:43:06 UTC: recorded start of the pre-fix 11-vector test replay.
- 09:10:46 UTC: corrective implementation commit 316f149; remote push receipt
  follows at 09:10:48 UTC.
- 09:17:36 UTC: clean-install full unit run started; browser/controls/baseline
  completed by 09:19 UTC. Exact commands/results are retained above.

Work continued through reproduction, implementation and validation; it was not
waiting for another review report. Later progress requests elicited status/PR
body updates rather than restarting or discarding work. One transient execution-
environment availability failure resolved on the next available execution; no
access denial or approval bypass occurred. The missing-dist setup timeout is
identified above. No private session files were examined and no cause is inferred
for differences between visible UI messages and parent API observations.

Same-reviewer focus: absent/zero capabilities with still input and moving playhead;
actual sampled image bound to SDK execution; pixel changes/dimensions versus equal
content; first image and bootstrap/late notification; stop/restart/generation and
owned image/token release; full metric aging; test-only observer adaptation and
its synthetic timestamp; pixel readback CPU cost. Confirm F01 closure independently
while retaining F02 closure. No merge, auto-merge, deployment, branch protection,
rebase/force-push or unrelated branch changes are authorized or performed.
