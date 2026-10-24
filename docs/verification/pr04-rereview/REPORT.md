# Independent re-review of PR-04 / PR #7

**Verdict: FAIL. PR04-F02 is closed at this checkpoint; PR04-F01 remains open
with one medium-severity blocking compatibility counterexample.** No new
critical/high finding is claimed. The correction works for the tested immutable
VideoFrame paths, but does not satisfy the original no-repeated-frame contract
through every compatibility path it still executes.

| Binding | Exact value |
| --- | --- |
| Reviewed implementation HEAD | `f81ce6102ec4f459ef7a0c3fb7e55d126b157be5` |
| Actual GitHub and local source tree | `d15536393c00c55c16c8654612d13d8e7e5692a7` |
| Corrective parent commit | `89e444820b6befbd333236e65697b5bcf8f5e15a` |
| Implementation base main | `f1a0d0bbe343b666c75eae053340ceefc6730d5d` |
| Previous reviewed implementation | `2f4cdea1bc3e6706f49e37116ede8b1881472f7f` |
| Preserved original [review PR #8](https://github.com/TakashiSasaki/vision-edge/pull/8) | `c632cf0cbce82b48503540651120c88fff75c8ef`, tree `b382360d930ca20a71a5f2e7ae99b764f2250564` |
| New evidence branch / direct PR base | `review/pr04-independent-rereview` / `fix/inference-scheduler-metrics` at the reviewed checkpoint |

The live target matched the frozen SHA at startup. GitHub's actual commit API,
local checkout and hosted checkout log agree. Latest PRs were inspected: #8 is
the original FAIL, not an equivalent corrected-checkpoint review. Final source
movement checks and the new evidence HEAD/tree are recorded in the re-review
PR description. This review does not alter PR #8, its branch or its evidence.

This is the same separate reviewer execution continuing the approved correction/
re-review cycle, not the implementation author accepting their own work. A new
clean checkout was used at `/workspace/pr04-rereview`. No AGENTS.md or repository
.agents instructions were present; workspace instruction directories were empty.
Only new files under this evidence directory are proposed. No product fix,
existing test/workflow/evidence edit, merge, force-push, release or deployment
was performed. Publication uses the owner's connected GitHub account.

**Per-finding closure.**

| Finding | Decision | Evidence |
| --- | --- | --- |
| PR04-F01 | **Partially addressed; open, medium/P2 blocker** | Original reproductions pass and immutable delayed actual-input races pass; currentTime-only compatibility still consumes one unchanged source image repeatedly. |
| PR04-F02 | **Closed at this checkpoint** | Original native one-frame test now gives rVFC 1 / RAF 1; real app model-gated first frame, equal-time restart and bootstrap races pass. |

[findings.json](findings.json) provides stable IDs, locations, precise triggers,
reproduction steps, observed/expected results and correction acceptance.
[The matrix](requirement-test-matrix.md) retains every original acceptance
contract in scope.

**Remaining PR04-F01: playhead progress still substitutes for source progress.**

Location: [frameScheduler.ts:73](https://github.com/TakashiSasaki/vision-edge/blob/f81ce6102ec4f459ef7a0c3fb7e55d126b157be5/src/services/frameScheduler.ts#L73),
with the live compatibility return at line 61 and input-age observation at
lines 82–84. Contracts 1 and 7.

When `VideoFrame` is unavailable and `getVideoPlaybackQuality` is unavailable or
its count is unusable, the code compares successive `video.currentTime` values.
A native MediaStream's playhead can keep moving while its source image remains
unchanged. The fallback then executes successful inference on that same image
on successive RAF ticks and refreshes input age each time.

The independent actual-app reproduction uses a native
`canvas.captureStream(0)`: draw one red frame, request one frame and wait for
native presentation `mediaTime 0 / presentedFrames 1`. It then releases model
initialization. The explicit capability-absence case removes VideoFrame, rVFC
and playback-quality access from the product-visible environment. A saved
native constructor is used **only at the SDK observation boundary** to inspect
the actual argument. No scheduler, engine or camera hook is substituted.

Observed after **2300 ms**:

- **140 successful SDK calls**, rather than one.
- Bounded retained input observations all have intrinsic image timestamp **0**
  and the same red pixels; the source produces no second image.
- HUD displays **60.6 inference FPS** and **計測中**, despite no new source input
  for longer than the 2000 ms stalled-input threshold.
- Immutable and live/playback-quality controls each call the SDK **once**.

An earlier 650 ms run observed 41 calls and 61.1 FPS. The independent native
still-source matrix also repeats the image in RAF/time-only mode. Separately,
forced delayed native-video tests show byte-identical repeated SDK-boundary
input at the same intrinsic timestamp in **both rVFC and RAF** when VideoFrame
and quality information are absent. Those tests deliberately reach delayed work,
rather than letting bootstrap finish before the contention window.

See [actual app observations](logs/app-observations.json),
[final app log](logs/app-compatibility-aging.txt),
[native still-source matrix](logs/native-boundaries.json) and
[forced delayed races](logs/forced-native-race-final.json).

This is not a claim that collector arithmetic is wrong: the SDK really executes
the repeated successful calls, and its event-span FPS reflects them. The defect
is recognizing an unchanged image as new input and preventing stalled-input
aging. The author's disclosed possibility of coalescing coarse/incomplete
progress does not describe this unbounded repeated-work behavior. There is no
declared unsupported/stopped state for this implemented capability combination.
Tests emulate API absence in Chromium; they do not claim certification of a
particular older browser or physical device.

Correction acceptance: preserve the working immutable/F02 paths and repair the
explicit compatibility behavior, including delayed callbacks and no-new-input
aging. If reliable identity cannot be supported, expose a deliberate unsupported/
stopped state; an owner-approved narrowing of compatibility would be needed
instead of silently treating the original contract as waived. Return new
regressions, frozen HEAD/tree and exact-HEAD hosted validation for re-review.
No implementation repair is included here.

**What the correction successfully changes.**

The production scheduler now captures a VideoFrame after job-token acquisition,
uses that exact object's image timestamp for consumption checks and passes the
same immutable object to VisionEngine/SDK. It closes temporary arrival handles
immediately and execution handles only in their owning finish. The type/engine
change admits VideoFrame and uses its display dimensions; the installed SDK's
size helper also handles displayWidth/displayHeight. That static compatibility
check does not certify real model/GPU execution.

Bootstrap shares the pending slot, capture/dedup and generation checks, so an
already-presented ready first frame is acquired once. Subsequent source progress
is accepted. New model/camera sessions reset identity, including zero-time images.
The original threshold reentry/generation assertions still pass. No camera/model
creation dependency was added for statistics or appearance.

The 11 new independent unit cases verify bootstrap/arrival coalescing, actual
captured identity despite divergent metadata, subsequent immutable images,
failed-image no-retry/recovery, 100-arrival bounded pending work, queued
cancellation, capture-time invalidation, stale async completion, foreign-token
release, exactly-once closes and post-capture-failure retry. All pass.
Original independent drawing/parent/TTS exceptions, model/camera/threshold/
unmount guards, StrictMode, smoothing repair and complete-call timing also pass.
PR04-R01 remains the previously disclosed nonblocking future async wakeup risk;
production inference is synchronous and no Worker migration is requested.

**Source inspection and independent execution.**

The correction delta has **86 paths: 77 new evidence files, four product source
files and five test/fixture files**. All were read/classified, separating earlier
failed development receipts from final receipts. The full PR now has **160
changed paths**, including unchanged paths reviewed in PR #8. See the
[correction inventory](correction-inventory.json) and
[full inventory](full-pr-inventory.json). All **55 non-evidence tracked files**
were independently hashed and match the corrected frozen source receipt.
Dependency/lockfile, CI, metrics implementation, tracking/coordinate/dictionary/
speech-service code and historical implementation evidence remain unchanged.
No test skip, weakened substantive assertion, broad lint suppression, new
binary artifact or unrelated product edit was found.

| Execution | Actual outcome |
| --- | --- |
| Clean npm ci; typecheck; lint; current unit/component suite; production build | All exit 0; **109/109**, nine files. |
| Original main test suite | Exit 0; **61/61**, seven files; clean main checkout with identical locked dependencies. |
| Complete ordinary browser suite | Exit 0; **8/8**, explicit system Chromium. Includes original cases and actual-input correction cases. |
| Fault controls | Exit 0; **7/7**, two healthy passes and five diagnostic-specific child exit-1 failures. |
| Full main-to-HEAD and correction whitespace checks | Exit 0. |
| Original independent vectors, exact bytes from PR #8 | Exit 0; **29/29**. Separate replay checkout uses the complete actual corrected commit, not selected copied product files. |
| Original native first-frame probe | Exit 0; **rVFC 1 / RAF 1**. |
| Original six native race trials | Exit 0; no duplicate observations. Bootstrap can precede contention, so this alone is insufficient proof. |
| New ownership/capture unit boundaries | Exit 0; **11/11**. |
| New native still-source capability/path matrix | Exit 1; **5 pass / 1 fail** for remaining F01. |
| New forced delayed actual-input matrix | Exit 1; **4 pass / 2 fail** for remaining F01. |
| New independent actual-app capability matrix | Exit 1; **2 pass / 1 fail** for remaining F01. |
| Evidence-inclusive typecheck/lint | Exit 0; no source rule changes. |

Environment: Linux x86_64, Node 24.19.0, npm 11.9.0, Playwright 1.64.1 and
system Chromium 154.0.8037.57. Required command logs, explicit exit codes and
additional probe outcomes are in [commands.json](commands.json).
Original vectors are retained byte-for-byte under original-vectors and compared
to the original evidence Git blobs in [bindings](logs/original-vector-bindings.json).

The unchanged independent metric oracle still independently calculates irregular
event spans, low-rate input, 300-entry/5-second boundaries, 2-second expiration,
long idle/resume, invalid/reversed clocks and unknown versus zero. Detector
SDK+post-processing is independently 16 ms; segmenter SDK+cleanup is 18 ms.
Capture acquisition is outside the explicitly named detectVideoFrame-call
duration and is not advertised as total pipeline latency. Missing optional
SDK-only timing is not a finding.

Exploratory limits are retained honestly: the first independent app harness had
an incorrect web-server cwd and did not reach product tests; that setup was
corrected without product changes. An optional native generated-track timestamp
collision experiment failed at initial-source setup before its assertions; it is
**inconclusive**, not a product defect or passing boundary. Source timestamp
collision handling beyond the tested native inputs is not claimed certified.

**Hosted source binding and test isolation.**

Independently fetched [run 40076315419 / job 120820304359](https://github.com/TakashiSasaki/vision-edge/actions/runs/40076315419/job/120820304359),
the exact-commit check-run, all job steps and actual final-head logs. Metadata
and the literal checkout command bind to `f81ce6102ec4f459ef7a0c3fb7e55d126b157be5`.
The completed/success job records 109 tests, matched Chromium/headless shell
v1248 / 156.0.8078.4, eight browser passes, actual-input F01/F02 observations
and all seven controls. See raw decoded [hosted log](logs/hosted-job.raw.json),
[run](logs/hosted-run.json), [steps](logs/hosted-jobs.json) and
[check-run](logs/hosted-check-runs.json).

The previously denied local matched Chromium CDN download was not retried:
same selected environment and known HTTP 403 restriction, without materially
new access evidence. Local system-browser results remain distinct from verified
matched hosted results. No restriction was bypassed and no alternate environment
was used. New adversarial probes ran locally; hosted normal CI does not contain
the remaining capability-absence regression.

The normal production artifact contains neither author e2e SDK markers nor the
independent SDK/observer markers. Author e2e and independent re-review SDKs each
appear only in their own explicit separate build outputs. See
[isolation receipts](logs/artifact-isolation.json). The independent SDK observes
actual inputs; it is not a substitute correctness oracle for the scheduler.

**Required next author response and limits.**

Address the residual **PR04-F01** capability/input-aging case with an ordinary
new correction, regression links and the exact new HEAD/tree. Preserve F02's
closure and all existing success/failure/ownership controls. Run full validation
and provide successful exact-new-HEAD hosted matched Chromium logs, then request
independent re-review in this same cycle. Do not merge or deploy. An author-only
replay or green current CI does not close the newly demonstrated case.

Uncertified: physical cameras/permission UX, actual downloaded models and GPU
execution/utilization, audible TTS, real-device background/resume behavior,
cross-platform equivalence, detection/geometry accuracy and deployment. These
remain nonblocking scope limits. Synchronous main-thread inference remains a
disclosed architectural limit. There is no publication blocker; implementation
acceptance remains blocked by the proven compatibility behavior.
