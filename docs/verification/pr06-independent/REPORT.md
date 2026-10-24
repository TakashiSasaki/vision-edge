# Independent PR06 review

**Scoped verdict: FAIL. PR06-F01 (P2) remains open against the explicitly requested invalid-statistics fault tolerance.** The ordinary product suite passes; this verdict does not assert a spontaneous malformed output in normal collector operation. The reproduced failure is a genuine uncontained React render failure under a one-field statistics fault injection. Author correction and independent re-review are required for that acceptance item.

## Source and independence

| Binding | Exact value |
| --- | --- |
| Product PR | #14, Draft, open, unmerged |
| Actual branch | `feat/pipeline-performance-instrumentation` |
| Frozen reviewed HEAD | `04a3530c6ec9755bb7b89c3e31c1c3e15c9b1ea9` |
| Frozen reviewed tree | `7edc57e2c004d8a85502209b0e2541bd8b11ec07` |
| Product-only parent | `9865050bd1ee025ecc3e39021dfcd103b6f7acc7` |
| Product-only tree | `48e7896b7f9dc96f9579d06a080adf7933f77967` |
| Main/base | `18e94a4508a0fa316cf31171789c3c10a0a041c3` |
| Base tree | `f8e78b24e0bf27393ca9266c4d64bf039d232516` |
| New review branch | `review/pr06-pipeline-independent` |

Separate selected cloud environment; author implementation was not edited. Coordination skill read. No AGENTS.md or relevant `.agents/skills` files were present. Live GitHub PR metadata, actual remote branch refs, commit/tree and base were checked before tests. Main-only fetch configuration initially needed an explicit product-branch fetch. Existing open PRs were #2/#4/#8/#9/#10/#12/#13/#14; none was a PR06 independent review. Initial generic issue search returned empty; the dedicated PR search returned those eight entries and is the duplicate-check evidence. No stale HEAD evidence is transferred to moved code. Final readback is recorded at publication.

Reviewer evidence is added only under `docs/verification/pr06-independent/`. Dedicated detached worktrees `/tmp/pr06-product` and `/tmp/pr06-baseline` ran ordinary source tests/build/browser. Locked dependencies were installed once with Node24.19.0/npm11.9.0 and shared via symlink; package/lock files are byte-identical between base and product. Unit/baseline processes initially overlapped; browser suites and controls ran serially. No ordinary browser test failed in this review, so no retry was required. Product screenshots created by existing tests were copied into new evidence, then the disposable source worktree restored and checked clean. No author evidence, existing tests, workflows or product files enter the evidence commit.

The entire product diff, measurement contract, author README/matrix, raw overhead data, experiment source, changed and relevant existing scheduler/engine/Canvas/FPS/coordinate/browser tests and CI workflow were reviewed. All original assertions remain; scheduler test fixture changes add an optional observer. Source SHA256 bindings and complete product/test diff are retained. Product-only-to-final-HEAD changes are evidence-only. The unchanged SDK factory/version, camera hook, dictionary, speech, coordinates and performance estimator are separately bound. No Worker/hash/thinning/algorithm or deploy workflow was added.

## Finding F01

The actual VisionCanvas bridge with ready synthetic camera and real VisionEngine is rendered, and the initial SDK input executes. A spy calls the real `PipelineMetrics.snapshot()` then replaces only `stages` with `null`, retaining all other actual fields. At the 250 ms HUD publication, telemetry returns a non-null object; `setPipelineStats` schedules child React rendering. `PipelineDetails.tsx:21` indexes `snapshot.stages[stage]` without validation or a details-only render boundary. Rendering throws **Cannot read properties of null (reading 'capture')**, unmounting the video and cancelling all RAF callbacks. Observed: `videoMounted=false`, `pendingRAF=0`, `cameraRequests=1`, `sdkCalls=1`.

Expected under acceptance5: invalid statistics must become unavailable or fail within the details feature while retaining existing camera/SDK/resource ownership. The control where `snapshot()` itself throws remains mounted, demonstrating the uncovered boundary is invalid returned data/rendering, not computation exceptions. Collapsed details still render their children, so disclosure state does not contain the fault.

This injection intentionally violates the TypeScript return shape to exercise the user's required invalid-statistics case. Normal collector code constructs a complete shape; **no naturally reachable stages=null path was identified**. Treat F01 as an explicit fault-containment acceptance gap, not a claimed ordinary-use production crash or SDK/model defect. Author can address containment and add an equivalent regression, or return a reasoned contract response for the parent to decide whether the requested acceptance should be narrowed. This reviewer does not silently waive it or change the product.

Reproduce: `npx vitest run --config docs/verification/pr06-independent/vitest.config.ts`. Final independent run: **30 pass / 1 fail**, exit1. `findings.json` supplies stable ID, severity, expected/actual, locations, reproducer and author action. `logs/independent-all.txt` preserves the actual failure and mounted/RAF observations. The separate ordinary `npm test` includes src tests only, so evidence CI success would not close F01.

## Positive acceptance evidence

Independent nonuniform1703-event completion-time oracles compare all40 stage/outcome cells, including missing durations, median, nearest-rank p95, sum/mean, latest300 coverage and exact expiry. A2050-event ring overflow uses a binary-exact .125ms boundary; no bucket approximation. Pressure200000 stage records and10000 identities verifies every map/ring cap and age expiry. No duration is replaced with0 when missing; no-execution has count0/null. Valid equal clocks preserve measured0. Bad/throwing/backwards clocks and throwing observer property getters are contained; reset rejects old generation identities.

Independent full8-byte RGBA fixture changes the last byte of imageA before executionB. Actual work receives B; captureId2/job1 bind its17.25ms start/end; queue=17.25ms and firstdraw20ms gives latency2.75ms. Bootstrap+execution+600duplicate polls produce602captures/cleanup invocations and602samePixels calls, including the cheap null call;601actual full-array scans; one inference. Private canvases are released. The live quality-counter path retains original input and explicitly unbound physical-image latency.

Independent detector/segmenter factories assert exact source reference and original product timestamp1234. Enabled/disabled output/exception/resource comparisons cover SDK throw, mask access/transform throw and close throw; deterministic segmenter stages SDK7ms, transform16ms, close13ms remain separate. SDK failure executes no transform; mask failure still closes once; swallowed close failure preserves ready return. Original synchronous subscribers/reentrant generation tests pass. Separate scheduler work/adopt/close exceptions under valid, throwing-observer and throwing-clock modes preserve errors/results/release/lock counts. Cancelled/stale async owners keep foreign lock and input until completion, never adopt retired results, and occupy their own outcome stage.

Firstdraw accepted-reference/generation matching, repeated copy, exact5000ms eligibility and later expiry, overwritten/hidden/cancelled/drawfailed no-zero controls pass. Existing actual bridge tests cover empty successful draws once, latest result overwrite, hidden/stop/draw failure and fatal-fault snapshot reset. HUD browser tests cover immutable/RGBA/live-quality paths, model/facing/camera transitions, expiry and disclosure without restarting camera/model. Original real CSS coordinate/margin/ripple/TTS tests pass, along with unchanged exact-fractional-edge unit controls. These are synthetic camera/SDK/audio probes, not hardware validation.

The contract accurately calls the measurements JS wall intervals, distinguishes inclusive job/engine from exclusive postprocess segments and separate input cleanup, and forbids summing overlaps. CaptureDuty explicitly excludes outside comparison/cleanup and attributes whole completed intervals to the window (possible boundary bias); it is not CPU utilization. Quantiles openly use latest300 while count/mean/sum/max use up to2048 retained events and explicit partial coverage. Draw latency ends at synchronous Canvas API return plus immediate bookkeeping; physical exposure/presentation is unknown.

## Ordinary checks and CI

| Check | Exact base | Frozen product |
| --- | --- | --- |
| Unit | 187/187 | 250/250 |
| Focused pipeline/Canvas | n/a | 121/121 |
| Real React/CSS browser, system Chromium154.0.8037.57 | 16/16 | 19/19 |
| CSS controls | 7/7 (2intact,5specific expected-fault exits1) | 7/7 |
| Build | exit0 | exit0 |
| Locked install/typecheck/lint | shared byte-identical lock install | exit0 |
| Production test-double marker audit | unchanged prior foundation | no test marker in normal production JS |
| Diff/source preservation | checked | checked |

Evidence-inclusive typecheck/lint also pass. Command vectors, working directories, timestamps and exits are in `commands.jsonl`; raw logs and browser HTML/report/attachments are retained. Initial reviewer oracle/type/lint/path setup issues are disclosed in `setup-history.txt`; no product failure was relabelled a harness issue. The initial decimal-boundary oracle and failed log remain. There were no new T01-like browser failures to archive or rerun.

Final-product hosted CI was independently fetched through the GitHub connector: **run 40106484147 / job 120909128538 completed/success**. Raw decoded logs explicitly name checkout ref/fetch/checkout/git-log HEAD `04a3530c6ec9755bb7b89c3e31c1c3e15c9b1ea9`. Actual steps show locked install, typecheck, lint,250unit, build,19browser, seven stylesheet controls and artifact upload. Hosted matched Chrome for Testing156.0.8078.4/Playwright v1248 is distinct from local system Chromium154; the earlier local matched-browser403 restriction was respected, with no retry/bypass. No evidence-PR CI is used as substitute for product CI.

## Experiments and limits

See `experiment-report.md` for all96reviewer trials, independent reanalysis of all288author trials, unfavorable observations, all-stage medians/p95/variance, exact work counts, warmup/order balance and biases. All12configurations cover640x480/1280x720/1920x1080, RGBA/VideoFrame, both model doubles. Disabled instrumentation omits collection/clocks/identities/snapshots while actual work counts match. The inspected rebuilt workload bundle is byte-identical to the author's frozen bundle. An independent81920-event snapshot/probe pressure experiment supplies a separate saturation cost, without claiming browser/React/GPU equivalence.

No arbitrary1ms threshold is used. Manual scheduling, representative overlays, full-resolution synthetic segmentation mask, cloud CPU noise, short trials, limited paired-tail precision, clock quantization, publication-per-eight-simulated-frames bias, shared stopwatch/counter overhead and synthetic SDK are disclosed. Nonzero instrumentation cost affects existing wall-duration values, not their estimator/eligibility semantics. Real physical camera/model/GPU/audio, mobile/background hardware and public/deployed-version equality remain **unverified**.

Prior PR04 expensive fullRGBA comparisons and identical-content coalescing limits remain. Prior PR05 **T01 P3 timing sensitivity** remains: original normal15/16 and replay23/24 initial failures, screenshots/traces and unmodified serial16/16 and24/24 reruns are preserved in [PR13's immutable evidence](https://github.com/TakashiSasaki/vision-edge/tree/9c0637f7671f0e0bdac1f782516ef414597207f3/docs/verification/pr05-rereview). Concurrency causation was not established. This review does not erase or generalize that history; F01 is a different deterministic invalid-statistics failure.

## Handoff

Return PR06-F01 to the same author for a new corrective commit and regression, full appropriate validation and exact-new-HEAD/tree/CI checkpoint, then return to this same reviewer. No implementation repair, author PR modification, merge, auto-merge, production workflow or deployment was performed. The review ends at publication of this checkpoint. Exact evidence commit/tree and final remote/PR readback are recorded in the new Draft review PR and final handoff. No authorization denial or uncertain external write is outstanding at this report checkpoint.
