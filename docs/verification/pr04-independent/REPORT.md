# Independent acceptance review of PR-04 / PR #7

**Verdict: FAIL. Two medium-severity blocking contract defects: PR04-F01 and
PR04-F02.** No critical/high defects are claimed. Successful existing CI does not
close either independently reproduced failure.

This review was performed in a separate execution and clean clone by a reviewer
who did not author PR #7. No product code, existing tests, workflow, historical
evidence, main, other branches or settings were changed. The only proposed
changes are new review evidence under this directory. Publication uses the
owner's connected GitHub identity; this is a separate agent review, not a claim
of approval by a different human/GitHub account.

| Binding | Exact value |
| --- | --- |
| Reviewed PR | [TakashiSasaki/vision-edge #7](https://github.com/TakashiSasaki/vision-edge/pull/7) |
| Reviewed implementation HEAD | `2f4cdea1bc3e6706f49e37116ede8b1881472f7f` |
| Reviewed implementation tree | `aaf506b0b70499da4a9474fc39b062841ed06965` |
| Implementation base main | `f1a0d0bbe343b666c75eae053340ceefc6730d5d` |
| Base tree | `941c586c76bec45fff68d10cc8cb5b6761bd59d4` |
| Evidence branch / intended PR base | `review/pr04-independent-acceptance` / `fix/inference-scheduler-metrics` |
| Date / clean review checkout | 2026-10-24 UTC / `/workspace/pr04-independent` |

The live PR, all latest PRs, remote branch refs, commit/tree, check-run, Actions
run, job steps and actual job log were read independently. No equivalent PR-04
review existed: open #2 and #4 are historical PR-01 evidence. PR #7 still
matched the frozen HEAD at the last pre-publication read. The exact evidence
commit/tree and final readback are recorded in the review PR description to avoid
self-reference. See [source bindings](source-bindings.json) and
[preflight PR snapshot](logs/preflight-pr7.json).

**PR04-F01 — medium, blocking: deduplication identity and consumed input diverge.**

Location: [frameScheduler.ts:40](https://github.com/TakashiSasaki/vision-edge/blob/2f4cdea1bc3e6706f49e37116ede8b1881472f7f/src/services/frameScheduler.ts#L40),
with identity recorded at line 52 and deferred execution at line 30. The
production bridge passes the live video at
[VisionCanvas.tsx:138](https://github.com/TakashiSasaki/vision-edge/blob/2f4cdea1bc3e6706f49e37116ede8b1881472f7f/src/components/VisionCanvas.tsx#L138).
Contracts 1 and 6.

Arrival A records A's key and queues a timer. Before the timer runs, the decoded
video advances to B without B's JavaScript callback being delivered. The timer
infers live B. B's subsequent callback sees a different key and queues another
inference of B. The implementation has no consumed-frame identity check.

Minimal independent scheduler reproduction: deliver frame 1; update live video
to frame 2 without delivering its callback; flush the timer; deliver frame 2;
flush again. **Observed inputs [2, 2]; expected [2].** Both rVFC and fallback
assertions fail. A separate test through the real VisionCanvas and VisionEngine
records two SDK reads of frame 2, two TTS decisions and `推論: 2.0 FPS`.
The TTS service is observed at its boundary; this does not claim audible
duplicate speech through its real cooldown.

A native Chromium decoded-video probe uses the production scheduler source
(TypeScript transpilation only), a changing test clip, 320 ms of bounded
main-thread contention after the first arrival callback, and pause after the
first consumed image to model stalled input. **All six fixed trials, three per
input path, read byte-for-byte identical pixel buffers twice.** Equality checks
compare every pixel byte; the retained hashes are just convenient diagnostics.
This does not depend on the author's SDK fixture or scheduler helper.

Impact: repeated expensive inference on the main thread, duplicated adoption/
TTS decisions and duplicated image samples in inference measurements. The two
SDK calls are real successful calls; the event-span arithmetic itself is not the
cause. The new-frame acceptance contract fails.

Correction acceptance: bind/reconcile consumed input identity at execution,
retain bounded latest-only work and all generation/readiness/lock checks, and
pass both deterministic paths plus real bridge/native delayed-execution
regressions. Do not merely remove the failing assertions or infer candidate
identity from an unrelated timestamp.

**PR04-F02 — medium, blocking: an already-presented first frame is never acquired
by rVFC startup.**

Location: [frameScheduler.ts:72](https://github.com/TakashiSasaki/vision-edge/blob/2f4cdea1bc3e6706f49e37116ede8b1881472f7f/src/services/frameScheduler.ts#L72),
which only registers a future callback at line 57. Contracts 1 and 9.

The camera can present its first image while the model is still loading. Once
the scheduler starts, registering rVFC does not deliver an image already
presented. If the source pauses/stalls before another image, inference never
starts despite readyState and dimensions passing readiness.

The independent native reproduction creates `canvas.captureStream(0)`, requests
one frame and waits for native rVFC to confirm `mediaTime: 0, presentedFrames: 1`
before starting the production scheduler. No further frame is produced. After
400 ms, **rVFC work count is 0; fallback control is 1**, with real video
`readyState: 4` and 64×64 dimensions in both. A deterministic startup assertion
also fails. Expected: consume that first usable image once.

Impact: no result, overlay or processing measurement for an available first image
until the next arrival; eventual input-stalled display. Correction acceptance:
bootstrap the already-ready first frame once, cover scheduler startup after
presentation and equal-timestamp new sessions with native input, and verify
bootstrap/arrival races do not introduce F01 duplicates.

Detailed stable records, minimal steps and correction conditions are in
[findings.json](findings.json). [The matrix](requirement-test-matrix.md) maps
every acceptance contract to evidence.

**Source, scope and actual call graph.**

No ancestor/repository AGENTS.md or repository .agents instructions were present;
`/workspace/.agents` and `.codex` were empty. Read PR-04's report, earlier
integration, PR-02/PR-03 records, all changed source/tests/configuration and the
existing App, camera hook, engine, speech and UI consumers.

All **81 changed paths** were inspected/classified: **65 evidence files,
4 product source files, 8 test/fixture files and 4 configuration/control files**.
The initial conversational estimate of 64 evidence/17 other files was off by
one; [the complete inventory](changed-files.json) is authoritative. Every changed
file has byte/line counts and SHA-256. The evidence includes an executable
author verification Python script; it was read, not executed. Earlier failed
development receipts were distinguished from final receipts.

All 54 tracked non-evidence files were independently hashed and match the
frozen source, including the author's final hash receipt; that comparison
establishes byte identity, not correctness. No dependencies/lockfile, CI workflow,
tracking IDs, coordinate calculations, dictionary or speech-service source
changed. Engine `git diff -w` confirms output-generation guards and error repair,
with inherited mask/box/smoothing algorithms preserved. Old assertions remain;
the obsolete unavailable-metrics text assertion was appropriately updated.
No skipped tests or weakened CSS diagnostics were found. The sole new ESLint
ignore is generated `.cache/**`; source hook rules remain enforced and the old
canvas exhaustive-deps suppression was removed.

There are no new binary artifacts, generated product builds or identified secret
markers in the implementation diff. The existing .aistudio/public images are
baseline files. Paths/hashes in receipts are scoped provenance, not newly
published image content. Historical evidence files were not altered.

| Production path | Observation |
| --- | --- |
| App owns engine, subscribes to model state, initializes chosen model | Camera stop does not dispose it; same ready model is reused. |
| Camera hook owns stream and playback readiness/request ID | Independent camera generation is carried into scheduler current checks. |
| Canvas layout invalidates scheduler generation and temporary outputs | Parent/scene/metrics clear on model/camera/threshold/retry changes. |
| rVFC or RAF → latest pending bit → owned timer/job → engine call | No FIFO; F01 is the gap between arrival identity and live SDK input, F02 is absent initial acquisition. |
| Engine SDK → conversion/filter/smoothing/result close → typed result | Engine model/output generation rejects stale output; errors repair smoothing. |
| Adoption → completion metrics/latest result/dirty flag/TTS | Current model/camera/scheduler checks protect adoption. |
| Separate overlay RAF → offscreen drawing → visible copy → draw completion | A callback without a usable result, missing context or failed copy is not a successful draw sample. |
| 250 ms HUD → fresh clock snapshot → dirty parent callback | Parent/TTS work is restrained; metric aging happens without new samples. |

**Independent execution and arithmetic.**

Environment: Linux x86_64, Node 24.19.0, npm 11.9.0, Playwright 1.64.1,
system Chromium 154.0.8037.57, ffmpeg 7.1.5. Clean target `npm ci` used a fresh
workspace cache. A second clean checkout of the actual main base ran its original
61 tests using the same locked dependency installation; no independent baseline
`npm ci` is claimed.

| Command/check | Actual result |
| --- | --- |
| Clean npm ci; typecheck; lint; npm test; production build | All exit 0; **86/86** tests, 9 files. |
| Original main test suite | Exit 0; **61/61**, 7 files. Baseline checkout clean. |
| Playwright matched Chromium install | Exit 1: CDN HTTP 403 `Domain forbidden`. Installer's own bounded retries are retained. No restriction bypass or alternate environment used. |
| Default matched-browser e2e / controls attempts | Both exit 1 because matched browser was unavailable; these are not assertion-validation passes. |
| Explicit existing system-browser override, full e2e | Exit 0; **4/4**, including both original cases and both positive inference paths. |
| System-browser fault controls | Exit 0; **7/7**: two healthy exit-0 controls and five intended diagnostic-specific child exit-1 controls. |
| Main-to-frozen-HEAD whitespace check | Exit 0. |
| Independent acceptance vectors | Exit 1: **25 pass, 4 fail / 29 total**. Three F01 assertions, one F02 assertion. |
| Native first-frame stream probe | Exit 1 for F02's explicit 0-versus-1 assertion. |
| Native delayed-execution probe | Exit 1 for F01; all six final fixed trials duplicate consumed pixels. |
| Evidence-inclusive typecheck/lint | Both exit 0; no suppressions added. |

Raw required command receipts: [checks.json](logs/checks.json),
[system-browser-checks.json](logs/system-browser-checks.json),
[independent acceptance](logs/independent-acceptance.txt) and
[command receipt](logs/independent-acceptance-command.json).

The independent metric oracle uses its own arrays, literal contract windows and
count/elapsed-seconds calculation, without the production collector's constants
or helpers. It verifies irregular/low-rate input, latest-300 retention, inclusive
5-second boundaries, 2-second expiration, long idle/resume, invalid/reversed
clocks and unknown versus numeric values. Event-span FPS is `(N-1)/elapsed`,
not fixed-window throughput; stalled/no-sample output is unknown, not fabricated
zero. Complete-call duration is independently injected at separate stages:
5 ms SDK + 11 ms category conversion = **16 ms**; 5 ms segmenter SDK +
13 ms result close = **18 ms**. Empty successes count; failure/refusal and
pre-result draw callbacks do not.

Six separately injected draw/callback/TTS faults each reached an explicit stopped
state and recovered through retry without recreating the model or camera.
Model/camera/threshold/unmount reentry, foreign-token release, old async
completion, latest-pending coalescing and StrictMode cleanup were also tested.
No author test fixture/helper/collector algorithm was adopted as an independent
correctness oracle.

Exploratory receipts are retained honestly: one independent duration expectation
initially included a zero-duration call in its mean, and another demanded
bit-identical floating arithmetic from different operation order. Those harness
expectations were corrected to the stated positive-duration and tolerance
contracts. An exploratory 400 ms native race trial duplicated fallback input but
landed on a new rVFC image; final fixed 320 ms trials all duplicate. These do not
replace the deterministic failures, and no product/test source was changed.

**Hosted binding and browser isolation.**

[Run 40071091971 / job 120804489259](https://github.com/TakashiSasaki/vision-edge/actions/runs/40071091971/job/120804489259)
is completed/success and bound to the exact reviewed SHA by the API run,
check-run and actual checkout log. Raw [run](logs/hosted-run.json),
[job/steps](logs/hosted-jobs.json), [check-run](logs/hosted-check-runs.json)
and [downloaded job log](logs/hosted-job.txt) are retained.
The readable text logs have trailing whitespace/terminal blank lines normalized
for Git diff checks; [the raw decoded hosted log](logs/hosted-job.raw.json)
is preserved losslessly as a JSON string. [Normalization receipts](logs/log-normalization.json)
record before/after hashes for new review logs only.
The log confirms exact SHA checkout, Node 24.19.0/npm 11.17.0, 86 tests,
matched Chromium/headless shell **v1248 / 156.0.8078.4**, four browser passes and
all seven diagnostic controls. No system executable override is present in that
job. Uploaded artifact ID 12551382365 is recorded in its log; the artifact's
binary contents were not downloaded. Local alternate-browser results are
additional evidence and do not substitute for this hosted binding.

Positive e2e uses native MediaStream/video progression and actual production
engine/scheduler with an explicit build-time SDK double. Normal production
artifacts independently contain neither `visionSdkProbe` nor
`Test-only SDK fault`; the separate e2e output contains both. See isolation
JSON receipts. There is no runtime test injection in normal product behavior.

**Nonblocking risks, gaps and limits.**

PR04-R01: with hypothetical Promise-valued production work, an old job's lock
release does not wake a replacement rVFC candidate; another arrival is required.
This is independently reproduced with Promise work. Current production engine
work is synchronous, so it is a future asynchronous-path risk, not an additional
blocking defect or a request for Worker migration.

PR04-C01: existing tests immediately flush callbacks/timers and use continuous
native input, leaving the two proven timing cases uncovered. Their successful
results are valid within those paths, not an independent acceptance oracle.

Uncertified: physical cameras, permission prompts/devices, actual downloadable
model inference, GPU utilization/delegation execution, audible TTS, hardware
background/resume behavior, cross-platform equivalence, detection/geometry
accuracy and deployment. Synchronous main-thread inference is a disclosed scope
limit. SDK-only timing is optional and its absence is not a finding. No
Worker/YOLO/SegFormer/PR05/catalog/deploy work was performed.

**Required response from the implementation author.**

Respond to **PR04-F01 and PR04-F02 individually** with the root cause, ordinary
new corrective commits on the implementation branch, regression-test links and
the exact corrected HEAD/tree. Preserve the independent single-consumption and
ready-first-frame expectations. Run clean install, typecheck, lint, the complete
baseline/current unit and browser suites, controls and diff checks, and provide
successful **exact-new-HEAD hosted matched Chromium logs**. Return those receipts
for independent re-review in this thread. Neither finding is closed by an
author-only assertion or green pre-fix CI. Do not merge or deploy.
