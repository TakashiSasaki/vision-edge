# PR-04 independent review response: PR04-F01 and PR04-F02

Both Medium blocking findings are accepted and corrected in implementation commit
[`89e444820b6befbd333236e65697b5bcf8f5e15a`](https://github.com/TakashiSasaki/vision-edge/commit/89e444820b6befbd333236e65697b5bcf8f5e15a).
**Independent closure is pending.** These are Author corrections and probe replays,
not a substitute for the separate reviewer's confirmation.

| Source binding | Frozen value |
| --- | --- |
| Reviewed implementation | `2f4cdea1bc3e6706f49e37116ede8b1881472f7f` / tree `aaf506b0b70499da4a9474fc39b062841ed06965` |
| Independent evidence [PR #8](https://github.com/TakashiSasaki/vision-edge/pull/8) | `c632cf0cbce82b48503540651120c88fff75c8ef` / tree `b382360d930ca20a71a5f2e7ae99b764f2250564` |
| Corrective implementation | `89e444820b6befbd333236e65697b5bcf8f5e15a` / tree `426e315744e1e4538b4afc70906011c095958ef5` |
| Actual main base | `f1a0d0bbe343b666c75eae053340ceefc6730d5d` |
| Implementation PR / branch | [#7](https://github.com/TakashiSasaki/vision-edge/pull/7) / `fix/inference-scheduler-metrics` |

The final evidence-bearing HEAD/tree and its exact-HEAD hosted Run/Job/log
receipts are in PR #7's description to avoid a self-referential source commit.
PR #7 was returned to Draft on receipt of the FAIL and remains Draft until the
new final-HEAD validation succeeds. PR #8 and all historical evidence are preserved.

## Evidence read and reproduced before repair

Read the actual independent REPORT.md, findings.json, REPRODUCE.md, requirement
matrix, scheduler/canvas/metrics vectors, both native probes and raw observations.
Fetched the evidence commit, checked its tree, and extracted a read-only working
copy under /workspace/scratch. A detached checkout of the actual reviewed
implementation received unchanged review vectors. It reused the repository's
locked dependency installation; no separate baseline npm ci is claimed.

`pr04-corrections/logs/before-checks.json` binds the commands and exits:

- Independent acceptance: **25 pass / 4 fail**, exit 1. F01 is reproduced in both
  scheduler modes and the actual Canvas/Engine bridge: consumed input `[2,2]`,
  two TTS decisions and a misleading two-event inference estimate for one image.
- Native first frame: **rVFC 0 calls, RAF 1 call**, exit 1, source mediaTime 0,
  presentedFrames 1, readyState 4 and positive dimensions.
- Original native delayed-execution probe: exit 1; **5 of 6 local trials** repeat
  byte-identical pixels. One trial advances to another image instead. This is
  timing variability, retained in before-03.txt; the reviewer's original six
  duplicate trials are a separate unchanged evidence set.
- Additional actual App/Canvas/Engine native tests fail on frozen product source:
  **three failures / one fallback first-frame control pass**. The final observer
  records two SDK entries at the same intrinsic image timestamp in both paths;
  rVFC still misses its first frame. See before-browser-acceptance.txt.
  That observer's readback hashes can differ for those equal image timestamps;
  this report does not label those particular arrays byte-identical. The original
  unchanged native probe above provides the exact-byte reproduction.

Before-browser runs copy only test/SDK observation boundaries to the detached
checkout; its scheduler, canvas, engine and types remain the reviewed production
source. The new observer can snapshot the image at SDK entry, but never replaces
production scheduling. Product repair is not applied until the after-replay.

## PR04-F01: bind identity to consumed input

Cause: the timer read live video B while deduplication remembered arrival A.
B's later notification therefore caused another inference. Arrival counters,
playback-quality totals and currentTime cannot simply be compared numerically:
they describe different things, and the playhead continues moving on a still
MediaStream. The native diagnostic in video-frame-identity.txt records intrinsic
image timestamps alongside rVFC metadata, playback position and quality totals.

The production scheduler now captures the current usable **immutable VideoFrame
at execution**, after acquiring its job token. That exact object and its own
source-image timestamp travel together into work. The same timestamp is used for
consumed-input deduplication, so a delayed notification of B cannot resubmit B.
A newer C is still accepted even if its playback-quality total equals B's.
Temporary arrival samples are closed immediately; there is still one latest
pending opportunity, no image queue and no retained capture history.

`InferenceInput` admits HTMLVideoElement or VideoFrame. VisionEngine routes the
actual supplied image to its selected SDK task and uses that image's intrinsic
dimensions. The installed MediaPipe 0.10.14 typings define ImageSource as
TexImageSource, including VideoFrame; its existing image-size helper handles
displayWidth/displayHeight. No SDK/dependency/backend change is needed. Object
conversion, masks, smoothing, tracking IDs and coordinate formulas are unchanged.
The frame's image timestamp is an identity, not SDK/GPU duration: the SDK still
receives its existing monotonic call timestamp.

The execution owns its immutable image until success, exception or completion of
an asynchronous test double. Only its own finish releases that image and token.
Pending cancellation creates no owned execution image; all temporary captures
close. Generation validity is checked after capture and immediately before work,
as well as reservation/adoption. Capture exceptions stop acquisition deliberately
and expose `映像フレーム取得エラー` with the existing explicit retry flow; retry does
not recreate the model or camera. Tests include capture reentry, capture failure
after lock acquisition, stale async completion, invalid image timestamps, failure
followed by a late callback, and exactly-once close assertions.

When VideoFrame is unavailable, the compatibility path samples live playback
progress immediately at execution, using the same quality-counter domain across
executions and currentTime otherwise. It reconciles late rVFC metadata with an
input already consumed ahead of that notification, retaining coarse-timestamp
presented-frame support. This remains a progress-based approximation: browsers
without immutable image capture and with coarse/incomplete progress information
can coalesce distinct images. We do not claim exact source-frame identity there.
Native Chromium tests exercise both rVFC and RAF with actual immutable inputs;
controlled no-VideoFrame tests exercise compatibility behavior. No Worker or
asynchronous model migration is introduced.

Regression links: [scheduler tests](../../src/services/frameScheduler.test.ts),
[real Canvas/Engine test](../../src/components/VisionCanvas.test.tsx),
[SDK routing](../../src/services/visionEngine.test.ts),
[native actual-input tests](../../e2e/frame-identity.spec.ts).
The Canvas regression asserts one SDK read, one TTS decision, one parent result
and unavailable one-sample inference FPS, rather than the prior duplicate 2 FPS.

## PR04-F02: acquire the ready first image once

Cause: startup registered only a future rVFC notification. The camera's first
image could already be presented while the model loaded, with no later input.

Startup now also reserves the currently usable image through the same bounded
pending slot, readiness/generation checks, token and consumed-input comparison.
Bootstrap and a racing arrival coalesce to the latest captured input. Every new
camera/model session resets identity; a new source image at timestamp zero is
processed even if the preceding session also ended at zero. Explicit recovery
can bootstrap again. A threshold-only restart does not reprocess the attempted
old image as a new scene, preserving the independent threshold-reentry control.
Appearance/callback/statistics changes do not restart the scheduler or acquisition.

Native tests first hold model initialization until a one-frame captureStream(0)
has already presented a usable image. Both scheduler paths then produce exactly
one successful result without another source frame. Stop/restart presents another
zero-time image; a third scene advances during a delayed bootstrap and produces
one latest result. Each path asserts three camera acquisitions, one model creation,
native first mediaTime `[0,0]`, and no fabricated numeric FPS for a single sample.
The SDK-entry image timestamps are also zero for the first two corrected scenes.

Both findings use the same corrective commit because bootstrap must share F01's
consumed-input identity and ownership path. They are not independently closed by
that commit or by successful CI.

## Final local validation and unchanged reviewer replay

Linux x86_64, Node **24.19.0**, npm **11.9.0**, Playwright **1.64.1**, explicit system
Chromium **154.0.8037.57**. The previously documented matched-download HTTP 403
restriction remains; no bypass was attempted. Hosted matched Chromium validation
is separate and must bind to the final published HEAD.

`pr04-corrections/probes/verify.py` runs clean npm ci, typecheck, lint,
**109 unit/component tests**, build, **all 8 browser cases**, **all 7 fault controls**,
working/full-base whitespace checks, and the original **61-test main baseline**.
All exit **0**: checks.json and final-00 through final-11 preserve receipts.
The baseline uses the same newly installed locked node_modules via a symlink;
its original source remains clean. verified-source.json hashes all non-evidence
product/test/config files, including the exact corrective source.

The four additional browser cases run actual production App, scheduler, engine,
camera hook and collector against native MediaStreams/video, with only the
explicit e2e SDK observation boundary. For F01 a clip is encoded by native
MediaRecorder in the browser, then decoded independently of JS contention.
A real pending timer and a real native arrival are delivered in adversarial order,
with bounded 320 ms contention. The producer is paused and pending presentations
drained before reading the final stalled image; the scheduler is never replaced.
Tests prove both intrinsic image timestamp and full pixel data advance from the
reserved image, then exactly one SDK entry survives the delayed notification.
Successful JSON summaries include image times and hashes; the test compares the
full pixel arrays, not just hashes. For F02 a single native canvas frame is already
presented before releasing the model gate. The test-only observer is bounded to
16 input records and is absent from normal production builds.

`replay-review.py` checks each independent vector against the exact evidence Git
blob, copies only the four corrected product dependencies into the detached
replay checkout, and executes the original commands unchanged. This is an Author
replay. Source and vector SHA-256 values are in replay-source-binding.json.

| Unchanged independent command | Before | Corrected source |
| --- | --- | --- |
| 29-case acceptance vectors | 25 pass, 4 fail; exit 1 | **29 pass, exit 0** |
| Native pre-presented first frame | rVFC 0 / fallback 1; exit 1 | **1 / 1; exit 0** |
| Six native pixel-comparison trials | 5 local duplicates; exit 1 | **No duplicate observation; exit 0** |

After receipts are after-checks.json and after-00 through after-02. Bootstrap can
make the old native race probe finish before its contention window; that green
probe alone would not establish F01's fix. The new held-job, real-image browser
cases explicitly require delayed execution and source-image advancement.

Development receipts remain disclosed. Early counter-only repairs were rejected
in favor of actual image binding. A first native test compared running currentTime
values across scenes, which are not image timestamps. Another test assumed a
consumer pause drains a MediaStream producer; final fixtures freeze/drain the
producer. Capture/DOM mock typing and bootstrap-era test sequencing were corrected.
No independent vectors, expected single-consumption counts, first-image conditions,
source lint rules, CSS controls, retries, CI requirements or test skips were changed.
Trailing whitespace and terminal blank lines of new text logs are normalized
solely for Git whitespace checks. Pre-line-normalization strings and hashes are
retained in raw-log-whitespace.json and log-normalization.json; historical evidence remains byte-for-byte unchanged.

## Re-review focus and remaining limits

Ask the same separate reviewer to verify both IDs against the final published
HEAD, including SDK input identity, exact handle ownership, bootstrap/arrival races,
threshold/camera/model separation and the unchanged probe replay. The complete
metric estimator/window/expiration contract remains unchanged from pr04.md;
independent metric/duration/fault controls pass again.

PR04-C01's coverage gap is addressed with ordinary CI regressions. PR04-R01 remains
a disclosed **nonblocking future async-path risk**: a replaced rVFC candidate can
need another arrival after an older asynchronous job unlocks. Production inference
is synchronous; no Worker migration is attempted. A never-settling test Promise
retains its owned lock/image until completion rather than closing another job's
resource. Physical cameras, real downloaded models/GPU execution, audible TTS,
hardware background behavior and cross-platform equivalence remain unverified.
Captured-frame acquisition/cleanup is outside the complete detectVideoFrame-call
duration; it is not reported as GPU/SDK-only timing or total pipeline latency.

PR #8 and its immutable evidence remain unchanged. No force-push, rebase, unrelated
branch update, merge, auto-merge, deployment or PR-05+ work occurred.
