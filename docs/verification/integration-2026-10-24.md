# PR-02 / CSS controls / PR-03 integration review

Date: 2026-10-24. The user authorized completing the pending corrections,
verification and main-branch integration in the current conversation. This work
is limited to roadmap PR-02, the existing PR-01 F01 CSS correction, and PR-03.
It does not implement PR-04 or deploy the public application.

## Evidence and source bindings

| Stage | Source | Hosted evidence |
| --- | --- | --- |
| Original PR-02 | `538d25157a5d87feac2da4496e354cbf8b9bdf5f` | CI `40061066760` |
| Reentrancy regression only | `ae7f2ce2e7223efadbe9d1bfa9c9c5dcdd9b8efd` | CI `40066135991`, job `120789678099`: 39 passed, 3 expected regression failures |
| Reentrancy correction | `5ed8109f67768f7684af91104868fc0e0678adbd` | CI `40066224753`, job `120789941072`: all required steps passed |
| PR-02 main merge | `2d167050920268f7a8281453dfb47c37e510a455` | PR #5 merged after exact-head verification |
| Original F01 CSS correction | `accfb1e895f16b4aec32db5dd5025a1bf2782898` | Original independent review in PR #4 at `fc0e643afef17fdcd7ab888e59e34991b85cd2a7` |
| CSS correction reconciled with PR-02 | `27a5c034cdc7f1e367b704a5f38c6998077d7de7` | CI `40066525531`, job `120790836509`: all steps, including stylesheet controls, passed |
| CSS main merge | `d3cafd523617d77c75bcb919da72dce033b5632a` | PR #3 merged after exact-head verification |
| Original PR-03 | `a5efdd17ed2956959862ebbced15776893ce4e80` | CI `40063538658`, job `120781914307`: 55 unit/component tests and 2 browser cases |

Final integration HEAD/tree, final PR CI, and post-merge main CI receipts are
recorded in PR #6 review/conversation comments after execution. These receipts,
not the historical author counts in pr02.md/pr03.md, determine final acceptance.
A source-bearing commit cannot include its own final SHA without self-reference.

## Reentrancy finding: reproduced and corrected

The original engine published loading before assigning its shared Promise.
Synchronous listeners could request the same model, replace it, or dispose it
before that assignment. This was tested against the real engine using injected
SDK boundaries, not an independently reimplemented state machine.

The regression-only commit passed install, typecheck and lint. Exactly three new
cases failed: same-model Promise identity, replacement Promise ownership, and
no factory load after disposal. The original 36 cases and three new positive
controls passed. The positive controls exercise ready-notification replacement,
error-notification retry, and a synchronous factory exception.

The correction establishes the shared Promise before publishing, checks its
captured generation before factory invocation, and returns its own local Promise.
Generation-gated cleanup cannot clear a replacement request. The unchanged six
new regression cases and existing suite are retained. The correction passed the
full hosted pipeline. Actual commit diff readback confirmed only initialization
ownership logic changed.

## CSS integration

PR #3 was integrated without reverting PR-02's supported-model UI expectations.
The latest production smoke retains CSS MIME, CSSOM rules, applied styles,
model-error/retry, disabled choices and fatal-JavaScript assertions. Its original
control runner, npm command and required CI step are retained.

An integration defect was prevented: the runner assumed exactly one browser test
but previously selected all e2e files. PR-03 introduces a second browser test.
The control runner now explicitly selects e2e/smoke.spec.ts and uses zero retries;
ordinary CI still runs every browser test. Two healthy and five faulty builds must
produce the original specific outcomes. Arbitrary child failures do not pass.

Source comparison against the PR-02 main merge found no product-source, HTML,
lockfile or smoke changes in the CSS-only integration. Original evidence files
remain unchanged and histories are preserved through two-parent merge commits.

## Independent review of the camera implementation

The original PR-03 source and actual tests were inspected independently of the
author's summary. App callbacks are stable. The hook's effect does not depend on
model/settings/frame renders or callback identity. Each request owns its stream;
cleanup cancels before releasing, removes ended listeners, and detaches only an
identical srcObject. Late acquisition is stopped before attachment. Obsolete
rejection or playback completion cannot affect a replacement session.

Camera readiness requires successful play(), separately from model readiness and
video readyState. Acquisition/playback/ended failures release owned resources and
preserve diagnostic causes. Stop/restart and facing changes invalidate scene
buffers, overlays and selected object ID while retaining the ready model task.
The existing queued-inference cancellation remains; scheduler redesign is not
introduced here. clearDetections() is retained alongside the reentrancy fix.

The 17 hook cases include stop/unmount, out-of-order acquisition, stale rejection
and playback, explicit restart, missing/synchronous APIs, ended tracks and actual
React StrictMode setup/cleanup. Canvas integration tests also require positive
inference after readiness, no inference from stale video during pending playback,
and scene/selection invalidation without recreating the model.

The browser camera test controls acquisition but uses native canvas-capture
MediaStreams, video playback, srcObject and track.readyState. It checks parent
rerenders, stop/restart, reverse-order completions, rapid facing changes and final
track release. External model resources remain blocked. This is native-browser
synthetic-stream verification, not physical-camera or model/GPU certification.

## Final verification requirements and limitations

The integrated source retains 55 PR-03 unit/component cases plus the six new
reentrancy cases (expected total 61), both browser tests, and seven stylesheet
controls. Required pipeline: npm ci; typecheck; lint; unit/component tests; build;
matched Chromium installation; complete browser suite; stylesheet controls.
A failing required command must stop verification. Final HEAD and main receipts
must be checked separately and attached to PR #6 before reporting completion.

The review container could not resolve github.com for git clone. No clean local
install or local browser rerun is claimed in this integration pass. Execution
uses hosted Actions with repository-pinned Node/locked dependencies; exact commit
IDs, job steps and logs are checked through the GitHub connector.

The original agent implementations were independently reviewed. The reentrancy
correction and merge reconciliation authored in this pass are regression-tested
maintenance, not falsely described as third-party approval of my own changes.

Not verified: physical cameras/permission prompts, actual GPU/model inference,
Windows/macOS, audible TTS, tracking/geometry accuracy, broad accessibility, or
production deployment equivalence. PR-04 retains scheduling, repeated-frame
handling, synchronous main-thread inference and valid FPS/latency measurement.
Historical evidence PRs #2/#4 are not product changes and are left untouched.
