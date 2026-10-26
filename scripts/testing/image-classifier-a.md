# Phase A: whole-image classification foundation

Base/main `bebc22a7d11e2bcbd0a63fa1cd141c0a8359dceb`, tree
`7d49c16d0a50b1ba3a6b6a0b43c6cf00fd40aa6c`, version1.0.5. This coherent PR uses
npm's PATCH1.0.6; no dependency change, tag, merge or deployment. Phase B is deferred
until independent A review. This is backend foundation acceptance, not completed
UI or successful real inference.

## Contract and integration boundary

`ALL_MODEL_REGISTRY` discovers five families; `IMAGE_CLASSIFIER_MODEL` provides
Lite0/Lite2 whole-image metadata and explicitly `availableInUI:false`. Existing
`MODEL_REGISTRY`, detection-only RunnableModelType, parameter maps and current UI
retain their four families/nine variants. Do not send classification into
VisionEngine/detection rendering. No fake rectangles, masks or per-object crops.

`ImageClassificationEngine.initialize(variant, delegate)` owns one model task and
one initialization generation. GPU is this application's explicit default; CPU
requires an explicit request. No GPU-to-CPU delegate fallback is performed.
State separates requestedDelegate, configured sdkDelegate (unreported for an
injected factory without that report), and actualBackend:unmeasured. SDK creation
failure is an error, not CPU success. A mismatched reported delegate is rejected.

The default factory resolves an approved exact variant, loads MediaPipe WASM,
then calls installed ImageClassifier.createFromOptions with VIDEO/maxResults:-1
and the explicit delegate. The current asset resolver intentionally blocks before
WASM/SDK startup. Tests substitute this resolver and SDK boundaries; they do not
enable an acquisition switch in production. A future reviewed asset-policy change
is necessary before real default initialization can run.

`classifyVideoFrame(input, {scoreThreshold,maxResults}, instrumentation)` returns
a dedicated kind:whole-image-classification frame or not-ready. The actual
VideoFrame/owned Canvas/live video is passed whole to classifyForVideo without
manual resizing/ROI. Timestamps submitted to VIDEO are strictly increasing;
optional SDK timestamps remain separate. Heads retain their headIndex/headName;
categories retain actual index/categoryName/displayName/score. Categories are
copied, thresholded inclusively and stably sorted by descending score, then capped
per head (default5). IDs are never converted to ranking or shifted by ±1. Empty
names and empty legitimate results are preserved. Finite scores are not rescaled,
clamped, translated or converted to a fabricated confidence. The SDK/model metadata
may itself filter returned categories; this API does not claim all ImageNet labels
were returned. maxResults accepts1..10000 as an API bound, not a model class count.

Malformed output and current frame errors throw typed input/run/output errors,
invalidate prior frame ownership, and retain an initialized task for an explicit
next attempt. SDK/engine metrics distinguish failure/cancellation/success. Old
completion/clear/model switch cannot publish a stale frame. `isCurrentFrame` binds
the exact latest frame, request, variant and output generation. `clearResults`
invalidates output without destroying the model.

Pending old initializations close once and discard their source journals.
Successful current initialization publishes ready ownership and model/variant/WASM
provenance atomically before observers. Source callbacks cannot publish a partial
batch. A synchronous run holds a lease: dispose/switch delays release until that
run exits. Sync/async cleanup failures are handled without unhandled rejection.
Custom factories remain test/embedding boundaries; independent global mutation by
an injected factory is not made safe by this engine.

`startImageClassificationScheduler` reuses the actual latest-frame scheduler,
JobLock and capture/release owner. It requires ready initialization, suppresses
obsolete adoption, and clears only its own request on stop. Restart must create
a new scheduler; stop does not dispose the initialized model. Native video fallback
remains live/unbound input; this phase does not claim display latency, presentation,
GPU execution or immutable input where none was captured.

## Asset and specification gates

The exact reference URLs come from Google's
[official Web sample](https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/image-classifier.ts):
EfficientNet-Lite0/Lite2 under image_classifier/efficientnet_lite{0,2}/float32/1.
They are versioned reference URLs, not verified or acquired bytes. Both SHA256,
size and exact-weight license remain null, status BLOCKED/acquisitionEnabled:false.
They are outside the nine-entry acquisition catalog. Status exposes them in a
separate classification map; startup prefetch, explicit cache prefetch, disk reuse
and model HTTP serving cannot acquire or redistribute them. Existing nine variants
and their integrity/provenance guards remain unchanged.

[Official task guide](https://developers.google.com/edge/mediapipe/solutions/vision/image_classifier/web_js)
and the installed0.10.14 vision.d.ts expose ImageClassifier, VIDEO classifyForVideo,
category IDs/names/scores and CPU/GPU options. SDK code license is not proof of
these exact model weights' redistribution terms. No SDK upgrade is needed for the
inspected API surface; successful real SDK/WASM/model execution is still unverified.
The overview and Model Maker list differing Lite2 input sizes:
[overview](https://developers.google.com/edge/mediapipe/solutions/vision/image_classifier)
versus [SupportedModels](https://developers.google.com/edge/api/mediapipe/python/mediapipe_model_maker/image_classifier/SupportedModels).
They may describe different assets. No224/260 tensor assumption or manual reshape
is encoded. Exact asset metadata, label count/index alignment, GPU support and
model results must be validated after lawful reviewed bytes/terms/pins exist.

| Model | VideoFrame | Owned Canvas | Live video |
| --- | --- | --- | --- |
| Lite0 | BLOCKED assets | BLOCKED assets | BLOCKED assets |
| Lite2 | BLOCKED assets | BLOCKED assets | BLOCKED assets |

This real-inference matrix applies to either requested delegate; no real case was
run. Model-free input/delegate tests are separate and do not turn these into PASS.
Six older MediaPipe CPU cases also remain BLOCKED. No binary request, dev-server
prefetch, denied-route/ZIP retry, alternative host/account/machine or Windows work.

## Verification and Phase B handoff

70 new model-free controls cover both variants/delegates and actual default-factory
SDK options; blocked assets/no I/O; malformed/unknown identity; failed init/retry;
ABA, repeated cancellation, stale WASM and atomic provenance; score/order/head
preservation; malformed results/options; run failure and metrics; exact input types;
lease/close/reentrant release behavior and validated option snapshots; rAF/rVFC, bootstrap/dedup, native/Canvas capture, stop/switch,
error/adoption and cleanup. New tests use controlled results/generated bytes,
not models. Existing621 repository cases and57 guard controls are preserved.

Two development typechecks initially failed: widened unreported literal, then mock
function signatures/nullable metric snapshot. These were corrected with precise
types and an explicit missing-snapshot assertion, without suppression or weakened
behavioral assertions. Direct first npm typecheck exit2 remains in the command
transcript; the second captured failure and successful focused/full receipts are
retained separately. Typecheck exit2 here is FAIL, not BLOCKED inference.
An intermediate full688 pass was followed by three additional adversarial controls
that failed the development implementation: option-object mutation and initialize/
dispose cleanup reentry. Their failed3/skipped53 filtered receipt is retained.
Ownership is now established before cleanup callbacks and validated options are
snapshotted before SDK work. Final exact source/local/hosted binding is in the
receipts and Draft PR body.

For B: use the separate classification state/frame contract and route the fifth
family deliberately. Dispose the previous engine on cross-family changes; use the
shared camera owner and classification scheduler; clear displayed output on error,
stop and camera replacement. Render per-head Top5 raw scores/IDs without boxes.
Use an independently validated Japanese map keyed to the exact model labels,
with raw categoryName/displayName fallback; never an external-list index offset.
Manual TTS requires explicit user action; do not reuse detection auto-readout.
Keep unavailable assets visible as BLOCKED, not selectable inference success.
Add final UI/E2E and comparison documentation in separate stacked B after A review.
Neither phase may assert successful inference from API inspection or negative tests.

Local and hosted platform coverage is Linux only. Windows/macOS/physical GPU and
large-model runtime remain untested. The production JS includes added registry
metadata; byte-equivalence of the entire build is not claimed. CSS and static
asset parity are recorded separately. Injected factories must return a fresh owned
task; reusing an already closed SDK task is outside the ownership contract.
