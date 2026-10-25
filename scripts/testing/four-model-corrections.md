# Four-model correction checkpoint

Author response to F1–F8 at baseline `a8c8cd9f8caeb76bb051e0a004942ca04600d7f5`
(tree `bfdfa8a4d2027f7e578c2873d693b4f9ce33e695`). Implementation changes and
regressions are ready for **independent re-review, PENDING**. This is not an
independent closure verdict. No merge or deployment is authorized by these checks.
Final correction HEAD/tree and exact-HEAD hosted CI are bound in the Draft PR;
this file deliberately does not embed a self-referential commit hash.

| ID | Severity | Original reproducer / observed failure | Author correction and regression |
| --- | --- | --- | --- |
| F1 | P1 | Main CI [40180757546 / 121131130790](https://github.com/TakashiSasaki/vision-edge/actions/runs/40180757546/job/121131130790) checked out a8c8cd9 and failed setup-node because no npm lock existed; subsequent checks did not run. | Restore npm lock from 63bd3429, include the already-added ONNX dependency, pin ORT1.30.0 to match its WASM version, remove unused Bun lock, align AGENTS Node24. Canonical-registry lock generation and a clean `npm ci` succeeded. 355 lock entries, 22 additions, zero prior dependency version changes. |
| F2 | P1 | Controlled `session.run` rejection became ready/empty; YOLO retained old confidence about .765 and success telemetry, SegFormer retained inspector data. | Typed run/output errors reach the existing frame-failure owner; current outputs/inspector reset and failed SDK/engine probes are asserted. Initialized model remains usable. Valid zero detections keep the existing bounded policy; malformed output is not zero detection. |
| F3 | P2 | YOLO cx0,w128 at input640 yielded width .2 rather than .1 and nontransparent mask beyond right edge. | Clamp both endpoints, use the same clipped geometry and pixel-center crop for the mask. Edge/corner/offscreen regressions inspect actual mask bytes. Keep instance mask and its paired box together instead of smoothing only the box. |
| F4 | P2 | Uniform 150-class logits gave .99; dominant class bypassed threshold; clicked pixel invented .95. | Stable softmax at each pixel; list score is the mean winning-pixel score for that class, area is separate. Every class obeys threshold. Full argmax mask is separate from list selection. UI/readout explicitly name uncalibrated mean/pixel softmax scores. Uniform logits give 1/150; strongly separated equal-area classes both pass .9 rather than retaining the old invented .845. |
| F5 | P2 | Clear during an in-flight person result returned not-ready but published inspector person .95. Clear after an accepted wall frame also left wall visible to inspection. | Per-result immutable snapshots publish only through generation-checked acceptance. Task, exact result array, request and output generation bind inspection. Clear/error/switch invalidate it. Small-class pixel inspection works even when the accepted prominent-object list is empty; old arrays cannot inspect a new frame. |
| F6 | P2 | WASM-only ONNX configuration appeared as GPU delegate. | Both factories report requested provider, configured priority list and observed fallback reason; internal backend remains unmeasured. ONNX ready state carries this metadata instead of GPU delegate. MediaPipe GPU request remains unchanged. Explicit WASM, unavailable WebGPU and creation-failure retry have distinct controls. |
| F7 | P2 | Async release rejection escaped synchronous catch in both services; repeated close released early/more than once. | Session lease spans preprocessing/run/output/tensor cleanup. Close denies new work and releases once after active work settles. Async release/engine cleanup rejection is caught and warned. Pending-run, rejection-during-close, repeated-close, switch and actual-service child-process controls cover ordering and zero unhandled rejections. |
| F8 | P2 | 48 old PR06 raw logs were deleted while immutable documents still referenced them. | Restore the exact 48 blobs from 63bd3429. Compare all 702 historical files against their original SHA256 values before/after verification: zero mismatches. No old evidence was regenerated. |

## Execution and reproduction

Use Node24/npm11 and the tracked npm lock. Local browser checks used the already
installed system Chromium154.0.8037.57, not a new browser installation.

```sh
npm run --silent ci:install
npm run --silent test -- src/services/fourModelCorrections.test.ts src/components/SemanticPresentation.test.tsx
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run --silent verify:offline
```

The two new test files contribute 38 cases. Full unit/component selection is
361 =323 existing cases (including the original317) +38 new cases. Existing
test assertions, timings and retries were not weakened. Full offline exit0:
unit361, foundation73, delegate40, output37, browser19 (zero flaky/skip), and
stylesheet controls7 (two healthy +five specifically diagnosed expected faults),
plus typecheck/lint/production build. Compact receipts and failure history are in
[`four-model-corrections-results.json`](four-model-corrections-results.json).
Full raw stdout/stderr/reporter files remain in the ignored run
`.cache/test-runs/2026-10-25T10-26-22.438Z-89d041be/`; hosted CI publishes its own
corresponding raw files as artifacts. No byte count is presented as token count.

Before implementation, the initial 23-case regression selection against the
unchanged product source failed20/passed3 (exit1). The old async-release child
independently recorded two unhandled rejections (exit1). The first lint of the
new fixture failed `no-this-alias`; the fixture was corrected, not suppressed.
Intermediate focused green selections were23,27,36 and UI2; one earlier full
unit selection was350 before the last11 regressions. These are separate attempts,
not additional cases in the final361. Exact old failures are encoded in the new
tests and the retained initial reporter. The model-switch result guard was already
working; the new release-order assertion exposed the separate early-release flaw.

For F1, clean `ci:install` exit0 evidence is the ignored run
`2026-10-25T09-58-43.701Z-8dbb04d3/install`; lock generation used only
`https://registry.npmjs.org`. Installed prior versions did not change. Baseline
and corrected production CSS built with this same dependency installation are
both27898 bytes, SHA256
`3c60bb7d9314d6bc99c7d29821748ea2b56c49eeaa3ba29f5ea6c8b3cefba5a4`.
Production JavaScript intentionally changes for these correctness fixes.

## Limits and prerequisites

These are model-free controlled ORT-boundary tests of actual service/engine code,
not successful real ONNX inference. They preserve positive preprocessing controls
(YOLO planar640x480 and SegFormer normalized512x512), actual mask byte operations,
and the existing asynchronous model-switch guard. No model was acquired, no denied
403 request or ZIP download was retried, and no alternative route was used.
Hosted artifact publication metadata can be verified; hosted ZIP contents remain
unverified under the earlier restriction. Windows/macOS cleanup remains untested.

Release errors are reported, not proof that a failed backend release freed memory.
An underlying run that never settles leaves close pending; this correction does
not invent ORT cancellation or release a session still in use. Actual backend,
physical GPU, browser/model compatibility, real output quality/performance and
long-running resource behavior still need real-device/model validation. Existing
Hugging Face resolve/main model references and weight provenance/licensing are
unchanged; they are not reproducible byte pins. The user's Windows success is user
reported and is not independently re-certified by these model-free tests.

The earlier six CPU scenarios remain BLOCKED: EfficientDet-Lite0 and Selfie
Segmenter each on VideoFrame, owned Canvas and live HTMLVideoElement. They are
not YOLO/SegFormer cases. Lawful model bytes, applicable terms and reviewed
integrity pins remain prerequisites; no successful inference is inferred here.
