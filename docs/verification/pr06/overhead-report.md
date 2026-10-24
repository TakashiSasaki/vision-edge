# PR06 matched-work synthetic overhead

Measurement implementation built at `c4ce53557e0e74798ff69dddf529ba2c5a447ad6`; final product `9865050bd1ee025ecc3e39021dfcd103b6f7acc7` changes only the fatal-fault snapshot reset and its test. The benchmark imports no VisionCanvas module; all benchmark-imported product modules are byte-identical across these commits, and the rebuilt bundle hash is checked. Same production-compiled experiment bundle for both arms. Raw per-operation samples and all trials: [overhead.json](overhead.json); exact aggregate/count validation: [overhead-summary.json](overhead-summary.json).

12 paired trials per resolution/path/model (24 measured runs); 16 processed jobs per RGBA run and 32 per VideoFrame run. Each configuration warms both arms first (8/16 jobs), excluded from statistics. Pair order alternates disabled/enabled then enabled/disabled, six of each ordering. Each job changes only the final source pixel, then polls a duplicate. The full-array scan callback and dimensions/bytes algorithm are unchanged. Actual production-boundary work counts must match in both arms or the runner fails. The comparison counter measures actual TypedArray.every full-array scans in both arms; nullable samePixels invocations also include cheap checks and have a different count. No lifetime count is inferred from five-second snapshot retention.

For RGBA each arm/configuration processes 192 jobs, captures 576 times, performs 924 full-array comparison scans and copies 192 results. For VideoFrame each arm/configuration processes/copies 384 jobs and captures/releases 1152 frames, with no pixel comparisons. The raw counts.released counter covers VideoFrame.close only; RGBA Canvas release is verified by ownership tests and is not counted by that VideoFrame-specific counter. Both model doubles use the same scripted images; the segmenter supplies a uniform full-input-resolution mask, deliberately synthetic. Snapshot/quantile computation plus actual React/DOM detailed-panel preparation executes once per eight synthetic 30fps source frames (approximately the app 250 ms publication cadence). Disabled runs have zero instrumentation clock reads, zero event/identity allocation through the optional probe paths, and zero snapshots/panel updates. Shared external stopwatch/counter overhead applies to both arms. Work is synchronous and timer callbacks are manually driven to avoid timer-clamp noise; native queue delay is covered separately by deterministic scheduler and real-app browser tests.

Numbers below are **external observed milliseconds**. Median and nearest-rank p95 pool all per-operation samples (192 RGBA or 384 VideoFrame per arm); paired deltas are enabled-minus-disabled **per-trial means**, with median, p95 and population variance (ms²) across 12 pairs. With 12 pairs the nearest-rank p95 of paired mean deltas is the maximum pair. Engine includes production wrapper/probes. Draw includes shared real Canvas renderer plus enabled draw/first-draw bookkeeping. Total includes capture/comparison/engine/draw and the periodic snapshot/React/DOM cost. HUD is snapshot plus detailed-panel preparation, with 24/48 updates per enabled arm. Disabled HUD timings are external stopwatch measurements of the bypass branch, not 0 ms snapshot executions; the actual snapshot count is zero. The synthetic every-eight-frame publication schedule is explicit and is not a measured real-time 4 Hz scheduler under slow RGBA work; separate HUD costs are provided so the weighting is visible. This harness does not reproduce every VisionCanvas overlay label/ripple, React App notification, browser RAF gap, physical camera or model/GPU operation. It is a local instrumentation-overhead experiment, not a device guarantee or speedup claim.

| Resolution / path / model | Total disabled median / p95 | Total enabled median / p95 | Paired total mean delta median / p95 / variance |
| --- | --- | --- | --- |
| 640×480 / rgba / mediapipe | 54.3000 / 69.1000 | 55.7000 / 64.3000 | 0.7906 / 5.1500 / 9.6039 |
| 640×480 / rgba / selfie-segmenter | 55.9000 / 62.5000 | 56.3000 / 62.6000 | 0.1219 / 3.3812 / 2.2231 |
| 640×480 / video-frame / mediapipe | 0.1000 / 0.2000 | 0.1000 / 0.6000 | 0.0453 / 0.1906 / 0.0020 |
| 640×480 / video-frame / selfie-segmenter | 2.0000 / 2.4000 | 2.0000 / 2.9000 | 0.1188 / 0.2469 / 0.0092 |
| 1280×720 / rgba / mediapipe | 162.9000 / 180.2000 | 164.0500 / 178.4000 | 0.1406 / 7.8125 / 27.5189 |
| 1280×720 / rgba / selfie-segmenter | 168.9500 / 185.2000 | 169.1000 / 183.3000 | -0.5188 / 7.2625 / 20.7612 |
| 1280×720 / video-frame / mediapipe | 0.1000 / 0.3000 | 0.1000 / 0.7000 | 0.0750 / 0.1969 / 0.0026 |
| 1280×720 / video-frame / selfie-segmenter | 6.5000 / 11.5000 | 6.5000 / 11.2000 | 0.0031 / 1.0281 / 0.5727 |
| 1920×1080 / rgba / mediapipe | 358.0000 / 392.3000 | 361.6000 / 396.6000 | 3.0719 / 19.2188 / 73.1416 |
| 1920×1080 / rgba / selfie-segmenter | 383.4500 / 423.0000 | 378.4000 / 420.1000 | -2.4625 / 11.4688 / 92.9947 |
| 1920×1080 / video-frame / mediapipe | 0.2000 / 0.4000 | 0.2000 / 1.2000 | 0.1297 / 0.2000 / 0.0024 |
| 1920×1080 / video-frame / selfie-segmenter | 14.7000 / 24.4000 | 14.4000 / 19.3000 | -0.3656 / 1.1531 / 1.6399 |

## Engine

| Resolution / path / model | Disabled median / p95 | Enabled median / p95 | Paired mean delta median / p95 / variance |
| --- | --- | --- | --- |
| 640×480 / rgba / mediapipe | 0.0000 / 0.1000 | 0.0000 / 0.2000 | 0.0063 / 0.1937 / 0.0030 |
| 640×480 / rgba / selfie-segmenter | 2.0000 / 2.9000 | 2.0000 / 2.8000 | -0.0344 / 0.2188 / 0.0102 |
| 640×480 / video-frame / mediapipe | 0.0000 / 0.0000 | 0.0000 / 0.1000 | 0.0062 / 0.0156 / 0.0000 |
| 640×480 / video-frame / selfie-segmenter | 1.9000 / 2.2000 | 1.9000 / 2.3000 | 0.0344 / 0.0687 / 0.0013 |
| 1280×720 / rgba / mediapipe | 0.0000 / 0.1000 | 0.1000 / 0.2000 | 0.0187 / 0.0687 / 0.0003 |
| 1280×720 / rgba / selfie-segmenter | 6.0000 / 8.4000 | 6.0000 / 8.5000 | -0.1406 / 0.6438 / 0.1310 |
| 1280×720 / video-frame / mediapipe | 0.0000 / 0.1000 | 0.0000 / 0.1000 | 0.0000 / 0.0156 / 0.0001 |
| 1280×720 / video-frame / selfie-segmenter | 6.0000 / 9.0000 | 6.0000 / 8.8000 | -0.0500 / 0.8469 / 0.4241 |
| 1920×1080 / rgba / mediapipe | 0.0000 / 0.1000 | 0.0000 / 0.1000 | 0.0063 / 0.0688 / 0.0005 |
| 1920×1080 / rgba / selfie-segmenter | 13.5000 / 18.1000 | 13.5000 / 18.8000 | -0.2719 / 1.3250 / 0.5075 |
| 1920×1080 / video-frame / mediapipe | 0.0000 / 0.0000 | 0.0000 / 0.1000 | 0.0000 / 0.0094 / 0.0000 |
| 1920×1080 / video-frame / selfie-segmenter | 13.6000 / 18.8000 | 13.3000 / 16.7000 | -0.5375 / 0.9625 / 1.2628 |

## Canvas draw and first-draw bookkeeping

| Resolution / path / model | Disabled median / p95 | Enabled median / p95 | Paired mean delta median / p95 / variance |
| --- | --- | --- | --- |
| 640×480 / rgba / mediapipe | 0.1000 / 0.8000 | 0.2000 / 1.0000 | 0.0375 / 0.2563 / 0.0059 |
| 640×480 / rgba / selfie-segmenter | 0.2000 / 0.8000 | 0.2000 / 0.9000 | 0.0219 / 0.8688 / 0.0573 |
| 640×480 / video-frame / mediapipe | 0.0000 / 0.1000 | 0.0000 / 0.1000 | 0.0094 / 0.1000 / 0.0010 |
| 640×480 / video-frame / selfie-segmenter | 0.1000 / 0.2000 | 0.1000 / 0.2000 | 0.0094 / 0.0312 / 0.0011 |
| 1280×720 / rgba / mediapipe | 0.2000 / 2.5000 | 0.3000 / 2.5000 | 0.0125 / 0.0875 / 0.0018 |
| 1280×720 / rgba / selfie-segmenter | 0.4000 / 2.6000 | 0.4000 / 2.5000 | -0.0094 / 0.1062 / 0.0060 |
| 1280×720 / video-frame / mediapipe | 0.1000 / 0.2000 | 0.1000 / 0.2000 | -0.0094 / 0.1125 / 0.0022 |
| 1280×720 / video-frame / selfie-segmenter | 0.4000 / 1.0000 | 0.4000 / 0.9000 | -0.0078 / 0.1687 / 0.0103 |
| 1920×1080 / rgba / mediapipe | 0.4000 / 5.3000 | 0.4000 / 5.1000 | 0.0063 / 0.8438 / 0.0624 |
| 1920×1080 / rgba / selfie-segmenter | 0.9000 / 5.6000 | 0.9000 / 6.0000 | -0.0313 / 0.3625 / 0.0460 |
| 1920×1080 / video-frame / mediapipe | 0.1000 / 0.3000 | 0.1000 / 0.3000 | 0.0000 / 0.0562 / 0.0007 |
| 1920×1080 / video-frame / selfie-segmenter | 0.9000 / 1.7000 | 0.9000 / 1.5000 | 0.0187 / 0.1313 / 0.0395 |

## Snapshot, quantiles and React/DOM preparation

| Resolution / path / model | Disabled median / p95 | Enabled median / p95 | Paired mean delta median / p95 / variance |
| --- | --- | --- | --- |
| 640×480 / rgba / mediapipe | 0.0000 / 0.0000 | 0.7500 / 1.9000 | 0.7000 / 2.2500 / 0.2000 |
| 640×480 / rgba / selfie-segmenter | 0.0000 / 0.0000 | 0.5000 / 0.9000 | 0.5750 / 1.7000 / 0.1059 |
| 640×480 / video-frame / mediapipe | 0.0000 / 0.0000 | 0.3000 / 0.6000 | 0.2875 / 0.5250 / 0.0081 |
| 640×480 / video-frame / selfie-segmenter | 0.0000 / 0.0000 | 0.4000 / 0.7000 | 0.4000 / 1.3000 / 0.1198 |
| 1280×720 / rgba / mediapipe | 0.0000 / 0.1000 | 0.6000 / 0.9000 | 0.6250 / 3.2500 / 0.5364 |
| 1280×720 / rgba / selfie-segmenter | 0.0000 / 0.0000 | 0.6000 / 0.9000 | 0.6500 / 2.2000 / 0.1958 |
| 1280×720 / video-frame / mediapipe | 0.0000 / 0.0000 | 0.4000 / 0.7000 | 0.4250 / 0.5500 / 0.0060 |
| 1280×720 / video-frame / selfie-segmenter | 0.0000 / 0.0000 | 0.6000 / 4.1000 | 0.7750 / 2.7000 / 0.3733 |
| 1920×1080 / rgba / mediapipe | 0.0000 / 0.1000 | 0.6000 / 0.8000 | 0.6000 / 0.7500 / 0.0033 |
| 1920×1080 / rgba / selfie-segmenter | 0.0000 / 0.0000 | 0.5500 / 0.9000 | 0.6000 / 1.0500 / 0.0218 |
| 1920×1080 / video-frame / mediapipe | 0.0000 / 0.0000 | 0.4000 / 3.4000 | 0.7375 / 1.3000 / 0.1244 |
| 1920×1080 / video-frame / selfie-segmenter | 0.0000 / 0.0000 | 0.6000 / 5.0000 | 1.2750 / 2.1000 / 0.2243 |

## Interpretation and retained limits

Clock resolution creates many legitimate 0 ms intervals for the cheap deterministic SDK. Added instrumentation is measurable in the cheaper paths and in HUD preparation; percentages relative to a near-zero SDK are unstable. Positive and negative paired deltas and outliers are preserved. Negative values indicate local noise/order/GC variation and are not evidence that instrumentation optimizes inference. The unchanged full-resolution RGBA identity scans dominate the slower paths. No arbitrary 1 ms threshold or universal acceptance budget is imposed. The implementation keeps fixed event rings, a separate latest-300 quantile cap, fixed job retention, no RAF sorting, and 250 ms snapshots; these are instrumentation design bounds, not an image-algorithm optimization.

`overhead-initial-partial.json` preserves the original harness failure when rolling counts were mistaken for lifetime counts. `overhead-pre-occupancy.json` is the complete earlier diagnostic sweep before the final job-occupancy probe, with concurrent author validation during some trials; it is not substituted for this final-source sweep. Its unfavorable observations remain. The bulk browser/controls suites completed before this sweep. A 2.3-second focused fault-UI regression ran at 15:07:26 UTC during the sweep; this possible external noise is disclosed and its observations are retained. No effect is causally attributed to it. System Chromium and cloud CPU results are distinct from hosted matched-browser CI and from untested physical devices. Raw environment/browser version and warmup data are included in overhead.json.

Reproduce: `npx vite build --config docs/verification/pr06/experiment/vite.overhead.config.ts`; serve `.cache/pr06-overhead` at 127.0.0.1:4175; `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium node docs/verification/pr06/experiment/run-overhead.mjs`; `python3 docs/verification/pr06/experiment/summarize-overhead.py`.
