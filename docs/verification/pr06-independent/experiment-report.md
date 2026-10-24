# Independent experiment assessment

The author workload and separate build configuration were inspected, then rebuilt from frozen product HEAD. All rebuilt experiment artifacts are byte-identical to the author bindings (`experiment-build-binding.json`). The reviewer runner supplies its own balanced order, assertions and output destination; `analyze-overhead.py` independently recomputes every raw per-operation mean/median/nearest-rank p95/variance and actual work count. It does not call the author summarizer. The browser workload itself is the inspected author production-compiled scheduler/engine/Canvas workload. Independent core fixtures and the separate snapshot/probe experiment supplement it; this is not presented as an entirely independently implemented camera benchmark.

The original **288 measured trials / 24 warmup trials / 12 configurations / 6912 jobs** all validate. The reviewer reproduction ran **96 measured trials / 24 warmups / 12 configurations / 1152 jobs**, four pairs per configuration, 8 RGBA or16 VideoFrame jobs per trial. Both orders occur twice; each arm warms8/16 jobs before measurement. All captures, scans, SDK/process/copy counts, VideoFrame closes and segmentation mask reads/closes match. Disabled arms execute zero metric clocks/snapshots. Warmups remain in raw JSON and are excluded. Reviewer browser timing ran after the ordinary product/base browser and controls suites, without concurrent test commands; no causal noise attribution is made.

| Configuration | Total off median/p95 ms | Total on median/p95 ms | Paired mean delta median/p95 ms | Delta variance ms² |
| --- | --- | --- | --- | --- |
| 640x480/rgba/mediapipe | 53.9500/62.0000 | 53.0000/60.7000 | -0.2750/1.3500 | 3.9191 |
| 640x480/rgba/selfie-segmenter | 55.3500/60.0000 | 57.4000/63.6000 | 2.4500/4.9500 | 3.8568 |
| 640x480/video-frame/mediapipe | 0.1000/1.3000 | 0.1000/1.4000 | 0.0844/0.4312 | 0.0239 |
| 640x480/video-frame/selfie-segmenter | 2.1000/4.6000 | 2.1500/4.3000 | 0.1156/0.1813 | 0.0482 |
| 1280x720/rgba/mediapipe | 161.0500/178.0000 | 158.0000/176.6000 | -0.4750/1.0250 | 17.1737 |
| 1280x720/rgba/selfie-segmenter | 165.0000/171.3000 | 167.4500/181.2000 | 3.2062/11.7750 | 22.9768 |
| 1280x720/video-frame/mediapipe | 0.1000/3.1000 | 0.1000/3.5000 | 0.0938/0.4000 | 0.0190 |
| 1280x720/video-frame/selfie-segmenter | 6.1000/11.3000 | 6.3000/11.4000 | 0.2187/0.3813 | 0.0647 |
| 1920x1080/rgba/mediapipe | 363.7000/397.2000 | 363.6500/394.9000 | 1.5688/12.2875 | 54.1030 |
| 1920x1080/rgba/selfie-segmenter | 371.8000/400.3000 | 381.2500/418.9000 | 3.2625/15.9375 | 40.9503 |
| 1920x1080/video-frame/mediapipe | 0.2000/8.1000 | 0.2000/7.6000 | 0.1125/0.2187 | 0.0070 |
| 1920x1080/video-frame/selfie-segmenter | 14.3000/25.8000 | 14.6000/26.0000 | 0.1688/1.1437 | 0.5489 |

## engine raw sample aggregation

| Configuration | Off median/p95 ms | On median/p95 ms | Paired mean delta median/p95 ms | Delta variance ms² |
| --- | --- | --- | --- | --- |
| 640x480/rgba/mediapipe | 0.0000/0.1000 | 0.0000/0.1000 | -0.0125/0.0250 | 0.0006 |
| 640x480/rgba/selfie-segmenter | 2.1500/2.9000 | 2.2000/3.1000 | 0.1187/0.7375 | 0.0990 |
| 640x480/video-frame/mediapipe | 0.0000/0.0000 | 0.0000/0.1000 | 0.0063/0.0312 | 0.0001 |
| 640x480/video-frame/selfie-segmenter | 1.9000/2.9000 | 2.0000/2.6000 | 0.0187/0.0687 | 0.0350 |
| 1280x720/rgba/mediapipe | 0.0000/0.1000 | 0.1000/0.1000 | 0.0125/0.0500 | 0.0008 |
| 1280x720/rgba/selfie-segmenter | 6.0000/8.3000 | 6.0000/7.8000 | -0.0813/0.0750 | 0.0262 |
| 1280x720/video-frame/mediapipe | 0.0000/0.1000 | 0.0000/0.1000 | 0.0031/0.0063 | 0.0000 |
| 1280x720/video-frame/selfie-segmenter | 5.8000/8.8000 | 5.9000/7.7000 | -0.0281/0.1063 | 0.0445 |
| 1920x1080/rgba/mediapipe | 0.0000/0.1000 | 0.0000/0.1000 | 0.0250/0.0375 | 0.0000 |
| 1920x1080/rgba/selfie-segmenter | 13.5000/18.2000 | 14.0000/17.4000 | 0.3438/0.7875 | 0.0666 |
| 1920x1080/video-frame/mediapipe | 0.0000/0.0000 | 0.0000/0.1000 | 0.0094/0.0250 | 0.0001 |
| 1920x1080/video-frame/selfie-segmenter | 13.4000/17.7000 | 13.4500/17.9000 | -0.1219/1.0187 | 0.5714 |

## draw raw sample aggregation

| Configuration | Off median/p95 ms | On median/p95 ms | Paired mean delta median/p95 ms | Delta variance ms² |
| --- | --- | --- | --- | --- |
| 640x480/rgba/mediapipe | 0.2000/0.8000 | 0.1500/0.9000 | -0.0062/0.0375 | 0.0007 |
| 640x480/rgba/selfie-segmenter | 0.2000/0.8000 | 0.2000/1.0000 | 0.0625/0.0875 | 0.0019 |
| 640x480/video-frame/mediapipe | 0.0000/0.8000 | 0.0000/0.7000 | 0.0344/0.1687 | 0.0057 |
| 640x480/video-frame/selfie-segmenter | 0.1000/0.7000 | 0.1000/0.8000 | 0.0125/0.0375 | 0.0003 |
| 1280x720/rgba/mediapipe | 0.2000/2.5000 | 0.2000/2.4000 | -0.1250/0.0125 | 0.0107 |
| 1280x720/rgba/selfie-segmenter | 0.4000/2.5000 | 0.4000/2.4000 | 0.0125/0.1250 | 0.0032 |
| 1280x720/video-frame/mediapipe | 0.1000/2.0000 | 0.1000/2.2000 | 0.0281/0.2687 | 0.0116 |
| 1280x720/video-frame/selfie-segmenter | 0.3000/2.3000 | 0.3000/2.3000 | 0.0313/0.1312 | 0.0024 |
| 1920x1080/rgba/mediapipe | 0.4000/5.3000 | 0.4000/5.2000 | -0.0125/0.0625 | 0.0017 |
| 1920x1080/rgba/selfie-segmenter | 0.8500/5.6000 | 0.9000/5.7000 | 0.0375/0.2500 | 0.0192 |
| 1920x1080/video-frame/mediapipe | 0.1000/5.1000 | 0.1000/4.9000 | -0.0188/0.1062 | 0.0036 |
| 1920x1080/video-frame/selfie-segmenter | 0.9000/5.3000 | 0.9000/5.4000 | 0.0125/0.0688 | 0.0010 |

## hud raw sample aggregation

| Configuration | Off median/p95 ms | On median/p95 ms | Paired mean delta median/p95 ms | Delta variance ms² |
| --- | --- | --- | --- | --- |
| 640x480/rgba/mediapipe | 0.0000/0.0000 | 0.7000/1.0000 | 0.7000/1.0000 | 0.0225 |
| 640x480/rgba/selfie-segmenter | 0.0000/0.0000 | 0.7500/3.7000 | 0.7500/3.7000 | 1.7369 |
| 640x480/video-frame/mediapipe | 0.0000/0.0000 | 0.4000/1.7000 | 0.5250/1.0000 | 0.0662 |
| 640x480/video-frame/selfie-segmenter | 0.0000/0.0000 | 0.5000/0.8000 | 0.5000/0.7000 | 0.0169 |
| 1280x720/rgba/mediapipe | 0.0000/0.0000 | 0.7000/0.9000 | 0.7000/0.9000 | 0.0119 |
| 1280x720/rgba/selfie-segmenter | 0.0000/0.0000 | 0.6500/0.8000 | 0.6500/0.8000 | 0.0069 |
| 1280x720/video-frame/mediapipe | 0.0000/0.0000 | 0.3500/0.8000 | 0.3500/0.5500 | 0.0092 |
| 1280x720/video-frame/selfie-segmenter | 0.0000/0.0000 | 0.7000/4.0000 | 0.7750/2.2500 | 0.4231 |
| 1920x1080/rgba/mediapipe | 0.0000/0.1000 | 0.6500/0.8000 | 0.6000/0.8000 | 0.0169 |
| 1920x1080/rgba/selfie-segmenter | 0.0000/0.0000 | 0.7500/0.9000 | 0.7500/0.9000 | 0.0125 |
| 1920x1080/video-frame/mediapipe | 0.0000/0.0000 | 0.4500/0.7000 | 0.4500/0.5500 | 0.0092 |
| 1920x1080/video-frame/selfie-segmenter | 0.0000/0.0000 | 0.7000/4.3000 | 1.9000/2.4500 | 0.5156 |

## Interpretation and bias

The disabled arm omits probe/identity/sample allocation, metric clocks, aggregation, firstdraw bookkeeping, snapshot and details updates. It preserves the same captured source script, full-byte comparisons, real Canvas copies and SDK-double work. It still contains identical external counters/stopwatches and an initial null-details render, so it measures marginal instrumentation cost, not an alternate pristine application binary. No zero-cost claim is made. SDK model compute is synthetic. Draw is actual Canvas API submission and representative overlay work; deferred raster/GPU/composition/presentation are not timed. Overlay label/ripple and full App notifications are outside this workload. RGBA Canvas release is absent from the raw `released` counter, which is explicitly VideoFrame-specific; ownership is independently checked in the unit fixtures.

The synthetic manual scheduler avoids browser timer clamping and queue/RAF gaps. HUD executes every8 simulated30fps frames, not actual4Hz wall-time; for slow RGBA trials this understates the real HUD publication frequency per elapsed second. The report therefore exposes HUD cost separately. Short trials do not fill every store to maximum capacity. Browser clock precision leaves cheap SDK samples at0ms, so near-zero percentages are unstable. Four reviewer pairs give weak tail precision: paired p95 is the maximum of4 means, not an estimated population tail. The author twelve-pair raw results provide a distinct, larger local sample; p95 there is also the maximum pair. Neither is a hardware budget or confidence interval.

Unfavorable results are retained. Author1080p RGBA detector paired mean-delta median/p95 is +3.0719/+19.2188ms; reviewer is +1.5688/+12.2875ms. Reviewer1080p RGBA segmenter is +3.2625/+15.9375ms, whereas the author observed a negative median. Reviewer1280x720 RGBA segmenter is +3.2062/+11.7750ms. Negative medians in other configurations are variation/order/GC observations, not demonstrated speedups. Cheap VideoFrame detector paired medians are +0.0844/+0.0938/+0.1125ms across the three sizes. Measurements demonstrate nonzero overhead and considerable variability; no arbitrary1ms threshold is imposed. Existing processingMs wall measurements naturally include added probes inside their old boundaries, so values can rise while eligibility/estimator semantics stay the same.

## Independent pressure experiment

Node24 pure collection experiment preloads all40 stage/outcome stores with2048 values (81920 retained events);10 snapshot warmups followed by100 retained raw samples. Snapshot-only median/p95 is 4.3707/5.1726ms, maximum9.1793ms. At a hypothetical4Hz this mean is about18.03ms/s (1.80% wall time), excluding React/DOM. This is synthetic worst-retention collection on cloud Node, not a browser measurement or acceptance budget.

The separate integer-work loop alternates8 enabled/disabled pairs of10000 operations. Output work sums match. Per-event paired median/p95 overhead is 0.3425/0.8085 microseconds. Ring truncation is explicit, never a lifetime count. This isolates collection/probe overhead, excludes the SDK, and is not substituted for the browser experiment.

## Replay

From the frozen product worktree, build and serve its existing experiment:

```sh
npx vite build --config docs/verification/pr06/experiment/vite.overhead.config.ts
npx vite preview --outDir .cache/pr06-overhead --host 127.0.0.1 --port 4175 --strictPort
```

Use spaces after CLI option names (`--host 127.0.0.1 --port 4175`). From the review branch:

```sh
node docs/verification/pr06-independent/reproduce-overhead.mjs
python docs/verification/pr06-independent/analyze-overhead.py docs/verification/pr06-independent/review-overhead.json review-overhead-analysis.json
npx tsx docs/verification/pr06-independent/snapshot-pressure.ts
```

The runner deliberately names `/tmp/pr06-product` for exact-source provenance; create that frozen worktree as recorded in REPORT before replay. All raw output is reviewer-only. Physical camera, model/GPU, audio, mobile/background hardware and public deployment remain unverified.
