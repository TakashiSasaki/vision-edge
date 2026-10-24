# PR06-F01 HUD preparation cost

The new validation/copy runs only at HUD publication, not per inference/capture/draw. This production-compiled browser comparison measures the changed boundary explicitly: old frozen PipelineDetails receives the real raw snapshot; corrected publication validates/copies the snapshot then renders the corrected memoized details. Both arms use the exact same unchanged PipelineMetrics implementation, React installation, CSS and deterministic valid input profile. The frozen before source differs only by its relocated metrics import. Healthy rendered HTML is identical after equalizing native disclosure state.

144 measured runs: six profiles (fresh/300/truncated × closed/open), 12 paired trials per profile, 32 real snapshot+React/DOM updates per trial, 384 updates per arm/profile, 2304 per arm total. Warmup16 per arm/profile excluded. Order alternates before/after then after/before (six pairs each ordering). A fresh snapshot object is created each update; no memoized no-op is timed. Actual snapshot/render counts match. The 300 profile populates all40 stage/outcome cells; the truncated profile sends2051 events per cell, retaining2048 with missing-duration entries and latest300 quantiles. This deliberately bounds the row reader while independently exercising unchanged expensive snapshot preparation.

All values are JS wall milliseconds. Median/p95 pool384 operation samples per arm/profile. Paired mean delta is after minus before, summarized over12 per-trial mean differences; nearest-rank p95 with12 pairs is the maximum, and variance is population ms². Zero due to clock quantization is retained. This is not CPU/GPU utilization, physical display latency or a device guarantee. Native disclosure state is kept equal, and no other test suite ran concurrently with the final sweep.

| Profile / disclosure | Prepare before median / p95 | Prepare corrected median / p95 | Paired prepare mean delta median / p95 / variance |
| --- | --- | --- | --- |
| fresh / closed | 0.10000 / 0.30000 | 0.10000 / 0.30000 | 0.01406 / 0.20625 / 0.00375 |
| fresh / open | 0.10000 / 0.20000 | 0.10000 / 0.20000 | 0.00469 / 0.07187 / 0.00101 |
| 300 / closed | 0.10000 / 0.30000 | 0.10000 / 0.30000 | 0.00625 / 0.03125 / 0.00193 |
| 300 / open | 0.10000 / 0.30000 | 0.10000 / 0.30000 | -0.00156 / 0.10313 / 0.00202 |
| truncated / closed | 0.20000 / 0.30000 | 0.20000 / 0.30000 | 0.01250 / 0.05625 / 0.00063 |
| truncated / open | 0.15000 / 0.30000 | 0.20000 / 0.30000 | 0.01094 / 0.03438 / 0.00042 |

| Profile / disclosure | Snapshot+prepare before median / p95 | Corrected median / p95 | Paired total mean delta median / p95 / variance |
| --- | --- | --- | --- |
| fresh / closed | 0.10000 / 0.30000 | 0.10000 / 0.30000 | 0.00937 / 0.20625 / 0.00369 |
| fresh / open | 0.10000 / 0.20000 | 0.10000 / 0.30000 | 0.00156 / 0.08750 / 0.00150 |
| 300 / closed | 0.80000 / 1.50000 | 0.80000 / 1.40000 | 0.00313 / 0.22500 / 0.03895 |
| 300 / open | 0.80000 / 1.40000 | 0.80000 / 2.10000 | -0.00156 / 1.00312 / 0.08164 |
| truncated / closed | 3.50000 / 4.40000 | 3.50000 / 4.70000 | 0.04219 / 0.52500 / 0.02667 |
| truncated / open | 3.40000 / 4.40000 | 3.50000 / 4.40000 | 0.01719 / 0.38750 / 0.04246 |

## Retention of earlier experiments

The collector/probe, scheduler, engine, Canvas renderer, existing FPS estimator and the actual image/SDK/ownership paths are byte-identical to reviewed04a3530; exact source bindings accompany this report. Their author288 and reviewer96 trials remain evidence at their original heads. Their old HUD/end-to-end values are not represented as measurements of the corrected HUD. This targeted experiment supplies the new changed-path measurement, so the long resolution/model pipeline sweeps were not repeated. No material change was made to measurement work per capture/comparison/SDK/draw or to source/processed/drawn counts; only the fixed40-row HUD reader is added at250ms publication. No arbitrary1ms acceptance threshold or improvement of the image algorithm is imposed.

Both positive/negative deltas, maxima, variance and raw samples remain in hud-cost-final.json/hud-cost-summary.json. Small timing changes, GC, browser scheduling and clock quantization limit attribution and tail precision. The full pressure profile is synthetic and is not a prediction for normal camera rates. The earlier native-open comparison mismatch and prior complete renderer-boundary sweeps are preserved in separate logs/JSON and are not substituted for this publication-boundary sweep.

Reproduce: node experiment/materialize-before.mjs from repository root; npx vite build --config docs/verification/pr06-f01-correction/experiment/vite.config.ts; serve.cache/pr06-f01-hud-build at127.0.0.1:4176; PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium node docs/verification/pr06-f01-correction/experiment/run-hud-cost.mjs; python3 docs/verification/pr06-f01-correction/experiment/summarize-hud.py.
