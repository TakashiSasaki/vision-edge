# Independent corrected-HUD cost assessment

The correction changes only publication/display preparation. Historical author288 and reviewer96 pipeline trials remain at their original heads: collector/probe, scheduler, engine, Canvas renderer, FPS estimator, camera hook, source identity and SDK modules are byte-identical (source-bindings.json). Their original SDK/capture/draw observations remain applicable to those unchanged modules, but old HUD/end-to-end timings are **not relabelled corrected-head measurements**. This report measures the changed path separately.

## Independent browser comparison

The reviewer implemented a separate production-compiled entry and paired runner. Its before component is extracted from04a3530 with only the metrics import relocated, independently hash-verified. Both arms execute the same current unchanged collector, CSS, installed React, real flushSync/DOM work and snapshot calls. The after arm additionally runs the corrected reader before rendering. Counts are incremented at actual snapshot/read/render invocations and checked externally; fresh objects prevent memo no-ops. Disclosure state is equalized outside timing. Exact healthy HTML matches for all six profile/disclosure pairs, including open states independently checked here.

72 measured trials, six pairs per profile,32 updates per trial:2304 operations total,1152 per arm. Each profile warms16 updates per arm (12 warmup trials/192 operations excluded). Pair order alternates before/after then after/before (three of each order). Empty profile has no events; quantile300 has375 events/cell with75 missing and300 measured; truncated has3000 events/cell, retains2048 with partial coverage and latest300 quantiles. All40 stage/outcome cells are exercised. The author300-event profile instead has240 valid values; it is not called an exact match to our quantile300 profile.

| Profile / disclosure | Prepare before median/p95 ms | Prepare after median/p95 ms | Paired prepare mean delta median/p95 ms | Delta variance ms² |
| --- | --- | --- | --- | --- |
| empty/closed | 0.10000/0.20000 | 0.10000/0.30000 | 0.00156/0.02500 | 0.00128 |
| empty/open | 0.10000/0.20000 | 0.10000/0.20000 | 0.00156/0.05313 | 0.00043 |
| quantile300/closed | 0.10000/0.30000 | 0.20000/0.30000 | 0.01250/0.08437 | 0.00185 |
| quantile300/open | 0.10000/0.30000 | 0.10000/0.30000 | 0.02344/0.05938 | 0.00129 |
| truncated/closed | 0.10000/0.30000 | 0.20000/0.30000 | 0.01094/0.05625 | 0.00087 |
| truncated/open | 0.10000/0.20000 | 0.10000/0.20000 | 0.01094/0.02500 | 0.00013 |

| Profile / disclosure | Snapshot+prepare before median/p95 ms | Corrected median/p95 ms | Paired total mean delta median/p95 ms | Delta variance ms² |
| --- | --- | --- | --- | --- |
| empty/closed | 0.10000/0.30000 | 0.10000/0.30000 | 0.00313/0.02500 | 0.00153 |
| empty/open | 0.10000/0.20000 | 0.10000/0.20000 | 0.00469/0.05313 | 0.00044 |
| quantile300/closed | 1.70000/2.40000 | 1.70000/2.60000 | 0.02500/0.44688 | 0.05518 |
| quantile300/open | 1.60000/2.20000 | 1.50000/2.00000 | -0.02188/0.05000 | 0.00671 |
| truncated/closed | 3.80000/5.20000 | 3.90000/5.00000 | -0.00469/0.64375 | 0.15511 |
| truncated/open | 3.80000/4.40000 | 3.70000/4.40000 | 0.02187/0.06250 | 0.00835 |

Raw `read`, `render`, `snapshot`, `prepare` and `total` samples, extrema and population variance are retained. Prepare includes reader + rendering; total adds snapshot. Per-operation pooled median/p95 use192 samples/arm/profile. Paired summaries use six differences of32-update trial means; nearest-rank p95 is the largest pair, not a population-tail estimate. Added stopwatch boundaries/counters apply to both arms. Setup/null render/native disclosure and profile construction are excluded; this is marginal recurring preparation work, not startup or full-App cost.

## Author raw evidence checked independently

The independently written analyzer validates all144 author final trials,4608 equal snapshot/render operations,12 excluded warmup trials, balanced ordering, finite nonnegative raw samples, and snapshot+prepare=total. Recomputed medians/nearest-rank p95/population variance agree with the report. Prepare paired mean-delta medians span -0.00156 to +0.01406ms; the largest author paired-prepare p95 is +0.20625ms. Author300/open total paired p95 is +1.00312ms, retained rather than dismissed. Original native-open setup mismatch and incomplete first timing samples remain disclosed in author historical evidence; they are not substituted for final data.

## Bounded reader / Node cost

Generic recursive proxies with forbidden ownKeys/set traps verify known-field access without enumerating unknown keys, cycles, or mutation. The saturated fixture performs 455 property gets. The output consists of fixed40 owned rows and known primitive fields, plus up to8 outcome counters; no raw source/proxy is retained. The source is byte-equivalent after reading. Bounded here means a fixed number of relevant accesses; arbitrary nonterminating user-supplied getter code is not made preemptible by this reader.

Separate serial Node24 experiment:100 warmups,30 batches of1000 reader copies. Median/p95 **batch mean per copy** is 0.003721/0.009251ms, maximum batch mean0.009916ms. These include a small Node assertion each copy and exclude snapshot sorting, React/DOM/browser work. At4Hz the observed mean would be about0.018560ms/s on this cloud Node fixture; that is not a device prediction. The earlier sample overlapped typecheck and is preserved; the later serial p95 is worse and remains reported.

## Interpretation and limitations

The added recurring reader cost is observable and fixed in size. Independent paired preparation medians span +0.00156 to +0.02344ms/publication, with noisy positive and negative per-trial differences. Independent truncated/closed total paired p95 is +0.64375ms; a negative median there does not demonstrate a speedup. Browser clocks quantize many individual reads to0ms, and scheduling/GC/snapshot variance limits causal attribution. The observed bounded HUD cost supplies no evidence of material hidden overhead on this synthetic cloud fixture; it is not a universal negligible-overhead claim or arbitrary1ms acceptance threshold.

Only the250ms publication boundary adds validation/copy work; no per-capture/comparison/SDK/draw/RAF reader call was added. Browser/control suites completed before the HUD experiment. Short metadata/hash work overlapped the first~0.6seconds; all samples and setup details are retained with no causal noise claim. No extra synthetic camera/GPU/model/audio or display-presentation claim is made. Real physical hardware/mobile/background/public-deployment equality remain unverified.

## Replay

```sh
npx vite build --config docs/verification/pr06-rereview/experiment/vite.config.ts
npx vite preview --outDir .cache/pr06-rereview-hud --host 127.0.0.1 --port 4177 --strictPort
# in a separate terminal
node docs/verification/pr06-rereview/experiment/run.mjs
python docs/verification/pr06-rereview/analyze-hud.py docs/verification/pr06-rereview/hud-independent.json hud-independent-analysis.json
npx tsx docs/verification/pr06-rereview/projection-cost.ts
```
