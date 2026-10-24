# PR06-F01 independent re-review

**F01 CLOSED at the corrected frozen HEAD. Scoped PASS WITH CONDITIONS; no new product findings.** The correction meets the requested invalid-statistics isolation, owned display snapshot and recovery contract in independent tests. Remaining conditions are inherited scope/device/validation limits, not a newly discovered blocking defect.

## Binding and preserved history

| Binding | Exact value |
| --- | --- |
| Product PR / actual branch | #14 Draft / `feat/pipeline-performance-instrumentation` |
| Reviewed corrected HEAD | `7056c67cd1f59c8d3e2237b84df05c223d99a25e` |
| Reviewed corrected tree | `815f55307389f842d636f0381c206253630baabd` |
| Author product correction | `1678f8c23f0fdc96cc666600ae1ec3b910743ad2` |
| Product correction tree | `9463090735f5e9c4816286159235b5ba129c68da` |
| Previous reviewed HEAD / tree | `04a3530c6ec9755bb7b89c3e31c1c3e15c9b1ea9` / `7edc57e2c004d8a85502209b0e2541bd8b11ec07` |
| Preserved PR15 evidence HEAD / tree | `3d6bf05fcef72ede7d66d03a2ef34185ea802f86` / `6fe9a2806e377804ea7c99aee828e24fd632b999` |
| Main/base / tree | `18e94a4508a0fa316cf31171789c3c10a0a041c3` / `f8e78b24e0bf27393ca9266c4d64bf039d232516` |
| New review branch | `review/pr06-pipeline-rereview`, direct parent is corrected product HEAD |

Actual PR/head/branch/base/tree, remote heads and duplicate PRs were checked, and are rechecked before publication. Review uses the selected separate cloud environment, the same independent reviewer, and the previously read coordination skill. No AGENTS.md or relevant `.agents/skills` instruction was present. Prior PR15 and historical author evidence remain unchanged. Exact new evidence HEAD/tree and final live readback appear in the new Draft PR and final handoff, avoiding a self-hash claim inside its own commit.

Source bindings cover all product/tests/configuration. The full04a→7056 source diff and the author correction docs, response, raw experiments, tests and workflow were inspected. The product correction changes only the display reader, PipelineDetails, publication line and appended/new boundary tests. No original test assertion was weakened. Collector/probe, scheduler, engine, Canvas renderer, existing FPS, camera, dictionary, speech, coordinate helpers, SDK config/version, packages/lock and workflow are byte-identical to04a. Correction-commit→final-HEAD changes are evidence-only. The correction is not an image/scheduling/model/algorithm optimization.

## Original failure and closure

Exact original PR15 probe/configuration bytes were extracted and hash-verified, then materialized only into disposable fixed-source worktrees. Old04a F01 again produces one pass/one fail, TypeError, videoMounted=false and pendingRAF=0. Corrected7056 passes all31 original cases unchanged. Old FAIL verdict and original PR15 commit/tree remain intact as historical evidence; this review closes F01 only at the new target.

The corrected VisionCanvas publication passes `pipeline.snapshot()` through `readPipelineSnapshot` before React state. It reads only bounded known fields and owns plain primitive display values. Null/missing/malformed rows become unavailable; valid neighbors survive; invalid duration values become null; real0ms is preserved. No raw getter/proxy/object reaches JSX. The view object is fresh each publication, so the existing memoized component can recover even when the producer reuses one raw object. Later raw mutations cannot alter an already published view.

**Qualification remains:** the original one-field `stages=null` injection violates the TS return shape. No naturally occurring collector route to it, or ordinary-use production incidence, has been established. F01 closure concerns the explicitly requested synthetic fault-containment requirement; it does not claim a previously observed physical-device crash was fixed.

## Independent new evidence

Forty new cases pass, independent of author fixture helpers. The pure suite starts from hand-specified40-row partial statistics, then validates all cells, finite/invalid duration fields, counts/missing/quantile consistency, bad nested containers, safe zero/unexecuted values and recovery. Actual collector snapshots independently exceed300 and2048 events, retain missing durations/truncation, preserve overflowed null sum/mean and valid maximum, and respect exact5000ms expiry followed by zero recovery.

Generic source proxies forbid enumeration and writes, provide unknown/cyclic fields, and count relevant property accesses. Saturated input is read455 times in the measured fixture; no raw unknown-key traversal, recursive graph walk, source mutation or source/proxy retention occurs. Counter getters are read once, hostile object coercion is avoided, throwing accessors/descriptors and revoked proxies are caught. Frozen plain sources validate without writes; every owned row/outcome/results object is independent of the source. These are bounded access claims, not a promise to preempt arbitrary nonterminating getter code.

Nine live payload paths (null root/stages, array stage, missing row, toxic duration, invalid counters, throwing stage getter, throwing results descriptor, revoked root) use the actual VisionCanvas, camera hook, scheduler, VisionEngine and Canvas bridge with controlled camera/SDK/VideoFrame fixtures. Each survives four faulty publications: same video, two RAF callbacks, five SDK calls, four successful result notifications, unchanged ready model/request and one camera/model creation; no track/task closes early and every input owner closes once. Valid snapshots restore displayed0.00ms, another SDK call proceeds, stop cancels loops/releases the track, restart runs SDK again without model recreation, unmount closes tracks/inputs and explicit model disposal closes the task once. A tenth live case verifies the same reused raw object can go invalid→valid→invalid→valid, and raw mutation between publication and React rerender cannot poison the view.

No new finding remains. `findings.json` records F01's original severity, expected/before/after behavior, locations, exact target, reproducers and qualification. `matrix.md` maps all required re-review areas to evidence.

## Validation and exact product CI

| Validation | Independent result |
| --- | --- |
| Locked install | npm ci, exit0; Node24.19.0/npm11.9.0; lock unchanged |
| Typecheck/lint | fixed source and evidence-inclusive, exit0 |
| Normal unit/component |317/317 across14 files |
| Focused pipeline/display/Canvas |188/188 |
| Original unchanged reviewer suite |31/31 corrected; old F01 control reproduces failure |
| New independent suite |40/40 |
| Production build / isolation | exit0; no SDK fixture/experiment markers in normal JS |
| Browser |19/19, system Chromium154.0.8037.57, serial, retries0 |
| CSS controls |7/7: two intact successes, five specific expected-failure child exits1 |
| Source preservation / diff | evidence-only; raw bindings retained |

Ordinary tests/build/browser ran in detached `/tmp/pr06-rereview-product` at7056. Locked install is shared by symlink from the exact lock-installed reviewer workspace. Existing screenshot-producing tests ran only in the disposable worktree; new outputs were copied into this new evidence directory and its tracked screenshots restored. Original probes are preserved separately from ordinary src-only test discovery. Command vectors/cwd/timestamps/exits, all failed setup commands, raw browser archive/screenshots and experiment samples are retained. No ordinary browser failure/retry occurred. The setup path error is distinguished from the real old F01 reproduction; no assertion/wait/timeout was weakened.

Final-product hosted **run40110781409/job120921859541 completed/success**, independently fetched through GitHub. Raw logs explicitly show checkout ref, fetch, checkout and git-log SHA7056c67cd1f59c8d3e2237b84df05c223d99a25e. Actual steps pass locked install, typecheck, lint,317unit, build,19browser,7controls and artifact upload. Hosted Chrome for Testing156.0.8078.4/Playwright v1248 is distinct from system Chromium154 used locally. Prior local matched-browser403 was respected without download retry or bypass. Evidence-PR CI is separate and not substituted for product CI.

## Overhead and bounded work

The reader loops over exactly10 known stages×4 outcomes/eight known result counters; it has no growing store and is called only at the existing250ms publication. No per-capture/SDK/comparison/draw work changes. Projection input/output ownership and bounded reads are independently checked.

The reviewer created a separate production-compiled paired HUD entry, frozen-before source copy and runner. Seventy-two measured trials/2304 updates cover empty/exact300-valid-quantile/truncated profiles and both native disclosure states; all healthy HTML is exactly equal. Actual snapshot/read/render counts match and both arms use fresh snapshots. Twelve warmup trials are excluded; six pairs/profile balance both orders. Independent analysis recomputes all operation medians/nearest-rank p95, trial mean differences, extrema and population variance. Independent paired preparation mean-delta medians span +0.00156 to +0.02344ms/publication. Unfavorable truncated/closed total paired p95+0.64375ms and negative noisy differences are retained; negative values do not establish speedups.

All144 author final HUD trials/4608 updates were independently reanalyzed; totals/counts/order/statistics agree. Old author288 and reviewer96 pipeline trials remain at their original heads, justified for unchanged imported pipeline modules by exact byte bindings. **Their old HUD/end-to-end values are not measurements of the corrected HUD.** The changed path has separate new measurements. A separate serial Node reader experiment retains30 batch means,1000 copies/batch after100 warmups; median/p95 mean-per-copy0.003721/0.009251ms. This excludes snapshot/React/browser and includes assertion cost; it is not a single-operation tail estimate or device budget. Earlier concurrent sample and its better p95 remain preserved; the worse serial p95 is reported.

See `overhead-report.md` for all tables and raw data. Clock quantization, GC/scheduling variance, six/twelve-pair tail limitations, fixture/profile differences, startup exclusions and brief metadata-read overlap are explicit. There is no arbitrary1ms threshold or universal negligible-overhead claim. The bounded recurring cost observed here supplies no evidence of material hidden overhead in this synthetic cloud scope; physical-device behavior remains unverified.

## Conditions and handoff

No physical camera, real model/GPU, audible speech, mobile/background hardware or deployed-public-version equality is certified. PR04 expensive RGBA/full-byte comparison and identical-content coalescing limits remain. PR05 T01 initial timing failures/traces and unmodified serial reruns remain in immutable [PR13 evidence](https://github.com/TakashiSasaki/vision-edge/tree/9c0637f7671f0e0bdac1f782516ef414597207f3/docs/verification/pr05-rereview); no new failure is labelled T01 and concurrency causation is not asserted.

Publish this checkpoint as a new Draft directly on the corrected PR14 branch, preserving PR15. No product repair, original evidence/test/workflow edit, merge, auto-merge, deploy or next-phase action. F01 is closed; no further author correction is requested by this review. Parent may assess this scoped checkpoint within existing authorization. Any later product movement requires new SHA-specific assessment. Exact new evidence commit/tree and pending evidence-CI run (if any) are included in the final handoff. No denied or uncertain write is outstanding at this report checkpoint.
