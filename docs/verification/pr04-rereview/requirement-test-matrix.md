# Re-review acceptance matrix

Frozen target `f81ce6102ec4f459ef7a0c3fb7e55d126b157be5`, actual GitHub tree
`d15536393c00c55c16c8654612d13d8e7e5692a7`.

| Contract | Independent evidence | Outcome |
| --- | --- | --- |
| 1: arrivals, first frame, same-frame dedup, latest-only, readiness | Original vectors unchanged 29/29; first-frame native 1/1; forced delayed actual-input probes; six capability/path combinations; actual App static-source controls. | **FAIL: residual PR04-F01 in currentTime-only compatibility. PR04-F02 closed.** Immutable and quality-supported controls pass. |
| 2: model/camera/scheduler generations and token ownership | Original canvas reentry/model/camera/threshold/unmount and StrictMode vectors pass; 11 new capture/async/cancel/foreign-lock/handle cases pass. | Pass within synchronous production scope. R01 future async wakeup risk retained. |
| 3: retain model across camera stop and clearDetections | Original independent checks, retained current component suite and native equal-time camera restarts pass. Engine initialization ownership unchanged. | Pass. |
| 4: failed inference excludes success, releases ownership, no unchanged retry, smoothing repair | Original real engine post-processing/empty-recovery vectors plus new immutable failed-input ownership tests pass. | Pass for tested paths; F01 concerns repeated successful compatibility work. |
| 5: draw/TTS/parent/capture exceptions | Original independent six-stage exceptions/retry pass; new capture failure/cancellation tests and retained capture UI case pass. | Pass for tested synchronous paths. |
| 6: required measurements and complete detectVideoFrame duration | Original independent array oracle, success/draw refusal cases, detector 16 ms SDK+conversion and segmenter 18 ms SDK+cleanup vectors pass unchanged. | Arithmetic and call-duration semantics pass. Capture cost is outside named engine-call duration, not claimed as total pipeline latency. |
| 7: finite windows, minima, expiration and actual input aging | Original independent irregular/low-rate/idle/resume/clock/boundary metrics pass; currentTime-only native still input runs 2300 ms. | Collector passes; **F01 keeps input age fresh with an unchanged source frame.** |
| 8: latest appearance/ripple, bounded parent updates, no lifecycle recreation/lint weakening | Original independent canvas/fault vectors plus 109 current tests; source diff inspected. | Pass for tested behavior; no source lint suppression or camera/model recreation added. |
| 9: actual production scheduler/engine, native input and SDK isolation | 8 normal browser cases, exact hosted raw logs, independent SDK boundary in a separate explicit build, artifact marker checks across normal/e2e/review outputs. | Required positive paths pass; capability absence is a remaining blocking boundary. No real model/GPU certification. |
| 10: baseline and scope | Main 61 tests rerun; current 109 tests, 8 browser cases, 7 controls; full and correction diff checks; 86-path correction inventory (77 evidence, 5 tests, 4 product). | Pass. Original review PR #8 untouched; no product fixes or unrelated changes made. |

New native still-source matrix: **5 pass / 1 fail** (RAF currentTime-only repeats).
New forced-delay actual-input matrix: **4 pass / 2 fail** (currentTime-only in
both scheduler paths repeats). New actual App matrix: **2 pass / 1 fail**.
Each failed case supports the same original finding PR04-F01; no duplicate
finding IDs were introduced.

An exploratory native generated-track timestamp-collision test failed during
source setup before its product assertions. It is recorded as inconclusive and
is excluded from pass/fail product counts.
