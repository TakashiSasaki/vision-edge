# PR-04 second correction: independent acceptance matrix

Target `d92b7b40c7d2dc11572147d052f69f11657540e2`, tree
`e357f512a371df5c86950913cf3a43208f0321ce`.

| Contract | Independent evidence | Result |
| --- | --- | --- |
| 1: arrivals, first/equal-session frames, no duplicate inference, latest-only/readiness | PR #9 native/app/forced-delay replays; four new real-app missing/zero capability cases; 17 independent byte/dimension/queue cases. | **F01 closed; F02 preserved.** Pixel fallback deliberately coalesces identical content, including fresh native presentations. |
| 2: model/camera/scheduler generations, token-owned cleanup, StrictMode | Prior 11 capture/ownership cases unchanged pass; 17 new cases include copy-time invalidation, old Promise snapshots, foreign token, replacement sampler and stop/finish ownership. Original 29 logical assertions pass with disclosed quality-fixture adapter. | Pass in synchronous production scope. R01 future async wakeup risk remains nonblocking. |
| 3: stop/restart retains ready model, clearDetections narrow | Four real-app cases recover from readback error with one camera request/model creation; original lifecycle assertions, 116 current tests and native equal-time restart cases pass. | Pass. |
| 4: inference error ownership, no same-image retry, no stale smoothing | Four real-app cases inject one SDK failure, present identical content again without retry, then change content and recover. Old logical/engine smoothing tests retained. | Pass. |
| 5: draw/TTS/parent/capture exceptions recover or stop visibly | Original six stage-fault assertions pass under their explicit logical capability adapter. New actual-app readback failures stop visibly, release samplers, retry and retain resources correctly. Unit cases cover read/length/context/copy failures. | Pass for tested paths; no real audio certification. |
| 6: successful inference/draw FPS and complete engine-call duration | Independent original metric/duration assertions retained; actual app shows one-sample inference unknown, processing expiration, numeric draw FPS while redraws continue. Capture cost is disclosed separately from detectVideoFrame duration. | Pass. SDK-only timing optional; no false camera/GPU timing claim. |
| 7: finite samples/work, named windows, input aging with no new content | Existing independent arithmetic oracle plus 1000 equal-pixel arrivals: one inference/input event, no pending timer, two created canvases. Native fresh identical presentation near expiry still produces expired/stalled metrics. | Pass, with content-age semantics on the pixel path and measured O(width*height) cost. |
| 8: fresh properties/ripple, restrained parent/TTS, no lifecycle recreation/lint weakening | Existing tests and source call graph preserved; own capture-retry tests retain one model/camera; no rule suppression added. | Pass. |
| 9: native stream/video, real production engine/scheduler, explicit SDK doubles, isolation | 14 ordinary browser cases; prior capability native/app probes; four independent actual-app tests with direct Canvas observation and null source PTS; normal/e2e/review artifact marker checks. | Pass within synthetic/native input scope. Matched hosted final-HEAD logs verified. |
| 10: original baseline and scope | Fresh locked install; 116 current tests, 61 main baseline, 14 browser cases, 7 controls, full/correction diff checks. All 67 correction paths and full 219-path PR inventory bound. | Pass. No product fix, historical evidence edit, merge or deploy by reviewer. |

Raw PR #8 replay is **8 pass / 21 fail**, because those unchanged abstract
fixtures cannot perform pixel readback. Its separately inspected quality-
capability adapter gives **29/29** with unchanged assertions; it is not absence-
path proof. See observer-adaptations.md.

Positive independent counts: prior PR #9 unit boundaries **11/11**; adapted
native matrix **6/6**; adapted old actual-app matrix **3/3**; adapted forced
native matrix **6/6**; new pixel unit cases **17/17**; new direct-observer native
app cases **4/4**; four resolution/resource probes pass all structural assertions.
Original native first-frame remains **1 inference in each path**.

Verdict **PASS WITH CONDITIONS**: no open blocking product finding, but retain
the content-coalescing semantics, significant high-resolution CPU cost and
real-device/model/GPU/audio limitations. This is not merge/deployment authority.
