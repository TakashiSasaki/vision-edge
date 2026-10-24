# Requirement and independent test matrix

Reviewed target: `2f4cdea1bc3e6706f49e37116ede8b1881472f7f`, tree
`aaf506b0b70499da4a9474fc39b062841ed06965`.

| Contract | Evidence and path actually reached | Outcome |
| --- | --- | --- |
| 1: rVFC/fallback, first/equal timestamps, dedup, latest-only, readiness | Independent scheduler vectors call the real scheduler with separate arrival/execution events. Native first-frame stream, native decoded-video race, and real canvas/engine duplicate SDK-input probe. Positive readiness/coarse/equal-session/latest-100-arrivals tests. | **FAIL: PR04-F01 and PR04-F02.** Positive subcases pass. |
| 2: independent generations, adoption/locks, restart/unmount/StrictMode | Independent model disposal, committed camera stop, threshold generation and unmount during SDK; assert no parent result, TTS gate, or inference metric. Async cancellation/foreign token/latest pending tests. Real StrictMode hook cleanup. | Pass for synchronous production path; PR04-R01 is an asynchronous future-path risk. |
| 3: model retained on camera stop and clearDetections | Existing retained integration test independently rerun; new fault recovery and engine state/task-close checks. Static engine resetOutputs changes output generation only. | Pass. All 54 non-evidence source/test/config hashes independently bound. |
| 4: inference errors release lock, no failed FPS, no same-frame retry, repair smoothing | Independent throwing scheduler work; real engine post-processing getter throws after an earlier dog; subsequent empty SDK result remains empty. Four unchanged arrival ticks do not retry. Actual SDK-exception baseline regression retained. | Pass outside duplicate new-frame identity defect F01. |
| 5: stage faults stop visibly or recover, bounded retry/logging | Six independent stage injections: offscreen roundRect, visible drawImage, missing context, parent callback, canSpeakLabel, speak. Each stage is asserted reached, loops stop, explicit retry works, camera/task created once. | Pass. Ordinary error path logs once per scheduler session. Native TTS/audio not certified. |
| 6: successful inference/draw FPS and complete call duration | Separate completion count/span oracle; detector 5 ms SDK plus 11 ms category conversion = 16 ms; segmenter 5 ms SDK plus 13 ms result cleanup = 18 ms. Draw callbacks before a usable result have no sample. | Arithmetic/duration/draw semantics pass. F01 permits repeated image inference samples. |
| 7: bounded metrics, windows/minimum/expiration/clock/current-time aging | Independently computed irregular event rates, 1/1.5 Hz resume, 1000 events with latest 300, inclusive 5000 ms boundary, 1999/2000 ms expiration, 90-second idle, reversal/nonfinite clock, invalid durations, zero-duration success/unknown latency. | Pass. Event-span estimate is explicitly distinguished from fixed-window throughput. |
| 8: current props/ripple, restrained parent/TTS updates, no recreation/lint weakening | Original retained freshness/ripple/bounded-notification tests independently rerun; new six fault retries preserve camera/task; full diff and current-prop refs reviewed. | Pass outside F01 repeated TTS decisions. No broad source-rule suppression. |
| 9: real production scheduler/engine, native stream, explicit SDK test variant, isolation | Existing four browser tests independently executed; positive paths use native captureStream/video, real engine/scheduler and explicit e2e SDK alias. Independently inspected built artifacts: both test markers absent in normal dist, present only in separate e2e output. Native ready-first-frame vector bypasses author fixture entirely. | Required positive coverage exists and passes; startup/race gaps are proven by F01/F02. Matched hosted browser verified, local matched install blocked. |
| 10: preserve baseline and scoped changes | Clean base checkout: 61/61 pass. Frozen target: 86/86 pass, four browser passes including both old browser cases, seven diagnostic controls. All 81 changed paths classified and hashed; geometry/tracking/dictionary/TTS source and CI workflow unchanged. | Pass. No skips/weakened substantive assertions or new binary/generated product artifacts found. |

Independent final Vitest command: **29 cases: 25 pass, 4 fail**, with three
failure assertions for F01 and one for F02. The two native probe commands also
exit **1 for their explicit contract assertions**, not because browser startup
failed. They run the real production scheduler source, transpiled without a
scheduler substitute; their work callback is an observation boundary.

The normal `npm test` configuration intentionally includes only existing
`src/**/*.test.*`. Review probes live under this evidence directory and have
their own explicit config. A green evidence-PR CI therefore does not execute or
close these acceptance findings.
