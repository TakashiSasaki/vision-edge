# Physical-device handoff — all NOT RUN

Use a separate authorized desktop/mobile session. This cloud task did not access a physical camera, the owner's computer, or a public deployment. Record exact source/build equality separately before attributing a public-app observation to this commit.

Record browser/OS/device, HTTPS origin, source SHA/tree/public bundle hashes, SDK/WASM/model SHA256 (especially mutable Selfie `latest`), renderer/delegate evidence and unknowns, input dimensions, current model, permission state, exact actions, expected/actual result, screenshot where licensed/consented and logs. Avoid saving private faces/camera recordings. Use a consenting tester or the licensed fixed fixtures for semantic checks.

| Scenario | Desktop Chrome/Edge physical GPU | Desktop Firefox/Safari | Android Chrome | iOS Safari |
| --- | --- | --- | --- | --- |
| Permission allow / deny / dismiss / retry; busy/missing device; no audio capture requested | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| EfficientDet GPU requested, real model ready, recognizable cup/dog/person boxes, finite confidence and valid coordinates | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Selfie model ready, person foreground mask versus background; remove/re-enter person; empty controls | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Initial pre-presented frame, native VideoFrame support, observed SDK input; record actual fallback conversion where necessary | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Stop/restart repeatedly (bounded3 cycles), camera indicator/tracks released, same model retained; model switch closes old task | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Front/rear switch, portrait/landscape, intrinsic resize and letterbox clicks | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Tab background/foreground, screen lock/unlock; no stale-result overlay; recovery/visibility outcomes | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Details panel toggles retain camera/model; genuine SDK/postprocess/draw samples, immutable first-draw latency; live video unbound | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Default audio off, explicit tap/manual TTS and selected Japanese voice audible, cooldown/stop; no unsolicited audio | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Controlled asset failure yields model-specific load diagnostic and recovery after intended retry | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Bounded performance sample: init separately, first inference warmup, then30 successful steady frames; retained/missing/truncated counts and full distribution | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

No universal millisecond pass threshold. JS SDK wall time is not internal GPU duration; requested GPU may be software. Never infer CPU utilization, GPU utilization, physical camera FPS, physical display completion or camera-to-display latency from these metrics. Cloud samples supply no physical-device guarantee. Do not enable CPU silently to make a product GPU path pass. A product incompatibility needs an unchanged minimal reproducer and a separate proposed correction, not a hidden foundation repair.

Inherited PR04 full-RGBA readback/all-byte comparison cost and identical-content coalescing remain. PR05-T01 timing sensitivity/history remains. PR16's malformed-statistics synthetic fault closure does not establish an ordinary malformed collector path and does not certify these real-device scenarios.
