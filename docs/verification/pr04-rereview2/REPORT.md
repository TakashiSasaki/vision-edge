# Independent re-review: residual PR04-F01 correction

**Verdict: PASS WITH CONDITIONS. PR04-F01 is independently closed at this
checkpoint; PR04-F02's closure is preserved. No new blocking product defect
was found.** Conditions below are explicit nonblocking semantic/performance/
validation limits, not unfinished corrective work or permission to merge/deploy.

| Source binding | Exact value |
| --- | --- |
| Reviewed PR #7 HEAD / direct evidence-PR base | `d92b7b40c7d2dc11572147d052f69f11657540e2` |
| Actual GitHub and local tree | `e357f512a371df5c86950913cf3a43208f0321ce` |
| Correction commit | `316f149d68e08e50705e00869455835a28ea483d` |
| Main base | `f1a0d0bbe343b666c75eae053340ceefc6730d5d` |
| Prior rejected checkpoint | `f81ce6102ec4f459ef7a0c3fb7e55d126b157be5` |
| Preserved FAIL PR #8 | `c632cf0cbce82b48503540651120c88fff75c8ef` |
| Preserved FAIL PR #9 | `232b2104a2564d96a01e354bf7aeee7a380733f2` |
| New evidence branch | `review/pr04-independent-rereview2` |

This is the same independent reviewer continuing the owner's authorized
review-response cycle, not implementation-author acceptance. New clean checkout
`/workspace/pr04-rereview2`; separate complete-source replay checkout
`/workspace/pr04-rereview2-replay`. No repository/ancestor AGENTS.md or repository
.agents instructions were found; workspace instruction directories were empty.
The coordinate-codex-review-cycle skill was read after it became available and
used to preserve fixed-SHA role/evidence separation, not as product evidence.

The target matched at initial and resumed reads. No equivalent new review was
published during the platform-capacity interruption. Partial logs were preserved,
no model change or alternate environment was used, and review resumed from the
saved checkout. Final source movement and published evidence HEAD/tree are
recorded in the new PR description to avoid self-reference. Prior FAIL PRs and
their evidence remain unchanged. Only a new evidence directory is proposed;
no product/source/test/workflow/settings fix, force-push, merge or deployment.
GitHub publication uses the connected owner's identity; independence here is
separation of reviewer execution from implementation authoring, not a claim of
approval by a different human/GitHub account.

**Independent closure, not just successful CI.**

The remaining failure in PR #9 was a running playhead repeatedly submitted as
new source input. Actual-app observation had shown 140 SDK calls in 2.3 seconds
for one native frame, with continuously renewed input age. The new compatibility
path stops using currentTime as its identity when neither immutable VideoFrame
nor usable positive playback-quality counts exist.

At [frameScheduler.ts:31](https://github.com/TakashiSasaki/vision-edge/blob/d92b7b40c7d2dc11572147d052f69f11657540e2/src/services/frameScheduler.ts#L31),
it compares every sampled RGBA byte and dimensions. At lines 80–91 it samples
the live image and copies those precise pixels to a private execution canvas.
Deduplication at line 98 and observation at line 115 use the sampled data, while
the actual SDK receives that owned canvas at line 148. No hash, sparse pixel
selection or timestamp proxy is used on this path.

I reran PR #9's native still-source, actual-app and forced delayed-execution
probes against the **complete actual corrected commit**. All now pass with only
the necessary Canvas observation-constructor adaptation; assertions are retained.
The original native first-frame probe still gives rVFC **1** / RAF **1**.

More importantly, four newly constructed independent native App → Canvas →
Engine → SDK-boundary cases pass for **missing and zero quality, in both rVFC
and RAF paths**, without inserting a positive quality capability:

- One pre-presented usable image causes one successful inference.
- A new native presentation with identical content causes no extra inference.
  Another identical presentation near the input-age deadline also does not
  renew content age: inference remains unmeasured, processing duration expires
  and the HUD reports stalled content. Actual overlay redraw FPS remains numeric.
- Swapping colored pixels with the same histogram/sums produces a second
  inference; changing 16×8 to 8×16 with the exact same byte sequence produces a
  third. **Every byte and the dimensions of actual SDK input match a separately
  captured live-video reference**, not merely callback metadata.
- One SDK exception consumes its attempted image; identical subsequent content
  is not retried. Changed content recovers. A readback exception stops visibly,
  zeros the sampler and retries explicitly without another camera/model
  creation. Stop ends native tracks and zeros all recorded sampler/owned canvases.

These cases use a new direct Canvas observer that reports source timestamp
**null** for canvas input, avoiding synthetic source PTS. See
[native observations](logs/independent-app-observations.json),
[final log](logs/independent-pixel-app-aged-identical.txt) and
[probe](vectors/pixel-app.spec.mjs). Four passing cases each recorded one camera
request, one model creation, one deliberately failed SDK call and five successful
SDK calls across the complete change/error/retry sequence.

Seventeen additional independent unit cases verify all-byte/dimension identity,
late image capture, failed-image no-retry/recovery, invalidation during copying,
read/length/context/copy failure, unusable quality returns (undefined/null/zero/
negative/NaN/Infinity), foreign tokens, old Promise ownership and replacement
sampler safety. A thousand duplicate arrivals produce exactly one execution and
one input event, zero pending timers and two created canvases; all are zero-sized
after stop. These exercise the real production scheduler; the pixel-capable DOM
boundary is separately defined and expected arrays are independently calculated.

F01 is closed on this evidence. F02 remains closed through the shared bootstrap
at line 191, the original pre-presented-frame probe, current native equal-time
camera-restart/bootstrap-race cases and the new capability-absence startup cases.
Stable per-ID decisions and all conditions are in [findings.json](findings.json);
[the matrix](requirement-test-matrix.md) retains all ten acceptance contracts.

**Observation adaptations were audited explicitly.**

PR #9's old observer could construct a VideoFrame from live video, but Canvas
needs an explicit constructor timestamp. In three copied observation files, only
that constructor option was added for Canvas, with synthetic timestamp zero;
assertion/exit-condition lines were compared and unchanged. This zero is not
source identity or SDK/GPU timing. Original/adapted hashes and exact adapted
sources are retained.

Raw unchanged PR #8 abstract fixtures now give **8 pass / 21 fail**: their canvas
stubs cannot do the newly required readback, so capture fails before many old
assertions are reached. That raw result is not concealed. The author's separate
logical adapter adds a positive mocked quality capability and delegates to the
real scheduler; its **29/29** unchanged assertions pass on independent replay.
It selects another capability path, so I **do not use it as evidence for
missing-capability correctness**. It supports the prior logical lifecycle,
metric/duration and stage-fault assertions. Actual pixel-path closure rests on
the native and new independent cases above. See
[observer-adaptations.md](observer-adaptations.md), including precise changes.

The new app fixture initially requested a frame without repainting its producer.
Those four fixture timeouts are retained. Repainting identical bytes before the
explicit request fixed the setup without weakening the single-inference/aging
assertions. A final strengthened run presents identical content near expiry.
An initial lint error in the new review recorder was fixed without suppression;
final evidence-inclusive lint/typecheck and all meaningful product assertions
pass. No existing test assertions or product code were changed by this reviewer.

**Nonblocking conditions and measured cost.**

1. **Content change is the compatibility identity.** Fresh native frames with
   byte-identical pixels deliberately coalesce. On that path input age/stalled
   means no changed sampled content, not proof that the camera stopped emitting
   frames. The author source/report documents this accurately. Retain that
   distinction; inference FPS is not camera FPS, and no source timestamp is
   invented for canvas input.
2. **High-resolution equal-content polling is expensive.** Full readback and
   exact comparison are O(width×height) on the main thread, although application
   history/storage is bounded. In a local synthetic 20-poll probe:

   | Dimensions | Mean equal-content poll |
   | --- | ---: |
   | 64×64 | 0.475 ms |
   | 640×480 | 16.165 ms |
   | 1280×720 | 43.940 ms |
   | 1920×1080 | 94.015 ms |

   These are observations in this environment, **not universal latency bounds,
   model/GPU measurements or a device real-time guarantee**. Each run performs
   22 readbacks (bootstrap/execution plus 20 polls), one inference, one input
   notification, one sampler and one owned canvas. The owned canvas is already
   zero-sized before stop; both are zero-sized afterwards.
3. **Real-device/model limits remain.** Physical camera/permission UX, actual
   downloaded model inference/GPU execution/utilization, audible TTS, hardware
   background behavior, cross-platform accuracy and deployment equivalence are
   not certified. SDK-only timing is optional and its absence is not a finding.

The source retains only consumed/observed/current pixel snapshots, one reused
sampler/context and at most one token-owned execution image; no FIFO/history is
introduced. Constant-count state can still consume O(width×height) bytes, and
readback allocates temporary arrays. Total browser/SDK/GC heap limits were not
measured; zero canvas dimensions demonstrate disposal intent/resource sizing,
not a universal heap guarantee. Capture/compare cost remains outside the
accurately named complete detectVideoFrame-call duration; it must not be
advertised as total pipeline latency.

These limits do not violate a specified latency threshold, and this review does
not invent a new budget or request Worker migration. PR04-R01's existing
nonblocking future async lock-wakeup limitation is retained: production work is
synchronous; future asynchronous work needs another ownership/wakeup review.

**Execution and source inspection.**

| Check | Actual outcome |
| --- | --- |
| Fresh locked npm ci, typecheck, lint, unit/component tests, production build | All exit 0; **116/116**, nine files. |
| Original main source suite | Exit 0; **61/61**, seven files; unchanged clean baseline with identical locked dependencies. |
| Full ordinary native browser suite | Exit 0; **14/14** using explicit system Chromium. |
| Fault controls | Exit 0; **7/7**: two healthy successes, five diagnostic-specific expected child failures. |
| Full main diff and correction diff whitespace | Exit 0. |
| Prior PR #9 capture/ownership vectors | Exit 0; **11/11**, unchanged. |
| Prior PR #9 native/app/forced probes, observation-only adaptation | Exit 0; **6/6**, **3/3**, **6/6**. |
| Original native first frame | Exit 0; one inference in each path. |
| New independent pixel unit/native app tests | Exit 0; **17/17** and **4/4**. |
| Resolution/resource probe | Exit 0; four resolutions, all structural assertions pass. |
| PR #8 raw / disclosed logical adapter | Raw exit 1, **8/29**; adapter exit 0, **29/29**; scoped as explained above. |

Commands, exit codes and raw logs are in [commands.json](commands.json).
Linux x86_64, Node 24.19.0, npm 11.9.0, Playwright 1.64.1, system Chromium
154.0.8037.57. New source hashes match all **56** frozen non-evidence files
independently. The correction has **67 paths: 58 evidence, three product and six
test/fixture files**; the full accumulated PR has **219 changed paths**. All
correction source/evidence and the inherited call graph were inspected. Prior
inventories remain valid for unchanged paths. No dependency/lock/CI, unrelated
tracking/coordinate/dictionary/TTS change, skipped assertion, broad lint
suppression, product artifact leak or historical evidence rewrite was found.

**Exact final-HEAD hosted verification and isolation.**

Independently read [run 40081669748 / job 120836331099](https://github.com/TakashiSasaki/vision-edge/actions/runs/40081669748/job/120836331099),
commit check-run, every step and raw downloaded logs. All bind to the literal
`d92b7b40c7d2dc11572147d052f69f11657540e2` checkout. Completed/success logs show
Node 24.19.0/npm 11.17.0, **116 tests**, matched Chromium/headless shell
**v1248 / 156.0.8078.4**, **14 browser passes**, all **7 controls**, and the
missing/zero-quality native observations each changing from one red input with
expired/stalled metrics to new blue input. Raw decoded log is preserved as JSON;
readable logs have whitespace-only normalization receipts.

Known local matched-browser HTTP 403 access failure was not retried or bypassed.
The selected execution environment was kept; local system-browser results are
separate from hosted matched-browser evidence. No independent physical-model/
GPU claim is inferred from either.

Normal production JS contains neither author e2e observer/probe markers nor the
new independent SDK markers. They appear only in their respective explicit test
builds. See [artifact isolation](logs/artifact-isolation.json). There is no
runtime test injection in normal product behavior.

**Outcome and required response.**

No further implementation correction is required for F01/F02 at this frozen
checkpoint. Retain the stated nonblocking conditions and raw/adapted evidence
distinctions. Any new implementation HEAD needs source binding and appropriate
re-review; owner integration/merge/deploy decisions remain outside this review's
authority. Original FAIL reports continue to describe their original SHAs.
