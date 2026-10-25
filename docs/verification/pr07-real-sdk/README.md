# PR07 author checkpoint: actual MediaPipe verification foundation

**Foundation implemented; successful real inference is BLOCKED, not certified.** The selected cloud environment returned HTTP403 CONNECT denial for both exact production model URLs. No model bytes or model SHA256 were obtained, and no bypass, alternate model, SDK double, CPU delegate rewrite or fabricated ready/result was used. This Draft is ready for independent review of the foundation and its honest blocked findings, not a declaration that all PR07 runtime coverage passed.

Phase: author implementation. Parent assigns a different independent reviewer after this frozen checkpoint. No main push, merge, auto-merge, deployment, prior PR14–16 branch/evidence edit or later roadmap work was performed.

## Binding and scope

Latest fetched main remained `9dd0f7998a8c700c2f19c770e31dc790c4774de4`, tree `815f55307389f842d636f0381c206253630baabd`. PR14 is merged; PR15/16 remain open historical review Drafts. All branches and open PRs were checked; no competing real-SDK work was found. GitHub CLI API reads returned Forbidden; the authorized GitHub connector succeeded. No AGENTS.md or relevant checkout `.agents/skills/SKILL.md` was present. [Starting state](starting-state.json) and re-fetched main-CI raw excerpt bind the supplied baseline. Main run40115018649/job120934406523 is completed/success; raw checkout SHA, Node24.19.0/npm11.17.0 and317unit/19browser are verified. That baseline run is not substituted for final PR CI.

Author source freeze: `03762228e55c3db38943013a06e74547d4a3eb9d`, tree `6451a6a56f211cdbaba4c16391672b4f4a8e7daf`; [source binding](source-binding.json). Subsequent publication changes only new evidence under this directory. Final pushed HEAD/tree and exact final CI run/job/checkout are recorded in the Draft PR body and final handoff, avoiding self-hash claims inside their own commit.

All50 protected product/legacy source/config paths match main byte-for-byte, including VisionEngine, scheduler, collector/view/Canvas, camera, SDK doubles and legacy E2E, lockfile, original Vite/Playwright and CI workflow. Only package scripts, isolated new test/config/tool files and this new evidence directory are added. SDK/WASM remain0.10.14; EfficientDet float16/1 and Selfie float16/latest remain the production references; GPU remains the product delegate. No pixel/identity/Worker/frame-thinning/model/tracking/dictionary/TTS change is included.

The new optional command is `npm run test:e2e:real-sdk`. It has its own config, directory, build output and4175 preview, real unaliased SDK imports and default App/engine. Observation records forwarded actual calls/results/postprocessing and matching visible Canvas copies. It cannot fabricate detections/masks/readiness. Read [architecture and reproduction instructions](../../../e2e-real-sdk/README.md) (repository path: `e2e-real-sdk/README.md`). The existing4174/e2e-alias suite remains synthetic evidence.

## Actual execution matrix

| Model | VideoFrame, GPU requested | Owned Canvas, GPU requested | Live HTMLVideoElement, GPU requested | Controlled model HTTP503 diagnostic |
| --- | --- | --- | --- | --- |
| EfficientDet-Lite0 float16/1 | BLOCKED; inference0 | BLOCKED; inference0 | BLOCKED; inference0 | PASS: genuine SDK create attempt, exact503 error + expected task-generation UI, no page crash |
| Selfie Segmenter float16/latest | BLOCKED; inference0 | BLOCKED; inference0 | BLOCKED; inference0 | PASS: genuine SDK create attempt, exact503 error + expected task-generation UI, no page crash |

Successful model initialization, positive/negative semantics, real-result postprocessing/mask rendering, SDK/postprocess/draw metrics under inference, first-draw binding, stop/restart, model switch, stale-result rejection and successful-task disposal are therefore **not executed with actual models**. Assertions exist in the six isolated inference cases, but their source is not evidence that those behaviors pass. No product input incompatibility has been established or ruled out. Their input/API errors will be retained as minimal App-path reproducers if execution becomes possible; corrections require a separate justified product PR.

[Machine matrix](run-0376222/matrix.json), [actual command/environment](run-0376222/environment.json), [Playwright report](run-0376222/playwright.json), [raw command output](run-0376222/command.txt), [timing population](run-0376222/measurements.json). Source0376222 real command exits **2**, child diagnostics exit0,2passed. No skip/xfail or arbitrary millisecond pass threshold. The timing population is empty, not0ms. The initial unfrozen run and development mistakes remain separately preserved; see [setup observations](setup-observations.json) and logs.

Both diagnostic runs serve verified genuine WASM/loader files and licensed PNG inputs while aborting other external browser requests. The actual SDK throws `Failed to fetch model ... (503)`, the App displays the expected model-generation error and retains the video/App, with zero inference calls and no pageerror. This proves the controlled failure boundary; loading bytes alone is not claimed as successful WASM/model initialization or a continuous inference/rendering success.

Runtime: Linux x86_64, Node24.19.0/npm11.9.0, explicit `/usr/bin/chromium` **154.0.8037.57**. Playwright's Desktop Chrome context user-agent advertises156; actual `browser.version()` is separately recorded and authoritative. WebGL reports **ANGLE Vulkan SwiftShader Device (Subzero)**, a software renderer. GPU was requested in observed creation options; no physical GPU use or real inference is proven. Local matched Chromium is unavailable; the inherited403 download restriction was respected without retry. Final hosted CI's matched browser is a separate environment.

## Assets and licensing

[Asset lock](../../../e2e-real-sdk/assets.lock.json) (repository path `e2e-real-sdk/assets.lock.json`), [license decisions](LICENSES.md), [direct acquisition report](acquisition-with-proxy.json), [cached verification](run-0376222/assets.json). SDK module, four CDN WASM/loader assets and person/cup PNGs have reviewed size/SHA256/version/source/retrieval records. CDN bytes equal the installed locked npm counterparts, and input hashes independently equal scikit-image's registry. Model pins remain null and blocked; their mutable/latest identity and model-specific redistribution terms are unknown. No binary assets were committed.

The direct command initially failed to use this environment's proxy; its report is preserved as `acquisition.json`. The corrected standard `--use-env-proxy` command acquired/verifies all six available source assets, checks the npm SDK pin, refuses to request either unpinned/previously denied model and exits2. This is a client configuration correction, not a bypass of model403. Model acquisition failure is explicitly separate from local route execution. Every cache use rechecks size/hash; malformed/stale bytes fail, and null pins are never implicitly enrolled. Future model acquisition requires bounded direct access and a reviewable manifest update before an offline rerun.

Person input is NASA's public-domain Eileen Collins photo; cup input is Rachel Michetti's CC0 coffee image. Fixed images become a controlled native512×512/5Hz Canvas.captureStream with one changing corner pixel. Textured and empty controls supplement these meaningful positives. There are no private user photos/recordings or physical camera access. Screenshots are selected diagnostic evidence only. No model or fixture download is added to mandatory CI.

## Validation and evidence protection

| Check | Result |
| --- | --- |
| Locked install | PASS, npm ci with writable `/tmp/vision-edge-pr07-npm-cache`; lock unchanged. Initial default-cache ENOENT exit254 retained in observations. |
| Typecheck / ESLint / production build / diff checks | PASS; development failures retained separately |
| Unit/component |317/317,14files, unchanged baseline; final-source run retained |
| Normal browser E2E |19/19, serial, retries0, system Chromium154; unchanged assertions/waits |
| CSS/JS fault controls |7/7:2normal success +5specific expected fault detections |
| Additional asset foundation controls |7/7 Node tests, separate from317baseline; tamper/missing/unpinned/latest-change/403/no-retry/size bounds |
| Optional real-SDK run | exit2 BLOCKED;2actual-SDK controlled asset diagnostics PASS;6inference scenarios BLOCKED |
| Direct production URL acquisition | six available assets PASS + npm module pin PASS; both model URLs BLOCKED; exit2 |
| Production isolation | actual SDK graph/loader present, no SDK-double or PR07 observer/global markers; no SDK production alias |
| Source/historical evidence |50product/legacy paths unchanged;559old evidence file hashes unchanged |
| Physical desktop/mobile/camera/GPU/audio/public equality/CPU diagnostic | NOT RUN |
| Optional manual real-SDK workflow | NOT ADDED / NOT RUN; mandatory workflow unchanged |

[Command ledger](commands.jsonl) includes actual argv/cwd/time/exit; [logs](logs) retain failed development checks, corrected checks and final-source results. Nonsecret browser/acquisition environment overrides are recorded by run metadata and setup observations. Ordinary browser/controls ran at source8b92b46 in detached `/tmp/vision-edge-pr07-regressions`; all product/legacy source paths are identical at0376222. Later source changes affect only this new harness/acquisition/docs. Production build bytes match across the source/disposable build. Final-head hosted CI verifies the final evidence-inclusive commit separately.

Existing `e2e/pipeline-metrics.spec.ts` still writes six PR06 screenshot paths. It ran only in the disposable worktree, never against the protected source checkout; regenerated historical paths there were restored after archiving the ordinary report. [Preservation/bundle comparison](preservation-and-bundles.json) and [559 protected hashes](protected-evidence-before.json) show zero source evidence changes. The new suite writes only `testInfo.outputPath` beneath unique per-run cache directories. Selected immutable PR07 outputs are promoted here. [Normal browser archive](normal-browser-artifacts.tar.gz) preserves that run's report/results; earlier real diagnostic run is preserved separately. No broad output migration or PR05 stabilization was made.

## Limits and reviewer handoff

[Physical desktop/mobile matrix](manual-device-matrix.md) is practical but entirely NOT RUN. It covers permissions, actual models/foreground/background, input paths, stop/switch/resize, background transitions, explicit audio and bounded performance samples. Public application code equality is separate and not tested. Cloud requested-GPU/software-renderer behavior cannot certify physical cameras or GPUs.

Initialization, warmup and steady populations are separate in the harness; none has real-inference samples here. PR06's5s/2048event/300quantile population, missing/truncated accounting, inclusive-stage limits and immutable-image-only first-draw rule remain unchanged. Observer copies/readbacks add cost and are disclosed. No CPU/GPU utilization, GPU internal timing, physical camera FPS, physical display completion or camera-to-display estimate is inferred. Never sum overlapping wall intervals or relabel these diagnostics as model performance.

Inherited PR04 full-RGBA scan/copy cost and identical-content coalescing, PR05-T01 timing history and PR16's scoped conditions remain. PR16 closed an explicitly injected malformed-statistics fault; it did not establish a natural collector path or certify real-device behavior. No new ordinary browser failure occurred in this author run, and no old failure was reclassified.

Independent reviewer: inspect source0376222 plus final evidence-only HEAD/tree, verify real import/default App/engine and observer forwarding, byte pins/provenance and null-model refusal, exit/matrix semantics, input-path assertions and teardown, normal build exclusion, and protected hashes. Reproduce the two diagnostics with genuine cached assets and external browser network blocked. Do not claim the six blocked inference paths pass until reviewed model bytes and a real run exist. Parent assigns the independent reviewer; author does not self-declare independent PASS and ends at this published Draft checkpoint.
