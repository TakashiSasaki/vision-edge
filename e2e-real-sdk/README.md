# Real MediaPipe integration verification

This is an optional, explicit path. Mandatory CI retains the offline SDK-double suite. Neither the ordinary `test:e2e` command nor CI discovers this directory. Product source, delegate, models, dependencies and lockfile are unchanged.

From repository root, with Node24/npm11:

```sh
npm ci --cache /tmp/vision-edge-npm-cache
npm run test:e2e:real-sdk:assets
npm run test:e2e:real-sdk:acquire
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run test:e2e:real-sdk
node --test scripts/real-sdk/assets.test.mjs
npm run build
node scripts/real-sdk/audit.mjs
```

Omit the executable override to require the Playwright-matched browser. A system override is explicit and recorded; do not call its context user-agent the actual installed browser version. `runtimeEvidence` also records `browser.version()`. Use an already available matched browser; respect download denials.

The acquisition command uses Node's normal environment proxy support, 30s/asset timeout, maximum15MB, exact expected size/hash, no redirects and no retries. It writes no changed/unpinned bytes. A missing cache is blocked; a mismatching existing pin is a failure. SDK/WASM source is the pinned normal npm registry/CDN. No private assets or credentials are needed. The browser cached run routes **only exact manifest URLs** to hash-verified local bytes, with CORS/MIME; other external requests (including fonts) are aborted. Local requests to `/fixtures/astronaut.png` and `/fixtures/coffee.png` use the same verified input bytes. Nothing is added to the production public directory. Route fulfillment is genuine asset serving, not SDK result replacement.

`assets.lock.json` deliberately has null model pins at this checkpoint: both exact production model URLs returned HTTP403 in the selected cloud environment. These are blocked assets, not reproducible acquired fixtures. The acquisition command refuses to request unpinned/previously denied models or enroll arbitrary local bytes. In a separately authorized reachable environment, first check model-specific license/redistribution terms and retrieve each exact URL once using bounded tools; record source URL, UTC retrieval time, size and SHA256 as a **reviewable manifest change**, then place those bytes in `.cache/real-sdk-assets/`. Do not use a different host/version, bypass403, or quietly update `latest`. Verify the reviewed pin on every rerun. A changed `latest` response fails against the old pin. SDK/module and available WASM/input pins are complete; unavailable model identity is explicitly unknown.

The runner always creates `.cache/real-sdk-runs/<UTC>-<random>/`. It records source HEAD/tree/worktree state/diff hash, manifest hash, Node/npm/OS, asset status, exact browser command/exit, Playwright results, scenario matrix and measurement populations. `environment.json` may report dirty author work while developing; promoted final runs bind to their actual frozen source SHA. `.cache/real-sdk-last-run.txt` only points to a run and is not the evidence itself.

Exit0 means all scenarios passed; exit1 means a failed integrity/test/startup check; exit2 means blocked prerequisites even if the independent failure diagnostics pass. With missing models, the runner executes only the two named asset diagnostics, records all six inference scenarios as blocked with zero executed inference, and exits2. It uses no skips, expected failures or inference success substitutions. Direct `npx playwright test --config playwright.real-sdk.config.ts` instead fails the missing-asset prerequisite explicitly. The suite has one worker, zero retries, 60s/test, 10min total and45s preview startup; the outer process bound is11min. Browser contexts and preview servers are Playwright-owned and teardown on failure; tests stop UI acquisition, tracks, VideoFrames and tasks in finally.

## Authenticity boundary

`vite.real-sdk.config.ts` extends the normal config in production mode, never `mode=e2e`, builds `.cache/vision-edge-real-sdk`, and previews only on4175. The actual root main mounts the real App without a supplied engine: App constructs the default VisionEngine with the existing GPU factory. A build-only plugin imports the observation module before App, and records the actual Canvas copy after it completes. The SDK import resolves to the actual lock-installed `@mediapipe/tasks-vision`0.10.14.

Observation proxies forward original receiver and arguments exactly once and return the exact original result. They record original task creation/errors, real SDK input constructor, raw detections/masks, postprocessed output and result identity at visible Canvas copy. A maskCanvas identity observer also records real Canvas drawImage calls. No detector/segmenter factory, detection/mask, ready state, running mode or delegate is substituted. One explicit stale-result control invokes `clearDetections` after a genuine SDK result, testing reentrant invalidation; it never supplies a made-up result. Observer code and capability controls are confined to the separate build/browser context. Normal production has neither observer globals nor imports; audit it and compare production hashes.

The optional positive suites require meaningful cup/person detections, finite scores and boxes, mask dimensions/ranges plus a person-region/background contrast, real mask draws and a matching result with nontransparent visible overlay. Person and cup inputs are licensed fixed photos; textured/empty generated controls cannot substitute for positives. Each test starts with actual inference after ready, then tests matching rendering/metrics, a real-result stale invalidation, camera stop/restart without model recreation, background/empty transitions and model switch/disposal. These assertions are **unexecuted in the blocked checkpoint**; their implementation is not acceptance evidence.

| Product input path | Control used to reach it | Latency eligibility |
| --- | --- | --- |
| Native VideoFrame | Browser capability intact; constructor/close observed and forwarded | actual immutable execution input to first matching Canvas copy |
| Owned Canvas fallback | VideoFrame unavailable, quality counter reports0 and rVFC unavailable in test context | actual private RGBA execution copy to first matching Canvas copy |
| Live HTMLVideoElement | VideoFrame unavailable, native quality counter retained | unavailable/unbound; no fabricated image timestamp |

The camera source is native Canvas.captureStream at512×512/5Hz, with fixed licensed input resized to the square and one changing corner pixel. Camera hook, real video playback, scheduler capture/conversion, default engine, postprocessing, metrics and App Canvas all remain in the path. This is a controlled native MediaStream, never a physical camera. Input mode must match the observed SDK argument; a fallback mismatch fails.

Initialization records are separate. The first positive SDK call is warmup; subsequent positive calls form steady-state samples, with no arbitrary millisecond threshold. Raw SDK-call wall samples exclude later observer copying; production `sdk` metrics include observation wrappers/readback and differ for that reason. Other stages retain the PR06 contract; draw observations add readback work. Do not sum overlapping stages. Raw arrays cap80 with dropped coverage; metric snapshots cap24, each exposing the production5s/2048/300 retained/quantile coverage. Small samples and retained-first warmup after truncation are explicitly qualified. Diagnostic-only runs have no inference timing population. No CPU utilization, GPU internal duration, camera FPS, physical display or camera-to-display claim is possible here.
