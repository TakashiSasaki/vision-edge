# Vision Edge

VisionStudio AI is a React/TypeScript browser demo for webcam object detection,
person segmentation, Japanese labels, and optional speech synthesis. It uses Vite,
Tailwind CSS, MediaPipe Tasks Vision, and ONNX Runtime Web. Published app: https://vision-edge.ai.studio.

The runnable models are **Selfie Segmenter** (MediaPipe ImageSegmenter: person
region mask and enclosing box, not per-person instance segmentation),
**EfficientDet-Lite0** (MediaPipe ObjectDetector: boxes and class labels),
**YOLO-seg** (ONNX Runtime Web YOLOv8n-seg: 80 COCO classes with instance segmentation masks), and
**SegFormer** (ONNX Runtime Web SegFormer-B0: 150 ADE20K classes with full-pixel semantic segmentation colormaps).
`src/services/modelRegistry.ts` defines the names, implementation status,
outputs and descriptions shared by the engine, selectors, model dialog and HUD.

Each App owns one engine and at most one active task. Model state is separate from
camera state: `idle`, `loading`, `ready`, or `error`. Only the selected task is
created; duplicate pending requests share a Promise. Failures retain their WASM or
task-creation stage and cause, appear in Japanese, and retry only on explicit user
request. Superseded/unmounted completions cannot become current and their tasks
are closed. Model changes clear old detections and overlays. Ready means model
initialization succeeded, not that the camera or inference has succeeded.

The read-only backend label distinguishes requested and configured paths from
unmeasured execution. MediaPipe keeps its GPU delegate request. ONNX defaults to
WebGPU preference when the API is available, otherwise WASM; an observed session
creation failure can trigger an explicit WASM retry. The selected provider list
and fallback reason are reported, while the actual internal backend/hardware is
unmeasured. There is no new user-facing backend selector.

SegFormer confidence is an **uncalibrated softmax score**, separate from class
area coverage. Prominent-class entries require the mean winning-pixel score to
meet the threshold, including the largest class. The full argmax semantic mask
remains visible even if no class meets the list threshold; it does not imply
high confidence. Pixel inspection uses the winning score at that exact pixel,
including small classes absent from the list, and only from the currently
accepted frame/generation. The labels/readout distinguish scores from detector
confidence. Scores are not a guarantee of accuracy.

## Classifier cache and inventory

EfficientNet-Lite0/Lite2 add whole-image Top5 classification through the pinned
MediaPipe API, with manual candidate speech and no detection boxes. Their exact
Apache-2.0 metadata, sizes and hashes are recorded in the
[classifier cache contract](docs/verification/classifier-cache.md). Real model
inference has not been validated by the model-free CI controls.

`npm run dev` and ordinary `npm run preview` / `npm start` supply the container
cache API and schedule startup prefetch for registered variants. Startup waits
for the server to listen and uses the configured delay/concurrency. Classifier
bytes must match their fixed size and SHA-256 on acquisition, reuse and serving;
there is no unverified upstream fallback. An ephemeral container's cache is lost
when it is destroyed. The static [/model-cache.html](public/model-cache.html)
inventory itself only reads status; starting a normal server can independently
start downloads. It separates validated files from configured models.

In preview, defining `CI` or `TEST_LOG_DIR` suppresses the default cache API and
startup prefetch. `CONTAINER_MODEL_CACHE=true` overrides that suppression, so
ensure it is absent for model-free checks; setting it to `false` alone is not
an off switch for ordinary preview. This suppression does not apply to the dev
server. Use the repository test wrappers or injected-fetch cache fixtures
rather than starting a normal server to verify documentation or npm contracts.
See the [server hooks](src/services/containerModelCacheServer.ts) and
[model-free cache tests](src/services/containerModelCache.test.ts).

A build or pure static host supplies no cache API by itself. Without a reachable
API, the inventory reports unknown/unavailable and classification stays blocked.
These repository hooks do not establish deployed startup or real-inference
acceptance; those remain separate verification steps.

## Requirements and setup

- Node.js 24 LTS, at least 24.15.0 and below 25; `.nvmrc` and CI use 24.19.0.
- npm 11; the lockfile was generated with npm 11.9.0.
- Node.js and npm are the only supported runtime/package manager. `package-lock.json` is the sole lockfile; the unused Bun bootstrap lock has been removed.
- A modern browser with WebAssembly and WebGL for real inference.

```sh
nvm use
npm ci
npm run dev
```

`nvm use` is optional if the required Node version is already installed. The dev
server listens on http://localhost:3000 (and all interfaces). No API key or `.env`
file is required by the current frontend. `.env.example` is inherited scaffold
metadata; its Gemini/APP_URL placeholders are not read by this app. Never commit
credentials; local `.env*` files are ignored except for `.env.example`.

Use `npm ci` for repeatable installs. Do not use `--force` or
`--legacy-peer-deps`. Update and commit `package-lock.json` intentionally when
changing dependencies.

## Commands

| Command | Purpose |
| --- | --- |
| `npm run dev` | Start Vite on port 3000 |
| `npm run typecheck` | Run TypeScript (`tsc --noEmit`) over source, tests, and TS configs |
| `npm run lint` | Run ESLint with TypeScript recommended rules and React Hooks rules; warnings fail |
| `npm test` | Run Vitest unit/component tests once and exit |
| `npm run test:watch` | Run Vitest interactively in watch mode |
| `npm run build` | Write production assets to `dist/` |
| `npm run preview` | Serve an existing production build locally (default port 4173) |
| `npm run test:e2e` | Run Playwright against production preview; build first |
| `npm run test:e2e:controls` | Check production-smoke sensitivity using disposable healthy and faulty builds |
| `npm run clean` | Remove `dist/` and legacy generated `server.js`, using Node filesystem APIs |

The clean script avoids shell-specific deletion commands. Linux was exercised in
PR-01; Windows and macOS portability is based on Node's filesystem API and has not
been executed on those operating systems.

## Camera, speech, and network dependencies

The app requests camera access on initial load. Grant permission in a secure browser
context: HTTPS or localhost. The camera session changes only for start/stop or a
facing-mode change, not model/status/settings rerenders. Obsolete/late acquisitions
are stopped without being attached, and owned tracks are released on replacement,
stop, unmount, playback failure or a track ending. Camera readiness waits for video
playback and stays separate from model readiness. A stopped/replaced session clears
old scene detections without disposing the model task. Real device behavior still
requires acceptance testing. Speech starts disabled and depends on the browser's
Web Speech API and installed voices; Japanese voices may not be available.

Runtime inference downloads MediaPipe WASM/loader assets from
`cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm`, and EfficientDet-Lite0
and Selfie Segmenter models from `storage.googleapis.com/mediapipe-models` (see
`src/services/visionEngine.ts` for exact URLs). The npm SDK is pinned to 0.10.14 to
match those WASM assets. The Selfie model URL contains `latest`, so model downloads
are not immutable. Google Fonts also loads external styles/fonts at runtime.
ONNX models use the existing Hugging Face resolve/main URLs in their service files
and version-matched ONNX Runtime1.30.0 assets from jsDelivr. Model references are
mutable; this correction does not acquire model bytes or resolve their weight
license/provenance questions. Camera/network access and a supported configured
backend are required for real inference;
builds and automated static/unit/smoke checks do not require these model services.

## Tests and CI

For compact agent-facing output, complete raw artifacts and targeted diagnostics, see
[Compact verification output](docs/test-output.md). With dependencies and an authorized installed browser available,
`npm run --silent verify:offline` runs the installed checks without downloading browsers or models.


```sh
npm ci
npm run typecheck
npm run lint
npm test
npm run build
npx playwright install --with-deps chromium
npm run test:e2e
npm run test:e2e:controls
```

Vitest uses jsdom and React Testing Library. Tests cover dictionary/speech boundaries,
supported model selection, staged initialization errors, explicit retry, shared
requests (including synchronous subscriber reentrancy), cancellation, task ownership
and controlled out-of-order completion. The actual engine/state/UI run with the
MediaPipe creation boundary injected. Canvas bridge tests use a controlled RAF,
timers and canvas context, and exercise successful inference dispatch as well as
old-result invalidation. Camera acquisition uses controlled Promises and synthetic
streams in tests; these are not device tests.
Unit tests do not evaluate real GPU inference, canvas rendering, or audible speech.

The camera lifecycle browser test controls only acquisition, using native
MediaStreams/tracks from synthetic canvas video. Actual app/UI rendering, video
playback and track stopping run in Chromium. It checks render stability,
stop/restart, delayed acquisition and out-of-order facing changes. It is not a
physical camera or inference test. Unit hook tests separately cover unmount,
StrictMode, stale rejection, playback failure and ended-track cleanup.

Playwright starts a fresh Vite production preview at `127.0.0.1:4173` and runs the
real built app in Chromium. The separate offline production smoke denies camera
access and blocks every external request, including model/WASM/CDN/font resources.
It verifies supported/disabled model choices, comparison UI, model-load error/retry,
continued camera controls, required stylesheet CSS MIME/rules/applied styles,
successful local resources, and no uncaught JavaScript exception. It does not
replace app modules or mock model outputs.

The stylesheet controls rerun `e2e/smoke.spec.ts` explicitly with retries disabled
on disposable copies of the build. Two intact builds must pass. Missing CSS,
removed stylesheet links, empty CSS, unrelated valid CSS, and fatal JavaScript
must each fail at the intended assertion. A setup failure or the wrong diagnostic
does not satisfy a control. The original build is preserved. Selecting the smoke
explicitly keeps these seven controls independent of other browser test files.

If a compatible Chromium is already installed, a local-only override is available:

```sh
# POSIX shell example; CI uses Playwright's matched Chromium instead.
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run test:e2e
```

`.github/workflows/ci.yml` runs on every pull request and pushes to `main`, using
the lockfile and pinned Node version. PR runs explicitly check out the PR head SHA.
The job runs install, typecheck, ESLint, unit tests, production build, browser
installation, all browser tests, and stylesheet controls in that order. Failures stop subsequent verification
steps and fail the job. Browser reports and failure traces/screenshots are uploaded
for 14 days. CI needs no application secrets and does not deploy or merge.

See [PR-02/03 integration verification](docs/verification/integration-2026-10-24.md)
for the reentrancy correction, CSS reconciliation, source bindings and CI receipts.

See [PR-03 camera verification](docs/verification/pr03.md) for session ownership,
controlled races and remaining device/scheduler limits.

See [PR-02 verification evidence](docs/verification/pr02.md) for state/task ownership,
regression coverage and remaining lifecycle/measurement limits.

See [PR-01 verification evidence](docs/verification/pr01.md) for executed checks,
dependency decisions, environmental limits, and known deferred issues. Real
camera/device acceptance, inference accuracy, tracking, timing semantics, full TTS
behavior, and accessibility are future work; a successful CI run does not certify them.
