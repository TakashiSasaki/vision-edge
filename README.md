# Vision Edge

VisionStudio AI is a React/TypeScript browser demo for webcam object detection,
person segmentation, Japanese labels, and optional speech synthesis. It uses Vite,
Tailwind CSS, and MediaPipe Tasks Vision. Published app: https://vision-edge.ai.studio.

The current engine implements MediaPipe object detection and Selfie Segmenter.
YOLO-seg and SegFormer are selectable UI entries, but currently reuse the object
detector with generated polygons; their advertised runtimes are not implemented.
Backend selection currently changes displayed labels, not the MediaPipe GPU delegate.
Model accuracy, performance claims, and runtime truthfulness require separate review.

## Requirements and setup

- Node.js 24 LTS, at least 24.15.0 and below 25; `.nvmrc` and CI use 24.19.0.
- npm 11; the lockfile was generated with npm 11.9.0.
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
| `npm run clean` | Remove `dist/` and legacy generated `server.js`, using Node filesystem APIs |

The clean script avoids shell-specific deletion commands. Linux was exercised in
PR-01; Windows and macOS portability is based on Node's filesystem API and has not
been executed on those operating systems.

## Camera, speech, and network dependencies

The app requests camera access on initial load. Grant permission in a secure browser
context: HTTPS or localhost. A plain HTTP LAN address generally cannot access the
camera. Actual camera acquisition/stop behavior and device support are outside the
PR-01 infrastructure checks. Speech starts disabled and depends on the browser's
Web Speech API and installed voices; Japanese voices may not be available.

Runtime inference downloads MediaPipe WASM/loader assets from
`cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm`, and EfficientDet-Lite0
and Selfie Segmenter models from `storage.googleapis.com/mediapipe-models` (see
`src/services/visionEngine.ts` for exact URLs). The npm SDK is pinned to 0.10.14 to
match those WASM assets. The Selfie model URL contains `latest`, so model downloads
are not immutable. Google Fonts also loads external styles/fonts at runtime.
Camera access, network access, and GPU support are required for real inference;
builds and automated static/unit/smoke checks do not require these model services.

## Tests and CI

```sh
npm ci
npm run typecheck
npm run lint
npm test
npm run build
npx playwright install --with-deps chromium
npm run test:e2e
```

Vitest uses jsdom and React Testing Library. Tests cover dictionary translation,
unknown labels, deterministic colors, initial/unavailable speech synthesis, and
app rendering/model-selection interactions. The app test mocks the external
MediaPipe loader, denies camera permission, and supplies a null canvas context.
Unit tests do not evaluate real GPU inference, canvas rendering, or audible speech.

Playwright starts a fresh Vite production preview at `127.0.0.1:4173` and runs the
real built app in Chromium. It denies camera access and blocks every external
request, including model/WASM/CDN/font resources. It verifies the header, model
controls, comparison UI interactions, successful local resources, and no uncaught
JavaScript exception. It does not replace app modules or mock model outputs.
If a compatible Chromium is already installed, a local-only override is available:

```sh
# POSIX shell example; CI uses Playwright's matched Chromium instead.
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run test:e2e
```

`.github/workflows/ci.yml` runs on every pull request and pushes to `main`, using
the lockfile and pinned Node version. PR runs explicitly check out the PR head SHA.
The job runs install, typecheck, ESLint, unit tests, production build, browser
installation, and smoke tests in that order. Failures stop subsequent verification
steps and fail the job. Browser reports and failure traces/screenshots are uploaded
for 14 days. CI needs no application secrets and does not deploy or merge.

See [PR-01 verification evidence](docs/verification/pr01.md) for executed checks,
dependency decisions, environmental limits, and known deferred issues. Real
camera/device acceptance, inference accuracy, tracking, timing semantics, full TTS
behavior, and accessibility are future work; a successful CI run does not certify them.
