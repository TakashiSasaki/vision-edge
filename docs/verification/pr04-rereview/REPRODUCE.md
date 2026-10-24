# Reproduce the corrected-checkpoint re-review

Reviewed source: `f81ce6102ec4f459ef7a0c3fb7e55d126b157be5`.
All commands run from the repository root unless noted. Install locked
dependencies with the repository's pinned Node/npm.

```sh
npm ci --cache /workspace/pr04-rereview-npm-cache
npx --no-install vitest run --config docs/verification/pr04-rereview/vectors/vitest.config.mjs
node docs/verification/pr04-rereview/vectors/native-boundaries.mjs
npx --no-install playwright test --config docs/verification/pr04-rereview/vectors/playwright.config.mjs
```

Expected on this checkpoint: new unit boundaries 11/11 pass; native capability/
path matrix has one failure; actual-app matrix has two controls pass and the
currentTime-only case fail. The latter displays approximately 60 inference FPS
from one native still frame. Exact call count depends on scheduling; the failing
single-consumption assertion is deterministic in meaning.

Native probes use `/usr/bin/chromium` explicitly, as disclosed in the report.
They do not download a browser. To run elsewhere, change that observer-only
executable setting in a disposable copy. Normal product/config/test files need
no changes.

Force the original delayed-work boundary with actual sampled input:

```sh
mkdir -p .cache
ffmpeg -hide_banner -loglevel error -f lavfi -i testsrc2=size=64x64:rate=20:duration=5 -c:v libvpx -y .cache/review-video.webm
node docs/verification/pr04-rereview/vectors/forced-native-race.mjs
```

Four immutable/quality controls pass; time-only rVFC and RAF can duplicate the
same intrinsic image/pixel buffer and make the command exit 1. This timing probe
is diagnostic, not a universal scheduling guarantee. Bootstrap is explicitly
disabled in this separate scheduler test so execution must cross the intended
arrival/timer contention boundary. The actual-app still-frame case is separate
and uses normal production bootstrap behavior.

The new app test creates a separate build using only the independent SDK
observation module in this directory. It runs the real App, VisionCanvas,
VisionEngine, scheduler, camera hook and collector. The SDK stores at most 40
input observations. Capability absence is represented by feature removal;
the saved native VideoFrame constructor is visible only to the observer.
The native scheduler probes also execute actual frameScheduler.ts, transpiled
without a scheduler double.

To rerun the original independent vectors **unchanged**, use a disposable clean
checkout of the exact corrected commit, copy the files from original-vectors
into `docs/verification/pr04-independent/vectors/`, and provide that checkout
with the identical locked node_modules installation. Their relative imports and
unchanged original config then point at that checkout's actual corrected source:

```sh
node node_modules/vitest/vitest.mjs run --config docs/verification/pr04-independent/vectors/vitest.config.mjs
node docs/verification/pr04-independent/vectors/native-first-frame.mjs
node docs/verification/pr04-independent/vectors/native-frame-race.mjs
```

The old race command also needs the generated clip at `.cache/review-video.webm`.
All original commands pass on this checkpoint. Byte hashes against evidence
commit `c632cf0cbce82b48503540651120c88fff75c8ef` are preserved in
logs/original-vector-bindings.json. No file in original PR #8 is edited.

For the unchanged ordinary suite, use npm test, npm run build, npm run test:e2e
and npm run test:e2e:controls. In this environment use the repository's existing
explicit `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium` override.
Matched hosted results are verified separately. A green normal CI check does
not run this evidence-directory capability regression.
