# Reproduce this checkpoint review

Use the repository root at reviewed source
`d92b7b40c7d2dc11572147d052f69f11657540e2`, with these evidence files present
and the pinned Node/npm plus locked dependencies installed.

```sh
npm ci --cache /workspace/pr04-rereview2-npm-cache
npx --no-install vitest run --config docs/verification/pr04-rereview2/vectors/vitest.config.mjs
npx --no-install playwright test --config docs/verification/pr04-rereview2/vectors/playwright.config.mjs
node docs/verification/pr04-rereview2/vectors/pixel-cost.mjs
```

Expected: 17 unit cases pass, four native actual-app cases pass, and the four
resolution/resource probes pass structural assertions. Cost numbers are
environment-specific. Native probes explicitly launch /usr/bin/chromium and do
not download browsers. A different local executable can be configured in a
disposable copy; no product file needs editing.

The app config builds a separate artifact under .cache/pr04-rereview2 and
substitutes only the explicit independent SDK observer. App, canvas, camera hook,
engine, collector and scheduler are the actual production source. The SDK
observes actual supplied image bytes and dimensions, retaining at most 16
observations; canvas source PTS is null. The test repaints identical bytes to
force a fresh native presentation, including one close to input expiry.

PR #9 exact original vectors were extracted from evidence commit
`232b2104a2564d96a01e354bf7aeee7a380733f2` into their original paths in a
separate complete-source checkout of the corrected commit. PR #8 vectors were
extracted from `c632cf0cbce82b48503540651120c88fff75c8ef` the same way.
Bind their bytes with logs/prior-vector-bindings.json.

For native-boundaries.mjs, forced-native-race.mjs and independentSdk.mjs only,
apply the Canvas observation-constructor change recorded in
observer-adaptations.md. Exact adapted copies are saved under
adapted-pr9-vectors. Place those copies at the original
docs/verification/pr04-rereview/vectors paths in the disposable replay checkout,
then run:

```sh
node node_modules/vitest/vitest.mjs run --config docs/verification/pr04-rereview/vectors/vitest.config.mjs
node docs/verification/pr04-rereview/vectors/native-boundaries.mjs
node node_modules/@playwright/test/cli.js test --config docs/verification/pr04-rereview/vectors/playwright.config.mjs
node docs/verification/pr04-rereview/vectors/forced-native-race.mjs
node docs/verification/pr04-independent/vectors/native-first-frame.mjs
```

The forced native race needs .cache/review-video.webm, generated without a
download:

```sh
mkdir -p .cache
ffmpeg -hide_banner -loglevel error -f lavfi -i testsrc2=size=64x64:rate=20:duration=5 -c:v libvpx -y .cache/review-video.webm
```

Expected: prior unit 11/11; native matrix 6/6; prior app 3/3; forced matrix 6/6;
first frame one inference per path. The observation timestamp zero used only
for a Canvas constructor is synthetic and must not be treated as source PTS.

The raw PR #8 29-case abstract suite does not supply readable pixel contexts and
produces 8 pass / 21 fail. Its separately disclosed logical-capability adapter
gives 29/29 without changing assertions; run the frozen source's
docs/verification/pr04-compatibility/probes/original-unit.config.mjs from the
replay checkout. Reference copies in inspected-logical-adapter are for audit.
That adapter selects positive quality capability and is not absence-path proof.

Ordinary validation uses the unchanged npm scripts. Build before ordinary e2e;
local execution uses PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium.
Matched hosted final-HEAD logs were checked separately. A green evidence-PR CI
does not, by itself, constitute this independent acceptance.
