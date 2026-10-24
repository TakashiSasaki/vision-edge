# Reproduce this review

Run from the repository root with the frozen implementation checked out, these
evidence files present, Node/npm as pinned in the repository and locked
dependencies installed.

```sh
npm ci --cache /workspace/pr04-npm-cache
npx --no-install vitest run --config docs/verification/pr04-independent/vectors/vitest.config.mjs
```

The second command exits 1 on the reviewed HEAD: 25 pass, 4 fail. Three failures
demonstrate F01 (two scheduler paths and the actual canvas/engine bridge); one
demonstrates F02 (native startup before any subsequent arrival). The other cases
are positive adversarial controls. No existing tests/configuration are replaced.

Native probes use the existing local system Chromium override explicitly. Edit
the executable path in a disposable copy of the probe if using a different local
installation. They do not download a browser or change access policy.

```sh
node docs/verification/pr04-independent/vectors/native-first-frame.mjs
mkdir -p .cache
ffmpeg -hide_banner -loglevel error -f lavfi -i testsrc2=size=64x64:rate=20:duration=5 -c:v libvpx -y .cache/review-video.webm
node docs/verification/pr04-independent/vectors/native-frame-race.mjs
```

The first command uses native canvas.captureStream(0)/video/rVFC and requires
no SDK fixture or generated clip. It exits 1 for F02 on the reviewed source.
The final command uses native decoding of the generated clip, runs three fixed
trials per path, and exits 1 when duplicate pixel consumption is observed. The
native timing test is diagnostic rather than a universal timing guarantee:
machine scheduling can change which decoded image is seen. The deterministic
Vitest cases establish the identity/execution defect without that uncertainty.

The native probes transpile the real production frameScheduler.ts with the
installed TypeScript compiler and execute it without substituting a scheduler.
The work callback only observes SDK-boundary input. The canvas Vitest cases use
the real production VisionCanvas, VisionEngine, camera hook and collector with
separately defined SDK/DOM boundaries.

For the ordinary current unit suite, build, browser suite and controls, use the
repository's unchanged npm scripts. A normal production build is required before
browser testing. In this environment matched-browser installation failed with
HTTP 403; explicit system-browser tests passed, and matched hosted Chromium was
verified separately against the exact frozen commit. See REPORT.md and
commands.json. Review probes deliberately remain outside normal npm test
discovery, so a green evidence-PR CI does not close their findings.
