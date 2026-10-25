# Compact verification output

With Node24/npm11 and locked dependencies installed, use:

```sh
npm run --silent verify:offline
```

This runs typecheck, lint,317unit/component tests, the existing73foundation controls, output-protocol regressions, production build, browser tests and CSS/JS controls, sequentially. It installs no dependencies, browser or model. A permitted installed browser is required. Local cloud runs use the already installed system browser explicitly:

```sh
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run --silent verify:offline
```

For a narrow change, select `npm test`, `npm run test:foundation`, `npm run test:output`, or `npm run test:e2e`; do not rerun everything just to retrieve a failure message. Test arguments may follow `--`. A filtered run is labelled `selection=custom`; counts are the framework's reported counts, not a claim that the whole suite executed. Node may report file-level records without counting filtered-out cases as skipped.

Each task prints status/exit/counts and its immutable artifact directory. Full child output, source HEAD/tree/dirty-diff hash, command/exit/signal, byte counts/SHA256 and machine reports remain under `.cache/test-runs/<UTC-UUID>/`. Reports are never reused from a previous invocation. A missing, corrupt, empty or contradictory test report cannot turn child exit0 into PASS. Unexpected exit2 is FAIL; only explicitly declared asset/real-SDK tasks interpret it as BLOCKED. Capture/report failures override a nominal success/blocked result with exit1; the actual child exit is retained. Expected negative-control child failures remain distinct from the successful control runner.

Use the exact printed task directory to retrieve a bounded diagnostic without rerunning tests:

```sh
npm run --silent test:logs -- <printed-task-directory>
npm run --silent test:logs -- <printed-task-directory> 'literal text'
# Also accepts <run-directory> <task> [literal text].
```

Failed tasks retain `diagnostic.txt` (including errors available only in Vitest/Playwright JSON), plus complete `output.log` and reporter files. Search checks the diagnostic and raw log; it shows at most10matching lines and a bounded excerpt. The reader's exit0 means the read succeeded, not that the recorded test passed. Use `TEST_OUTPUT=verbose` to stream the child output explicitly, or `npm run test:raw` / the native CLI for custom reporters. Do not paste whole reports/hash inventories into agent context by default.

Successful tasks emit two lines. Failure display is capped at about6KB; raw capture is limited to64MiB per task and15minutes, with SIGTERM→bounded SIGKILL escalation on POSIX. Exceeding a limit fails explicitly and marks capture incomplete. A hard host kill may leave RUNNING metadata; that is unconfirmed, never PASS. Windows process-tree termination is not independently verified. Existing test assertions, timeouts and retries are unchanged. Child npm verbosity is restored from inherited `--silent` to `notice`, so npm warnings/audit diagnostics are retained in the saved log rather than erased.

Ordinary browser JSON/HTML/results and individual control reports are inside each task's artifact directory. CI uploads `.cache/test-runs/` with hidden files enabled and `always()` collection, including failure/interruption evidence where the runner can still collect it; retention stays14days. No application secrets or model binaries are introduced. Existing screenshot writers now use `testInfo.outputPath`; historical `docs/verification/**` files stay read-only.

`test:e2e:real-sdk`, `test:e2e:real-sdk:assets`, and the explicit acquisition command retain their original exit semantics and full JSON on disk. The direct `scripts/real-sdk/assets.mjs` JSON interface is unchanged for callers/tests. The real-SDK runner still writes its detailed matrix/measurements/browser artifacts under its separate `.cache/real-sdk-runs/` path; the compact log links that path. Acquisition is never part of `verify:offline`. Both denied model URLs remain unrequested externally, with null pins; six inference cases stay BLOCKED. HTTP503 diagnostics and checker/native controls are not genuine inference passes.

For measurements, scope and deferred work see [the output audit](verification/test-output-analysis.md). The reusable agent procedure is [.agents/skills/reduce-test-output](../.agents/skills/reduce-test-output/SKILL.md).
