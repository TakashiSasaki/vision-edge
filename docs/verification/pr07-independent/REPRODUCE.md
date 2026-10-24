# Reproduce in a disposable checkout

Do not run ordinary browser tests in the protected source checkout: the inherited pipeline test writes historical screenshot paths. Do not retry denied model or matched-browser downloads. Use only an already available authorized browser.

1. Freeze PR17 at31721975 and main9dd0f79; create detached disposable worktrees. Run npm ci with a writable /tmp cache. Existing lockfile and all product/test/workflow files must remain unchanged.
2. Run typecheck, lint, npm test, production build, ordinary browser and controls serially. Run node scripts/real-sdk/audit.mjs on the production build. Preserve exact outputs independently, including any ordinary failure; retain PR05-T01 assertions/waits/retries0.
3. For the demonstrated defects, no asset/model download is needed:

```sh
python3 docs/verification/pr07-independent/probes/invalid-manifests.py /path/to/target-checkout
node docs/verification/pr07-independent/probes/checker-counterexamples.mjs /path/to/target-checkout
```

The Python probe copies the exact assets module into temporary roots and symlinks the already-installed node_modules. It never edits the real manifest/cache and never calls acquireAsset. It prints actual child exits. On the reviewed head, the first three cases exit0 incorrectly; malformed JSON exits1. The original executed copy was named probe-assets.py and used the fixed /tmp/pr07-review-current path. The published copy adds a source-directory argument; acceptance logic is unchanged.

The JS probe is the executed source reconstructed from the saved tool-call content. It extracts/transpiles the exact negative-scene loop and asserts the stale-state counterexample, then invokes the actual summarizeRun on a zero-call evidence fixture. Exit0 here means both defects reproduced, **not** that the real suite/inference passed. After correction, its old assertions should fail in the intended place; add new expected-behavior regressions separately.

4. Genuine negative diagnostics need verified available loader/WASM/PNG assets. The documented acquisition command refuses null model pins. Do not request model URLs or introduce another route. Run the supported real-SDK command using the explicit available browser; expected current exit2 with two diagnostics passed, six scenarios BLOCKED. Asset7controls are separate.
5. Native acquisition source is included as native-inputs.spec.txt. Copy only that review probe into the disposable e2e-real-sdk directory, compile the exact unchanged frameScheduler.ts as an ES library to .cache/independent-scheduler/frameScheduler.js with Vite build(configFile:false), then run only its named tests through the real config. From that disposable checkout, the exact build API command is:

```sh
node --input-type=module -e "import {build} from 'vite'; await build({configFile:false,build:{lib:{entry:'src/services/frameScheduler.ts',formats:['es'],fileName:()=> 'frameScheduler.js'},outDir:'.cache/independent-scheduler'}})"
```

Run:

```sh
REAL_SDK_RUN_DIR=/tmp/pr07-native-inputs PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium node node_modules/@playwright/test/cli.js test --config playwright.real-sdk.config.ts --grep 'independent native acquisition'
```

The probe intercepts the root with a minimal video page and serves the separately bundled actual scheduler. It uses original fixture installation/routing. It tests native input acquisition only; it neither loads a model nor replaces an SDK result. All3cases exited0 before disconnect, but their raw local reports were not recovered.

6. If the original executor recovers, inspect prior sessions and the ignored cache before any retry: additional corruption/promotion action outcome and baseline browser/control completion are uncertain. Recover raw logs and final715/559 preservation comparisons. No regenerated historical evidence should be committed.

Later lawful model import requires the target README's reviewable manifest change and exact-byte cache placement, model-specific terms and authorization. Null pins remain blocked now. No command in these independent reproducers obtains model bytes.
