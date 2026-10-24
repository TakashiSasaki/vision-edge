# Reproduction

Use a clean disposable worktree at21417613 and an old worktree at31721975, plus the immutable PR18 probe source ate1414072. Do not run legacy screenshot-producing tests in the protected evidence checkout. No denied model or matched-browser download is authorized. Use the already available explicit system browser or an already installed authorized matched browser.

The exact executed commands/cwd/times/exits/log hashes are in commands.jsonl. Log files are retained. Current-run outputs, recovered historical outputs and expected fault controls are separate.

1. Run locked npm ci with writable cache; typecheck/lint/npm test/build, normal browser and controls. Use one worker/retries0 and run browser commands serially. Run the production isolation audit. Do this separately for current21417613 and main9dd0f79.
2. Run the exact published Python and JS probes from PR18 against old/current sources. Read child exits in Python output. The old JS adapter's TypeError on the new negativeCheck bridge is expected adapter incompatibility and not closure evidence.
3. Execute the independent shared-checker vectors against actual correction modules:

```sh
PR07_REVIEW_SOURCE=/tmp/pr07-rereview-current PR07_REVIEW_OLD=/tmp/pr07-review-current node --test docs/verification/pr07-rereview/probes/adversarial.test.mjs
node --test scripts/real-sdk/assets.test.mjs scripts/real-sdk/corrections.test.mjs
```

The independent273vectors deliberately construct checker-only records; they do not acquire a model, import/call the SDK or send a made-up result into App/VisionEngine. The old summarizer import needs the old source file; the new source needs its unchanged lock-installed dependencies. Initial syntax-error source/log are retained separately; use the final adversarial.test.mjs.

4. Verify existing nonmodel cache bytes against the unchanged manifest before copying them into the new disposable source's ignored cache. Keep both model pins null; do not issue requests to either denied model URL. Cached-assets command must exit2 and distinguish7verified available pins from2blocked models.
5. Copy probes/native-adversarial.spec.txt to the disposable e2e-real-sdk/independent-provenance.spec.ts, then use a unique REAL_SDK_RUN_DIR:

```sh
REAL_SDK_RUN_DIR=/tmp/pr07-rereview-native PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium node node_modules/@playwright/test/cli.js test --config playwright.real-sdk.config.ts --grep 'input provenance|independent provenance'
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run test:e2e:real-sdk
```

The first6tests test native acquisition/provenance only, using the unchanged scheduler and original licensed fixture. Minimal video root and separate native-probe library exclude SDK/App. The supported real command then runs only2genuine-SDK failure diagnostics when assets remain blocked, records all6inference cases blocked and exits2. No skips, fabricated successful inference or CPU rewrite.

6. probes/cache-controls.py documents the executed independent fault setup at /tmp/pr07-rereview-current, with separate /tmp/pr07-rereview-unrelated-startup output. Run it only in a freshly created isolated worktree/cache and with unused output paths; do not overlap browser processes. Same-size cache corruption must yield1; missing cache gives2; unrelated startup must make the unchanged diagnostic child fail specifically for absent intended modelURL/503 evidence. The script restores bytes in finally and checks the hash. This is a new control, not the old uncertain operation. Full original trace stays ignored locally because it may include cache-only SDK/input bytes; selected JSON/UI evidence is retained here.

7. Archive each new browser run before restoring only regenerated historical PNG paths in the disposable worktree. Remove the copied independent test file. Rehash784target paths and635prior evidence files and verify clean disposable trees. Do not publish regenerated old evidence.

For eventual real inference, first provide explicitly authorized lawful exact production-model bytes, model-specific terms and a reviewable manifest pin/provenance change. The supported manual cache import revalidates hashes on every use. Current acquisition denials are not authorization to retry or use alternate hosts/accounts/machines/proxies/delegates. All model-enabled positive/negative/render/lifecycle/timing/device/public equality work remains unexecuted here.
