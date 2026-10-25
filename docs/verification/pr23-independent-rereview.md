# PR23 independent re-review — PASS WITH CONDITIONS

2026-10-25 UTC. Reviewed HEAD `5ee8d9f659bd08fac462671a829cb0dcbf537c49`, tree `f396f5d24a304ff08c5ecd17986da0d717175fdb`; actual base/main `695e87d4a946b13f3c7ef4fbbc074fc98fea6ff7`, tree `09491a793cf463c8e4d21976e372f14ab9e7a226`. Actual refs, author response and review history were re-read. Independent reviewer is separate from author; owner-account COMMENT is not a separate human Approve.

The [old FAIL review5642015377](https://github.com/TakashiSasaki/vision-edge/pull/23#pullrequestreview-5642015377) and [old evidence commit70a6018](https://github.com/TakashiSasaki/vision-edge/tree/70a601885a8ed81ab1321eec1f3efc90b121bbc2) remain unchanged historical evidence for old SHA a21e750. This report appends evidence without implementing a fix or altering the reviewed branch.

## F1 / P2 — CLOSED

Executed the original published five-vector reproducer against this exact HEAD, changing only its SHA assertion; byte comparison confirmed that sole substitution. Empty creates and wrong-model-only CPU creates now produce explicit missing matching model/delegate errors, inferenceExecuted false and zero populations. Wrong/missing delegate controls still reject. Matching-model CPU positive CHECKER data stays eligible. None of these records is genuine SDK/model inference.

Read all six correction-changed files and the full summarizer. Matching non-error creation is checked for each retained call model; invalid mode and evicted/unidentified-model counters cannot qualify a file. Raw/retained counts and initialization records remain visible separately. Mode evidence explicitly distinguishes retained-model verification, unverified evidence, zero-call diagnostics and undeclared legacy data; it disclaims creation/request-generation binding and internal backend proof.

- The25new dependency-free controls passed separately, including bound ready/adoption/draw negative fixtures and all qualified flag suppression. The original15controls remain byte-identical to a21e750; installed total40passed.
- Independently authored20CHECKER cases passed for CPU/GPU: matching positives; missing/empty/wrong-model/failed/missing-delegate/mismatched-delegate creation; evicted-only counters; evicted success with only a retained failed call; invalid mode; legacy. Missing bindings suppress all four qualified flags and populations. Evicted success retains generic lifetime success1 but qualifies0retained successes and inferenceExecuted false. Undeclared legacy stays null/not-declared. Every input evidence file was byte-identical before/after summary; generic counters stayed visible.
- `npm run --silent test:foundation`:73passed, confirming legacy/asset/acceptance regressions. No existing criteria, raw evidence, dependency lock or model pins changed.

[Small observations](pr23-independent-rereview-observations.json) and [executed portable independent checker](pr23-independent-rereview-eligibility.mjs) provide reproducibility without large raw log duplication. Run with Node24: `node pr23-independent-rereview-eligibility.mjs /absolute/frozen-checkout /tmp/pr23-independent-cases`. Positive checker flags are fixture assertions, never reported as real inference.

## B1 — CLOSED for the previously missing independent runtime/build checks

The clarified task explicitly permitted restoring existing locked dependencies through canonical npm ci. No previous restoration acquisition denial existed. npm registry config and all332resolved lock URLs were registry.npmjs.org; `npm_config_cache=/workspace/npm-pr23-review-cache npm run --silent ci:install` succeeded. No new project dependencies, lockfile/auth/network-setting changes, browser install or system package install occurred. Existing system Chromium154.0.8037.57 was used with Node24.19.0/npm11.9.0.

WASM/loader bytes were copied from that restored locked SDK package only after matching existing reviewed sizes/SHA256. The two pinned non-model input PNGs were absent locally; each was fetched once from its exact original manifest scikit-image URL with no redirect/alternate source, and matched existing size/hash before cache use. No model bytes, upstream model request/retry, fabricated pin or model enrollment occurred. Those preparation results are recorded separately from model verification.

Independent actual commands:

| Command / scope | Actual result |
| --- | --- |
| `test:e2e:real-sdk:cpu` with existing system Chromium | genuine SDK failure diagnostics2pass; observed create options CPU only; childExit0/signal:null; outer exit2/BLOCKED;6inference blocked/4other checks not-run |
| default `test:e2e:real-sdk`, deliberately with ambient REAL_SDK_DELEGATE=CPU | genuine SDK failure diagnostics2pass; observed create options GPU only; default stays GPU; childExit0/signal:null; outer exit2/BLOCKED;6inference blocked/4other checks not-run |
| direct CPU real config, unique run directory, `--grep 'input provenance'` |3native-input controls pass,0unexpected/skips/flaky; model-free/custom subset, not genuine inference |
| fresh corrected/main production builds, same restored dependencies | all6file paths/lengths/SHA256 match byte-for-byte: HTML/CSS/JS/3public images |
| production audit | unaliased genuine SDK0.10.14; forbidden observer/test markers0 |

Read both modes' actual source-bound environment, assets, command, native Playwright report, matrix, measurements and diagnostic JSON. Every create retained the requested option and actual SDK model-fetch503error. The503responses were controlled browser routes; neither denied upstream model URL was fetched. Calls and successful calls are0; inferenceAttempted/inferenceExecuted/resultAdopted/resultRendered false, eligible attempts/successes0, populations empty, mode status no-inference-call-models. SwiftShader renderer is separate from the SDK's unverified internal execution backend. This verifies actual option propagation into genuine SDK failure initialization, not successful task/model/kernel initialization or inference.

CSS27,898bytes/SHA2563c60bb7d9314d6bc99c7d29821748ea2b56c49eeaa3ba29f5ea6c8b3cefba5a4; JS423,041bytes/SHA2561d1d27901e2d1f589ac4b3555a36f5ba69e8760f7b4396582fd3006de2a489ba. Product source, normal Vite config, lockfile, model references/null pins and original criteria have no diff. All702existing main verification blobs were independently compared:0mismatches. All six compact task receipts have correct final raw byte counts/SHA256 and clean exact-HEAD/tree bindings; native subset also has explicit source/command/report binding. An auxiliary receipt snapshot ran before foundation completed and encountered RUNNING metadata/KeyError; it made no changes and was repeated only after confirmed completion, with all final receipts verified. No failed test was hidden or relabelled.

## Hosted confirmation and conditions

[CI40172484506/job121106077929](https://github.com/TakashiSasaki/vision-edge/actions/runs/40172484506/job/121106077929): independently retrieved decoded raw log and step metadata; full final checkout SHA verified, Node24.19.0/npm11.17.0, all required steps success and317unit/73foundation/37protocol/40model-free delegate/19browser/7controls (2healthy/5specific faults), zero unexpected counts. Local focused results, original25subset and20independent checker cases are distinct from these hosted groups and are not added to a real-model success count. No complete local offline run or measurement-only repetition was done; local317/37/19/7and typecheck/lint were not repeated this re-review because exact-HEAD hosted evidence plus focused affected checks covered the change.

[Artifact12590448770](https://github.com/TakashiSasaki/vision-edge/actions/runs/40172484506/artifacts/12590448770): metadata binds to target,5,425,952bytes, SHA25643a532da3231505213656be61c573eb8ee7f771a51996c7a748b62b62db4f3a7. Publication is verified; hosted ZIP contents remain **UNVERIFIED**. No ZIP acquisition/retry or bypass through another route/host/account/machine/proxy was attempted after the known distribution CONNECT403. No actual hosted failure/cancellation upload was induced.

Genuine six-case CPU/model inference remains **BLOCKED / NOT ACHIEVED under Issue21**; default-GPU model inference also remains unexecuted. No model download, alternate model source or pin fabrication occurred. Passing F1/model-free/native controls and genuine HTTP503 diagnostics are not task readiness, CPU kernel/backend/model compatibility, semantic inference/rendering or performance success. Request-generation binding, Windows/macOS cleanup, hard host kill, physical GPU/camera behavior and public-deployment equality remain unverified. No Windows-PC operation occurred.

Recommendation: F1 and the former B1 reviewer-environment blocker are independently closed for this fixed SHA. **PASS WITH CONDITIONS** for the Linux CPU verification infrastructure scope; no remaining blocking code finding identified in that scope. Proceed only to the separate human merge decision. No merge/deploy/auto-merge or product-branch edit is authorized/performed by this review.
