# PR22 independent re-review — PASS WITH CONDITIONS

2026-10-25 UTC. This is independent re-review by the original reviewer, not author self-acceptance. COMMENT publication through the owner's account is not a separate human Approve. No merge/deploy/auto-merge or implementation-branch edit was performed.

Reviewed HEAD `9dcf8115be613f4c696788068661577ea587eef1`, tree `09491a793cf463c8e4d21976e372f14ab9e7a226`. Actual main remains `c5e08cc20a66ea0a0e58e222f2e291a0c7c22e6d`, tree `1d6674260605529054c2a555ba55abe81587bed1`. Fetched actual main, author branch and PR ref, read correction comment6394211124 and review history. The previous [FAIL review5641646639](https://github.com/TakashiSasaki/vision-edge/pull/22#pullrequestreview-5641646639) remains valid historical evidence for old HEAD `1c9a485841d89edcd79dabb6a6d1efdd6e632630`; it was not edited/dismissed. The author correction checkpoint was treated as a claim to verify.

## Finding verdicts

| Finding | Independent observed result | Verdict |
| --- | --- | --- |
| R1 / P2, diagnostic discovery | Original four-native-Vitest-failures repro still exits1 with4failed; documented reader now finds `NATIVE_FILE_ONLY_SENTINEL_3` in unit.json and prints complete report paths. Independent11,024-byte long UTF-8 line: searching `CRITICAL_SUFFIX` retains it, labels both prefix/suffix truncation and produces4,391display bytes. Searching the beginning also explicitly labels its omitted suffix. | CLOSED |
| R2 / P2, nested timeout cleanup | Original actual-SDK/installed-CLI fixture,1,500ms timeout/100ms grace: wrapper FAIL and fixture PID absent at3,000ms. Original generic detached/inherited-pipe vector,200ms/50ms: returns FAIL within1,200ms rather than hanging. Its orphan PID is state Z, PPID1, not a running process; remaining observed live PIDs is empty. The new Linux controls and real production smoke/control children also finish. | CLOSED for verified Linux managed/observed scope |
| R3 / P2, production payload | Fresh main and corrected production builds use the same locked installed dependencies. All6output file paths, lengths and SHA256 hashes match, including HTML, CSS, JS and3public image assets. CSS27,898bytes/SHA2563c60bb7d9314d6bc99c7d29821748ea2b56c49eeaa3ba29f5ea6c8b3cefba5a4; JS423,041bytes/SHA2561d1d27901e2d1f589ac4b3555a36f5ba69e8760f7b4396582fd3006de2a489ba. Source changes are Tailwind exclusions, with no application logic/dependency/model-pin change. | CLOSED |
| R4 / P2, contradictory detail records | Both original injected unit/browser aggregate/detail contradictions now return FAIL/exit1. Independent native Vitest pass/skip/todo returns3total/1passed/1skipped/1todo. Independent native Playwright expectation fixture retains one actual pass, one explicitly expected failed assertion, and one skip; JSON aggregate is2expected/1skipped, and its actual failed attempt remains on disk. These are protocol fixtures, not product/inference passes. | CLOSED |

No remaining blocking finding was identified within this review's Linux/offline-output scope. Conditions are coverage limits below, not a claim of portable cleanup or successful model inference.

## Independent execution and source inspection

The original [reproducer](https://github.com/TakashiSasaki/vision-edge/blob/821e025d1a5238631c93455517cf21cfb7cb9d43/docs/verification/pr22-independent-reproduce.mjs) was fetched from its preserved evidence commit and executed with **only its frozen SHA constant changed**. The bytes were checked against exactly that substitution. Local fixtures never acquired model files or edited implementation source. The binary/ANSI/NUL/split-Unicode/stderr positive still preserved96,010bytes exactly. Full local logs remain in ignored/private run directories; this evidence stores small observations, not duplicated framework logs.

Node24.19.0/npm11.9.0/system Chromium154.0.8037.57. Focused commands executed on clean detached target:

- `npm run --silent test:output`:37passed,0failed/cancelled/skipped/todo. Original22protocol tests unchanged;15additional correction regressions. Read the new tests and process tracking/parsers directly rather than relying only on their pass result.
- `npm run --silent test`:317passed,0failed/skipped/todo.
- `npm run --silent test:foundation`:73passed,0failed/cancelled/skipped/todo.
- `node scripts/testing/run.mjs build`:PASS; complete6-file parity checked against a fresh main build.
- `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run --silent test:e2e -- e2e/smoke.spec.ts`:1passed, marked selection=custom. This is a subset, not a19-test local result.
- `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run --silent test:e2e:controls`:7controls,2healthy/5specific expected faults.
- Separate native positive fixtures and original negative reproductions as above. Intentional child failures remain failures; the enclosing observation script's exit0 merely means observation completed.

All8focused/native-positive task receipts have matching saved raw byte counts/SHA256, exact target HEAD/tree and dirty:false. All698pre-existing docs/verification files were checked against main after execution:0mismatches. Re-read all13correction-changed files, revised AGENTS/skill/usage and source detection; the original assertion/control files, lockfile, original22protocol controls and CI permissions/required commands were not weakened. The protocol command now includes both old and new test files. The old evidence branch and old review were preserved.

[Small machine observations](pr22-independent-rereview-observations.json) record source bindings, findings, counts, hashes and unexecuted coverage. No complete offline suite or repeated suite was run solely to measure output/tokens. Local typecheck/lint and all19browser cases were not repeated in this turn; their exact-HEAD hosted steps/logs were independently inspected.

## Hosted verification and conditions

[CI40167527615/job121091528010](https://github.com/TakashiSasaki/vision-edge/actions/runs/40167527615/job/121091528010): independently retrieved decoded raw job log and job-step metadata. Verified full checkout9dcf8115be613f4c696788068661577ea587eef1, hosted Node24.19.0/npm11.17.0, every required step success,317unit/73foundation/37protocol/19browser/7controls, zero unexpected counts, and successful artifact upload. These hosted results are distinct from the local browser subset and native fixtures. Source CI still includes73model-free foundation regressions, always() upload, hidden log paths and read-only contents permission. No inference acceptance or token savings is inferred from job status or byte counts.

[Artifact12588454790](https://github.com/TakashiSasaki/vision-edge/actions/runs/40167527615/artifacts/12588454790): API metadata binds to reviewed HEAD; size3,721,240bytes, SHA256f2e68e7aa8618ea2279159225b60e91b3d56665c403a91cad632ae9455bbfb5e. Publication/configuration is verified; **hosted ZIP contents remain UNVERIFIED**. No ZIP acquisition, retry, alternate route/host/account/machine or proxy was attempted after the known distribution CONNECT403 restriction. No actual hosted failure/cancellation artifact upload was induced.

Process cleanup is verified on Linux for the actual nested runner and observed descendants. `/proc` identities and100ms observation do not establish cleanup for an immediately unobserved escaping daemon. The generic vector's non-running orphan zombie is disclosed, not called a live worker or absent PID. Windows/macOS cleanup and hard-host-kill behavior remain **UNTESTED**, with no portability certification; two Linux-only regression tests visibly skip on unsupported platforms.

Both denied model acquisitions remain unrequested. Genuine inference across six model/input scenarios remains **BLOCKED / NOT ACHIEVED under Issue21**. No real-model inference or fresh genuine-SDK HTTP503 diagnostic run was executed by this reviewer during re-review; model-free foundation tests and the no-network lifecycle/expectation fixtures do not satisfy that objective.

Recommendation: R1–R4 are independently closed for this fixed SHA. PR22 may proceed to the separate human merge decision within the documented Linux/offline-output scope. Do not treat this COMMENT, the same owner account, unsupported platforms, or blocked real inference as human approval. **No merge is authorized or performed by this review.**
