# PR22 independent review — FAIL (2026-10-25)

Reviewed as a new agent, independently of the implementation author and their self-check. This is a COMMENT assessment, not a separate human Approve. Do not merge this evidence or PR22 on the strength of this report.

- Repository/PR: https://github.com/TakashiSasaki/vision-edge/pull/22 (Draft).
- Frozen HEAD: `1c9a485841d89edcd79dabb6a6d1efdd6e632630`.
- Frozen tree: `76e6bededea6984378173279535f30621dfd13e0`.
- Actual main: `c5e08cc20a66ea0a0e58e222f2e291a0c7c22e6d`, tree `1d6674260605529054c2a555ba55abe81587bed1`.
- Branches/PR ref were re-fetched, and reviews/comments re-read before publication: no existing review to duplicate; final refs still matched. No reviewed branch, permissions, merge settings, deployment, asset pins or historical evidence were changed.

## Findings requiring separate author correction

**R1 — P2: diagnostic search falsely reports no match for preserved failures.** `scripts/testing/results.mjs:34–36,41` keeps only the first three failures, while `scripts/testing/read-log.mjs:20–27` searches only that excerpt and `output.log`. Run four real Vitest failures through the documented unit wrapper, with distinct assertion messages `NATIVE_FILE_ONLY_SENTINEL_0` through `_3`, then run `test:logs -- <task-dir> NATIVE_FILE_ONLY_SENTINEL_3`. Actual: reader exit0, `No match`, and a link to the 112-byte output.log; the fourth failure is in unit.json. Expected: search the complete structured failure evidence, or explicitly disclose excluded reports and point to the exact full report instead of an unqualified negative. A matching line longer than1,500characters is also silently sliced; a critical suffix disappears with no truncation marker. Full files survive; this is a diagnostic discovery failure, not loss of unit.json or a green test result.

**R2 — P2: timeout cleanup leaves the actual real-SDK runner's detached child alive.** `capture.mjs:34–35,54,59` terminates only the wrapper process group; `scripts/real-sdk/run.mjs:37–43` launches Playwright in a separate group and forwards only SIGTERM. In a disposable frozen worktree, replace only the installed CLI with a no-network fixture that records its PID, ignores SIGTERM and stays alive. Execute the unchanged real-SDK runner through actual capture with1,500ms timeout/100ms grace. Actual at3,000ms: wrapper has FAIL receipt, descendant still alive until reviewer SIGKILL cleanup. Expected: bounded descendant cleanup. A generic detached descendant retaining inherited output pipes additionally leaves capture RUNNING past its timeout until external cleanup; no infinite wait was allowed in the review. No model or browser inference was performed by either lifecycle fixture.

**R3 — P2: the change alters the production CSS payload despite no product source edits.** With identical locked dependencies, run Vite production builds from clean main and frozen PR worktrees. CSS grows27,898→29,121bytes (+1,223), adding `.grow`, `.filter` and filter properties. Newly added prose contains these Tailwind candidate words (e.g. audit line18 `grow`, skill line19 `filter`); automatic source detection includes such prose. Actual clean PR build matches the integrated build. Expected under the requested scope: agent/doc additions should not add unused application CSS. JavaScript bytes are identical; filenames and index.html references change along with CSS.

CSS SHA256:
- main `3c60bb7d9314d6bc99c7d29821748ea2b56c49eeaa3ba29f5ea6c8b3cefba5a4`
- PR `5e1881f7ea2b9b2d1c16da9cd5cf37f1d0205be61a52a2920e6d1662c90782f1`
- identical JS content `1d1d27901e2d1f589ac4b3555a36f5ba69e8760f7b4396582fd3006de2a489ba`

**R4 — P2: contradictory structured failure details can be called PASS.** `results.mjs:15–23` checks aggregate arithmetic/global flags but not actual records. Independent protocol vectors give the unit parser success:true/pass1/fail0 plus an assertion with status failed; or browser stats expected1/unexpected0 plus an unexpected failed result. Actual through execute: PASS exit0/passed1/failed0 for both. Expected: reject conflicting summaries/details under the stated contradictory-report acceptance requirement. These are injected malformed protocol reports, not a claim that the hosted native reporters produced such reports.

## Independent validation and limits

Node24.19.0/npm11.9.0; installed system Chromium154.0.8037.57. A clean detached worktree ran the documented `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run --silent verify:offline` once: typecheck/lint/build PASS;317unit/component,73foundation,22protocol,19browser,7controls (2healthy/5specific expected assertion faults), all expected totals and zero unexpected failures/skips/cancellations. These are distinct groups, with22additional output controls, not22extra product/inference tests. All eight task log byte counts and SHA256 receipts matched saved bytes. Seven underlying control JSON reports were inspected; their exact assertions/exits are still required by the unchanged runner. The22author controls all pass but do not cover R1's fourth native failure, R2's nested process group, or R4's conflicting detail records.

Additional independent vectors checked native four-failure reporting; native pass/skip/todo (3total/1pass/1skip/1todo); exit7 and unexpected exit2 remaining distinct failures; child SIGTERM recorded as signal with childExit:null; SIGTERM-resistant timeout escalation; cancellation/report contradiction; missing machine report; nonexistent executable/ENOENT; artifact collision rejection without replacing the original; Unicode/space paths;96,010bytes containing split UTF-8, ANSI, NUL and stderr preserved exactly; bounded output; and targeted reader behavior. Warnings survive in raw logs but successful compact receipts do not flag their presence. Initial install failed254 because this executor's default npm cache was unwritable; using `npm_config_cache=/workspace/npm-review-cache` succeeded, with the original failure receipt/log preserved. No browser install or model acquisition was attempted.

All698pre-existing docs/verification files were byte-checked against main with zero mismatches. Six normal/details screenshots exist under the unique browser run/results paths. Source/diff review covered all21changed files, AGENTS.md, .agents skill and agent metadata, output docs, wrappers/parsers, subprocesses, screenshots and CI. Existing assertions, selection, fault controls, waits/retries, dependency lockfile, product src and read-only CI permissions were not weakened. CI still integrates73model-free foundation regressions. AGENTS/skill instructions favor focused checks, integrated checks when warranted and preserved raw diagnostics; they do not authorize bypassing failures or required validation. Their diagnostic-discovery claims need R1 correction. No claims of exact token savings are made.

Hosted evidence is distinct from local browser execution: https://github.com/TakashiSasaki/vision-edge/actions/runs/40159397318/job/121067573785 . The independently retrieved raw log confirms checkout of the full frozen HEAD, Node24.19.0/npm11.17.0, every required step success,317/73/22/19/7counts, hidden `.cache/test-runs` artifact upload and always() collection configuration. The decoded job log is23,920UTF-8 bytes/287newline characters. The historical74,103bytes is a different run; this is not a controlled performance or tokenizer comparison. Local docs'15,184→671byte arithmetic is consistent, but its old benchmark raw executions were unavailable here; it is author evidence tied to ba06d5d, not independently remeasured final-HEAD savings.

Artifact12585820356 upload and API metadata bind to frozen HEAD (3,568,919bytes, digest6a9b7fa91036873f89c58a66880112868c7ada94a04ed4eae7a0d0c4deec8857): https://github.com/TakashiSasaki/vision-edge/actions/runs/40159397318/artifacts/12585820356 . Hosted ZIP contents remain **UNVERIFIED**. The prior distribution-host CONNECT403 acquisition denial was respected; no ZIP was requested, retried or routed elsewhere. Failure/cancellation upload is configured with always(); an actual hosted failure/cancellation upload was not induced. Windows cleanup and hard host kill were not tested.

Both denied model downloads were not retried or bypassed. Genuine six-scenario real-model inference remains **BLOCKED / NOT ACHIEVED**, Issue21. Foundation/model-free checks and synthetic lifecycle fixtures do not satisfy it; no fresh real-SDK inference run was performed.

## Reproduction

`pr22-independent-reproduce.mjs` contains small fixtures only, not duplicated framework logs. Against a disposable checkout of the frozen SHA with locked dependencies installed:

```sh
node /path/to/pr22-independent-reproduce.mjs /absolute/path/to/pr22-worktree /tmp/pr22-review-evidence
```

It exercises actual capture/parsers, four real native Vitest failures, exact raw binary capture, both injected report contradictions, and the actual real-SDK runner with a temporary installed-CLI fault fixture. It creates/removes its own detached nested worktree, manually kills the escaped descendant, never edits implementation files, and retains small observations/full local logs in the chosen output path. The portable script was executed successfully by this reviewer. CSS comparison requires two clean builds using the same node_modules/lockfile; compare CSS bytes/content, not filenames alone. The evidence branch is based on main and is not an implementation fix or acceptance target.

Recommendation: **do not merge**; return R1–R4 for separate author correction and independently re-review the new full HEAD/tree and its actual hosted checkout/logs. No merge, deployment or auto-merge was performed.
