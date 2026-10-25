# Test/CI output audit

Base: main `c5e08cc` after the externally completed PR17/20 integrations. Work is isolated on `chore/compact-test-output`; old branches and verification evidence are preserved. Product `src`, dependencies, lockfile, model pins, SDK/delegate, assertions, waits and retry policy are unchanged.

## Measured terminal output

Same installed Linux Node24.19.0/npm11.9.0/system Chromium154.0.8037.57 environment, original test counts. Before uses the original npm/direct-Node commands; after uses the recommended silent task invocation and per-task slices from `verify:offline`. These are UTF-8 bytes/lines entering the terminal, not exact billed tokens, runtime savings or disk savings. Full JSON/HTML/raw artifacts are deliberately excluded from console input and retained separately.

| Task | Before bytes /lines | After bytes /lines | Byte reduction |
| --- | ---: | ---: | ---: |
|317unit/component |760 /22 |162 /2 |78.7% |
|73foundation |5,095 /81 |184 /2 |96.4% |
|19browser |8,625 /48 |167 /2 |98.1% |
|7controls |704 /16 |158 /2 |77.6% |
|Production build |396 /14 |117 /2 |70.5% |
|Real SDK, blocked |7,317 /207 |742 /8 |89.9% |

The four existing test groups together fall from15,184bytes/167lines to671bytes/8lines (**95.6% fewer bytes**). The complete after offline pipeline, including the additional output-protocol checks, emits1,204bytes/16lines. Already-quiet typecheck/lint grow from49/57bytes to125/115bytes because a durable receipt path is now printed; this small cost is disclosed rather than presented as a universal reduction. Their existing direct npm commands remain available. Asset prerequisites now print four lines with two distinct blocked causes instead of a full JSON inventory.

[Machine measurements and methods](test-output-measurements.json) bind the representative benchmark to implementation `ba06d5d`; later reader/diagnostic refinements are separately covered by protocol tests and exact-head CI. No final-HEAD benchmark is fabricated. Original raw executions and later refinements are kept in ignored local directories, not committed as another large evidence archive. The historical hosted run40141811269/job121014945080 was74,103bytes/735lines; a new hosted whole-job reduction must be measured on its own actual final SHA, not inferred from the local table.

## Sources of excess and changes

1. **Passing-case chatter.** Default Node output lists all73passing controls, and Playwright lists every successful case. Compact tasks retain TAP/JSON as evidence and print actual totals only. Vitest uses its built-in JSON reporter, Playwright JSON plus HTML; no custom test-result generator replaces the frameworks.
2. **Duplicate observations.** Four E2E console statements repeated data already attached as JSON. Only console duplicates are removed. Geometry, frame/scene and compatibility attachments and all assertions remain.
3. **Large repeated status JSON.** The real-SDK console repeated the same missing-model reasons across six scenarios and dumped the full matrix. The matrix/measurement files remain complete; console now emits status counts, one cause per unavailable asset and a link. Asset CLI JSON remains compatible; npm entrypoints provide compact views.
4. **Tool/build/install noise.** CI now captures npm installation, compiler/linter, build and browser installation output with the same commands and failure propagation. No audit, warning, check or browser requirement is disabled. Parent npm `--silent` no longer erases child npm diagnostic output. Local denied browser/model downloads were not retried.
5. **Lost diagnostics after quieting.** File-only Vitest/Playwright assertion errors are extracted into a bounded saved diagnostic; Node failure blocks are extracted without passing-case chatter. Complete original streams and structured reports remain reachable. Startup failure, invalid/missing reports, contradictions, unexpected exit2, interruption and capture quota are not successes.
6. **Artifact overwrite/copy work.** Ordinary tests formerly rewrote six historical PR06 PNGs, forcing disposable-worktree recovery and large preservation reports. They now use per-test output paths. Control children keep individual stdout-JSON/stderr/results. CI explicitly includes hidden `.cache` files and collects with `always()`; otherwise quiet logs could disappear on failure/cancellation. Hard host termination remains a disclosed collection limit.
7. **Agent input amplification.** Downloading a complete PR body, review history, CI log or hundreds of hashes repeatedly can cost more context than the tests. The new skill and root AGENTS instructions require tool-side persistence/projection, exact source binding, summary-first inspection and targeted follow-up. Raw logs are not removed merely to avoid reading them.
8. **Redundant execution.** A completed raw report should be queried, not reproduced solely to get a shorter display. Focused checks precede a complete integration check; no cache of old test success replaces a new source-specific result. Raw historical evidence is not reformatted, deleted or relabelled.

## Integrity checks

The original baseline and modified source run the same317/73/19/7 checks successfully. New output protocol cases cover large success/failure streams, exact log hashing, nonzero/blocked exits, missing/contradictory reports, global startup errors, quotas, SIGINT/deadlines, report collisions, subset selection, nested failure details, inherited npm verbosity and bounded retrieval. A real intentional Vitest failure still exits1, exposes expected43/actual42 in1,135display bytes, and is retrieved in417bytes without re-execution.

An independent agent used the skill on temporary Node tests: success raw327,860bytes→167display bytes, intentional failure raw328,507bytes→4,168display bytes; diagnosis lookup402bytes and log integrity confirmed. That forward test is bound to source `5a4276e`, not relabelled final-head coverage. It reported no functional blocker. A later local regression caught inherited Node test context in a nested CLI fixture; that fixture now clears only its own child test-context variable. No production/test assertion was weakened.

All698pre-existing historical evidence hashes remain unchanged. No regenerated historical screenshot, model, private image or huge raw log archive is added by this work. Raw/full logs remain in ignored run artifacts. Real-SDK diagnostics still pass2, command exits2 with6BLOCKED scenarios and empty inference populations; real model/device/public-equality acceptance remains unachieved.

## Scope limits and further opportunities

- Exact total token savings depend on the calling model/tokenizer and which excerpts it reads. Bytes/lines are measured, not converted into a claimed token bill. Saved artifacts can be larger than the old human output; this change optimizes agent input and diagnostic access.
- GitHub checkout/setup/upload boilerplate, connector response envelopes, system catalogs and past discussion size are outside the repository's test reporter. Do not rewrite trusted action implementations or historical evidence to hide those lines. Project/skill guidance avoids injecting complete responses into model context.
- Structured JSON schema changes in future Node/Vitest/Playwright versions require validation; missing totals fail closed. Custom framework reporters should use the explicit raw CLI. Filtered Node files may yield file-level records; selection is disclosed and missing skip counts are not invented.
- More aggressive caching, fewer tests, retries, parallelism and altered timeouts are deliberately excluded: they affect coverage or timing rather than log verbosity. There is no reason to rerun all273historical independent vectors for a display-only change.
- CI source checks, current-head CI, human approval and later merge/deploy decisions remain separate. This branch does not merge, deploy, change protection settings or close the outstanding real-model objective. When lawful model pins are eventually enrolled, the existing “unexplained null model” fixture must be made explicitly paired-null; no such test edit belongs here.
