# Verification and logs

- Read `.agents/skills/reduce-test-output/SKILL.md` when working on tests, CI, or noisy verification output.
- Prefer `npm run --silent verify:offline` for the complete installed offline checks, or the relevant individual task. Dependencies and a permitted installed browser are prerequisites; this command does not download them.
- Keep raw output and reporter files in the printed `.cache/test-runs/<unique-run>/` directory. Use `npm run --silent test:logs -- <run-directory> <task> [literal-search]` to inspect only the relevant log. Do not paste every passing case, PR body, CI log or historical hash list into agent context.
- Preserve exit codes, actual test/skip/failure counts and raw failure diagnostics. Never hide failures using `|| true`, discard raw logs, weaken assertions/timeouts/retries, or call BLOCKED inference a pass.
- Keep historical verification evidence read-only. Test screenshots belong in `testInfo.outputPath`, not `docs/verification/**`. Do not retry or bypass the recorded model-download403 restrictions. No merge/deploy follows from a test PASS.
