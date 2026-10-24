# PR06 independent review checkpoint

**Verdict: FAIL for required invalid-statistics isolation (PR06-F01 / P2).** Normal source checks pass. This is a synthetic malformed-statistics fault; no spontaneous collector path to the malformed shape is asserted.

Reviewed product PR: [#14](https://github.com/TakashiSasaki/vision-edge/pull/14), branch `feat/pipeline-performance-instrumentation`. Frozen HEAD `04a3530c6ec9755bb7b89c3e31c1c3e15c9b1ea9`, tree `7edc57e2c004d8a85502209b0e2541bd8b11ec07`; base `18e94a4508a0fa316cf31171789c3c10a0a041c3`, tree `f8e78b24e0bf27393ca9266c4d64bf039d232516`. Review branch `review/pr06-pipeline-independent`, directly stacked on that actual product branch. Exact published evidence HEAD/tree are recorded in the review Draft PR and reviewer handoff (a commit cannot contain its own hash).

- [Report and author action](REPORT.md), [stable finding](findings.json), [acceptance matrix](requirement-test-matrix.md).
- [Independent probes](independent.test.ts), [scheduler faults](scheduler-adversarial.test.ts), [actual component invalid-statistics reproducer](ui-adversarial.test.tsx), explicit [probe configuration](vitest.config.ts). Ordinary `npm test` intentionally includes only existing `src` tests; green evidence CI cannot close F01.
- [Raw commands/exits/timestamps](commands.jsonl), [source bindings](source-bindings.json), [complete product/test source diff](source.diff.gz), [environment](environment.json), [reviewer setup errors](setup-history.txt).
- [Experiment assessment](experiment-report.md), independent raw trial JSON and recomputation scripts, snapshot pressure results.
- [Product browser archive](product-browser-artifacts.tar.gz), [baseline browser archive](baseline-browser-artifacts.tar.gz), actual `browser-screenshots/`; all original author evidence remains unchanged.
- `live-pr14.json`, `open-prs.json`, remote heads and `product-ci-*` contain independent GitHub readback; `logs/product-ci-job.txt.gz` contains raw decoded final-product job logs.

Reproduce from this branch after `npm ci --cache /tmp/pr06-review-npm-cache`:

```sh
npx vitest run --config docs/verification/pr06-independent/vitest.config.ts
```

Expected at frozen product: **30 pass, 1 fail**, exit1. The failing assertion is required isolation, not an expected-success negative control. Targeted core suites pass29; malformed-result suite passes its throwing-computation control and fails its malformed-result case. No source repairs, test assertion weakening, merge or deploy.
