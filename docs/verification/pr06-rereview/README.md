# PR06-F01 independent re-review checkpoint

**F01 CLOSED. Scoped PASS WITH CONDITIONS; no new product findings.** The original invalid-statistics injection now stays within telemetry/display handling. Neither this review nor the original establishes a natural collector path to the malformed shape.

Reviewed [PR14](https://github.com/TakashiSasaki/vision-edge/pull/14) actual branch `feat/pipeline-performance-instrumentation`, HEAD `7056c67cd1f59c8d3e2237b84df05c223d99a25e`, tree `815f55307389f842d636f0381c206253630baabd`. Previous reviewed HEAD `04a3530c6ec9755bb7b89c3e31c1c3e15c9b1ea9`; preserved PR15 evidence `3d6bf05fcef72ede7d66d03a2ef34185ea802f86`. New evidence only under this directory on `review/pr06-pipeline-rereview`, directly stacked on the corrected product branch. Exact evidence commit/tree are recorded in the new Draft PR and final reviewer handoff; a commit cannot contain its own hash.

- [Report and scope](REPORT.md), [closure/findings](findings.json), [matrix](matrix.md), [overhead assessment](overhead-report.md).
- [Commands/exits](commands.jsonl), raw `logs/`, [source bindings](source-bindings.json), compressed complete correction source diff, exact product CI logs/metadata, browser archive/screenshots.
- Independent [reader cases](boundary.test.ts), [live camera/engine cases](live-boundary.test.tsx), [configuration](vitest.config.ts): **40/40 pass**. Run `npx vitest run --config docs/verification/pr06-rereview/vitest.config.ts`.
- Original PR15 probe bytes preserved as `.txt` under `original-probes/` with [hashes](original-probe-bindings.json). [Replay script](replay-original.py) verifies/materializes them in a disposable old/corrected worktree. Old F01 still reproduces one fail; corrected original31/31 pass unchanged. Original tests are not silently repointed or edited in PR15.
- Product317 unit /188 focused /19 browser /7 controls, install/typecheck/lint/build and evidence-inclusive typecheck/lint pass. Exact final-product hosted checkout7056c67 and steps verified. Evidence-PR CI is separate.
- Independent72 matched HUD trials/2304 operations; author144 trials independently recomputed; fixed-size copy/ownership/455-access pressure and raw timing evidence retained. Prior wider pipeline overhead stays bound to original heads and unchanged modules; old HUD timings are not relabelled corrected.

Remaining conditions: physical camera/model/GPU/audio/mobile/background/public-deployment equality unverified; inherited PR04 RGBA/coalescing limits and PR05 T01 timing sensitivity remain. No implementation repairs, original evidence edits, merge/deploy or next phase. Setup failures and all timing samples are retained in `setup-history.txt` and raw logs.
