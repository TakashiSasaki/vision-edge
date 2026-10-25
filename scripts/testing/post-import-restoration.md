# Post-import npm/evidence restoration and prevention proposal

This is a bounded restoration, awaiting independent review. It does not authorize
merge/deployment or implement prevention settings/automation.

## Source binding and restoration

- Verified PR24 integration: `b2b7874f0a40608415e76d17fcdaf13824ba179d`,
  tree `a2da49e360da608d55a546a795fe0b05bed6fb88`.
- Regression: `11381a6572c84426ecb3f65b04efacc185124289` added Bun lock and
  deleted npm lock plus48 PR06 logs, without product-source/test changes.
  [CI40188288159/job121153646032](https://github.com/TakashiSasaki/vision-edge/actions/runs/40188288159/job/121153646032)
  checked out that SHA, then failed setup-node lockfile detection. Actual checks
  were skipped; this is not a failing inference/test result.
- Latest inspected main/base: `7b1ea4394fc23ae1fe6a7f5c017f9dc8e4fc9505`,
  tree `a35aa70f0a91a4f0f63bbc10bffea452c90d552e`. Its model resolver/cache/UI
  changes are preserved byte-for-byte, including all four models and PR24 fixes.
- `package.json` and `.nvmrc` did not change after PR24. Restore its exact
  compatible npm lock and48 original log blobs, remove the competing Bun lock.
  No whole-commit revert or lock regeneration is needed. All702 historical
  files match the PR24 integration's original bytes.

Clean canonical-registry `npm run --silent ci:install` passed on Linux,
Node24.19.0/npm11.9.0. Full offline receipts and actual counts are in
`post-import-restoration-results.json`; frozen HEAD/tree and hosted CI are in
the Draft PR. Logs stay in ignored unique run directories. Model acquisition,
licenses, cache integrity/security and the new feature's full acceptance are
not independently certified by this restoration or by mocked cache tests.

The new cache plugin automatically prefetches models in its development
`configureServer` hook. That server was not launched. Build/preview do not call
that hook; cache unit tests inject/mock fetch and SDK boundaries. This distinction
was checked before running the existing offline pipeline. No denied model/ZIP
request was retried, no alternate acquisition route was used, and no Windows
operation was performed; no dependencies were added or browser/system packages
installed. Earlier six real
MediaPipe CPU scenarios remain BLOCKED; model-free tests are not inference proof.

## Observation, documentation and uncertainty

The user reports recurrence while working in Google AI Studio. The commit graph,
changed files and setup-node diagnostic above are directly observed. The commit
message describes removing stale logs; it does not establish which person,
agent, export filter or synchronization step selected those deletions. No export
trace, project UI inspection or controlled round-trip is available here.

All48 deleted logs match the existing `.gitignore` rule `*.log`; `package-lock.json`
is not ignored. Git itself retains tracked ignored files. An exporter using its
own ignore filter is a possible explanation for logs, not demonstrated causation,
and does not explain the lock deletion. Do not call this a universal platform bug
or infer malicious intent. Recommitting restorations alone has already failed to
prevent recurrence.

Official-documentation findings supplied by the coordinating parent on2026-10-25:
[AI Studio Build mode](https://ai.google.dev/gemini-api/docs/aistudio-build-mode)
describes GitHub import/link, bidirectional push/pull and per-file conflict
resolution. Do not assume AI Studio cannot pull. Arbitrary branch selection and
automatic npm-lock/log deletion were not established by that research. Verify the
actual project's available pull/sync/conflict UI before choosing a procedure.
[Bun lockfile documentation](https://bun.sh/docs/pm/lockfile) describes migration
retaining the original lock; standard migration alone therefore does not explain
its deletion. These documentation findings do not prove this project's behavior.

## Prevention alternatives (proposal only)

| Option | Practical change to consider | Acceptance criteria / limits |
| --- | --- | --- |
| Early read-only contract check | After checkout, fail explicitly if the sole npm lock is absent, a competing lock exists, or package-manager/runtime declarations conflict. For an exact Node check before caching, separate pinned runtime setup from cache setup: setup Node without cache, run a dependency-free contract script, then initialize the npm cache. A smaller shell existence check can precede current setup-node, followed by runtime validation afterward. | Negative fixtures for missing lock, competing Bun lock, wrong Node/npm declaration and package/lock root-dependency mismatch fail with specific diagnostics; unchanged valid checkout passes. No install, network, write-back or auto-repair. Keep `npm ci` as the authoritative complete lock-graph consistency check, not a handcrafted substitute. No continue-on-error or skip success. |
| Immutable evidence check | Maintain a reviewed path/SHA256 ledger and verify required historical blobs before ordinary build/test work. Consider explicit ignore exceptions for intentionally tracked evidence logs. | Deletion or changed byte fails; original702 pass; empty files are legitimate and retain their original hash. Ignore exceptions help tooling that respects them but cannot stop explicit deletion and do not prove an exporter issue. No regenerated historical evidence. |
| Deliberate evidence relocation | If a controlled project round-trip proves certain paths/extensions are omitted, migrate bytes to a supported path/extension, or keep immutable raw evidence in a separately retained commit-pinned archive with small in-repo receipts. | Preserve original bytes/hashes, publish an old→new mapping, update every immutable reference and verify retrieval/retention.14-day CI artifacts are not a durable historical archive. A rename, archive or alternate repository is not guaranteed to survive untested export behavior. Do not migrate/delete/upload anything as part of this restoration. |
| Reconcile AI Studio imports explicitly | Record the integration base SHA; verify available project pull/sync and per-file conflict controls, pull current canonical work before editing, and inspect the export diff before publication. If supported, reconcile the export as a patch/three-way merge in an isolated integration branch and review it before main. | Keep source edits while reviewing package/lock/CI/evidence deletions separately. Do not assume arbitrary branch selection exists. If a source-only round-trip regenerates a competing lock or drops evidence, stop and retain its manifest/diff as evidence; do not repeatedly publish it. No force push, blanket overwrite, credentials or settings changes are proposed as an automatic operation. |
| One correction owner; dependency-aware reconciliation | Use one active restoration PR for a frozen main, and have a failing guard report rather than create auto-repair commits. Before restoration compare dependency definitions with the known-good lock's source; regenerate/reconcile canonically only if definitions actually changed. | A concurrent package.json change blocks blind old-lock copying. New main changes require re-fetch/rebase-by-review or fresh reconciliation and exact-HEAD checks. No two competing repair writers, CI write permissions, bot loops or standing automatic reverts. |

Recommended order: agree on the synchronization procedure and inspect the actual
project UI; then implement the small read-only contract/evidence guards in a
separate reviewed change. Test a controlled disposable round-trip with ordinary
source, lock, `.log` and `.txt` fixture paths to locate the omission, without
starting model-prefetching application code. Relocate evidence only if the measured
behavior warrants it. These trials/settings need their own scoped authorization.

CI guards diagnose and prevent a green pipeline; they do not by themselves prevent
an unchecked direct-main write. Making checks mandatory through branch protection
would require a separately approved access/policy change. No branch protection,
permissions, credentials, automation, AI Studio setting or workflow was changed
here. No broader model-acquisition or architecture redesign is included.
