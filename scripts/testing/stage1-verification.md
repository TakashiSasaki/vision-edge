# Stage1 npm/evidence contract checkpoint

Base `fdd37780930ef83431edcca404447b72a9c21976`, tree
`c16e5399e811a44a6ce5cbeb85527d149b57051e`. Implementation is bounded to
restoration, dependency-free preflight/controls, CI order and import guidance.
Independent review is pending. Integration is **BLOCKED by existing base failures**;
this report does not claim a green offline pipeline or model readiness.

Restored48 original logs and702 reviewed historical hashes from c1678e2; no old
evidence was regenerated. Dependency declarations match that lock's graph. Per the
new mandatory AGENTS rule, `npm version patch --no-git-tag-version` generated
package/lock1.0.4 from current package1.0.3, synchronizing both root lock versions.
All nonroot dependency entries remain identical. No tag was created or pushed.
Product source, variants, per-model settings and manual tag workflow are unchanged.

The guard runs before npm cache/install, after exact Node setup without cache.
47 independent Node controls cover valid/read-only checks, compatible npm minors,
legitimate version/dependency/source additions, missing/invalid/conflicting locks,
root/version/registry/runtime inconsistencies, ledger loss/count/path/hash issues,
missing/changed/empty/symlink evidence, and actual CLI success/exit1. Full JSON
failures are retained with bounded console diagnostics. See `ai-studio-sync.md`.

## Actual local verification

- Canonical clean locked installation: exit0 (Node24.19.0/npm11.9.0).
- Preflight: protected702, violations0. Final guard controls47 pass,0 other states.
  Earlier focused selection41 passed before6 additional negative controls.
- `verify:offline`: preflight/initial41 controls passed; **typecheck exit2**, so
  the remaining phases were not run by that invocation. They were then executed
  as separate diagnostic tasks, without changing the CI gate.
- Separate tasks: lint pass; unit369/369; foundation73/73; delegate40/40;
  output37/37; build pass. Browser19:17 pass,2 fail,0 skip/flaky. Controls stopped
  at the first intact smoke failure:0 healthy/0 confirmed expected faults,
  remaining6 controls not run. The old wrapper's `expectedFaults=1` counts a
  nonzero child generically here; it is **not** a confirmed expected fault.
- Independent builds of base and stage1 with the same installed dependencies:
  all7 production files byte-identical, despite the required package PATCH.
  Existing large-chunk build warning remains in raw output. Source is unchanged.

## Baseline blockers and reproducible evidence

1. Typecheck fails in both the unchanged base and stage1:
   App.tsx:10 and ControlPanel.tsx:15 import nonexported `ExtraInferenceOptions`;
   VisionCanvas.tsx:32 references the missing name. Baseline direct `tsc --noEmit`
   independently exits2 with the same3 diagnostics. No type suppression or
   product type repair is included in this stage1 scope.
2. Camera lifecycle test times out waiting to click the old `音声・設定` button;
   current UI labels the tab `音声`. The finalizer's page-closed error also appears;
   saved local trace identifies the pending click before that cleanup error.
3. Production smoke's `/SegFormer/` button locator resolves3 variant buttons,
   failing strict selection. The same smoke blocks the controls' initial intact
   case. This is not an acquisition failure or a passed negative control.

Both browser failures were rerun as the two targeted tests on the unchanged base:
0/2 pass,2 fail, exit1. No waits/assertions/locators were weakened in this PR.
Required product typing and test/UI-contract reconciliation need a separately
scoped correction; stage2's known cross-variant cache identity defect is also
unmodified and must remain separate. No claim of causal repair is made here.

Commands: `npm run --silent preflight`, `npm run --silent test:preflight`,
canonical `npm run --silent ci:install`, `npm run --silent verify:offline`, then
individual `node scripts/testing/run.mjs <phase>` diagnostic tasks. Compact receipts
and exact raw-run paths are in `stage1-results.json`; frozen HEAD/tree and actual
hosted CI outcome are bound in the Draft PR. The hosted workflow retains the
typecheck gate; downstream steps are expected to remain skipped on this source.

No merge, deployment, public review comment, tag request, setting/permission or
prevention auto-repair workflow change occurs. No development-server model
prefetch, denied model/ZIP retry, alternate acquisition or Windows work occurs.
Real-model inference and six old CPU scenarios remain BLOCKED; cache acceptance,
licenses, actual hardware and hosted ZIP contents remain unverified. Stage2 has
not started on this branch.
