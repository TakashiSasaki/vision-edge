# npm and historical evidence restoration

This bounded restoration is stacked on Node-runtime PR33, frozen at
706f07c9dadb785fed33d92d000451904b224398 (tree
40f881362dcf223e098fdde2b79574ffcc2ceadd). Its main ancestor is
1fea494c711901cad214d31456e20762f211b2c8. PR33 remains unchanged and unmerged.
The eventual integration must refresh main, review the cumulative delta and
retarget this PR only after its dependency is integrated; old checks cannot
substitute for changed integration content.

## Restoration

- Restore exactly48 missing PR06 logs from verified commit
  479c74527a5b5e13f4f5938c6f26787e60890f14. Each recovered byte sequence matches
  its existing SHA256 in scripts/testing/protected-evidence.json. The other654
  protected files already matched; neither those bytes nor the702-entry ledger
  is rewritten. Historical failures remain historical failures.
- Restore the same source commit's npm lock as the seed, then reconcile it with
  current package.json using Node24.22.1/npm11.19.1 and canonical npm registry.
  All354 non-root lock entries retain their package versions: no dependency
  declaration or locked dependency version changes. The lock's root version and
  engines match the current package. Restore packageManager as npm@11.19.1.
- Use npm version patch --no-git-tag-version for1.0.16. No tag, competing lock,
  broad commit revert, guard exception or regenerated historical evidence.
- Preserve all current PWA, SEO, catalog/download, cache, classifier and other
  model functionality. No application, workflow, test assertion or model pin
  changes. SSD PR32 remains separate and untouched.

## Reproduction and verification

Use the Node24.22.1 runtime pinned by .nvmrc and npm11. Before this restoration,
PR33 preflight failed50 violations: missing lock, missing npm declaration and48
missing logs. The original red CI and local failure receipts are preserved.

1. npm install --package-lock-only --ignore-scripts --registry=https://registry.npmjs.org
2. npm run --silent preflight && npm run --silent test:preflight
3. npm run --silent ci:install
4. CI=true PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run --silent verify:offline

Local Linux uses existing Chromium154.0.8037.57; no browser/system installation.
CI/test context prevents the preview cache server's automatic model prefetch.
No development server or model acquisition command is launched. Full raw output
and per-task reports remain in unique ignored .cache/test-runs directories.
The complete local run is2026-10-27T00-51-54.562Z-69b13a45.

Preflight702/0,60 guard controls, typecheck, lint,809 unit/component,73 foundation,
40 delegate,37 output-protocol, production build and33 browser cases pass.
The seven stylesheet controls also pass (two healthy, five intended faults).
No failed/skipped/cancelled/todo/flaky case occurs in the final populations, as
applicable to each reporter. Hosted exact-head CI is recorded in the PR body
once terminal; no pending result is counted as passed.

An isolated checkout with restored metadata/evidence and only its lock absent
passes missing-lock-control.mjs: expected child exit1, sole intended
LOCK_MISSING_OR_INVALID,702 protected. No setup error is accepted as that control.

Clean npm ci exits0. npm11.19.1 warns that esbuild@0.28.2 and protobufjs@7.6.6
install scripts lack allowScripts approval; this restoration does not approve
scripts or suppress that warning. The actual type/build/test outcomes determine
whether installed dependencies are usable. Existing bundle-size and color-env
warnings remain in raw logs.

## Product equivalence and limits

A separate disposable baseline checkout of PR33 uses the same installed lock
contents and normalizes only its package version from1.0.15 to1.0.16 with npm.
Its build and this restoration's build contain23 files, all byte-for-byte equal,
including application JS, CSS, PWA/static assets and the cache inventory. This
comparison isolates the intentional version increment; it does not claim that
a build with an older displayed version has identical bytes.

No real-model inference, GPU/device acceptance, performance benchmark, deployed
API operation or artifact-ZIP contents are validated here. Offline/mock and
intended-failure tests are not successful inference. Known model-download/ZIP
restrictions remain; no denied route is retried. Independent review, integration
approval and any eventual deployment remain separate. No merge is authorized.
