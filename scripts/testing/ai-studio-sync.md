# Safe npm/evidence reconciliation

The root package.json is the version authority. Use the pinned Node in .nvmrc
and npm11; compatible npm11 minor versions are accepted. This repository uses
only package-lock.json, with canonical HTTPS registry.npmjs.org resolved packages.

1. Record the GitHub main SHA and inspect the project's actual GitHub pull/sync
   and per-file conflict controls before editing. Official AI Studio Build mode
   documentation describes bidirectional sync; availability in this project's UI
   and arbitrary branch selection must be verified rather than assumed.
2. Pull/reconcile current canonical work before importing an older exported tree.
   Review the diff, preserving legitimate source changes. Resolve lock, version,
   CI and historical-file deletions explicitly instead of publishing a wholesale
   replacement. Do not treat an untracked/export-omitted file as intentional removal.
3. If dependency declarations changed, reconcile the lock using canonical npm;
   never restore an obsolete graph just because it passed earlier. For a coherent
   agent code/CI change follow AGENTS.md: `npm version patch --no-git-tag-version`,
   synchronize the generated lock, and recheck latest target version before PR
   submission. MINOR/MAJOR and tags require explicit human requests. Do not create
   tags during import, build, verification or routine PATCH changes.
4. Before dependency cache/install, run `npm run --silent preflight` and
   `npm run --silent test:preflight`. Both need only Node/npm builtins, not installed
   project dependencies. Then use canonical `npm ci` and normal verification.
5. A failing guard reports; it does not restore files, rewrite the lock, create
   repair commits, acquire models or change permissions. Assign one restoration
   owner and reconcile new main changes rather than running competing repair loops.

The reviewed `protected-evidence.json` ledger covers702 historical files from
c1678e2. It checks original bytes, including empty logs, and rejects missing files,
changed bytes and symlink substitutions. New source and new evidence files may be
added freely; they are not automatically declared immutable. A deliberate ledger
extension or authorized path migration requires review, retained original bytes
and explicit mappings. Do not delete ledger rows or regenerate old files to silence
a failure. Schema/count/path checks detect accidental ledger loss; this is not a
signed security boundary against someone deliberately changing the guard/ledger.

The dependency-free guard checks root package/lock identity, version, declared
dependencies/engines, sole-lock policy, canonical resolved URLs and Node/npm
contracts. It accepts whitespace comparator engine ranges (the current >= / <
form); policy/range syntax changes need explicit review. `npm ci` remains the
complete dependency graph validator. The current ledger cap is10000 entries,
contract JSON limit4MiB, npm-version probe10 seconds and captured preflight budget
60 seconds; preflight controls have120 seconds. Full JSON diagnostics are retained
in the printed unique .cache/test-runs directory; ordinary display is bounded.

The first setup-node must explicitly set `package-manager-cache: false`: omitting
`cache` alone still enables automatic npm cache when packageManager declares npm.
An isolated CI negative checkout removes its lock before that action and requires
the guard's exact missing-lock diagnostic; setup/cache failures cannot pass it.
CI pins Node without dependency cache, checks the contract and controls, then sets
up npm cache and executes all original required checks. Existing readonly CI
permissions, timeouts and human-only tag workflow remain unchanged. This guard
cannot prevent a direct main push or deliberate workflow removal. Mandatory branch
checks/protection would be a separate authorized policy change; none is made here.

Source docs: https://ai.google.dev/gemini-api/docs/aistudio-build-mode and
https://bun.sh/docs/pm/lockfile (standard Bun migration retains the original lock).
These descriptions do not establish the cause of this project's deletions or a
universal AI Studio bug. No AI Studio settings, credentials or automation are
changed by this procedure. Do not start the app's development-server model prefetch
or retry recorded model/ZIP denials during a synchronization check.
