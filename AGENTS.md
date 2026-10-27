# Verification and logs

- Read `.agents/skills/reduce-test-output/SKILL.md` when working on tests, CI, or noisy verification output.
- Prefer `npm run --silent verify:offline` for the complete installed offline checks, or the relevant individual task. Dependencies and a permitted installed browser are prerequisites; this command does not download them.
- Keep raw output and reporter files in the printed `.cache/test-runs/<unique-run>/` directory. Use `npm run --silent test:logs -- <printed-artifact-directory> [literal-search]` to inspect only the relevant log. Do not paste every passing case, PR body, CI log or historical hash list into agent context.
- Preserve exit codes, actual test/skip/failure counts and raw failure diagnostics. Never hide failures using `|| true`, discard raw logs, weaken assertions/timeouts/retries, or call BLOCKED inference a pass.
- Keep historical verification evidence read-only. Test screenshots belong in `testInfo.outputPath`, not `docs/verification/**`. Do not retry or bypass the recorded model-download403 restrictions. No merge/deploy follows from a test PASS.
- Search complete structured reports when an excerpt omits a failure. Validate aggregate counts against detailed records. New subprocess commands must share capture ownership or implement bounded child cleanup; test the actual nested runner. Compare production CSS/JS after adding docs or skills because Tailwind scans prose too.

# Shared agent work

- OpenAI Codex and Google AI Studio agents work in this repository. At task start and before publishing/pushing or merging, inspect the current branch, live main/PR refs, uncommitted changes and relevant ongoing work. Missing visible progress is not evidence that another agent is idle; local worktrees and other environments need not share state.
- Preserve other agents' legitimate changes. Use an isolated branch/worktree; do not publish an old exported tree wholesale or force-push a stale snapshot. If unexpected updates or conflicts change the affected scope, stop that write. Reconcile safely within the existing task or clarify scope when needed, then validate the exact new candidate and renew review where affected.
- Use `.agents/skills/preserve-npm-evidence/SKILL.md` for npm/evidence recovery and `.agents/skills/review-fixed-commits/SKILL.md` for review/integration. These procedures do not expand the task's authorization.

# Model-free verification

- Ordinary `npm run dev`, `npm run preview` and `npm start` can trigger cache startup downloads. Do not start them just to check metadata, documentation or builds. Prefer the repository test wrappers and model-free fixtures.
- Preview suppresses its default cache API/startup prefetch when `CI` or `TEST_LOG_DIR` is defined, unless `CONTAINER_MODEL_CACHE=true` overrides it. Ensure that override is absent for model-free preview tests. `CONTAINER_MODEL_CACHE=false` alone does not disable ordinary preview, and the preview suppression does not apply to the dev server. Check `src/services/containerModelCacheServer.ts` when changing this behavior; do not retry restricted acquisitions to verify documentation.

# Runtime Environment & Package Manager Note (Google AI Studio)

- **Standard Runtime**: Node.js (`v24.x`, CI24.19.0) and `npm` are the canonical, active runtime and package manager for this application in Google AI Studio (`npm run dev`, `npm ci`, etc.).
- **Lockfile**: `package-lock.json` is the sole dependency lock. Do not use Bun or introduce competing lockfiles. Preserve the pinned Node24/npm11 setup and canonical registry URLs.
- Before cache/install, run `npm run --silent preflight` and `npm run --silent test:preflight` using the pinned Node. Follow `scripts/testing/ai-studio-sync.md` when reconciling imports. Preserve reviewed historical bytes; guard failures require reconciliation, not automatic repair or removal of ledger entries. CI guards do not prevent direct-main writes.

# Application versioning and tags (mandatory for all coding agents)

- **One authoritative version:** The application version is `version` in root `package.json` (SemVer `MAJOR.MINOR.PATCH`). Never maintain an independent version number in `metadata.json`, a PWA web app manifest, UI source, deployment config, or documentation. If the application needs its version at build/runtime, derive it from `package.json`.
- **Agent code changes require a PATCH bump:** Every agent-authored functional or code change (source, tests, scripts, CI/build configuration, runtime assets, or dependencies) must include a PATCH increment before its change set is committed or submitted as a PR, unless a human explicitly authorized a MINOR or MAJOR increment for that change set. Bump once per coherent change set/PR rather than every editor save. Pure prose/documentation-only changes need no bump.
- **Human-led MINOR/MAJOR:** Only an explicit human instruction authorizes a MINOR or MAJOR increment. Do not infer authorization from the perceived scope of a feature or breaking change; when absent, increment PATCH and flag any semantic-release concern to the human.
- **Command and lock synchronization:** Use `npm version patch --no-git-tag-version` (or `npm version minor|major --no-git-tag-version` ONLY when explicitly human-authorized). Commit the updated `package.json`; if `package-lock.json` exists, include npm's synchronized generated update. The lockfile is derived metadata, not another version authority; never edit its version manually. Do not generate a tag as a side effect of version changes.
- **Branch/PR monotonicity:** Compare against the latest target `main` just before submission/merge. If the target advanced or another agent already used the intended number, rebase/reconcile and advance PATCH again, keeping the version strictly above the target. Do not overwrite another agent's changes or silently resolve version conflicts by taking an obsolete version.
- **Git tags are human-requested only:** Never automatically create/push tags for agent code changes, merges, builds, deployments, or routine version bumps. Create an annotated `vMAJOR.MINOR.PATCH` tag only upon an explicit human request, pointing to the exact approved commit; verify that commit's `package.json` version matches the tag. Tag creation is independent of a production/release build and must not be used to imply release validation.
