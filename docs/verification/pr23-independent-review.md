# PR23 independent review — FAIL; additional runtime verification BLOCKED

2026-10-25 UTC. Reviewed Draft PR https://github.com/TakashiSasaki/vision-edge/pull/23 as a reviewer separate from its author. This is a COMMENT assessment through the owner's account, not a separate human Approve. No merge/deploy/auto-merge, implementation-branch edit, install or Windows-PC work was performed.

Frozen HEAD `a21e7509636c364650418ba236a7e330a813f0c2`, tree `2c7ae44ba55b8019cfebb871f9bdd875ee26a12d`. Actual base/main `695e87d4a946b13f3c7ef4fbbc074fc98fea6ff7`, tree `09491a793cf463c8e4d21976e372f14ab9e7a226`. Fetched actual main/PR head/author ref, read PR body/comments and all17changed files, relevant AGENTS/skill and surrounding engine, observer, asset/acceptance/summarizer paths. No existing review was found to duplicate.

## F1 — P2: missing create/delegate observations remain eligible for requested-mode inference evidence

Target `scripts/real-sdk/summarize.mjs:28–29,46–60`. The new requested/observed check rejects a nonmatching observed delegate, but an empty creates array produces an empty observed set and no error because `.some(...)` is false. That incomplete record remains eligible for execution flags and timing populations.

Minimal independent model-free reproduction through the actual summarizer:

1. Write environment.json with requestedDelegate:"CPU".
2. Write browser/evidence.json with probe.creates:[], calls:1, sdkSuccesses:1, rawDropped:0, one internally consistent retained successful mediapipe record (scene coffee.png, ms1), and empty processed/adoptions/draws.
3. Invoke actual summarizeRun(runDirectory).

Actual measurements: requestedDelegate CPU; observedCreateDelegates []; evidenceErrors []; inferenceExecuted true; one population, with no initialization observation. Control with the same record and creates:[{model:"mediapipe",delegate:"GPU"}] instead correctly yields a mismatch error, inferenceExecuted false and zero populations. Omitting delegate from a present create also correctly yields an error. Matching CPU create produces the expected positive control. A CPU create recorded only for selfie-segmenter also incorrectly leaves the successful mediapipe record eligible, despite no creation observation for that model.

Expected: when a requested mode is declared and successful calls/timing are being accepted, missing create/delegate evidence for that model must remain unverified/ineligible rather than silently passing the requested-mode evidence check. Add an explicit evidence error/eligibility check (and regressions for absent creates and wrong-model-only creation); retain raw generic call observations and legacy evidence semantics separately. This finding concerns malformed/incomplete evidence validation. It is **not** a claim that the author's actual HTTP503 runs executed inference or observed the wrong delegate, and the synthetic records never enter SDK/App/model caches or pins.

[reproduce-summary.mjs](pr23-independent-reproduce-summary.mjs) is a portable, executed Node24 reproducer that requires no installed packages or model/network access. It checks the full target SHA and writes only checker JSON in its chosen output directory. Exit0 means observations completed, not acceptance. [Small observations](pr23-independent-observations.json) retain the actual results.

## Verified source/model-free behavior

- CPU mode is opt-in through a separate npm task and direct --delegate=CPU argument. Actual unavailable-runtime invocations confirmed CPU forwarding and that ambient REAL_SDK_DELEGATE=CPU does not change the default command's GPU selection. Invalid/extra modes fail rather than being ignored; CPU/GPU origins, build/probe/run roots and last-run pointers differ.
- Actual engine source was independently evaluated with Node's built-in TypeScript transform and a **local recording SDK boundary**. Unmodified defaultFactory passed GPU to both detector/segmenter; actual cpuEngineVariant passed CPU to both with unchanged model URLs, thresholds, running mode and segmentation flags. These are four synthetic boundary observations, not genuine SDK calls, CPU kernel execution or inference. No model bytes/pins were fabricated.
- Source inspection shows the CPU Vite pre-transform changes only the anchored engine delegate constant, before TS erasure; normal Vite config does not import this helper or observer. The real test config uses production mode, with no SDK alias. Observers use Reflect.apply with original receiver/arguments and do not rewrite delegates/results. Backend is explicitly unverified; WebGL renderer is separate. CPU/GPU positive semantic/lifecycle assertions are retained; diagnostic assertions now also require the requested option.
- All702existing main docs/verification blobs were checked against actual working-tree bytes:0mismatches. Product src, normal vite.config.ts, dependency lockfile and model manifest/pins have no diff from main; package.json changes only add scripts. CI adds15model-free controls without removing317/73/37/19/7checks or weakening permissions/assertions. The lifecycle fixture copy list adds the new helper without changing its assertions/time limits.
- A positively selected dependency-free subset of13delegate controls passed, marked selection=custom. All saved task receipts/raw hashes and exact HEAD/tree/clean-source metadata matched. The new15controls are model-free and distinct from genuine inference; the author's3native provenance controls likewise cannot demonstrate SDK CPU execution.

## B1 — local genuine SDK/build acceptance verification BLOCKED

Exact affected operations: supported CPU/default real-SDK commands, installed-SDK declaration/implementation check, actual Vite CPU/production builds, production byte-parity comparison, and Playwright native provenance/diagnostics in `/workspace/pr23-review`.

This fresh selected environment has no node_modules. Harmless filesystem lookup found no installed MediaPipe/Vite/Playwright/Vitest elsewhere in the permitted existing workspace/tool directories. Task explicitly forbids installs; **none was attempted or denied**. No replacement package/model acquisition or alternate environment was used. Verification remains pending availability of already installed authorized dependencies/build evidence, not permission to bypass the restriction.

Actual supported CPU command and default command (with ambient CPU) both correctly FAIL/exit1 on missing Playwright. Each records six inference rows blocked, two diagnostics not-run, zero inferenceAttempted/inferenceExecuted/resultRendered and empty populations; the exact source SHA/tree and forwarded CPU/GPU env are preserved. These local startup failures are not relabelled the author's successful2diagnostics/exit2 result.

The initial negative-name filter did not exclude the two dependency-requiring cases, so all15tests ran;13passed and two failed on absent vision.d.ts/Vite. Both errors/full logs are saved. A corrected positive-name subset then independently passed13/13. This is an environment limitation and transparent selection correction, not evidence of an implementation regression in the two hosted-pass cases. Node24.19.0/npm11.9.0 are available; no dependencies were installed.

Consequently, this reviewer has **not** dynamically established that the built CPU browser variant reaches the genuine SDK, independently repeated the two genuine CPU/GPU HTTP503 diagnostics or3native controls, or built/compared production bytes. Source preservation alone is insufficient to attest byte-identical output (Tailwind can scan non-product files). The author's production parity and local runtime results remain author evidence, not independent measurements.

## Hosted evidence and remaining limits

[CI40170433242/job121099987877](https://github.com/TakashiSasaki/vision-edge/actions/runs/40170433242/job/121099987877): independently fetched decoded raw log plus step metadata. Verified checkout of the full frozen SHA; Node24.19.0/npm11.17.0; all required commands/steps succeed;317unit/component,73foundation,37output protocol,15delegate model-free,19browser,7controls (2healthy/5specific expected faults), with zero unexpected failures/skips/cancellations. The hosted15includes SDK API-file and actual CPU-bundling controls. It does not run the optional genuine CPU diagnostics or real-model inference; its build regression checks build completion, not a genuine CPU inference call. No full test rerun was done for token/output measurement.

[Artifact12590205384](https://github.com/TakashiSasaki/vision-edge/actions/runs/40170433242/artifacts/12590205384): metadata binds to target HEAD, size5,433,210bytes, SHA25662f1e04b6dff9f9ebe206fb8636c4bc8749937328f501787d4353ff66d9c5613. Upload/source metadata is verified; hosted ZIP **contents remain UNVERIFIED**, with no acquisition/retry/alternate host/proxy/account/machine after the known distribution CONNECT403. Raw job logs were permitted and read. No actual hosted failure/cancellation upload was induced.

Both denied model files remain absent/null-pinned, with unchanged403 blockers and no request/retry/bypass/alternate sources. Genuine six-scenario CPU inference remains **BLOCKED / NOT ACHIEVED under Issue21**. API support, synthetic boundary/summary controls, native provenance and initialization-failure diagnostics do not prove task readiness, CPU backend/kernel execution, semantic/model compatibility, performance or displayed real inference. Windows/macOS cleanup, physical GPU/camera behavior and public-deployment equality were not tested; no Windows work occurred.

Recommendation: **do not merge**. Return F1 for a separate author correction/regression and then re-review the new full SHA/tree. B1 must remain explicitly pending until authorized existing dependency/runtime evidence permits the missing independent checks. This review does not authorize installation, acquisition, model enrollment, merge or deployment. Evidence is on a separate review branch, with no implementation changes.
