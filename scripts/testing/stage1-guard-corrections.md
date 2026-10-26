# Guard-only correction checkpoint

Responds to private independent review of
`fbe45b2a30bd1982444fe6f7b7fbd47bdaabe4a8`
(tree `676fa32aa7a2b5af3ca7acca55825429001046fb`). This correction leaves the
product source, B1/B2/B3 blockers, dependencies, historical bytes and stage2
unchanged. Independent closure is pending; no public review comment is published.

| Finding | Correction and evidence |
| --- | --- |
| S1-F1 P2 implicit early npm cache | Pinned setup-node defaults package-manager-cache to true and detects root packageManager. The initial action now explicitly disables it. A separate isolated CI checkout removes package-lock before initial setup with automatic cache disabled; setup must succeed, then the guard must return exactly one intended missing-lock violation and child exit1. There is no dependency cache step in that negative job. The main job may initialize its explicit npm cache only after successful preflight/controls. Node binary tool-cache use is distinct from dependency cache. |
| S1-F2 P3 non-string metadata crash | Version parsing now requires a string before regular-expression validation and splitting; packageManager is also type-checked. CLI controls cover array/object/number/boolean/null package versions, plus array runtime Node/npm, packageManager and engine range. They require structured intended diagnostics and nonzero exit, never a TypeError/empty stdout. |

The original47 preflight controls are retained. Nine metadata-type cases and one
actual missing-lock-control contract case bring the total to57. That last case
rejects a valid-lock precondition and an additional unrelated Bun-lock violation;
only the isolated intended missing-lock failure passes. The new metadata cases
were run against the old checker first:56 cases,53 pass/3 fail, exit1 (array package
version, array runtime Node and npm threw). Full red logs remain in ignored run
`2026-10-25T17-35-16.112Z-1309947c`. After the parser fix56 passed; final57 results
and hosted positive/negative job binding are in the PR correction response.

The CI negative job modifies only its disposable checked-out lock, not committed
source or historical evidence. It stores full child stdout/stderr, actual exit,
signal and error metadata under a unique .cache/preflight-negative directory.
It requires ENOENT on package-lock with no other guard violations. A setup failure,
startup crash, extra runtime/ledger error or wrong child exit fails the job.
There is no continue-on-error, retry, self-repair or ignored assertion.

Version remains the already generated1.0.4 for this same coherent PR; current target
main is1.0.3. Package/lock root metadata stay aligned, consistent with the AGENTS
once-per-PR PATCH rule. No dependency entry or manual tag workflow is changed and
no tag is created. Latest main is rechecked before submission. Existing B1 missing
ExtraInferenceOptions types, B2 old button locator and B3 ambiguous variant locator
remain unresolved because their correction is not yet authorized. Prior local
unit/browser results remain prior-head evidence, not new-head successes.

New CI is expected to pass guard/controls/cache/install then fail the unchanged B1
typecheck gate. Downstream skipped steps must not be called passes. The separate
negative job proves the missing-lock path before dependency cache; artifact
publication is not ZIP-content verification. No merge/deployment, permission or
tag-setting change, Windows work, model acquisition, denied-route retry or stage2
product correction is included. Existing real-inference limits remain unchanged.
