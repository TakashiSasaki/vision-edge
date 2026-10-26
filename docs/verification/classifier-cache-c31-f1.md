# C31-F1/P2 author correction

Independent review found a display-integrity defect in PR31 at
`9467d4071f9382946b8e585215addacc3d4f273d`, tree
`76edacdf339ee7c4f096c6ff65e19ab31cca92c0`. The old validator checked cached
rows thoroughly but treated any shaped non-cached row as coherent. Thus
status=unavailable, filePresent=false, size=null, sha256=null combined with either
integrity=pinned or integrity=self-recorded displayed a successful hash check
beside an absent file, without a malformed/unknown warning. This was a real
inventory defect; it did not bypass the model-serving pin checks. The prior
independent FAIL and the original implementation evidence remain historical.

## Reproduction and correction

Before changing the static JavaScript, two desired-behavior unit controls ran
against the reviewed source: total6, passed4, failed2, exit1. Both exact
contradictions incorrectly returned invalid=0. Two actual static-page controls
also failed, total2, passed0, failed2, exit1: the notice said a successful zero
cached/one configured observation instead of warning about an invalid row. These
are intentional red reproductions, not model failures or inference evidence.
Raw logs/reports/screenshots stay in their unique .cache/test-runs directories.

The static inventory now validates state relationships as well as field shape.
Verification success is confined to a cached, present row with matching verified
size/hash and no error. All non-cached states require size=null and sha256=null;
contradictions become visibly unknown with the existing invalid-row warning and
are excluded from cached counts. No malformed row is silently normalized into a
successful empty observation. Errors must be null or a nonempty diagnostic
string; raw values are still never rendered.

| State | Coherent presence/integrity/error |
| --- | --- |
| cached | Present, pinned or self-recorded, valid size/hash, no error; fixed pins must match where asserted |
| downloading | Presence unknown, pending, no error, no verified byte data |
| unavailable/absent | File absent, not-present, optional acquisition error, no verified byte data |
| unavailable/invalid | Known file presence or absence (including orphaned metadata), invalid, diagnostic present, no verified byte data |
| unavailable/unknown | Presence unknown, unknown integrity, no verified byte data |

Two exact finding controls and adjacent coherent and contradictory matrices
cover successful empty, failed acquisition, corrupt file, orphan metadata,
unknown, downloading, pinned cached and self-recorded cached records. A separate
integration control feeds real cache-manager snapshots through the actual static
validator at each lifecycle stage using injected fetch and synthetic bytes/pins.
It verifies all eleven initial entries, acquisition in progress, success,
corruption, model-file removal and a controlled HTTP403 acquisition failure.
That synthetic HTTP403 fixture makes no network request and is not a retry of
a denied production route.

The two page controls use the actual emitted static HTML/JS with a controlled
status API. They require an invalid warning, unknown integrity, no hash-success
text, no cached table, exactly one read-only status GET and zero model requests.
Existing escaped rendering, refresh, unavailable API, variant isolation, mobile
layout and classifier UI tests remain. No serving, resolver, SDK, asset/license
pin, app runtime, permission or dependency behavior is changed by this correction.

## Verification and limits

The same coherent PR retains npm-generated package/lock1.0.8 under AGENTS.md's
once-per-change-set/PR rule; main remains1.0.5 and the stacked parent1.0.7.
PR29/30 and main are unchanged. No tags, model acquisition, merge or deployment.
Prior reports are not overwritten. The final HEAD/tree and exact hosted checkout
are bound in the updated PR body; independent finding closure remains PENDING.
Final local receipts and red/green command selections are recorded in
[classifier-cache-c31-f1-results.json](classifier-cache-c31-f1-results.json).

All cache-positive data remain synthetic and SDK boundaries remain controlled.
Real classifier acquisition/initialization/inference, deployed cache API startup,
physical GPU and Windows/macOS behavior are unverified. The six older MediaPipe
CPU scenarios remain BLOCKED; hosted ZIP contents remain unacquired/unverified.
License and source-attestation facts already recorded in the original cache
report remain resolved and unchanged. No denied acquisition was retried.

Local final `verify:offline` completed exit0: **799 unit,73 foundation,37 output,
40 delegate,57 guard,32 browser,7 controls (2 healthy/5 expected faults)**, plus
typecheck/lint/build. Zero failed/skipped/cancelled/todo/flaky in final populations.
Focused48 overlaps these counts. Both original reproducer browser cases now pass,
including no model request and exactly one status GET; the corrected screenshot
was inspected. Historical702 hashes match before/after. Application JS/CSS are
byte-identical to the reviewed build; only the standalone inventory module changes.
Raw red and green evidence are retained separately. Exact corrected-head hosted
CI is recorded on the PR after terminal status; older source CI cannot close this
finding or substitute for that run.
