# B-F1/P2 mobile results correction

Rejected PR30 source: `3362f70f1ff59175ef6bdb8f8cb1ba3a4b5aae93`, tree
`499b120477564398f63c3820933ef5687ba9f6d1`. Stacked PR29 remains frozen at
`8b238424c2ca31b6d7fff7f33ff2da06377a9093`, tree
`06f19257b1e8911fce6303e071ed9fcce61fcec4`. Main remains
`bebc22a7d11e2bcbd0a63fa1cd141c0a8359dceb`. No retarget or integration.

## Finding and correction

The constrained main grid applied to the single-column mobile layout too.
The preview consumed its natural row height and the min-height:0/overflow:auto
results section collapsed into the remaining track. Outer scrolling did not make
that inner viewport usable. Independent B-F1 is a product usability defect; old
green CI/DOM presence did not establish readable candidates or usable controls.

Mobile now uses natural block flow with one outer vertical scroll owner. Results
keep their full content height with visible overflow. At md and above, main is
still a two-column grid and results retain their own overflow:auto. Video sizing,
camera hook, inference scheduler, source/asset/delegate contracts are unchanged.

| Chromium viewport | Old panel height / content | Corrected height / content | Outer usable height |
| --- | --- | --- | --- |
|320×640|34px /1252px|1254px /1252px|308px|
|375×667|110.5625px /1212px|1214px /1212px|434px|

Corrected mobile panel scrollTop stays0; outer scrollTop reaches1239/1103 when the
fifth candidate is focused. Each score and action is inside the outer viewport,
unoccluded, and has a measured target of at least44px wide/32px high. Every candidate
is tapped and must increment the manual-speech counter exactly once. Actual Tab
navigation from threshold reaches the fifth button; Enter operates it. Desktop
1280×800 preserves grid/top179 alignment, previewRight634/panelLeft646, results
clientHeight607/content1160 and overflow:auto. These are measured fixture values,
not guaranteed sizes for arbitrary labels or physical-device performance.

The first dimension-only correction exposed a related real touch-action race:
SDK acceptance updated engine frame ownership before React committed the new
candidate render. A tap could be rejected as stale although its button was visible.
A desired-behavior component control independently failed before correction.
Adoption now uses a bounded synchronous React commit for that accepted Top5 frame;
old-frame acceptance is not relaxed. No observer can fabricate SDK output, no
waiting/retry/rate cap is added, and manual speech still requires the exact current
frame/category. This ensures an input event cannot see a superseded render between
acceptance and publication. Real-model rendering throughput remains unmeasured.

## Reproduction and evidence

`PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium node scripts/testing/run.mjs browser e2e/classification.spec.ts -g B-F1`

Three new browser controls: two touch-enabled mobile viewports plus desktop split.
Each attaches geometry/scroll JSON and screenshots using testInfo.outputPath.
The reviewer screenshots/path were not assumed accessible; these are fresh author
reproductions. Old source:2 failed mobile/1 passed desktop. Intermediate layout
passed dimensions but failed a real tap race and a keyboard fixture that attempted
to refocus its already-focused button after programmatic scrolling. Keyboard now
uses actual five-Tab movement from a different control; no reachability assertion
is removed. Corrected focused3/3 pass. All screenshots (old collapse, corrected
fifth candidate at both widths, desktop) were visually inspected.

New component publication control: old behavior failed1/skipped11 in its filtered
run; corrected full component12/12 passes. No assertion, timeout, retry or existing
case was weakened. Final full local/hosted source binding is in the new receipts
and PR30 body. Historical red/intermediate and prior-source CI remain historical.

Same coherent PR retains npm-generated1.0.7 above A1.0.6/main1.0.5, with no lock or
dependency change. All702 protected hashes, original tests and failure records
remain. No new automatic speech, stale selection allowance, asset acquisition,
prefetch/serving permission, model retry, alternate route or Windows operation.
Both classifier assets and all actual model inference remain BLOCKED/unverified.
Touch/keyboard evidence is Linux Chromium emulation with controlled SDK/speech
boundaries, not physical mobile/audio or genuine inference acceptance. No merge,
public review, deploy, tag, settings/permissions change. Independent B-F1 closure
is PENDING; author does not self-certify independent acceptance.
