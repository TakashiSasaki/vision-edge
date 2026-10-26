# Phase B: whole-image Top5 UI

Stacked base PR29 remains frozen/unmerged at
`8b238424c2ca31b6d7fff7f33ff2da06377a9093`, tree
`06f19257b1e8911fce6303e071ed9fcce61fcec4`. Main remains
`bebc22a7d11e2bcbd0a63fa1cd141c0a8359dceb`. This PR targets
`feat/image-classifier-foundation`, not main. npm generates PATCH1.0.7 over A1.0.6;
no dependency change/tag. Independent B review and any integration are separate.

## Product contract

Header and comparison select the fifth family and Lite0/Lite2. Classification has
its own workspace/preview/panel and A engine/scheduler; it never enters detection
rendering, object picking, masks or legacy auto-readout. Existing four families and
nine variants, parameters and inference behavior remain. The existing camera hook
owns acquisition/release, including model-family unmount and late streams.

Top5 is per classification head, sorted/thresholded by A. UI rank starts1 but the
SDK index remains the raw model output ID. categoryName is preserved exactly,
including unknown case/commas/empty-name indication; raw numerical score is shown
without percent or a calibrated-correctness claim. The small human-authored
Japanese dictionary uses whole-string exact equality, never class IDs, offsets,
case folding, substring or COCO mapping. Unregistered labels retain their exact
raw text and show translation unavailable. This dictionary is not an asserted
ImageNet label list for uninspected weights. SDK displayName remains in A's result
contract; it does not silently replace categoryName in the raw-label UI.

Only the classification score threshold and explicit delegate request are active.
Detection boxes, masks, spatial tap and automatic speech are visibly disabled.
No spatial tap handler or fake object count is present. The mobile header wraps;
preview/panel can scroll within the viewport and native controls retain labels.

Production still resolves both classifier assets to BLOCKED before SDK/WASM model
startup. UI distinguishes the exact ClassificationAssetsBlockedError cause from
unrelated asset/init/run failures. BLOCKED explains pending exact-weight terms,
SHA256 and model verification; it offers no download/override. Init/run failures
have explicit separate retry actions. CPU/GPU requested and SDK-configured values
remain separate from unmeasured actual backend. B adds accepted request-owned
assetSource/wasmSource snapshots to ready state so UI never guesses delivery from
a cache default or an earlier variant's global record. Missing factory observation
is unreported, not a fabricated successful cache delivery.

Current frame identity, model request, variant, camera session and ready state
gate displayed results and speech. Switch/stop/threshold/failure clears outputs
and cancels owned speech. Stale initializations close once and cannot overwrite
current results. The new source fields are an additive A-contract extension on B;
A's remote branch/source is not modified. The two old assertions about enabled UI
families/availability now explicitly expect the fifth UI family, retaining the
four detection choices and the guard rejecting classification in VisionEngine.

Manual candidate buttons are the only classification speech entry point. They
ignore legacy autoReadout and require a currently owned frame/category. While an
utterance is queued/playing, scheduler work is held on that exact accepted frame;
video preview may continue, and UI explains the hold. Matching end/error resumes
classification. Cancellation tokens reject old browser events after replacement.
Stop, variant/family change, capture/run failure and unmount cancel the utterance.
Unsupported/throwing speech reports failure without changing classification into
an inference failure. No automatic announcement is added in classifier mode.

## Model-free verification boundary

Production preview:4173 is built normally with genuine MediaPipe imports and the
unchanged asset gate. The browser test aborts nonlocal requests, verifies both
variants' BLOCKED state, zero classifier-model URL requests, disabled controls,
no CI global/marker in same-origin production JS, and375px viewport containment.
Screenshots use testInfo.outputPath, never protected historical locations.

Only explicit --mode e2e uses the existing SDK double plus a narrow classifier
resolver alias. The resolver delegates catalog constants to A but does not fetch;
ImageClassifier is also the CI-only fake, on isolated build/output/port4174.
CLASSIFIER_CI_ONLY identifies those scenarios. No query parameter, production
asset flag, environment override or UI action can enable unverified weights in
production. Successful mock init/results are not real model/WASM/GPU evidence.

28 new unit/component controls cover exact translations, manual speech ownership,
late events/errors, actual workspace/engine/scheduler Top5/threshold, both asset
gates, distinct failures/retry, rapid ABA, stop/restart, stale speech, disabled
spatial behavior and request-scoped provenance. Existing691 cases remain; final
full population is719. Five added browser cases cover production BLOCKED/mobile/
source isolation and four clearly identified CI-double UI/lifecycle scenarios.
Original19 browser and7 healthy/intended-fault controls remain. The existing
smoke list now requires exactly five choices, including every original choice.

Failed attempts are retained: two typechecks used Playwright's exact option in
Testing Library; corrected to that library's exact-by-default string matching.
One browser startup ran before a production build and timed out with no tests;
this is startup FAIL. A later5-case run passed4/failed1 because source inspection
iterated an appended external WASM script. APIRequestContext GET to jsdelivr's
vision_wasm_internal.js failed DNS EAI_AGAIN. That external request was not retried.
Inspection is now restricted to same-origin production bundles; its focused1
passed. No model-weight request occurred. These failures do not count as BLOCKED
or successful inference. The first full24 browser run passed23 and failed the inherited four-choice
smoke expectation. It now requires the exact five-choice feature list; no timeout,
retry, skip or original behavior assertion was removed. Exact final local/hosted
receipts are separately bound.

## Remaining acceptance and integration

Exact weights' license/hash/size/tensor/label alignment and real CPU/GPU inference
remain unverified; the production acquisition/prefetch/redistribution gate is
unchanged. The overview/Model Maker input-size discrepancy remains unresolved;
Tasks preprocessing is used without manual reshape. All new real classifier
cases and six earlier MediaPipe CPU cases remain BLOCKED/not run. Windows/macOS,
physical GPU and hosted ZIP contents remain unverified. Linux model-free checks
cannot replace those acceptance conditions.

Before integration: independently review B's exact source; integrate A only with
separate permission, refresh actual main, explicitly retarget/reconcile B and
recheck ancestry/cumulative diff/version monotonicity and exact-source checks.
If code/base/tree materially changes, return to review. No merge, auto-merge,
force push, deployment, tags, settings/permissions or acquisition is authorized
by this Draft. Do not bypass previous model/ZIP denials or start Windows work.
