# S2-F1 initialization-owned asset provenance

Correction to the independently rejected Stage2 source
`a0dfa4b76ea4cb00019708b1b8bf5d88dc51a622` (tree
`2738131525532859fc0fdb8989a911a5d347d540`). Stacked base remains PR27
`5fc8b7f32859e071e9c74d1269429ba945c4b3af`; main remains
`fdd37780930ef83431edcca404447b72a9c21976`. No retarget or integration.

## Finding and behavior

S2-F1/P2: default MediaPipe factories and ONNX session creation recorded global
asset sources before VisionEngine's generation acceptance. Delay A's cached task,
switch to B, accept a new A via upstream fallback, then finish the original A:
the old result correctly returned cancelled and closed its task, yet global model,
variant and/or WASM provenance could overwrite the current observation. A variant
key alone cannot distinguish different requests for the same variant.

The engine now creates a private source journal and default factory for each
initialization request. MediaPipe task/WASM and both ONNX services write through
that request's recorder. Only the existing current-generation task acceptance
commits the journal. Failed and cancelled requests discard it. Ready ownership
and all model/variant/WASM source entries are set before source observers run;
one batch notification avoids partial model/variant publication during synchronous
observer reentry. SDK errors, selected model URLs, default GPU request, cancellation
and exactly-once stale-task cleanup remain intact.

The optional ONNX recorder also covers the engine's built-in creator fallback
when a custom factory omits an ONNX creator. Standalone service/session callers
retain their prior immediate source-recording behavior; they do not claim engine
request ownership. Arbitrary injected factories that independently mutate global
state are outside the default production-factory guarantee. Last accepted source
records are not rewritten into a successful observation when a new request fails.
This correction does not change global ORT configuration or claim to measure the
SDK's actual internal backend.

## Reproduction and coverage

Run `npm run --silent test -- src/services/provenanceOwnership.test.ts`.
The 28 controls use the actual resolver, default factories, engine and ONNX service
wrappers with controlled SDK/session promises and status fetch. No model bytes,
real inference or external acquisition are involved.

- Four models: same-variant A/B/A with obsolete success and terminal failure.
- Four models: stale cross-variant completion cannot rewrite current model/UI source.
- Four models: repeated dispose/restart with two obsolete successes completed in
  reverse order; each stale task/session closes/releases exactly once.
- Four models: obsolete local failure followed by successful upstream fallback.
- Four models: current task failure discards earlier successful WASM observations.
- MediaPipe: obsolete primary WASM success and obsolete fallback WASM success.
- Atomic accepted-source observation, including synchronous selection of another
  variant inside the source notification.

Initial 22 controls against the rejected source passed8/failed14, exit1. After
adding further failure/fallback/reentrant vectors, the final identical28-case file
against that source passed6/failed22, exit1. These are expected red reproductions,
not successful application checks. Their full ignored raw/report files are kept.
The fixed focused population is261 =28 new +224 existing Stage2 +3 cache +6 engine
reentrancy cases. Final complete offline and exact-source hosted results are bound
in the new receipts and PR28 body; earlier failed and successful records remain
historical. Independent closure remains PENDING.

The coherent PR keeps its existing npm-generated1.0.5 PATCH, above target1.0.4 and
main1.0.3; no additional version authority, dependency or tag is introduced. All702
protected historical hashes and the original guard/CI/configuration/UI assertions
remain. CSS parity is checked against the prior reviewed build. No merge, deploy,
retarget, settings/permissions change, real model/denied ZIP request, alternate
acquisition route or Windows work. Six old real MediaPipe CPU cases remain BLOCKED.
