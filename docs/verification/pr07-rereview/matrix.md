# Independent re-review matrix

Exact target21417613 / tree1abe52da. All executions below are NEW unless explicitly labelled historical. PASS is limited to the named check.

| Check | Result | Qualification |
| --- | --- | --- |
| F01 original invalid manifest probe old/new | CLOSED | Child0/0/0/1 →1/1/1/1; outer script0 both, read actual children |
| F02 original old checker | REPRODUCED old failure | Old source accepts stale-empty counterexample |
| F02 original adapter on new checker | ADAPTER ERROR | TypeError retained, never used as closure |
| F01 independent controls |38PASS | Required nine-asset inventory and invalid variants |
| F02 independent controls |174PASS | Both models ×3input kinds; fresh vs stale/failed/unknown/incorrect chains |
| F03 independent controls |61PASS | Actual old/new identical zero-call fixture plus stage/status/counter variants |
| Author focused controls independently rerun |73PASS |7original integrity +30F01 +26F02 +10F03; separate from317unit |
| Target install/static/unit/build/isolation | PASS |317unit/component; source unchanged during these checks |
| Target ordinary browser/control |19PASS /7PASS | Serial, workers1/retries0, unchanged assertions/waits |
| Fresh main baseline install/static/unit/build/browser/control | PASS /317/19/7 | New main9dd0f79 executions; recovered old results not substituted |
| Author native provenance rerun |3PASS | Actual unchanged scheduler; model-free |
| Additional independent native scenes/restart |3PASS | Requested-vs-actual source identity, stream2, multiple scenes, blank marker rejection, frame/track cleanup |
| Actual cached inventory | exit2 |7available pins pass;2null model prerequisites blocked |
| Same-size changed WASM | asset1 /runner1 | Corrupt preflight fails; no browser child, no inference |
| Missing WASM | asset2 | Missing prerequisite blocked |
| Missing WASM with actual model503 diagnostic | child1 EXPECTED | Exact required modelURL/503 assertion fails; generic startup error cannot count as success |
| EfficientDet actual SDK controlled503 | PASS | Genuine SDK creation attempt, exact model error, no inference |
| Selfie actual SDK controlled503 | PASS | Same scope |
| EfficientDet ×VideoFrame | BLOCKED |0inference |
| EfficientDet ×owned Canvas | BLOCKED |0inference |
| EfficientDet ×live HTMLVideoElement | BLOCKED |0inference; latency remains unbound later |
| Selfie ×VideoFrame | BLOCKED |0inference |
| Selfie ×owned Canvas | BLOCKED |0inference |
| Selfie ×live HTMLVideoElement | BLOCKED |0inference; latency remains unbound later |
| Actual new diagnostic summary | all flags false /no populations | No file-existence or zero-call success claim |
| Protected target /prior author evidence |784/635unchanged | No old screenshot regeneration published; disposable trees restored |
| Production JS /CSS | JS payload identical; CSS differs | Exact target adds one `.transition` rule; filenames differ; observer/double isolation pass |
| Target CI40119850663/job120948814304 | completed/success | Raw exact21417613 checkout;317/19/7; optional real models not run |
| Physical-device/model/GPU/audio/mobile/background/public equality | NOT RUN | Cannot be established by cloud negative or native-input tests |
| CPU diagnostic | NOT RUN | Product GPU request unchanged |
| Recovered old baseline/raw logs | HISTORICAL ONLY |21log hashes recovered; former uncertainty at PR18 checkpoint remains documented; fresh runs above establish current baseline |
| Old uncertain corruption/promotion submission | NO EXECUTION RECEIPT | Old script/log absent and cache matched pin; no retrospective test-success claim |

Foundation PASS WITH CONDITIONS, F01–F03 CLOSED; actual real-inference objective BLOCKED / NOT ACHIEVED. No merge/deploy.
