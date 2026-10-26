# SSD MobileNetV2: gated third ObjectDetector variant

Base/main:479c74527a5b5e13f4f5938c6f26787e60890f14, tree
72d346c1158abf3dbef6073e31fccc2721364985, version1.0.8. This bounded Draft
increments npm package/lock to1.0.9 without dependency changes or tags. Final
HEAD/tree and hosted CI binding are recorded in the PR; independent review is
PENDING. No merge/deployment is authorized.

## Behavior and boundaries

The existing mediapipe family is named MediaPipe Object Detector, with Lite0
still default and Lite2 retained. ssd-mobilenet-v2 is a third variant, not a new
engine or model family. Four detection/segmentation engines plus whole-image
classification remain five families; the catalog now has twelve variant rows.
Existing family-scoped parameter controls, scheduler, generation/provenance
ownership, camera, boxes, labels, scores, and speech paths are reused. No cropping,
segmentation, index offsets, second sigmoid, hand-written SSD decoder/NMS, silent
CPU fallback or performance assertion is introduced. GPU remains the product
request; it is not a measurement of the actual execution backend.

SSD is selectable but explicitly BLOCKED before production WASM/SDK or model I/O.
Its default initialization and explicit retry cannot silently load Lite0 or Lite2.
The API exposes its unique file/variant/pins, actual file presence, null verified
byte data and a coherent blocked integrity state. Neither a filename nor forged
sidecar/status can enable acquisition, cached readiness or serving. Startup keeps
the same eleven enabled requests and never fetches the twelfth/SSD variant. The
static inventory distinguishes policy block from absence/corruption, and keeps
C31-F1's contradictory-state validation. Its successful-cache display also refuses
an acquisition-disabled/policy-blocked row even if it asserts matching hashes.

The pinned-variant resolver supports only matching container identity/pins,
validated cached bytes or truthful pending acquisition states, and never direct
unverified upstream fallback. The current SSD gate remains mandatory; test-only
catalog substitutions and injected SDK/task boundaries exercise later wiring
without introducing a production override flag. Legacy nine enabled variants'
existing fallback rules and both EfficientNet pins are preserved.

## Evidence, license and SDK compatibility

The official [overview](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector#ssd_mobilenetv2_model)
documents256x256/float32/COCO80. The official
[sample](https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/object-detector.ts)
references the exact candidate GCS URL in the catalog. That moving sample is not
proof of compatibility with our installed tasks-vision0.10.14.

The owner supplied an actual SSD file. Independent inspection of that upload
reported11316189 bytes, SHA256
b8ccb1a25d45455ba52e85f26531948e1cb75efeb94c7c3d456d54fd4d6fbdd2,
TFL3/schema3, ModelMetadata name ObjectDetector/author MediaPipe/version1. Its
license field is absent; no embedded Apache declaration is claimed. A MediaPipe
collaborator explicitly includes models in the project's Apache-2.0 coverage,
unless otherwise noted, in the official
[2026-10-05 statement](https://github.com/google-ai-edge/mediapipe/issues/6355#issuecomment-5997480830).
This is stronger than assuming weight licensing from SDK licensing alone. No
SSD-specific exception was established; exact-file metadata and project-level
license evidence are kept separate. Public attribution/Apache terms are linked.

Uploaded bytes have FLOAT32 input[1,256,256,3], mean/std127.5 and **two RAW SSD
outputs**, location[1,12276,4] and score[1,12276,91]. The91 label positions contain
background0 and gaps12,26,29,30,45,66,68,69,71,83, leaving80 semantic classes.
No guessed contiguous80 map is created. SDK Category.index is a position in its
model label file ([official contract](https://developers.google.com/edge/api/mediapipe/js/tasks-vision.category));
the existing engine takes SDK categoryName/score directly, not an external index
lookup. The model's score graph already applies LOGISTIC.

The294764-byte DETECTOR_METADATA custom metadata has identifierV001. The supplied
independent decoder inspection reports classes91, boxes12276, coords4, scales1,
exponential box size=true and sigmoid_score=false. The official
[v0.10.14 schema](https://github.com/google-ai-edge/mediapipe/blob/v0.10.14/mediapipe/tasks/metadata/object_detector_metadata_schema.fbs)
has V001 and the parser/anchors/decoding slots. Its
[native postprocessing source](https://github.com/google-ai-edge/mediapipe/blob/v0.10.14/mediapipe/tasks/cc/components/processors/detection_postprocessing_graph.cc)
explicitly supports two output tensors and consumes DETECTOR_METADATA. Installed
0.10.14 types expose createFromOptions and detectForVideo. These are static/API
compatibility evidence, **not a successful execution of this exact file in the
installed WASM**. Requiring four post-NMS outputs or adding another sigmoid/NMS
would misinterpret this model.

Provenance is user-uploaded file plus independent inspection results received by
this author. Equality with the official GCS object was not independently verified.
The current Library skill and exact resolved file were used for consumer-local
materialization, including one bounded fresh-reference retry. Both current
helper invocations exited1 with only `library file transfer failed: download
failed`; no installed file existed at either consumer-local destination. No
HTTP denial reason was returned, so this is not relabelled as a403 or a credential
problem. No generic downloader, guessed URL, reupload, other machine, mirror or
GCS retry was used. The author therefore could not independently rehash or run
the provided model in this coding executor. The supported handoff needs repair
before native execution. Private transfer references/URLs are not published.

## Verification scope and remaining work

Focused controls cover twelve-row registry identity, unique SSD pins, explicit
pre-I/O blocking, forged cache/status refusal, inactive SSD startup acquisition,
strict pinned resolution with controlled eligibility, exact GPU/VIDEO task
arguments, raw label/score/box behavior, Lite0->SSD->Lite2 switching, obsolete
initialization/task/frame disposal, errors/retry, blocked UI and camera stop/resume.
No synthetic result is called real TFLite inference. Browser tests distinguish
the normal production bundle from the existing explicit CI SDK double.

Legacy source-regression matrices retain all original nine enabled cases; only
the new blocked SSD row is excluded from those positive-enabled matrices and has
its own negative/controlled-positive controls. Classifier/static inventory and
all historical checks remain. Screenshots use testInfo.outputPath, not historical
paths. New API integrity='blocked' is policy unavailability, not a successful hash
check or a claim that absent bytes are corrupt.

The actual model0.10.14 initialization/inference, raw model outputs in the browser,
same-input latency/accuracy comparison, physical GPU and Windows/macOS acceptance
are NOT RUN. Six older MediaPipe CPU scenarios remain BLOCKED. Cache APIs are
still Vite-dev-only, absent from preview/static hosting; deployed behavior and
hosted ZIP contents remain unverified. Required next step is supported receipt
of the same lawful bytes followed by bounded genuine-SDK validation; any later
acquisition/serving enablement needs reviewed evidence and code, not a UI flag.

Final commands/counts, failed attempts and preservation/build observations are
recorded in the adjacent results JSON after verification. Source hashes in old
reports remain historical; those files are not regenerated.

Local `verify:offline` completed exit0 with826 unit,35 browser,73 foundation,
37 output,40 delegate,57 guard,7 controls (2 healthy/5 intended faults), types,
lint and build. Three later controls exercise generated bad bytes against actual
SSD pins plus controlled valid synthetic serving with SSD-specific attribution.
After that narrow header correction, focused51, final **829/829 unit**, typecheck
and lint pass. Original35 browser population is unchanged. Final hosted CI will
run the complete corrected source. Focused counts overlap full populations.
Zero failed/skipped/cancelled/todo/flaky in final checked populations.

One initial11-case SSD run had9 pass/2 failures because the newly pinned SSD URL
still lacked its hash query; the server correctly returned400 before the expected
policy503. The catalog URL was corrected; no assertion or timeout/retry was
weakened. Library transfer failures are separate from test/model failures.
702 historical hashes match; screenshots use unique test outputs. Production CSS
matches main exactly (29713 bytes); application JS intentionally changes for the
new registry/gate/UI. No dependency or historical evidence change. See
[receipts and model metadata](ssd-mobilenet-v2-results.json).

Final explanatory correction: after initial source e0e25f689885a189a74b2718553698341935ed1d
passed CI40290575748, the comparison dialog's superseded uninspected-bytes
explanation was clarified to the actual remaining0.10.14 execution prerequisite.
A new component control verifies that wording and selection of the same mediapipe
SSD variant through the specification dialog. Pins and availability gate are
unchanged. Fresh final830/830 unit, type/lint/build pass; earlier829 and full826
records remain historical. Final hosted CI binds the complete new source.
