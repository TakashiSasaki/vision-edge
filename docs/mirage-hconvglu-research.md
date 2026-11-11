# Mirage checkpoint and isolated HConvGLU verification

This is a **partial operator conversion**, not an image encoder, detector, OCR
system, or accepted Mirage replacement. `MIRAGE_ONNX_NOT_VALIDATED` remains
enabled and the official WASM model remains the default. No production model
route, decoder, cache key or availability decision is changed.

## Acquired official artifacts

The environment was restarted with public networking available. The owner's
resume request authorized a fresh official acquisition after the previous
environment's recorded CONNECT denial. No alternative model source or denial
bypass was used. The previous evidence is retained unchanged.

Live official Hub metadata and full local hashes agree at revision
`76c1e9e53381aca8fc3843c636216c96ff220122`:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| mirage.safetensors | 216370504 | 6ba97c5c1bef661c9b0c80782135682b38b4312fb33e0b645c93c10604bb16cd |
| mirage.cact | 21611043 | 4249a0f7515272ff76b9414bec17d7e7daf84b15b66c7fb4f17fce1e66f9f237 |
| config.json | 1430 | f7a3c37642086f954b1000fe80edb544a153c76fe68fb1b8173ddb97d465e915 |

`scripts/mirage/acquire_reference.py` explicitly checks the live revision,
artifact sizes/LFS hashes, config Git blob identity, and complete downloaded
files. It performs no retry or fallback. Existing corrupt artifacts are rejected.
Model binaries and ONNX graphs remain in the ignored cache.

The Safetensors inventory contains **583 FP32 tensors**. The complete
[`mirage-tensor-mapping.json`](verification/mirage-tensor-mapping.json) records
every name, shape, dtype, relative/absolute offset, and mapping status.
Exactly 112 tensors map to the 28 isolated mixers. The other **471 tensors are
explicitly out of scope and unmapped**, rather than silently dropped from a
supposed complete model. Config validation is limited to the pinned config and
the subset's required tensor set/shapes; full checkpoint/architecture
compatibility has not been established.

## Reference assessment

The official Cactus source at
`bafcf7348607c89f30ff9fde8e59d20b76baa86f` was fetched and inspected.
`cactus/agent/mirage.py` delegates to `cactus_describe`. The public source has no
complete checkpoint-matched Mirage neural forward implementation or intermediate
HConvGLU API in the inspected tree. This bounds the source assessment; it does
not prove the implementation is unavailable elsewhere.

The pinned [Oasis reconstruction](https://huggingface.co/phanerozoic/oasis/blob/75bf8f3047e88a21c10d1f4175450656704a3c1e/reference.py)
was obtained (SHA-256
`ce84a5da7016d141269ecce6bcb4eabcfcaada74a2ff189856a4e94c3913ff62`).
Its HConvGLU agrees with the published equations. Its `OPTS` explicitly leaves
stride-two padding, canvas fill, 2D rotary layout/axis order, Walsh ordering,
level gates, and resampling selectable. Those are unresolved full-model
assumptions. Oasis is not imported by this exporter and is not treated as an
official runtime reference.

The independent operator reference follows the
[official HConvGLU specification](https://cactuscompute.com/blog/hconvglu):
normalized sequency butterfly, output-channel-indexed zero-padded 65-tap bands,
gate bias, tanh-approximation GELU, value product, output band, inverse butterfly.
The exporter instead constructs the Walsh matrix from bit-reversed Gray codes
and explicit band matrices. A second export folds adjacent linear maps in FP64
before rounding constants to the selected arithmetic type. GELU remains
nonlinear. Normalization and residual wrappers are outside this operator scope.

Each mixer uses its original 512-channel tensors. Each band stores 33,280 taps;
1,056 boundary slots multiply zero by specification. Folding excludes only those
mathematically inactive slots. No learned tensor is synthesized or randomized.
Missing, extra, wrong-shape, wrong-dtype and nonfinite subset tensors are rejected
before any graph is exported.

## Actual numerical results

Both graph forms passed the full ONNX checker and executed with the explicitly
requested ONNX Runtime CPU provider. There are 28 blocks, two graph forms and
six immutable feature-vector fixture groups. The saved report identifies each
ONNX hash, converter/reference source hashes, checkpoint/config identities,
versions, input hashes, and every measured stage.

The fixtures are synthetic vectors with genuine pretrained weights. They include
zeros, all 512 channel impulses, a signed ramp, two modular patterns and 33
distinct integer-mixed vectors. The two 17-row modular patterns each contain
only two distinct vectors; repeated rows are not independent coverage. These
are not model-accuracy or image-encoder tests. Fixture identities are pinned in
`scripts/mirage/fixtures/hconvglu-inputs.json`.

| Experiment | PASS comparisons | FAIL comparisons | Worst output absolute error | Worst output RMS error |
| --- | ---: | ---: | ---: | ---: |
| FP32 explicit matrices | See detailed report | See detailed report | 0.046875 | 0.0036796504 |
| FP32 folded maps | See detailed report | See detailed report | 0.015625 | 0.0018295820 |
| FP32 combined | 1991 | **25** | 0.046875 | 0.0036796504 |
| FP64 diagnosis, combined | 2016 | 0 | 3.092282e-11 | 3.385177e-12 |

FP32 acceptance uses `abs(actual-reference) <= 2e-4 + 2e-4*abs(reference)` per
element, without changing the threshold after failures. The relative-error
denominator is `max(abs(reference), 1e-6)`; RMS uses FP64 differences. The report
includes shape, dtype, maximum absolute/relative errors, RMS, threshold and failed
element counts at each captured operation. Large relative errors near zero are
retained rather than suppressed.

All 1,680 comparisons on the five bounded fixture groups passed; the 25 failing
comparisons belong to the amplitude-eight stress pattern. The first threshold
crossing in block/stage traversal is `body_1/hconv`, explicit graph, stress
fixture, final output. Some later blocks first cross at product or output-band
stages. The complete FP32 experiment **exits 1 and reports FAIL**.

FP64 diagnosis promotes the same original FP32 checkpoint values and inputs to
FP64 arithmetic. With separate `atol=1e-8, rtol=1e-9`, all comparisons pass.
This supports the mathematical mapping and is consistent with accumulated FP32
rounding/cancellation being responsible for stress differences. It does not
validate the FP32 failures, establish actual encoder activation ranges, or
validate WebGPU. No tolerance was relaxed and no input was removed.

## Official native observations and limitations

The official 4.1.0 Linux x86_64 wheel was validated against its pinned release
manifest before extracting `libcactus.so`. Model, manifest, wheel, runtime
revision and extracted-library identities appear in
[`mirage-official-acquisition.json`](verification/mirage-official-acquisition.json).
`capture_native_reference.py` verifies the `.cact` and library again and calls
the public C ABI directly, with no provider fallback or model download.

Seven deterministic RGB images exercise square/landscape/portrait blank inputs,
small/boundary/overlapping shapes, masks and bitmap text. Six calls returned
parseable objects/text JSON. The portrait blank returned a positive native code
but **invalid JSON containing `"score":-nan`**. Its exact raw response is retained
in `mirage-native-portrait-invalid-response.txt`; the capture exits 1 and records
FAIL while retaining all other observations. Blank images also produce named
detections, and the bitmap text fixture produced no OCR result. None of those
observations is an accuracy PASS.

There is no ONNX image model to compare with those native boxes/scores, names,
masks or OCR. Bounding-box IoU, mask IoU and text agreement are explicitly null,
not zero or fabricated. Natural multi-category semantic fixtures are still
missing. The native API cannot expose matching mixer intermediates. Its mixed
quantization is distinct from the FP32 source checkpoint; no quantization-error
estimate is claimed from these isolated-operator results.

WebGPU was not run: the full FP32 experiment fails and encoder/head conversion
has not passed CPU acceptance. The next implementation step is to determine
checkpoint-matched preprocessing/stem padding and capture encoder activations,
then validate the surrounding body blocks and attention against a trustworthy
full forward reference. The stress failures require diagnosis using those
activation ranges before declaring even FP32 operator acceptance complete.

## Reproduction

Use Python 3.12 in an isolated ignored environment; the application's Node/npm
dependencies are unchanged. This environment used torch 2.6.0+cpu, NumPy 2.2.6,
ONNX 1.17.0, ONNX Runtime 1.20.1, Safetensors 0.5.3, Pillow 11.1.0.

```sh
python3 -m venv .cache/mirage-python
.cache/mirage-python/bin/python -m pip install -r scripts/mirage/requirements-hconvglu.txt
python3 scripts/mirage/acquire_reference.py --directory .cache/mirage-reference --evidence-dir .cache/mirage-acquisition-new
python3 scripts/mirage/verify_checkpoint.py --checkpoint .cache/mirage-reference/mirage.safetensors --config .cache/mirage-reference/config.json --cact .cache/mirage-reference/mirage.cact --output .cache/mirage-reference/inventory-new.json
python3 -m unittest discover -s scripts/mirage -p 'test_verify_checkpoint.py' -v
python3 -m unittest discover -s scripts/mirage -p 'test_acquire_reference.py' -v
.cache/mirage-python/bin/python -m unittest discover -s scripts/mirage -p 'test_hconvglu.py' -v
.cache/mirage-python/bin/python scripts/mirage/verify_hconvglu.py --checkpoint .cache/mirage-reference/mirage.safetensors --config .cache/mirage-reference/config.json --output-dir .cache/mirage-fp32-new
.cache/mirage-python/bin/python scripts/mirage/verify_hconvglu.py --checkpoint .cache/mirage-reference/mirage.safetensors --config .cache/mirage-reference/config.json --precision float64 --output-dir .cache/mirage-fp64-new
```

The FP32 command currently exits 1; retain its complete log and validation JSON.
Use fresh output/evidence directories. Acquisition is an explicit network action
and must not be used to retry a policy denial without authorization. None of
these commands is an npm startup hook. Ordinary CI runs the 13 checkpoint and
two metadata controls without real model/Python numerical dependency downloads.
The six operator unit controls are synthetic and separate from actual-weight
ONNX execution.

For the native experiment, check out the official Cactus frontend revision above
in an ignored directory, install it into the research environment, then invoke
its `cactus.agent.fetch.fetch_library(dest_dir=...)` explicitly. It verifies the
pinned runtime manifest and wheel. The library must have SHA-256
`2a4c9afa68aad5122dacb0adc60773988622debbffe4572f535404a676fb0e6a`.
Then run, using a fresh output directory:

```sh
.cache/mirage-python/bin/python scripts/mirage/capture_native_reference.py --cact .cache/mirage-reference/mirage.cact --library .cache/mirage-runtime/libcactus.so --output-dir .cache/mirage-native-new
```

This currently exits 1 for the recorded native JSON failure. The acquisition
report records the exact wheel identity; it is not inferred from version alone.
Raw logs, initial failures, generated fixtures and ONNX files are preserved under
`.cache/mirage-resume/`. Committed numerical reports contain measurements and
identities; model binaries are excluded from Git.

## Review and application regressions

The cumulative change from PR #36's `dc28e44` to implementation commit
`5795226eda397501d00c0d479506ed93b44ca08a` (tree
`7b79c69c126f5ea67d2974c1bfad7df1937273c5`) received independent review with no
blocking code findings for this research-only scope. The reviewer independently
checked the original artifact identities, all 583 inventory rows, all 112 saved
ONNX hashes/checkers, detailed comparison counts, and the native fixture hashes.
They reran all 21 controls and reproduced the first FP32 stress failure.

The prescribed `verify:offline` application checks all passed: preflight with
702 protected files unchanged, 60 preflight controls, typecheck, lint, 847 unit
tests, 73 foundation tests, 40 delegate tests, 37 output tests, build, 33 browser
tests, and seven stylesheet controls (two healthy, five expected faults).
The built CSS, application JavaScript and WASM-worker JavaScript are byte-for-byte
identical to the pre-documentation/version-bump build using the same dependencies.
The version is 1.0.32; no application source or npm dependency graph changed.

[`mirage-research-validation.json`](verification/mirage-research-validation.json)
records the commands, counts, raw-log hashes, independent review and asset hashes.
The local offline run started with the pre-commit dirty tree; that metadata is
retained accurately. Its tested source was then committed without modification.
Hosted PR checks are a separate verification of the final committed head; do not
substitute the old PR #36 CI or local launch SHA for those checks.
