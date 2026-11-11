# Mirage reference-order arithmetic and native geometry diagnostics

The candidate now reproduces the pinned Oasis PyTorch computation exactly at
all 2,640 captured eager/reference checkpoints in this experiment. This is
**third-party-reference agreement, not official Mirage equivalence**. ONNX
numerical verification still **FAILS** at the unchanged tolerances, and full
native equivalence remains **NOT_ESTABLISHED**. The application retains
`MIRAGE_ONNX_NOT_VALIDATED` and the native WASM default. No browser/WebGPU
acceptance is claimed.

This is an additive follow-up to [the folded encoder experiment](mirage-encoder-candidate.md).
Its historical reports and failures remain unchanged. The scope is still the
452 mapped encoder/dense-head tensors; 131 original checkpoint tensors remain
outside this partial conversion. Names, OCR decoding and the language/region
computation are not implemented by these graphs.

## What Oasis does and does not establish

[Oasis's reference.py](https://huggingface.co/phanerozoic/oasis/blob/75bf8f3047e88a21c10d1f4175450656704a3c1e/reference.py)
is an unofficial third-party PyTorch reconstruction, pinned to revision
`75bf8f3047e88a21c10d1f4175450656704a3c1e` and full-file SHA-256
`ce84a5da7016d141269ecce6bcb4eabcfcaada74a2ff189856a4e94c3913ff62`.
It provides inspectable forward calculations and intermediate tensors. Our
loader hash-checks the file and executes only audited forward definitions,
without its top-level program, imports or model constructor.

Reproducing that source removes differences between our eager candidate and
that particular reference. It cannot validate the reference's own choices.
Official published equations and the pinned weights support parts of the
mapping, but native resizing, RoPE/layout choices, activation arithmetic and
full-model behavior still require independent official evidence. The native
C ABI exposes final structured outputs rather than the required intermediate
tensors. Its quantized archive is also a different weight representation from
the original FP32 checkpoint. Final-output differences cannot therefore be
assigned uniquely to an Oasis error, preprocessing, quantization or a native
kernel approximation.

## Arithmetic changes

`encoder_reference_order.py` keeps the candidate's strict original-weight
validation but replaces folded HConvGLU matrices with explicit normalized
Walsh transforms, padded 65-tap Gather windows, multiply/ReduceSum, tanh GELU,
and the output transform. Pointwise layers use NHWC matrix multiplication.
RoPE frequencies use reciprocal-after-positive-power, matching the reference's
FP32 rounding rather than the mathematically equivalent negative-power form.
Unused folded matrices are not calculated or stored.

The complete combination reproduces Oasis exactly for the measured eager
cases. The unfused-only exploratory probe did not eliminate all differences;
this is not evidence that a single change explains every improvement. FP32
operation order matters even when the real-number formulas are identical.
ONNX export and runtime kernels can introduce different reductions/fusions,
so eager agreement does not imply ONNX agreement.

## Completed numerical measurements

Forty numerical cases cover two explicit weight sources (original FP32 and
archive-dequantized), two preprocessing hypotheses, and the same ten original
fixtures. Each captures 66 checkpoints for three pairwise comparisons. Both
graphs pass the full ONNX checker and run on `CPUExecutionProvider` only, with
ORT's recorded default optimization. All comparisons retain
`abs(actual-reference) <= 2e-4 + 2e-4*abs(reference)` and relative denominator
floor `1e-6`.

| Weights / resize rule | Eager vs Oasis PASS / FAIL | ONNX vs Oasis PASS / FAIL | ONNX vs eager PASS / FAIL |
| --- | ---: | ---: | ---: |
| Original FP32 / pixel | 660 / 0 | 616 / **44** | 616 / **44** |
| Original FP32 / nearest32 | 660 / 0 | 614 / **46** | 614 / **46** |
| Archive-dequantized / pixel | 660 / 0 | 621 / **39** | 621 / **39** |
| Archive-dequantized / nearest32 | 660 / 0 | 622 / **38** | 622 / **38** |

All eager/Oasis maximum absolute errors are actually zero, rather than merely
below tolerance. Each ONNX comparison label totals 2,473 PASS and **167 FAIL**;
the worst absolute error is 0.451171875. These are checkpoint comparisons,
not failed-image counts or failed-element counts. Detailed element counts,
shapes, dtypes, RMS and relative errors remain in the structured report.

Improvement is not uniform: for the same pixel/symmetric configuration, the
old folded graph had 55 failed ONNX/Oasis checkpoints with original FP32
weights and 29 with archive-dequantized weights; the new graph has 44 and 39,
respectively. Reference-order eager agreement is therefore not an ONNX
accuracy or performance win in every configuration. None of the numerical
failures has been accepted, removed or hidden by an increased tolerance.

## Separate native geometry observations

A new, explicitly supported native request asks only for `bbox` and polygon
`mask` fields in the `objects` and `text` sections, with confidence thresholds
0.3, at most 64 objects and 32 text lines, and language `en`. All 22 calls
produce observed, parseable outputs. This changes the requested API scope;
it does **not** fix the old two full-score invalid-JSON responses containing
`score:-nan`. Those original raw failures remain unchanged. Scores are neither
requested nor compared here; score differences are null, never invented zeroes.

The same ten fixtures plus twelve aspect-ratio probes compare the old folded
and new reference-order graphs, both with archive-dequantized weights and
symmetric padding. The probes resize one existing photo to short dimensions
264, 272, 280, 321, 327 and 400 at long dimension 512, in both orientations.
They are correlated variants of one photo, not twelve independent scenes.
Their historical `holdout_*` filenames do not make them acceptance holdouts:
these inputs were inspected while choosing the hypothesis and are now
**exploratory probes**. A fresh, independently chosen corpus is still needed.

Three preprocessing hypotheses use Pillow bilinear, top-left normalized-zero
padding and explicit valid extents: pixel rounding, nearest 32 with half-up
ties, and ceiling to 32. For snapped dimensions the inverse coordinate mapping
uses the actual horizontal and vertical scale separately. Nearest32 is an
empirical hypothesis, not a proven native rule or a production default. The
327-pixel probes remain counterexamples to treating it as universal. The
native resize filter itself has not been established.

The following totals combine object boxes and text-region boxes while keeping
the groups separately matched. They are **not object-only or OCR accuracy**.
Every row has 136 native regions. Values are for the new reference-order graph
across all 22 inputs:

| Resize rule | Candidate regions | Matched at box IoU >=0.5 | Unmatched native | Unmatched candidate | Mean matched box IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| pixel | 166 | 112 | 24 | 54 | 0.904224 |
| nearest32 | 169 | 124 | 12 | 45 | 0.981497 |
| ceil32 | 164 | 117 | 19 | 47 | 0.970865 |

Means are conditional on successful one-to-one matches and omit unmatched
regions; they must be read with the unmatched counts. Object-only nearest32
results are 107 matches among 113 native objects, with six unmatched native
and 43 unmatched candidate objects. Text-region results do not compare text.
The old folded graph has the same match counts for every rule, so the principal
geometry improvement here comes from preprocessing/inverse coordinates, rather
than reference-order arithmetic.

On the original portrait photo, pixel preprocessing matches four of seven
native objects with mean matched box IoU 0.952476; nearest32 matches six of
seven with IoU 0.999376, but produces nine candidate objects. On the rotated
landscape photo, the corresponding counts improve from six to seven matches
among seven native objects, with nine nearest32 candidate objects. This is
useful diagnostic evidence, not full native parity.

Matching remains greedy, class-agnostic, one-to-one maximum box IoU >=0.5.
Masks use the existing diagnostic 128-square original-coordinate raster
comparison, with invalid native polygons retained and null mask metrics.
Object names, confidence agreement, OCR text and native-resolution mask
identity are outside this measurement.

## Evidence, completion and reproduction

- [Complete aggregate](verification/mirage-refinement-results.json): all 40 numerical cases (7,920 detailed comparisons), 132 geometry records, exact fourteen-scope coverage, graph hashes, source hashes, inputs, runtime versions and unchanged tolerance.
- [Native geometry capture](verification/mirage-refinement-native-geometry.json): pinned model/library, options, 22 raw responses and hashes.
- [Execution record](verification/mirage-refinement-execution.json): commands, completion, controls, interruption and local regression evidence.

Original artifacts retain model revision
`76c1e9e53381aca8fc3843c636216c96ff220122`. Numerical tooling remains isolated
in `.cache/mirage-python`, with the pinned research requirements; Node/npm
application tooling is separate. No model is downloaded by CI controls.

An initial unbounded run stopped after eight cases with no retrievable executor
exit code or traceback. No cgroup OOM event was recorded; its cause remains
unknown. Its incomplete report and raw log are preserved and excluded from
acceptance/aggregation. Eight five-case numerical scopes and six eleven-input
geometry scopes then completed. Numerical scopes exit 1 for real retained
failures; geometry scopes exit 0 with `GEOMETRY_DIAGNOSTICS_ONLY`, not numerical
acceptance. Aggregation checks exact coverage, shared provenance, graph hashes,
status/element consistency, one-to-one matches and absence of fabricated scores;
its final **FAIL** also exits 1. Zero numerical coverage is rejected.

With the previously verified cached artifacts, one bounded example is:

```sh
.cache/mirage-python/bin/python scripts/mirage/verify_refinement.py \
  --checkpoint .cache/mirage-resume/mirage.safetensors \
  --config .cache/mirage-resume/config.json \
  --cact .cache/mirage-resume/mirage.cact \
  --photo .cache/mirage-encoder/coco-231049.jpg \
  --oasis-reference .cache/mirage-resume/oasis-reference.py \
  --geometry-report .cache/mirage-refinement/geometry-1/geometry.json \
  --baseline-onnx .cache/mirage-encoder/cact-dequantized-symmetric-final/partial-encoder-and-heads.onnx \
  --baseline-validation .cache/mirage-encoder/cact-dequantized-symmetric-final/validation.json \
  --output-dir .cache/mirage-refinement/reproduction-new \
  --weights cact-dequantized --rule pixel --numerical-only \
  --cases square_blank landscape_blank portrait_blank small_objects boundary_objects
```

The output directory must be new. The execution manifest records the exact
measured paths. To repeat the geometry phase use
`--geometry-only --weights cact-dequantized`, select one of the three rules and
an explicit subset of fixture names. Each geometry scope compares both graphs.
The aggregate's report list identifies all measured partitions and their hashes.
Cached graphs/raw logs are local research artifacts, not browser-shipping assets.

The next correctness step is to obtain independently chosen native observations
and, if available, official intermediate outputs or kernel source. ONNX
primitive-level localization remains necessary before considering browser GPU
acceptance. A custom WebGPU implementation would need the same reference and
numerical validation; changing runtimes does not establish Oasis's correctness.
