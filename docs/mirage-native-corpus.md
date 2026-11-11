# Official Mirage comparisons on new real images

This study compares the pinned official Cactus/Mirage native runtime with the
existing partial PyTorch/Oasis and ONNX encoder/dense-head candidates on twelve
distinct COCO val2017 source images. Eight development images select a
preprocessing/postprocessing hypothesis; four reserved images are evaluated
only after that selection is written and hashed. The source images are not
committed. Their encoded and decoded RGB identities, dimensions, source URLs
and split are fixed in `scripts/mirage/fixtures/native-corpus.json`.

**Official Mirage parity is not established.** The application still uses its
native WASM default and rejects the ONNX variant with
`MIRAGE_ONNX_NOT_VALIDATED`. These research scripts do not change the exporter,
model weights, runtime, application preprocessing or application decoder.
Only 452 of 583 tensors are mapped. Names, OCR strings, language/region
computation and final semantic selection remain outside the candidate.

## Fixed experiment

The protocol was recorded before any corpus inference. Both original FP32
weights and separately archive-dequantized weights use the exact existing ONNX
graphs and pinned unofficial Oasis forward definitions. Native inference uses
the pinned official mixed-precision `.cact` model and `libcactus.so`, with two
threads. All three runtimes receive the same checksum-verified decoded RGB
image; these twelve JPEGs have no EXIF orientation. Native internal resize and
kernel details remain unobserved.

The development grid comprises pixel rounding, nearest multiple of 32 and ceil
multiple of 32 for the resized short side, each with no NMS or greedy IoU 0.5
NMS. The canvas remains 512 square, with Pillow bilinear interpolation and
explicit independent inverse scales for the two axes. NMS, when enabled, runs
before the 64 object/32 text proposal caps. It suppresses strictly greater
than 0.5 IoU, separately by group, in score/cell order. It changes no raw head.
These are explicit hypotheses, not claims about official runtime internals.
Nearest32 rounds half up. Both policies use confidence 0.3 and 3×3 local peaks.

Selection uses only archive-dequantized development geometry: pooled
`2*matched/(candidate+native)` across separately matched object/text groups,
then mean matched box IoU and fixed grid order for ties. Matching is greedy,
one-to-one, class agnostic, with box IoU at least 0.5. This native-reference F1
is **not COCO accuracy or mAP**. Conditional mean IoU must be read alongside
unmatched counts. Object masks use the existing 128 square diagnostic grid,
not native-resolution pixel equivalence.

Each native image makes two separate requests. The full request includes
scores; geometry-only omits scores. Invalid JSON/nonfinite responses remain
FAIL, with their original bytes. A successful geometry request never repairs
or accepts a failed full response. Native output names and strings are
preserved as observations but are not compared to candidate semantics.

## Results

All 192 final raw-head comparisons pass the unchanged elementwise tolerance
`abs(actual-reference) <= 2e-4 + 2e-4*abs(reference)`: 64 image/weight/resize
conditions, each with three heads. Across these conditions, maximum absolute
errors are 0.0025482177734375 for detections, 0.000036597251892089844 for
prototypes and 0.00010311603546142578 for coefficients. These are tensor values,
not pixel shifts. The 112 decoder-policy conditions have the same proposal
counts, peak cells, order, integer boxes and diagnostic mask pixels between
PyTorch and ONNX. Maximum score errors are 1.7881393432617188e-6 for objects and
1.4603137969970703e-6 for text regions. Torch-native geometry comparisons also
match their ONNX-native counterparts exactly. Repeated weight/rule/policy
conditions are not independent scenes.

Development selection froze `ceil32` with no NMS before any reserved inference.
For archive-dequantized weights, development results were:

| Resize | NMS IoU | Candidate/native/matched | Geometric F1 | Mean matched box IoU |
| --- | --- | --- | ---: | ---: |
| pixel | none |316/159/121|0.50947|0.88571|
| pixel |0.5|306/159/117|0.50323|0.87997|
| nearest32 | none |330/159/142|0.58078|0.96365|
| nearest32 |0.5|321/159/137|0.57083|0.96016|
| ceil32 | none |330/159/144|0.58896|0.97618|
| ceil32 |0.5|321/159/139|0.57917|0.97227|

The frozen selection is timestamped 2026-11-11T07:34:54.254274+00:00, with its
byte identity recorded before reserved capture. Reserved results are:

| Weights | Resize (no NMS) | Candidate/native/matched | Unmatched candidate/native | Geometric F1 | Mean matched box IoU |
| --- | --- | --- | --- | ---: | ---: |
| Original FP32 | pixel |87/54/29|58/25|0.41135|0.82591|
| Original FP32 | selected ceil32 |90/54/32|58/22|0.44444|0.85617|
| Archive-dequantized | pixel |133/54/41|92/13|0.43850|0.88349|
| Archive-dequantized | selected ceil32 |132/54/52|80/2|0.55914|0.98542|

Thus the selected archive-dequantized policy covers 52 of 54 native regions but
still emits 80 unmatched candidates. Its object group matches 47 of 49 native
objects with 80 extra proposals; its text group matches all 5 native regions
with no extras. These are box correspondences, not name or OCR correctness.
The higher correspondence with archive-dequantized weights shows why original
FP32 versus the official mixed-precision model must be reported separately.
It does not identify any single native kernel or prove the hypothesized resize
is exactly the official implementation.

For the selected archive-dequantized reserved object matches, mean mask IoU is
0.88516 over 40 valid native polygon comparisons. Seven additional matched
native polygons are invalid/degenerate for this rasterization and retain null
mask IoU; they are not silently discarded as perfect masks. Matched native
scores still differ, with maximum absolute difference 0.05494 on these four
images. The masks and scores therefore do not establish native equivalence.

All twelve geometry-only native requests are observed. Of twelve full-score
requests, ten are observed and two development responses fail strict JSON
parsing because their raw score fields contain `-nan` (images 6818 and 480985).
Both original responses are preserved. Their cause remains unknown; the
separate geometry observations do not make those full responses pass.

NMS 0.5 worsened the prespecified development metric and is not selected.
The excess development proposals are neither empty-mask nor zero-area-box
artifacts in the separate exploratory check. The pinned Oasis full pipeline
contains later language/reference selection and drops empty OCR output; this
suggests work beyond dense-head decoding, but is not proof that the official
runtime uses the same final selection. The next priority is validating the
remaining language/region/names/OCR computation and final selection, while
preserving the original numerical failures. These results do not establish a
need to modify ONNX Runtime or build a custom WebGPU runtime.

## Evidence and reproduction

- [Complete measurements and raw native responses](verification/mirage-native-corpus-results.json).
- [Execution, source-freeze review and controls](verification/mirage-native-corpus-execution.json).

The measurement entry point is `scripts/mirage/native_corpus_study.py`.
Each inference process is limited to four images. The collector checks exact
coverage, raw-head shapes, runtime/source/graph/input identities, resize
metadata, unchanged numerical tolerances, native raw-response reparsing and
one-to-one matches. Comparison counts must equal the actual native arrays and
the corresponding PyTorch/ONNX decoded counts. A reserved inference first
recomputes the frozen selection from its development-only evidence.

With the checksum-verified model/runtime/graph cache already available, verify
the images without network access:

```sh
.cache/mirage-python/bin/python scripts/mirage/corpus_fixtures.py \
  --cache-dir .cache/mirage-native-corpus --verify-only \
  --report .cache/mirage-native-corpus/reproduction-image-verification.json
```

The destination must be new. Without `--verify-only`, only absent images are
downloaded from the manifest's public URLs; mismatching cached bytes are not
replaced, and a recorded HTTP403 is not retried. No model acquisition is part
of these scripts. The complete execution record enumerates bounded capture,
measurement, development selection and reserved summary commands, with exits
and raw-log identities. Reproduction must preserve the development-first
order and use new output destinations.

The original 167 intermediate failures out of 2,640 comparisons per historical
ONNX label remain FAIL. Final-head and decoded-output stability on these new
images do not waive that criterion, demonstrate WebGPU behavior, or validate
full official Mirage output.
