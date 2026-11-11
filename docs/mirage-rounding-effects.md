# Earlier FP32 differences and candidate decoded-output stability

The existing ONNX graphs can execute with the original learned computation in
scope, yet differ from the pinned unofficial Oasis reference in intermediate
FP32 values. This follow-up traces earlier numerical differences and measures
whether they change the fixed candidate decoder's discrete outputs. It changes
neither the learned weights nor the encoder/exporter/runtime implementation.

**The original full-model criterion remains FAIL:** 167 failed checkpoints out
of 2,640 per ONNX comparison label. All 167 are body/attention block outputs.
None is one of the three final raw heads. The unchanged gate
`MIRAGE_ONNX_NOT_VALIDATED` and native WASM default remain in force.
Only 452 of 583 tensors are mapped by the partial encoder/head candidate. Object
names, OCR decoding and language/region computation remain unsupported.

## What the existing failures mean

A comparison marks an entire tensor FAIL if any element exceeds
`abs(actual-reference) <= 2e-4 + 2e-4*abs(reference)`. A failed checkpoint is
neither a failed image nor an incorrectly named object. The reported maximum
intermediate absolute difference 0.451171875 is not a bounding-box pixel shift.

The previous [body_3 replay experiment](mirage-upstream-drift.md) showed that
upstream perturbations can reproduce the same failing positions even when
replayed through Torch. Small differences can accumulate through blocks, while
normalization can bring later tensors back within tolerance. This new evidence
quantifies final raw heads and the fixed candidate postprocessing separately;
it does not relax the intermediate criterion.

## Earliest observed inequality in an instrumented stem

The new `measure_numerical_effects.py` stem phase observes 48 named stages,
including convolution, SiLU, pointwise projection, square, mean, epsilon,
reciprocal square root, scaling and masking. It checks its five eager stem
outputs byte-for-byte against the pinned Oasis candidate and measures the
instrumented ONNX outputs against the original graph's five stem outputs.

The following is the **first numerically unequal captured stage in that
instrumented graph**, not a claim about an unobserved primitive in the original
optimized graph. None of these differences fails the existing tolerance:

| Weights | Fixture | First observed unequal stage | Maximum absolute error at that stage |
| --- | --- | --- | ---: |
| Original FP32 | blank | first convolution | 1.4901161193847656e-8 |
| Original FP32 | small objects | first convolution | 8.940696716308594e-8 |
| Original FP32 | square photo | first convolution | 5.21540641784668e-8 |
| Archive-dequantized | blank | following SiLU | 2.384185791015625e-7 |
| Archive-dequantized | small objects | first convolution | 2.9802322387695312e-8 |
| Archive-dequantized | square photo | first convolution | 4.470348358154297e-8 |

All 288 stage comparisons (48 × 3 fixtures × 2 weight modes) pass, as do all 30
stem-output instrumentation comparisons. The maximum instrumentation-only
absolute difference is 5.379319190979004e-6; it is nonzero. Exposing intermediate
outputs can change fusion/layout, so the measurement does not assume byte
identity with the original graph. These are numerical inequalities, not newly
identified unsupported operators or proof of a runtime defect.

## Forty output conditions

The output phase reuses the exact existing graph hashes. It compares both
original-FP32 and separately archive-dequantized weights, pixel and nearest32
preprocessing, and the ten existing fixtures: 40 conditions. These are not 40
independent scenes: seven fixtures are synthetic and three are variants of one
photo. No new independent accuracy corpus is claimed.

All 120 final raw-head comparisons pass at the unchanged tolerances. Their
metrics reproduce the historical aggregate exactly:

| Raw output | Comparisons | Maximum absolute error across 40 conditions |
| --- | ---: | ---: |
| Detection logits/regression | 40 PASS | 0.0021514892578125 |
| Mask prototypes | 40 PASS | 0.00003081560134887695 |
| Mask coefficients | 40 PASS | 0.000054955482482910156 |

Raw outputs remain numerically different. Their values have different scales;
these maximum differences must not be read as pixel displacement or a class
accuracy measure.

Both outputs then use the unchanged research decoder: confidence 0.3, 3×3 local
peaks, at most 64 object and 32 text-region proposals, deterministic score/cell
ordering, integer boxes on a 0..1000 coordinate grid, and binary masks on the
existing 128×128 diagnostic grid. Pairs are identified by (group, peak cell), not
by a forgiving nearest-box match. Missing/extra cells and order changes are
reported explicitly.

Across 40 conditions there are 156 object-proposal instances and 14 text-region
instances on each side. These totals include repeated fixtures/conditions and
are not counts of independent objects. The observed differences are:

- Proposal counts, selected cells and their order: none.
- Integer-coordinate bounding boxes: none.
- Object-mask pixels on the 128×128 diagnostic grid: none.
- Maximum object-score absolute difference: 1.7285346984863281e-6.
- Maximum text-region-score absolute difference: 1.3113021850585938e-6.

Thus the measured discrete outputs agree for this exact candidate decoder and
these inputs; scores agree within the existing criterion. This is not generic
zero output effect, native-resolution mask identity, semantic object-name/OCR
agreement or official Mirage parity. Different inputs, thresholds or a different
postprocessor can still expose differences. Degenerate box IoU is null; masks
with an empty union on both sides report `EMPTY_BOTH` and conventionally IoU 1.

## Graph-layout exploration

A separate unaccepted exploration kept the Torch normalization calculation
unchanged while lowering channel normalization through NHWC rows in ONNX. The
six inspected eager/reference examples stayed numerically exact. ONNX failed
checkpoint counts changed as follows:

| Weights | Blank: old → rows | Small objects: old → rows | Square photo: old → rows |
| --- | ---: | ---: | ---: |
| Original FP32 | 4 → 5 | 6 → 13 | 3 → 4 |
| Archive-dequantized | 15 → 5 | 5 → 6 | 3 → 4 |

This is mixed and generally worse in the inspected FP32 examples, so the
lowering is **not adopted**. The exported/optimized exploratory graphs and
complete measurements remain in the ignored cache, identified in the execution
record. This six-case exploration is not an accepted converter or a benchmark.

## Evidence and reproduction

- [Complete measurements](verification/mirage-rounding-effects.json): all 40 output cases, 6 stem cases, raw metrics, ranked cells, box/score/mask pair metrics, runtime/source/artifact/input identities and exact scope coverage.
- [Execution and control record](verification/mirage-rounding-execution.json): ten bounded commands and exits, raw-log identities,50 model-free controls, retained development collection failure and rejected-layout exploration.

Measurements run in ten bounded scopes: eight output scopes of five fixtures
and two stem scopes of three fixtures. They all complete and exit 0 with
`MEASURED_NOT_FULL_MODEL_ACCEPTANCE`; this means successful scoped diagnostics,
not acceptance of the existing 167 intermediate failures. The collector rejects
zero, missing, duplicate, mixed-provenance or contradictory records and includes
stem/instrumentation failures in its status and nonzero exit.

For example, with the existing checksum-verified cache:

```sh
.cache/mirage-python/bin/python scripts/mirage/measure_numerical_effects.py measure \
  --checkpoint .cache/mirage-resume/mirage.safetensors \
  --config .cache/mirage-resume/config.json \
  --cact .cache/mirage-resume/mirage.cact \
  --photo .cache/mirage-encoder/coco-231049.jpg \
  --oasis-reference .cache/mirage-resume/oasis-reference.py \
  --aggregate docs/verification/mirage-refinement-results.json \
  --production-onnx .cache/mirage-refinement/numerical-fp32-pixel-0-r3/fp32-reference-order.onnx \
  --output-dir .cache/mirage-reduction-order/reproduction-new \
  --phase outputs --weights fp32 --rule pixel \
  --cases square_blank landscape_blank portrait_blank small_objects boundary_objects
```

The destination must be new. The execution record enumerates all ten exact
commands/partitions and the combine command. Use `--phase stem --rule pixel`
with the three stated fixtures to repeat a stem scope. No model is downloaded.
The original graph, aggregate, all 14 original producer hashes and the additional
diagnostic sources must match before inference or collection.

These results support continuing with the existing runtime while evaluating
output stability on more independent images and clarifying official native
pre/postprocessing. Earlier convolution/activation kernel rounding and global
FP32 agreement remain research questions. No runtime fork or custom WebGPU
implementation has been established as necessary by this experiment.
