# Separating local ONNX arithmetic from upstream FP32 drift

The encoder/detection/mask candidate still fails numerical acceptance. A valid
ONNX graph and successful CPU execution do not establish numerical equivalence.
The existing full experiment retains 167 failed checkpoint comparisons out of
2,640 per ONNX comparison label, with maximum absolute error 0.451171875.
No threshold, export, production gate or historical result is changed here.

The original FP32 checkpoint and the separately archive-dequantized weights are
both preserved as explicit experiments. Only 452 of 583 tensors are mapped; the
language/region computation, object naming and OCR decoder remain outside the
partial conversion. Exact eager agreement with unofficial Oasis is still not
proof of official Mirage equivalence.

## New diagnostic

`diagnose_reference_body.py` extracts a selected body block (1..15; default3)
and observes 20 stages: normalization, depthwise mixing, SiLU, residual and
HConv normalization, explicit Walsh/band/GELU operations, GRN and block output.
It checks its eager final output for bitwise equality with the pinned candidate
reference before using any local measurements.

Two different input paths distinguish error sources:

1. **Common reference input:** Torch and an instrumented ONNX block receive
   exactly the same preceding-block reference tensor and valid-region mask.
2. **Production-input replay:** the unchanged full ONNX graph supplies its
   actual preceding-block output. The instrumented ONNX block and a Torch
   block both replay that tensor. Torch replay retains the reference input's
   strides, changing its values rather than its layout.

The instrumented ONNX output is also compared with the original full graph's
block output. Exposing intermediate tensors can affect fusion and layout, so
instrumentation identity is measured rather than assumed. The script checks
all 14 source identities from the production aggregate, official artifact pins,
reference hash, unchanged tolerances, exact production graph hash and explicit
CPU providers. Unknown/duplicate/zero cases are rejected; each run is bounded
to at most five fixtures and has a completion flag. Actual failures exit 1.

## Actual body_3 measurements

Both weight modes were measured on three previously used square fixtures;
these are diagnostic examples, not a new independent validation corpus.
All **120 common-input stage comparisons pass** (20 stages × 3 images × 2 weight modes).
The complete diagnostic still **FAILS**, preserving the production/replay
failures below:

| Weights | Fixture | Production body_3 failed elements | ONNX replay failed elements | Torch replay failed elements |
| --- | --- | ---: | ---: | ---: |
| Original FP32 | blank | 0 | 0 | 0 |
| Original FP32 | small objects | 1 | 1 | 1 |
| Original FP32 | square photo | 0 | 0 | 0 |
| Archive-dequantized | blank | 2 | 2 | 2 |
| Archive-dequantized | small objects | 1 | 1 | 1 |
| Archive-dequantized | square photo | 0 | 0 | 0 |

The complete failure masks, not just their counts, agree for production and
both replay paths in all six cases. Every production metric matches the
previous committed aggregate exactly. The maximum instrumentation-only output
difference is 0.000030517578125; it passes the criterion but is not bitwise zero.

For example, archive-dequantized blank output at index `[0,219,0,0]` has reference
-0.271744966506958 and ONNX value -0.2720418870449066. Its absolute difference is
0.0002969205379486084, exceeding the allowed error 0.0002543489933013916. Its other
failed element is retained in the report. This illustrates an actual
threshold failure rather than a missing/unsupported ONNX operator.

The upstream body_2 tensor passes its own threshold in every measured case.
Nevertheless, replaying that perturbed tensor through Torch is sufficient to
reproduce the same body_3 failing positions in these examples. Thus the first
failed *captured checkpoint* need not be the operation that introduced the
first rounding difference. This does not prove the source of every full-model
failure, identify the first global primitive, or establish native correctness.

## Runtime versus export

Both ONNX export and runtime execution can affect FP32 behavior: graph lowering
chooses expressions and layouts, while kernels choose reduction order, fusion
and function approximations. These experiments do not assign all differences
to ONNX Runtime alone.

An additional unaccepted exploration lowered SiLU as x/(1+exp(-x)) instead of
x*sigmoid(x), leaving learned weights unchanged. On the three inspected images,
original-FP32 failed checkpoint counts changed from 4/6/3 to 5/9/4; archive-dequantized
counts changed from 15/5/3 to 4/6/2. The outcome is mixed and remains failing. This
lowering was **not adopted**. Its raw script, graphs, log and measurements stay
in the ignored cache, with identities in the execution record; it is not an
accepted converter or a performance result.

## Reproduction and evidence

The original fourteen-scope baseline remains in
[the refinement aggregate](verification/mirage-refinement-results.json).
New evidence is additive:

- [Original-FP32 diagnosis](verification/mirage-upstream-drift-fp32.json).
- [Archive-dequantized diagnosis](verification/mirage-upstream-drift-cact-dequantized.json).
- [Commands, control counts and exploratory identities](verification/mirage-upstream-drift-execution.json).
- [Mainline integration record](verification/mirage-mainline-integration.json).

With the existing verified cache, reproduce one complete three-case scope:

```sh
.cache/mirage-python/bin/python scripts/mirage/diagnose_reference_body.py \
  --checkpoint .cache/mirage-resume/mirage.safetensors \
  --config .cache/mirage-resume/config.json \
  --cact .cache/mirage-resume/mirage.cact \
  --photo .cache/mirage-encoder/coco-231049.jpg \
  --oasis-reference .cache/mirage-resume/oasis-reference.py \
  --aggregate docs/verification/mirage-refinement-results.json \
  --production-onnx .cache/mirage-refinement/numerical-cact-dequantized-pixel-0-r3/cact-dequantized-reference-order.onnx \
  --output-dir .cache/mirage-upstream-drift/reproduction-new \
  --weights cact-dequantized --block 3
```

The output directory must be new. Default fixtures are the blank,
small-object and square-photo examples. For original FP32, change the weight
mode and production graph path together. Both measured runs exit 1 because
real numerical failures remain. No model download is required or attempted.

Next, trace the earlier production tensors and normalization/activation
lowerings under common inputs, preserving input layouts and every failure.
Global FP32 error localization and official-native validation remain necessary
before browser/WebGPU model acceptance. `MIRAGE_ONNX_NOT_VALIDATED` and the
native WASM default remain in force.
