# Built-in stereo matcher investigation — 2026-11-17

Baseline: `10f0c229a0fae5bbacb312da8b292da3fa1ee620` (main).
Working branch: `fix/stereo-invalid-matches`.

## Reproduction and findings

Decoded all three bundled real SBS WebM videos at 1, 4 and 6 seconds. Compared
CPU SAD and actual WGSL compute execution at a common 320×180 per-eye resolution,
maximum disparity 48, window setting 7 and confidence setting 0.35. This is 18
outputs per revision. WGSL uses Chromium's SwiftShader adapter here, not a physical
GPU; these results establish shader behavior, not hardware speed. The production
GPU profile normally uses 480×270, which was not measured in this comparison.

Both old matchers painted almost the entire image, including ambiguous regions.
The train interior shows patchy, scale-like depth before color mapping. The fault
is not primarily the heatmap renderer: it already makes negative disparity
transparent. The matching stage was accepting unreliable disparities:

- No uniqueness check for competing matches in repetitive or low-texture regions.
- No left/right correspondence consistency check for occluded/mismatched content.
- Invalid border correspondences were clamped and given finite costs.
- SAD automatically raised its rejection threshold to the 85th percentile when
  few matches passed, effectively accepting bad pairs despite the user's threshold.
- A small preference for low disparity broke cost ties without establishing a
  meaningful correspondence.

## Changes

Both matchers now retain four best costs, rejecting a competitor outside the
winner's ±1 disparity subpixel basin when its cost is within 10% of the winner
(with a small numerical tolerance). Four minima suffice because the basin can
contain at most three discrete disparities. Removed disparity tie penalties.
Unsupported windows and subpixel interpolation beyond the legal search interval
are excluded. SAD's adaptive threshold relaxation is removed.

SAD derives the reverse winners from the same window costs. WGSL computes both
directions in a two-layer dispatch using a doubled intermediate disparity buffer.
Correspondences must agree within one pixel. WGSL's median filter also excludes
inconsistent neighbors. Invalid centers remain invalid; holes are not fabricated
into depth. This adds CPU bookkeeping and roughly doubles the GPU matching work;
no real-time performance claim is made.

## Observed results

Mean across the nine video frames (each frame weighted equally):

| Method | Accepted pixels before → after | Horizontal jumps >3px before → after |
|---|---|---|
| CPU SAD | 99.96% → 69.26% | 2.174% → 0.103% |
| WGSL | 99.79% → 62.98% | 3.671% → 0.086% |

Jump rate counts only neighboring pixels that are both valid. This is a
roughness diagnostic, **not a depth accuracy metric**: rejecting pixels changes
its denominator, and real depth edges can also create jumps. The sparse output
is intentionally more conservative; some speckles and wrong matches remain.

Controls, evaluated away from unsupported borders:

- Uniform and eight-pixel-periodic pairs: before both engines accepted 100%;
  after both accept 0%. A unique disparity cannot be established for these pairs.
- Deterministic random texture shifted by eight pixels: after both engines retain
  100% of the evaluated interior, with no errors exceeding one pixel.
- CPU regression additionally checks unrelated pairs at strict confidence,
  border rejection, subpixel accuracy, and transparent rendering of invalid data.

Real videos have no pixelwise ground truth or calibrated camera parameters in
this experiment. Thus metric depth accuracy cannot be established. Rectification,
vertical mismatch, reflective surfaces, occlusion and repetitive texture still
limit local SAD/Census methods. Learned models or a stronger stereo optimizer
may produce denser results; these changes do not turn the baselines into SGM.

## Reproduce

From the repository root with installed dependencies and `/usr/bin/chromium`:

```sh
node scripts/stereo/diagnose-builtins.mjs 10f0c229a0fae5bbacb312da8b292da3fa1ee620
node scripts/stereo/diagnose-builtins.mjs
npm test -- src/services/stereoDepthService.test.ts src/services/stereoMatchingValidity.test.ts
```

The script uses a plugin-free Vite server and a blank diagnostic page, blocks
external browser requests, and downloads no models. Each invocation prints a
unique `.cache/stereo-diagnostics/run-*` directory containing JSON and PNGs.
Top row: input, SAD map, WGSL map. Bottom row: corresponding overlays (55% opacity).
Fixed Turbo scale 0–24 disparity pixels; transparent map regions show black in
standalone maps and the original image in overlays. Current-tree control failures
exit nonzero; git-ref mode records the baseline without requiring corrected behavior.

## Verification

- Targeted unit tests: 28 passed.
- Full unit suite: 900 passed, 3 skipped, 0 failed (903 total).
- Production build and lint: passed.
- Offline aggregate: stopped at typecheck, with diagnostics in the unchanged
  `e2e/mirage-parity-and-speed.spec.ts` (Mirage API/type mismatches).
  No diagnostics were reported for the changed stereo files. The offline suite
  is not claimed to pass, and its subsequent browser checks did not run.
- Real WGSL diagnostic: 18 video outputs and six controls completed; software
  adapter `google / swiftshader`, no CPU fallback.
