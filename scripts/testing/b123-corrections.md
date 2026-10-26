# Authorized B1–B3 correction checkpoint

Corrects the remaining blockers on reviewed guard HEAD
`d9c5a2b47da3eba491ab0bd4e4940a22a805ef33`
(tree `55059bb14c7bc786da8bf6f158f1902efc5f7e5e`), based on main
`fdd37780930ef83431edcca404447b72a9c21976`. Earlier failed type/browser/CI
receipts remain unchanged and apply to those earlier sources. Current results
are recorded separately; independent re-review is pending.

| Blocker | Minimal correction and retained checks |
| --- | --- |
| B1 missing/wrong shared type | App, ControlPanel and VisionCanvas import the single existing ExtraInferenceOptions from visionEngine with type-only imports. That interface now includes optional confidenceThreshold already consumed by the UI and callback. Remove App's old type assertion; infer its existing per-model parameter union. No duplicate type, new cast, suppression or runtime algorithm/configuration change. |
| B2 obsolete camera-test UI contract | Select the current exact 音声 tab and assert the TTS panel appears. Current UI has one cooldown slider, disabled until speech and auto readout are enabled; select those actual controls, require one enabled slider, and verify ArrowRight changes its value by its existing0.5 step. Keep every original acquisition count, srcObject identity, facing-mode, pending/replaced/late stream and stopped/live-track assertion. |
| B3 ambiguous SegFormer locator | Select the intended default SegFormer-B0 button by anchored displayed name, assert uniqueness and enabled state. Keep General selection, modal dismissal, model/backend/error/stylesheet/page-error checks and every healthy/fault control. |

The first focused two-browser run after import/tab/name correction passed smoke
but still timed out on camera's old nth(1) slider. Its saved trace identifies that
focus as pending before the finalizer's page-closed error. This exposed the second
obsolete UI assumption inside B2, not a need to raise timeout. After correcting the
current settings flow, the targeted native-camera test passed. The failed run is
retained under `2026-10-25T23-40-29.159Z-4a3d4db2`; focused camera pass under
`2026-10-25T23-42-26.791Z-e3f6e0b9`. No assertions, waits or retry rules were removed.

Final complete offline counts and exact run paths are in `b123-corrections-results.json`.
Final frozen HEAD/tree, both hosted CI jobs and actual command/checkout bindings
are in the PR response. Controls must reach two healthy and five intended faults;
the earlier first-intact failure is never relabelled as a passed negative control.

Version remains the already npm-generated1.0.4 for the same coherent PR, above
target main1.0.3, per the once-per-PR PATCH rule. No dependency or lock entry changes.
No tag is created. Guard code, all57 controls, cache-disabled initial setup, the
isolated missing-lock job,702 historical ledger entries and sync guidance remain
unchanged. Product changes are type-only and preserve the compiled production
output; no stage2 resolver/cache/variant behavior is changed.

This validation is model-free. Preview/native camera probes use synthetic input;
external model/SDK resources in production smoke are blocked by the test's route
before network acquisition. No development-server prefetch, denied model/ZIP retry,
alternate route or Windows operation occurs. Real inference, actual EP/GPU,
cache identity acceptance, rights/integrity and hosted ZIP contents remain outside
this correction. Six earlier MediaPipe CPU cases remain BLOCKED. No merge,
public review comment, deployment, tag, protection/settings change or stage2 work.
