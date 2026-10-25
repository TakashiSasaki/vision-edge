# PR24 F5 residual correction

Response to [independent review5642792403](https://github.com/TakashiSasaki/vision-edge/pull/24#pullrequestreview-5642792403)
of `b1913199f8851a8f3306bb513b16ca2889c8b387`
(tree `389837c847a517a9183f38cd02c61be6975508d0`). That review passed F1–F4
and F6–F8, and the original stale-generation F5 controls, but identified an F5
UI integration regression. The author's residual correction awaits independent
re-review; earlier review results remain bound to their earlier source.

## Cause and change

VisionCanvas published its React result state every250ms. Drawing already used
the latest accepted result, but tapping used the older HUD array. The engine's
correct exact-array check rejected that older array until the next HUD update.
This affected even same-model/current-generation results and empty prominent
lists with a valid semantic mask.

Pointer inspection now uses the exact frame successfully copied to the visible
Canvas, independently of HUD notification. It also requires that frame to remain
the latest adopted result and match the current model/request/camera lifecycle
and the engine's live ready state. A newly accepted but not-yet-drawn frame is
not inspected prematurely. Clear, frame failure, model/camera reset, draw failure
and scheduler teardown invalidate the drawn reference. The HUD's250ms callback
cadence is unchanged; its now-unused local result state is removed.

The engine's task/array/request/output-generation guard is unchanged. An explicit
clear or rejected delayed result cannot revive the inspector. During model
initialization, the engine state is checked before React has received new props,
preventing old semantic objects from falling through generic box hit testing.
No score is fabricated and no confidence/threshold policy changes.

## Executable controls and attempt history

`src/components/VisionCanvasInspector.test.tsx` uses the actual component,
scheduler, camera hook, renderer, VisionEngine and SegFormerEngine. Only ORT and
browser I/O are controlled. Logits are [1,150,10,10], class0/wall or class12/person
winning with10 or0.1 above the other logits. Canvas mask bytes are retained.
These are model-free controls, not completed model inference.

Five cases cover:

- A drawn frameB is inspectable before HUD advances from frameA, with either a
  prominent class or an empty prominent list. Readout uses actual pixel softmax.
- Clear during pending inference rejects its late result; a fresh accepted/drawn
  frame recovers without waiting for the HUD. Existing accepted pixels remain
  inspectable while an ordinary next inference is still pending.
- Accepted-but-undrawn frameB is rejected until it is actually copied.
- Model switch rejects taps immediately during initialization, before new React
  props, and rejects the old delayed result after the replacement model is ready.

Initial four-case run against b191319 product code: exit1, pass1/fail3, retained
run `2026-10-25T10-49-14.454Z-1d3f952d`. Initial component correction passed the
79-case focused scope after a fifth case was added. Strengthening the switch
case to tap before props update exposed one further stale-state gap: exit1,
pass4/fail1, run `2026-10-25T10-51-08.038Z-1d31934f`. Adding the live engine-state
guard fixed that gap; final focused79 pass, exit0, run
`2026-10-25T10-51-27.207Z-1f1b8691`.

An intermediate full offline run began before that last strengthening/guard.
Its results are retained separately and are not final-source certification.
Final full offline receipts are in `pr24-f5-residual-results.json`; exact frozen
HEAD/tree, hosted checkout/counts and CI URLs are in the PR correction response.

```sh
npm run --silent test -- src/components/VisionCanvasInspector.test.tsx src/components/VisionCanvas.test.tsx src/services/fourModelCorrections.test.ts
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run --silent verify:offline
```

No existing assertions, engine guards, timings, retries, SDK/provider/model paths,
dependency files or historical evidence changed in this residual correction.
All702 historical files remain checked against63bd3429. Real model inference,
actual EP/GPU, Windows/macOS cleanup and hosted ZIP contents remain unverified;
the earlier six MediaPipe CPU scenarios remain BLOCKED. No denied acquisition
was retried or bypassed. No merge or deployment was performed.
