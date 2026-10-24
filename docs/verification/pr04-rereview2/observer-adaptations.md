# Observation and fixture adaptations

Original PR #8/#9 evidence files and their Git branches remain unchanged.
The replay checkout contains the complete actual corrected commit; product
source was not selectively patched or replaced.

The PR #9 native observer accepted video/VideoFrame only. Canvas requires an
explicit timestamp when constructing a WebCodecs observation frame. Exactly
these source expressions were adapted in the copied observer files:

```js
new NativeFrame(input)
// becomes
new NativeFrame(input, input instanceof HTMLCanvasElement ? { timestamp: 0 } : undefined)

new Frame(input)
// becomes
new Frame(input, input instanceof HTMLCanvasElement ? { timestamp: 0 } : undefined)
```

Files: native-boundaries.mjs, forced-native-race.mjs, independentSdk.mjs.
Every assertion/exit-condition line was compared before and after and remains
unchanged. Original/adapted SHA-256 values are in logs/observer-adaptations.json;
the exact adapted copies are under adapted-pr9-vectors.

**The zero is a synthetic observation-constructor requirement. It is not source
PTS, SDK timing, or the production identity.** Production comparison uses the
actual sampled bytes/dimensions. The independent new SDK observer avoids this
constructor entirely: it directly draws the actual supplied image, compares full
bytes/dimensions with an independent video reference and reports canvas source
timestamp as null.

Original PR #8 abstract DOM fixtures do not implement readable pixel contexts.
Raw unchanged replay therefore gives 8 pass / 21 fail, with capture-context
failures before many original assertions can be meaningfully reached. The raw
log is retained.

The author-supplied logical adapter was read and separately rerun. It adds a
positive mocked quality capability derived from the fixtures' old logical frame
markers and delegates to the real scheduler; all 29 original assertions pass.
This deliberately selects the quality path. It **cannot establish missing-
capability correctness** and was not used as F01 closure evidence for that path.
It is limited evidence for the prior logical lifecycle, metric and fault
contracts. Its exact source/config are under inspected-logical-adapter; execute
them from their original paths in the replay checkout, not these reference copies.

Missing/zero-capability acceptance instead rests on the native PR #9 replays and
the independent four-case real-app tests, 17 pixel-boundary unit cases and
resolution/cost probe. Those exercise real production capture/deduplication
without inserting a positive quality capability.

The new app fixture's first run requested a frame without repainting the
producer; no fresh presentation arrived and all four cases timed out there.
The final fixture repaints identical bytes before requesting a fresh frame,
preserving the single-inference assertions. The old timeout log is retained.
The final strengthened run presents identical content again near input expiry
and still requires expired/stalled metrics. One initial lint error in the review
unit recorder was fixed without adding a suppression; final lint and all tests
pass. No product source or existing test assertion was changed.
