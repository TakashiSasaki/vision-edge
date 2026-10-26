# Verified classifier cache and static inventory

This change is stacked on PR30 at `2de974cb994042a0816ee5940017a0467c48ed77`
(tree `d97f215c878c0d0b2b29db5c3123ee648e5e1895`), which in turn depends on PR29.
Main was `bebc22a7d11e2bcbd0a63fa1cd141c0a8359dceb`. Neither parent PR is
modified or merged by this work. Version is derived from package.json (PATCH
increment via npm; no tag). The final source SHA/tree and hosted checkout are
bound in the new Draft PR, since a document cannot contain its own commit hash.
After parent integration, retarget and verify ancestry, cumulative delta,
version alignment and new integration CI before any merge decision.

## Exact-file provenance

The owner attests downloading both files from these exact official URLs. A
separate authorized local read-only inspection, task
`01a1768e-ce9f-76b0-90c6-165844c670df`, inspected those bytes and structured
ModelMetadata. It did not execute models or upload their binaries. This author
received the inspection results as text, not the binary files. This is **not** an
independent GCS download/hash comparison. The earlier cloud access denials have
not been retried, bypassed, or reclassified as an authentication requirement.

| Variant | Official URL | Bytes | SHA-256 | FLOAT32 input/output |
| --- | --- | ---: | --- | --- |
| lite0 | https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite | 18582189 | 6c7ab0a6e5dcbf38a8c33b960996a55a3b4300b36a018c4545801de3a3c8bde0 | [1,224,224,3] / [1,1000] |
| lite2 | https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite2/float32/1/efficientnet_lite2.tflite | 24325765 | 94b6c84a5d6de20f932558de213b50e0338dcc06ab83d8d64b12128a476b4b83 | [1,260,260,3] / [1,1000] |

Both declare author MediaPipe and `Apache License. Version 2.0
http://www.apache.org/licenses/LICENSE-2.0` in ModelMetadata. Both contain
identical labels_without_background.txt with 1000 labels, and neither contains
a separate LICENSE or NOTICE. Public attribution and the Apache-2.0 terms are
included under public/licenses and linked from the UI, inventory, and model HTTP
response. No copyright owner is invented, no weights are modified or committed,
and SDK licensing is not substituted for this exact-file license evidence.
The previously unresolved license/source-attestation prerequisites are resolved
by this evidence. Runtime input/output acceptance remains separate.

## Serving and acquisition contract

The supported mode in this repository is the Vite **development server**
(`npm run dev`, Node24/npm11). Its configureServer middleware supplies
/api/model-cache/status, models and installed WASM. No Dockerfile, process
manifest or externally deployed container startup was available to establish
that a deployed container starts this mode. No deployment was inspected or
changed. `npm run build` emits static assets; `npm run preview` and pure static
hosting do **not** supply these cache APIs. They truthfully block classifier
initialization instead of requesting unverified upstream weights. Supporting a
production backend is a separate runtime/deployment decision.

On a permitted runtime's startup, the existing prefetch manager now includes
all four legacy families/nine variants plus the two classifier variants. Cold
start has no persistent guarantee: ephemeral-container destruction removes the
cache; restart with retained bytes revalidates them. Per-variant promises dedupe
concurrent acquisitions within one manager. There are eleven finite startup
requests, each with a 45-second fetch/body deadline, bounded bytes, no automatic
retry loop, and cleanup of owned temporary files on failure. Multi-process
writers/shared persistent storage are not an implemented coordination contract.

Classifier acquisition must match fixed size and SHA-256 before publication.
Reuse and the exact buffer served recheck both the sidecar identity and the
reviewed pins, so forged self-consistent metadata cannot authorize other bytes.
Classifier URLs include the expected hash; missing/wrong/extra query identities
are rejected. Classifier responses use no-store so a browser cannot bypass
server revalidation with an old cached response. Invalid/unavailable assets
return truthful failures, not successful empty results. A failed acquisition
is not implicitly retried by repeated model requests. No unverified upstream
model fallback exists. Recovery requires an intentional operator restart or
explicit manager acquisition after resolving the cause; the inventory page is
not a repair or download interface.

The resolver accepts only the exact matching variant, URLs, license and pins
from the container API; cached rows also require actual size/hash/presence and
pinned integrity. A missing asset may be acquired through that pinned local
route, and a downloading request waits for its same-variant promise. The SDK
receives only the pinned local path. Initialization and accepted results retain
request-generation ownership, requested SDK delegate and unmeasured actual
backend. No delegate fallback, label index offset, fake box, or fake probability
is introduced. The legacy nine variant URLs/fallback policies and self-recorded
integrity rules are preserved, not relabelled as independently pinned weights.

## Static inventory

/model-cache.html, /model-cache.mjs and /model-cache.css are static public assets
copied into dist, with a clear application link. Runtime GET status observation
and the explicit refresh button do not prefetch, reset, delete, request model
bytes, or poll repeatedly. Startup prefetch is the server's separate behavior.
The status API hashes registered files read-only, reporting file presence,
verified size/hash, readiness, integrity, and an observation timestamp. An
unregistered or temporary file is outside this inventory, not silently adopted.

The page separates saved/validated files from configured unavailable/downloading
or invalid assets, distinguishes fixed pins from acquisition-time hashes, and
never counts a configured filename alone as cached. Missing, malformed, disabled
or unreachable APIs show unavailable/unknown, not a successful zero count.
Malformed individual rows are visibly unknown and excluded from cached counts.
Filename/error rendering uses safe text and bounded labels; no raw filesystem
paths, credentials or arbitrary server error strings are displayed. Public API
errors are sanitized; test raw output and failure history remain in per-run logs.
Observation is a point-in-time report, not a guarantee against later deletion or
proof of SDK initialization/inference. The mobile layout and keyboard refresh
are exercised in browser tests.

## Verification scope and remaining prerequisites

All new cache-positive tests use deterministic synthetic bytes, injected fetch,
and test-only module pins. Production has no alternate-pin flag or test fixture
inclusion. Separate tests assert the actual production pins and metadata. The
actual Vite plugin is exercised with synthetic fetch at a local ephemeral HTTP
port; no real upstream acquisition occurs. Model-free SDK boundary tests verify
that the actual resolver feeds the correct local path and requested delegate.
They do not demonstrate successful real SDK/model inference.

Covered negative controls include wrong size/header/hash, oversize and interrupted
body, forged cache bytes/sidecars, orphaned publication, variant isolation,
concurrent deduplication, unknown/unpinned routes, mismatched status identity,
missing APIs, stale UI/results, and inventory partial-invalid/XSS/path cases.
Historical preflight protects 702 hashes; no historical screenshots are rewritten.
Browser screenshots use testInfo.outputPath. Linux/Chromium is the tested platform;
Windows/macOS cleanup and physical GPU behavior remain unverified.

Actual model acquisition in this cloud, genuine classifier SDK initialization,
both classifier variants' inference and production acceptance are NOT RUN.
No real model bytes were supplied to this cloud. The earlier six CPU cases
(two older MediaPipe models times VideoFrame/owned Canvas/live video) remain
BLOCKED and are not satisfied by these synthetic cache controls or HTTP503
negative diagnostics. Next acceptance requires lawful bytes matching the pins
in an authorized cache-enabled runtime, then separate real SDK/inference/input,
lifecycle/delegate and user-device validation. No merge or deploy is authorized.

Local command receipts and initial failed attempts are recorded below after
verification; raw logs stay in ignored .cache/test-runs and CI artifacts.

Local final evidence: [machine-readable receipts](classifier-cache-results.json).
`PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run --silent verify:offline`
completed exit0: 57 guard, 774 unit at that point, 73 foundation, 40 delegate,
37 output, production build, 30 browser, 7 controls (2 healthy/5 expected faults),
and typecheck/lint. Two subsequently added serve-time mutation tests passed;
final `npm run --silent test` completed **776/776** after their assertions were
strengthened. No failed/skipped/cancelled/todo/flaky results in final populations.
Post-addition typecheck/lint also completed exit0. Focused 116 unit and 11 browser
populations overlap these counts; final integrity controls are 21/21.

Initial failures remain in unique raw logs: two mistaken tensor-shape test
expectations, one substring-based DOM test falsely matching a filename suffix
followed by its byte count (replaced with exact variant-cell assertions), and
one preview startup without its prerequisite dist build. The latter did not run
browser assertions; building the output resolved it without changing timeouts or
retries. Initial test fixture source types also required migration to the new
container-only contract. These failed attempts are not reported as model failures
or successful inference. No existing behavior assertion was weakened.

All 702 protected hashes match, no historical files changed. Static HTML/JS/CSS
and licenses are byte-identical between public and dist. Application CSS adds
only 36 bytes for the new link's padding rule; no rules removed. Application JS
intentionally changes for the resolver, manifest, link and accurate UI messages.
Full build equivalence is not claimed. Hosted exact-source evidence is recorded
on the Draft PR after its run completes; local runs reuse installed dependencies,
whereas CI performs a clean locked install. Hosted artifact ZIP contents remain
unverified under the existing acquisition restriction.
