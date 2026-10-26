# Stage2 exact-variant cache routing

Stacked on frozen Stage1 PR27, HEAD
`5fc8b7f32859e071e9c74d1269429ba945c4b3af`, tree
`27f710a3a587adc62b90830e301d7efbfae660ef`. Stage1 and main are not modified or
merged. This PR targets `fix/npm-evidence-preflight-stage1`, not main. Independent
Stage2 review is pending; Stage1's success is not Stage2 acceptance.

## Confirmed defect and correction

The server tracked variantStates internally but published only four model-level
defaults. The resolver used a model-level row when a requested variant row was
absent. Cached/downloading default entries therefore supplied standard weights
to all five nondefault requests while ready state retained their requested labels.
Initial21-case controls against Stage1 passed8/failed13: ten wrong-default routes
(five variants ×cached/downloading), absent9-entry status, corrupted restart cache,
and unsupported-variant fallback. These failed receipts remain separate evidence.

The server now maintains one variant-state/inflight source of truth and exposes
all9 entries, including model id, variant id, exact filename, local/upstream URL,
availability/error and byte size/time. The four legacy model rows are derived
default aliases, never independent state. Filename routes use only registered
variant entries; unknown keys/paths fail without fetching a default. Responses
identify model and variant in headers and relevant failures.

The resolver requires exact model/variant/file/local URL/upstream URL and valid
cached/downloading metadata. A present variants map is authoritative, including
missing/invalid rows. Legacy responses without that field may supply only an
unambiguously matching registered default; nondefault requests use their own
upstream. Disabled, unavailable, erroneous, malformed or mismatched responses
never silently use another variant. Status-fetch failures invalidate prior cached
responses. All primary/upstream fallbacks come from the same requested catalog
entry; redundant service-level default-model URL fallbacks are removed.

Explicit unsupported variants are rejected before replacing a valid task. Engine
ready labels use the normalized registered variant actually passed to the default
factory's selected URLs. ONNX provenance now records the corresponding variant
key as well as model-level source, including upstream fallback. Late completion,
SDK failure and existing lifecycle/provider/configuration behavior remain covered.
Custom task factories or explicit caller-supplied model URLs are outside catalog
routing guarantees; ready never certifies semantic model contents or real inference.

## Restart and disk cache contract

Successful cache writes atomically rename data and an `.identity.json` sidecar
through unique temporary files. The sidecar binds schema/model/variant/file/URLs,
actual size and SHA256 of acquired bytes. Status and HTTP serving revalidate that
binding; serving verifies its exact owned response bytes, catching post-status
corruption too. Interrupted pairs, missing/oversized/malformed sidecars, changed
bytes, symlink files and URL/identity mismatches are unavailable, not cached.
Unrecorded old files are not silently blessed by filename/size. No missing bytes
receive an invented hash. Validation itself never rewrites, deletes or relabels
old bytes.

Existing startup/on-demand acquisition calls can replace invalid cache entries
using the requested variant's existing upstream; no permission to run those real
requests is implied here. This task never launches the dev-server
prefetch hook or acquires models. Tests inject generated2048-byte payloads into
temporary directories and substitute fetch/SDK boundaries. The cache contract is
local consistency, not signed authenticity, licensing, model-format validation,
immutable upstream pinning or a guarantee that a mutable upstream serves a stated
architecture. Both maliciously replaced bytes+metadata and same-URL upstream
updates require a separately reviewed trust/pinning policy.

Acquisition has one attempt per inflight variant, deduplication,45-second fetch/body
deadline even for an ignoring mock,512MiB maximum body/file size and4KiB sidecar
limit. Streaming reads enforce the size limit; metadata records no model content.
Within a manager lifetime, failed requests retain their error and do not implicitly
retry on status/serve. A fresh process has no persisted failure history.
Temporary-file cleanup is best-effort on failure; incomplete pairs remain unusable.
Status validates up to9 variants concurrently and hashes cached bytes again; real
model-size throughput/long-run resource behavior remains unmeasured. Browser HTTP
cache freshness and semantic model correctness are not established by these tests.

## Validation and integration checkpoint

224 new model-free cases plus the3 existing cache cases give227 focused passes.
The full unit/component population is593 =369 inherited +224 new. Matrices cover
all9 cached/downloading/unavailable/missing/disabled cases; identity-field mismatch,
wrong-default negative controls, legacy responses, actual server status/HTTP bytes,
restart reuse/corruption/deletion/symlinks, errors/timeouts/temp cleanup/deduplication,
and actual engine/default-factory URL/ready/error/lifecycle routing through controlled
SDK boundaries. They are not real inference populations or license verification.
One new-fixture typecheck initially failed an optional index signature; its fixture
return type was corrected without suppression. All earlier failures are retained.

Fresh canonical npm installation and complete Stage2 offline/hosted checks are
recorded in `stage2-variant-cache-results.json` and the PR's exact HEAD/tree/CI binding.
Node24/npm11 and the sole npm lock remain. A separate coherent PATCH is generated
with `npm version patch --no-git-tag-version` (1.0.4→1.0.5); dependency entries and
manual tag workflow are unchanged. No tag is created. All702 historical hashes,
Stage1 guard/control/CI logic, model parameters, UI and existing assertions remain.
JS changes intentionally for routing/cache behavior; CSS parity is checked separately.

Before any eventual integration: independently review this exact stacked diff;
integrate Stage1 only with separate authorization; re-fetch actual main and Stage1
result, retarget/reconcile Stage2 explicitly, recheck the cumulative diff and version
monotonicity, then rerun exact-HEAD checks if its source/base changes. A green
stacked-head CI is not a main merge decision or successful retarget verification.
Do not merge, auto-merge, force-push, delete evidence branches or deploy from this
checkpoint. Six old MediaPipe CPU cases remain BLOCKED; no denied acquisition or
ZIP request was retried, no alternate route/account/machine or Windows work used.
