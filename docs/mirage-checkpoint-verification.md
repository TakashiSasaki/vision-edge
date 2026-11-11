# Mirage checkpoint verification foundation

This increment is stacked on draft PR #36 (`dc28e446a7e0fdb22aa82e9e3193cc7ab1b2bf9f`).
The observed main was `e6bf2899fd2e9d916f3904d344171f167588b8d4`.
It provides acquisition/integrity and inventory tooling, **not a faithful exporter**.
`MIRAGE_ONNX_NOT_VALIDATED` and the native WASM default remain in force.

## Reproducible offline workflow

The verifier uses Python standard-library modules only (Python 3.10+ recommended).
No Python dependencies are added to the application environment.
Keep model files in the ignored `.cache/mirage-reference/` directory or outside Git.
Given independently acquired pinned files:

```sh
mkdir -p .cache/mirage-reference
python3 scripts/mirage/verify_checkpoint.py \
  --checkpoint .cache/mirage-reference/mirage.safetensors \
  --config .cache/mirage-reference/config.json \
  --cact .cache/mirage-reference/mirage.cact \
  --output .cache/mirage-reference/inventory.json
python3 -m unittest discover -s scripts/mirage -p 'test_verify_checkpoint.py' -v
```

`--cact` is optional. Output creation is exclusive: use a fresh output filename
for a repeat run; existing reports and source files are never overwritten.
The CLI verifies the complete Safetensors length and SHA-256 before parsing its
header, rejects duplicate keys, invalid shapes/dtypes, gaps, overlaps, unindexed
bytes and truncated data, then enumerates every tensor with relative and absolute
byte offsets. Scalar and zero-element tensors are supported. Unsupported dtypes
fail rather than being guessed. Config bytes are hashed and retained in the report,
but architecture compatibility with the checkpoint is explicitly **NOT_VERIFIED**.
All tensors are initially **UNMAPPED**, including those not used by a future
partial exporter. An inventory is not a mapping or a conversion acceptance report.

In an authorized network environment, `--download-to DIRECTORY` may replace
`--checkpoint FILE` to explicitly acquire the pinned Safetensors file. It uses
normal proxy/TLS trust, validates a complete HTTP 200 response and final size/hash,
and publishes only verified bytes. Existing corrupt files are rejected without
fetching or replacing them. There is no fallback URL or automatic retry.
Do not execute acquisition to retry or bypass a previously recorded policy denial.
`config.json` must be supplied separately; no config hash is invented. The two
artifact identities are task-pinned values, not independently verified live Hub metadata.

## Observed blockers and reference assessment

On 2026-11-11 the pinned Hugging Face config request failed during proxy CONNECT
with HTTP 403, before an upstream response. No model bytes were downloaded, and
no acquisition was retried through another host. The original Safetensors and
`.cact` identities, complete tensor inventory, official/Oasis mapping and numerical
errors therefore remain unverified. The Oasis reconstruction is not vendored or
accepted as authoritative. Its supplied URL is also on the denied host.

The official Python frontend at Cactus commit
`bafcf7348607c89f30ff9fde8e59d20b76baa86f` was accessible and inspected again:
its `describe` method calls `lib.cactus_describe`, without exposing the neural
forward operations or intermediate tensors. See `mirage-onnx-status.md` for the
prior broader source assessment. This frontend cannot establish a PyTorch/ONNX
operation mapping. No native runtime was installed or measured in this increment.

The Git HTTPS proxy allowed current refs and checkout acquisition. PR #36's public
page reported Draft. The GitHub CLI open-PR query failed with Forbidden on
`api.github.com`; Git refs alone do not establish review/CI/open status for every
PR. The existing SSD MobileNet branch was observed and left untouched.

## Validation and next step

The 13 new standard-library tests use synthetic bytes, including malformed files,
mocked complete/truncated/oversized downloads, and CLI atomic-output controls.
They demonstrate integrity/format handling only. They perform no real-model,
PyTorch, ONNX checker, ORT CPU, native `.cact`, or WebGPU inference. No numerical
error measurements exist for this increment. Independent review found nonfinite
JSON overflow could leave partial output; finite-number parsing and atomic,
exclusive publication correct that finding (R1).

Once access or externally acquired files are available, verify the pinned files
with this tool, review live upstream metadata, inspect the actual config and every
tensor, and establish an independently checked operation mapping. Only then export
and compare the encoder, starting with the exact 512-channel/65-tap HConvGLU.
Do not enable production ONNX based on format checks or synthetic test success.
