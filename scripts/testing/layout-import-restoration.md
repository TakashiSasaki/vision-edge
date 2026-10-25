# Restoration after the responsive-layout import

This author checkpoint awaits separate independent review. No merge, deployment,
public review comment or recurrence-prevention implementation is included.

- Base: `d7577f18bc99b7087337c86df3928d94250a4399`, tree
  `28c22d88a0aeaacceeabe6dcad916a509edb78dd`.
- Restore from reviewed PR25 integration
  `460bdabb244d0958e78b60e364d81c2096e9b0b1`.
- Keep the base's App, ControlPanel and VisionCanvas responsive-layout changes,
  all four-model fixes and resolver/cache/UI source byte-for-byte.
- Package/runtime definitions remain unchanged. Restore the exact compatible npm
  lock and48 original PR06 log blobs; remove the unintended competing Bun lock.
  No whole-commit revert, dependency update or lock regeneration is used.
- All702 historical verification files are checked against original integration
  blobs, including empty logs. No screenshot/log is regenerated.

[Base CI40192055190/job121164937397](https://github.com/TakashiSasaki/vision-edge/actions/runs/40192055190/job/121164937397)
checked out d7577f1 and failed setup-node with missing dependency lock. Installation,
actual tests and build were skipped, not executed failures. Raw logs and step
metadata were inspected. The deletion mechanism is not established by that result.

Canonical locked installation uses Node24.19.0/npm11.9.0 locally. Full offline
verification uses existing system Chromium154.0.8037.57. Actual run receipts,
counts and production-output parity are in `layout-import-restoration-results.json`;
final HEAD/tree and hosted CI binding are in the Draft PR. Raw stdout/stderr and
structured reports remain in ignored unique `.cache/test-runs/` directories.
Build parity compares this layout base and the restored source using the same
locked dependencies, not the earlier pre-layout app.

```sh
npm_config_registry=https://registry.npmjs.org npm run --silent ci:install
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run --silent verify:offline
```

The existing cache development-server prefetch hook is not launched. Selected
cache unit controls substitute fetch/SDK, and build/preview do not invoke that
hook. No model bytes are acquired, no denied model/ZIP route is retried or bypassed,
and no Windows operation occurs. Real-model inference, actual backend/GPU, new
cache acceptance/licensing/provenance/security, and hosted ZIP contents remain
outside this restoration's certification. Earlier six MediaPipe CPU scenarios
remain BLOCKED. Existing prevention proposals/settings/CI/permissions are unchanged.
