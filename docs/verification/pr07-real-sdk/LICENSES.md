# Asset and fixture licensing/provenance

No SDK, WASM, loader JS, model or input binary is committed by PR07. Normal trusted npm installation is covered by the unchanged lockfile integrity. Locally cached assets remain ignored. Selected diagnostic screenshots show only the public-domain NASA fixture and application UI; there are no private photos, camera recordings, credentials or arbitrary network bodies in the new evidence.

| Asset | Source / license evidence | Storage decision |
| --- | --- | --- |
| SDK0.10.14 and four WASM/loader variants | [npm package](https://www.npmjs.com/package/@mediapipe/tasks-vision/v/0.10.14), installed package.json says Apache-2.0; each CDN variant was byte-compared with its installed npm counterpart | cache only; no vendored redistribution in this PR |
| `astronaut.png` | [scikit-image v0.24.0 data source](https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/data/_fetchers.py#L382), public-domain NASA photograph of astronaut Eileen Collins, source attribution retained | cache only; permitted public fixture visible in selected screenshots |
| `coffee.png` | [same official project source](https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/data/_fetchers.py#L912), Rachel Michetti / Pikolo Espresso Bar; photographer's CC0 release | cache only; attribution retained despite CC0 |
| Input hashes | [project registry](https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/data/_registry.py), independently agrees with both acquired PNG SHA256 values | source URL/hash/retrieval window in manifest |
| Textured and empty controls | simple deterministic Canvas commands authored in this repository, one changing corner pixel, no external photograph | source code only, no generated binary committed |
| EfficientDet-Lite0 float16/1 | exact production Google model URL in lock; [official guide/model references](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector) | HTTP403, no bytes/hash acquired; model-specific redistribution terms not verified; do not commit binary |
| Selfie Segmenter float16/latest | exact production Google model URL in lock; [official guide/model references](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter) | HTTP403, no bytes/hash acquired; mutable latest remains unpinned/blocked; model-specific redistribution terms not verified; do not commit binary |

A documentation site's general content/code license does not establish the license for a separately hosted trained model. The unknown model-specific terms and bytes are retained as a prerequisite for future acquisition, rather than assuming the SDK's license applies to them.
