# Vision Edge

VisionStudio AI is a React/TypeScript browser demo for webcam object detection,
person segmentation, Japanese labels, and optional speech synthesis. It uses Vite,
Tailwind CSS, MediaPipe Tasks Vision, and ONNX Runtime Web. Published app: https://vision-edge.ai.studio.

The runnable models are **Selfie Segmenter** (MediaPipe ImageSegmenter: person
region mask and enclosing box, not per-person instance segmentation),
**EfficientDet-Lite0** (MediaPipe ObjectDetector: boxes and class labels),
**YOLO-seg** (ONNX Runtime Web YOLOv8n-seg: 80 COCO classes with instance segmentation masks), and
**SegFormer** (ONNX Runtime Web SegFormer-B0: 150 ADE20K classes with full-pixel semantic segmentation colormaps).
`src/services/modelRegistry.ts` defines the names, implementation status,
outputs and descriptions shared by the engine, selectors, model dialog and HUD.

**Mirage** uses the original `.cact` model in a Cactus WASM worker. Its ONNX
variant is unavailable until a faithful export passes reference-output checks;
synthetic model generation and fallback are disabled. See the
[Mirage ONNX status and validation requirements](docs/mirage-onnx-status.md).

Each App owns one engine and at most one active task. Model state is separate from
camera state: `idle`, `loading`, `ready`, or `error`. Only the selected task is
created; duplicate pending requests share a Promise. Failures retain their WASM or
task-creation stage and cause, appear in Japanese, and retry only on explicit user
request. Superseded/unmounted completions cannot become current and their tasks
are closed. Model changes clear old detections and overlays. Ready means model
initialization succeeded, not that the camera or inference has succeeded.

The read-only backend label distinguishes requested and configured paths from
unmeasured execution. MediaPipe keeps its GPU delegate request. ONNX defaults to
WebGPU preference when the API is available, otherwise WASM; an observed session
creation failure can trigger an explicit WASM retry. The selected provider list
and fallback reason are reported, while the actual internal backend/hardware is
unmeasured. There is no new user-facing backend selector.

SegFormer confidence is an **uncalibrated softmax score**, separate from class
area coverage. Prominent-class entries require the mean winning-pixel score to
meet the threshold, including the largest class. The full argmax semantic mask
remains visible even if no class meets the list threshold; it does not imply
high confidence. Pixel inspection uses the winning score at that exact pixel,
including small classes absent from the list, and only from the currently
accepted frame/generation. The labels/readout distinguish scores from detector
confidence. Scores are not a guarantee of accuracy.

## ウェブアプリケーション全体の構造 (Web Application Architecture)

本ウェブアプリケーション（VisionStudio AI / Vision Edge）は、ブラウザ内のエッジ推論（WebAssembly / WebGPU / WebGL）を用いてリアルタイムの物体検出・セグメンテーション・ステレオ視差深度推定・画像分類を実行するモジュラー構造の React + TypeScript シングルページアプリケーション（SPA）です。

### 1. アーキテクチャ階層とデータフロー

アプリケーション全体は以下の 5 つの主要レイヤーで構成され、単一方向のデータフローと厳密なタスク所有権によって制御されています。

```text
+-----------------------------------------------------------------------------------+
| 1. プレゼンテーション＆UIレイヤー (src/App.tsx, src/components/*)                 |
|    - TopNavBar / VisionCanvas / ControlPanel / TelemetryDrawer / ModelCompareModal|
|    - ライブカメラ・静止画・SBSステレオ動画入力の切替とオーバーレイ描画              |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 2. メディア入力＆スケジューラ層 (src/hooks/useCameraStream.ts, frameScheduler.ts) |
|    - カメラストリーム管理 (MediaStream 取得・解放・世代管理・USBステレオカメラ対応)|
|    - 合成SBSステレオ動画ジェネレータ / フリーSBS動画プリセット / 動画ファイル読込   |
|    - requestAnimationFrame ベースの非重複推論ディスパッチ＆FPS/レイテンシ計測      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 3. 推論オーケストレーション層 (src/services/visionEngine.ts, modelRegistry.ts)    |
|    - 単一アクティブタスク管理 (idle -> loading -> ready / error)                   |
|    - 世代トークンによる古い非同期初期化や破棄済みタスク結果の無効化                |
|    - モデルメタデータ・対応バックエンド・フォールバック理由の一元管理              |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 4. モデル実行アダプタ群 (src/services/*Service.ts)                                |
|    - MediaPipe Tasks Vision: EfficientDet-Lite0/Lite2, Selfie Segmenter           |
|    - ONNX Runtime Web: YOLOv8n-seg (インスタンス分割), SegFormer-B0 (ADE20K 150類) |
|    - Cactus Mirage 51.3M: WASM Worker (.cact) / 4-Graph Experimental ONNX         |
|    - ステレオ深度推定 (Stereo Depth): ブロックマッチング SAD & Neural ONNX         |
|    - 画像分類ワークスペース: EfficientNet-Lite0/Lite2 (MediaPipe ImageClassifier) |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 5. キャッシュ検証＆補助サービス層 (containerModelCache*, speechService.ts)        |
|    - SHA-256 固定ハッシュ検証付きコンテナモデルキャッシュ API (/api/model-cache)   |
|    - 日英ラベル辞書 (dictionary.ts) ＆ Web Speech API 音声読み上げ (speechService) |
+-----------------------------------------------------------------------------------+
```

### 2. ディレクトリおよび主要モジュール構成

```text
/
├── index.html                  # アプリケーションのエントリHTML・メタデータ定義
├── vite.config.ts              # Vite設定および開発/プレビュー時のモデルキャッシュミドルウェア組込
├── src/
│   ├── main.tsx                # React ルートマウント
│   ├── App.tsx                 # アプリ全体の状態管理・2つのワークスペース（ライブビジョン / 画像分類）の統合
│   ├── types/
│   │   └── vision.ts           # モデルID、検出ボックス、セグメンテーションマスク、ステレオ深度・動画ソース等の型定義
│   ├── hooks/
│   │   └── useCameraStream.ts  # Webカメラ・USBステレオカメラのセッション寿命管理カスタムフック
│   ├── components/
│   │   ├── TopNavBar.tsx               # ヘッダー・モデル選択・ワークスペース切替・テレメトリ表示トリガー
│   │   ├── VisionCanvas.tsx            # ビデオ/静止画/SBS動画の描画、バウンディングボックス・マスク・視差ヒートマップ重畳、動画トランスポートHUD
│   │   ├── ControlPanel.tsx            # モデル別パラメータ調整、ステレオ動画デモ選択、キャリブレーション、検出結果インスペクタ
│   │   ├── ClassificationWorkspace.tsx # EfficientNet-Lite0/Lite2 静止画Top-5分類専用ワークスペース
│   │   ├── TelemetryDrawer.tsx         # 推論時間・FPS・プロバイダ状態・Mirage品質メトリクスの詳細ドロワー
│   │   └── ModelCompareModal.tsx       # 登録済み全モデルのアーキテクチャ・入出力仕様・ライセンス比較モーダル
│   └── services/
│       ├── modelRegistry.ts            # 全モデルの定義・ステータス・推論バックエンド仕様の単一情報源
│       ├── visionEngine.ts             # 推論タスク生成・切替・破棄・フレーム推論呼び出しのコアエンジン
│       ├── frameScheduler.ts           # 推論ループのスケジューリングと世代整合性チェック
│       ├── yoloOnnxService.ts          # YOLOv8n-seg (ONNX Runtime Web) の前処理・NMS・マスク復元処理
│       ├── segformerOnnxService.ts     # SegFormer-B0 (ADE20K 150クラス) の推論・確信度集計・ピクセルプローブ処理
│       ├── mirageService.ts            # Cactus Mirage (.cact WASM Worker) 推論および品質テレメトリ管理
│       ├── mirageOnnxService.ts        # Mirage 4-Graph Experimental ONNX パイプライン実行サービス
│       ├── stereoDepthService.ts       # ステレオカメラ視差・深度推定（SAD エンジン / Neural ONNX / 合成テストシーン生成）
│       ├── stereoVideoService.ts       # 合成SBSステレオ動画（3シーン）＆フリー素材SBSステレオ動画プリセット生成・再生時間制御
│       ├── imageClassifierService.ts   # MediaPipe ImageClassifier による静止画分類サービス
│       ├── containerModelCache.ts      # クライアント側モデルキャッシュインベントリ照会・検証クライアント
│       ├── containerModelCacheServer.ts# Node.js サーバー側モデルキャッシュ配信・SHA-256検証・プリフェッチ制御
│       ├── dictionary.ts               # COCO / ADE20K / ImageNet クラスの日本語翻訳辞書
│       └── speechService.ts            # Web Speech API を用いた検出ラベル・分類候補の音声合成
├── public/
│   ├── model-cache.html        # コンテナ内モデルキャッシュの検証ステータスを確認する静的インベントリ画面
│   └── stereo-videos/          # Wikimedia Commons (CC BY-SA 4.0) の実写撮影サイドバイサイドWebM動画アセット
├── docs/                       # モデル仕様書、ステレオ深度モデル解説、Mirage検証レポート、CI検証記録
└── e2e/                        # Playwright を用いた本番ビルドスモークテスト・カメラライフサイクル検証
```

### 3. ステレオカメラ深度推定とサイドバイサイド（SBS）動画デモンストレーション機能

物理的なUSBサイドバイサイド（SBS）ステレオカメラが手元にない環境でも、ステレオ視差（Disparity）および距離（Depth $Z = \frac{f \cdot B}{d}$）の推定実験を行えるよう、`stereoDepthService.ts` と `stereoVideoService.ts` により以下の入力モードをサポートしています。

1. **合成SBSステレオ動画プリセット（3シーン）**:
   - **移動ブロック (`synth_moving_blocks`)**: 異なる深度レイヤー（0.8m / 1.4m / 2.3m）を横移動・往復するターゲットとリアルタイム理論視差（GT）の比較。
   - **奥行き回廊 (`synth_corridor`)**: 前進カメラ視点によるパースペクティブ回廊と周期的な距離マーカーアーチ。
   - **接近障害物 (`synth_obstacles`)**: 遠方（3.5m）から手前（0.65m）へ接近する複数レーンの障害物による視差急増の検証。
2. **フリー実写SBSステレオ動画プリセット（Wikimedia Commons CC BY-SA 4.0 実写映像）＆ローカル動画ファイル読込**:
   - **フリー実写: 空中鉄道 (`free_outdoor_drive` / `wuppertal-schwebebahn-sbs.webm`)**: ドイツ・ヴッパータールで実写ステレオ撮影された懸垂式モノレール車両とトラス鉄橋・河川の映像（`File:Wuppertaler schwebebahn.sbs.webm`）。
   - **フリー実写: 踏切と列車 (`free_nature_park` / `bahn-latzen-crossing-sbs.webm`)**: 手前の架線柱・踏切設備と、中景を通過する2階建て自動車運搬貨物列車・遠景の樹林の実写ステレオ映像（`File:Bahnübergang latzen.sbs.webm`）。
   - **フリー実写: 広場とICE車内 (`free_tabletop_lab` / `segway-ice-real-sbs.webm`)**: 建築広場を移動するセグウェイ群と歩行者（`File:Segway.sbs.webm`）および時速250kmで走行する高速鉄道ICE 1134の車内速度表示盤・客室通路（`File:Ice geschwindigkeit.sbs.webm`）の実写ステレオ映像。
   - 手元の `.mp4` / `.webm` / `.mov` 等のサイドバイサイドステレオ動画ファイルをローカルから読み込んで連続推論・検証可能。
3. **ステレオ動画トランスポート制御＆ピン留め計測プローブ**:
   - 再生 / 一時停止、タイムラインシークバー、`-1F` / `+1F` の1フレーム単位コマ送り、再生速度切替（`0.25x` / `0.5x` / `1.0x` / `1.5x`）、ループ再生を搭載。
   - キャンバス上の任意座標（または合成・実写シーンの基準ランドマーク座標）をピン留めし、フレーム進行に伴う視差 `d (px)` と推定距離 `Z (m)` の時間変化をリアルタイムに追跡可能。

## Classifier cache and inventory

EfficientNet-Lite0/Lite2 add whole-image Top5 classification through the pinned
MediaPipe API, with manual candidate speech and no detection boxes. Their exact
Apache-2.0 metadata, sizes and hashes are recorded in the
[classifier cache contract](docs/verification/classifier-cache.md). Real model
inference has not been validated by the model-free CI controls.

`npm run dev` and ordinary `npm run preview` / `npm start` supply the container
cache API and schedule startup prefetch for registered variants. Startup waits
for the server to listen and uses the configured delay/concurrency. Classifier
bytes must match their fixed size and SHA-256 on acquisition, reuse and serving;
there is no unverified upstream fallback. An ephemeral container's cache is lost
when it is destroyed. The static [/model-cache.html](public/model-cache.html)
inventory itself only reads status; starting a normal server can independently
start downloads. It separates validated files from configured models.

In preview, defining `CI` or `TEST_LOG_DIR` suppresses the default cache API and
startup prefetch. `CONTAINER_MODEL_CACHE=true` overrides that suppression, so
ensure it is absent for model-free checks; setting it to `false` alone is not
an off switch for ordinary preview. This suppression does not apply to the dev
server. Use the repository test wrappers or injected-fetch cache fixtures
rather than starting a normal server to verify documentation or npm contracts.
See the [server hooks](src/services/containerModelCacheServer.ts) and
[model-free cache tests](src/services/containerModelCache.test.ts).

A build or pure static host supplies no cache API by itself. Without a reachable
API, the inventory reports unknown/unavailable and classification stays blocked.
These repository hooks do not establish deployed startup or real-inference
acceptance; those remain separate verification steps.

SSD MobileNetV2 is registered but acquisition-disabled pending genuine SDK
compatibility verification; startup does not download its weights.

## SSD MobileNetV2 availability

The MediaPipe Object Detector family keeps EfficientDet-Lite0 as its default,
retains Lite2, and adds the selectable SSD MobileNetV2 variant. It reuses the
same VIDEO detector, scheduler, boxes, labels and speech path; no new engine,
segmentation or delegate fallback is introduced. SSD is explicitly BLOCKED
before WASM/model I/O, cache acquisition or serving while actual SDK compatibility remains unverified. Fixed size/hash and metadata
from an independent inspection of the user upload are recorded; this executor
could not receive the file through the supported Library transfer. Inventory lists
it as configured/unverified, never cached merely because a file is present.
The official guide documents256×256/float32/COCO80; those are documentation-level
specifications, also confirmed by the independent uploaded-file inspection; actual execution
is still unverified. The RAW outputs are two SSD tensors, with91 label positions
including background and gaps; no contiguous80-index remapping is assumed. The official
project's model-inclusive Apache-2.0 statement is recorded separately from SDK
licensing. See [SSD verification contract](docs/verification/ssd-mobilenet-v2.md).

## Requirements and setup

- Node.js 24 LTS, at least 24.22.1 and below 25; `.nvmrc` and CI use 24.22.1 (security release).
- npm 11; the lockfile was generated with npm 11.9.0.
- Node.js and npm are the only supported runtime/package manager. `package-lock.json` is the sole lockfile; the unused Bun bootstrap lock has been removed.
- A modern browser with WebAssembly and WebGL for real inference.

```sh
nvm use
npm ci
npm run dev
```

`nvm use` is optional if the required Node version is already installed. The dev
server listens on http://localhost:3000 (and all interfaces). No API key or `.env`
file is required by the current frontend. `.env.example` is inherited scaffold
metadata; its Gemini/APP_URL placeholders are not read by this app. Never commit
credentials; local `.env*` files are ignored except for `.env.example`.

Use `npm ci` for repeatable installs. Do not use `--force` or
`--legacy-peer-deps`. Update and commit `package-lock.json` intentionally when
changing dependencies.

### Container size optimization & Cloud Run publishing limit

To keep Cloud Run service updates within the **250 MB publishing limit**, non-essential
build artifacts, test suites (`e2e/`), verification archives (`docs/`), test caches
(`.cache/`), development scripts (`scripts/mirage/`, `scripts/testing/`), and heavy binary
model caches are strictly excluded via `.gcloudignore` and `.dockerignore`. All agents and
contributors must ensure unnecessary files are kept out of the container image build context.


## Commands

| Command | Purpose |
| --- | --- |
| `npm run dev` | Start Vite on port 3000 |
| `npm run typecheck` | Run TypeScript (`tsc --noEmit`) over source, tests, and TS configs |
| `npm run lint` | Run ESLint with TypeScript recommended rules and React Hooks rules; warnings fail |
| `npm test` | Run Vitest unit/component tests once and exit |
| `npm run test:watch` | Run Vitest interactively in watch mode |
| `npm run build` | Write production assets to `dist/` |
| `npm run preview` | Serve an existing production build locally (port 3000 by default; override with `PORT` or `--port`) |
| `npm run test:e2e` | Run Playwright against production preview; build first |
| `npm run test:e2e:controls` | Check production-smoke sensitivity using disposable healthy and faulty builds |
| `npm run clean` | Remove `dist/` and legacy generated `server.js`, using Node filesystem APIs |

The clean script avoids shell-specific deletion commands. Linux was exercised in
PR-01; Windows and macOS portability is based on Node's filesystem API and has not
been executed on those operating systems.

## Camera, speech, and network dependencies

The app requests camera access on initial load. Grant permission in a secure browser
context: HTTPS or localhost. The camera session changes only for start/stop or a
facing-mode change, not model/status/settings rerenders. Obsolete/late acquisitions
are stopped without being attached, and owned tracks are released on replacement,
stop, unmount, playback failure or a track ending. Camera readiness waits for video
playback and stays separate from model readiness. A stopped/replaced session clears
old scene detections without disposing the model task. Real device behavior still
requires acceptance testing. Speech starts disabled and depends on the browser's
Web Speech API and installed voices; Japanese voices may not be available.

Runtime inference downloads MediaPipe WASM/loader assets from
`cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm`, and EfficientDet-Lite0
and Selfie Segmenter models from `storage.googleapis.com/mediapipe-models` (see
`src/services/visionEngine.ts` for exact URLs). The npm SDK is pinned to 0.10.14 to
match those WASM assets. The Selfie model URL contains `latest`, so model downloads
are not immutable. Google Fonts also loads external styles/fonts at runtime.
ONNX models use the existing Hugging Face resolve/main URLs in their service files
and version-matched ONNX Runtime1.30.0 assets from jsDelivr. Model references are
mutable; this correction does not acquire model bytes or resolve their weight
license/provenance questions. Camera/network access and a supported configured
backend are required for real inference;
builds and automated static/unit/smoke checks do not require these model services.

## Tests and CI

For compact agent-facing output, complete raw artifacts and targeted diagnostics, see
[Compact verification output](docs/test-output.md). With dependencies and an authorized installed browser available,
`npm run --silent verify:offline` runs the installed checks without downloading browsers or models.


```sh
npm ci
npm run typecheck
npm run lint
npm test
npm run build
npx playwright install --with-deps chromium
npm run test:e2e
npm run test:e2e:controls
```

Vitest uses jsdom and React Testing Library. Tests cover dictionary/speech boundaries,
supported model selection, staged initialization errors, explicit retry, shared
requests (including synchronous subscriber reentrancy), cancellation, task ownership
and controlled out-of-order completion. The actual engine/state/UI run with the
MediaPipe creation boundary injected. Canvas bridge tests use a controlled RAF,
timers and canvas context, and exercise successful inference dispatch as well as
old-result invalidation. Camera acquisition uses controlled Promises and synthetic
streams in tests; these are not device tests.
Unit tests do not evaluate real GPU inference, canvas rendering, or audible speech.

The camera lifecycle browser test controls only acquisition, using native
MediaStreams/tracks from synthetic canvas video. Actual app/UI rendering, video
playback and track stopping run in Chromium. It checks render stability,
stop/restart, delayed acquisition and out-of-order facing changes. It is not a
physical camera or inference test. Unit hook tests separately cover unmount,
StrictMode, stale rejection, playback failure and ended-track cleanup.

Playwright starts a fresh Vite production preview at `127.0.0.1:4173` and runs the
real built app in Chromium. The separate offline production smoke denies camera
access and blocks every external request, including model/WASM/CDN/font resources.
It verifies supported/disabled model choices, comparison UI, model-load error/retry,
continued camera controls, required stylesheet CSS MIME/rules/applied styles,
successful local resources, and no uncaught JavaScript exception. It does not
replace app modules or mock model outputs.

The stylesheet controls rerun `e2e/smoke.spec.ts` explicitly with retries disabled
on disposable copies of the build. Two intact builds must pass. Missing CSS,
removed stylesheet links, empty CSS, unrelated valid CSS, and fatal JavaScript
must each fail at the intended assertion. A setup failure or the wrong diagnostic
does not satisfy a control. The original build is preserved. Selecting the smoke
explicitly keeps these seven controls independent of other browser test files.

If a compatible Chromium is already installed, a local-only override is available:

```sh
# POSIX shell example; CI uses Playwright's matched Chromium instead.
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run test:e2e
```

`.github/workflows/ci.yml` runs on every pull request and pushes to `main`, using
the lockfile and pinned Node version. PR runs explicitly check out the PR head SHA.
The job runs install, typecheck, ESLint, unit tests, production build, browser
installation, all browser tests, and stylesheet controls in that order. Failures stop subsequent verification
steps and fail the job. Browser reports and failure traces/screenshots are uploaded
for 14 days. CI needs no application secrets and does not deploy or merge.

See [PR-02/03 integration verification](docs/verification/integration-2026-10-24.md)
for the reentrancy correction, CSS reconciliation, source bindings and CI receipts.

See [PR-03 camera verification](docs/verification/pr03.md) for session ownership,
controlled races and remaining device/scheduler limits.

See [PR-02 verification evidence](docs/verification/pr02.md) for state/task ownership,
regression coverage and remaining lifecycle/measurement limits.

See [Cactus Mirage Technical Reference](docs/mirage-integration.md) and
[Mirage Evaluation Report](docs/verification/mirage-evaluation-report.md) for the
51.3M ultra-lightweight perception model evaluation, comparative benchmarks, and
hybrid triage edge-routing architecture.

See [PR-01 verification evidence](docs/verification/pr01.md) for executed checks,
dependency decisions, environmental limits, and known deferred issues. Real
camera/device acceptance, inference accuracy, tracking, timing semantics, full TTS
behavior, and accessibility are future work; a successful CI run does not certify them.
