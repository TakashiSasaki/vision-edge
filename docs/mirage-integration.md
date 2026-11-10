# Cactus Mirage (51.3M) 技術知見・統合ドキュメント

本ドキュメントは、Cactus Compute が公開した知覚特化型超軽量ビジョンモデル「**Mirage**」の技術的仕様、実機検証（8枚のテスト画像）で判明した動作特性、エッジケースとその回避策、既存ビジョンパイプライン（VisionStudio AI）との統合方法、およびハイブリッドルーティングのベストプラクティスをまとめたリファレンスです。

---

## 1. モデル基本仕様とアーキテクチャ

| 項目 | 仕様・計測値 |
| :--- | :--- |
| **モデル名** | Mirage |
| **開発元 / リポジトリ** | Cactus Compute, Inc. / [huggingface.co/Cactus-Compute/mirage](https://huggingface.co/Cactus-Compute/mirage) |
| **Python ライブラリ** | `cactus-compute==4.1.0` (ライセンス: Apache-2.0) |
| **パラメータ数** | 51.3M |
| **重みファイル形式** | `.cact` アーカイブ（2-bit / 4-bit 量子化） |
| **ファイルサイズ** | **21,611,043 bytes（約 20.61 MB）** |
| **SHA-256 ダイジェスト** | `4249a0f7515272ff76b9414bec17d7e7daf84b15b66c7fb4f17fce1e66f9f237` |
| **ネイティブ共有ライブラリ** | `libcactus.so` (Linux x86_64 / arm64, macOS Metal) |
| **ウォーム初期化時間** | **約 310〜370 ms**（ローカル重みファイル指定時） |
| **推論時メモリフットプリント** | ロード時 RSS 増加 **+59.6 MB** / ピーク RSS 増加 **+174.4 MB**（総 RSS ~224 MB） |

### 既存モデル（VisionStudio AI）との決定的な違い
従来のパイプラインでは「矩形検出（EfficientDet-Lite0: 7.0 MB）」「人物セグメンテーション（Selfie Segmenter: 0.24 MB）」「インスタンスマスク（YOLOv8n/s-seg: 13.7〜45.9 MB）」「全画素セグメンテーション（SegFormer: 4.3〜27.6 MB）」「全体分類（EfficientNet-Lite: 17.8 MB）」が個別のタスク・別々の重みとして存在していました。

Mirage は、テキスト指示なしで「画像のみ」を入力し、**単一の 20.61 MB 重み・1 パス推論** で以下を網羅した構造化 JSON を直接出力します：
1. **オープン語彙の物体検出 (BBox)**
2. **ポリゴン輪郭マスク (Masks)**
3. **行単位の文字認識 (OCR) と BBox**
4. **物体・領域間の空間関係グラフ (Relations)**
5. **シーン分類・背景・スタイル記述 (Scene)**
6. **ゼロショット構造化フォーム判定 / 保留付き選択肢分類 (Decisions API)**

---

## 2. API 仕様と出力スキーマの詳細

### 2.1 高速初期化の要点 (`weights` 引数の指定)
`cactus.Mirage()` を引数なしで呼ぶと、毎回 `hf_hub_download` 経由で Hugging Face Hub 上の `config.json` をオンライン確認しようとします。ローカルキャッシュ済みのパスを明示的に渡すことで、オフライン環境でも完全ゼロレイテンシ（~312 ms）で起動できます。

```python
import cactus

# ローカルキャッシュ済み重みを直接指定（推奨）
vision = cactus.Mirage(weights="/path/to/mirage.cact")
```

### 2.2 全体記述 API: `vision.describe(image, **options)`

```python
results = vision.describe("path/to/image.png")
```

返却される JSON の主要セクションとデータ構造：

```json
{
  "scene": {
    "background": "white",
    "style": "screenshot",
    "setting": "mobile phone screen"
  },
  "objects": [
    {
      "id": 1,
      "name": "window",
      "bbox": [53, 161, 951, 789],
      "score": 0.742,
      "mask": [53, 161, 951, 161, 951, 789, 53, 789],
      "position": "center",
      "description": { "color": "white" }
    }
  ],
  "regions": [
    {
      "name": "status bar",
      "score": null
    }
  ],
  "text": [
    {
      "string": "VisionStudio AI",
      "bbox": [134, 77, 310, 98],
      "score": 0.615
    }
  ],
  "relations": [
    {
      "subject": 1,
      "predicate": "inside",
      "object": 4
    }
  ]
}
```

#### 【重要】座標系の仕様（0..1000 スケール）
- `bbox`: `[x1, y1, x2, y2]` は画像全体の幅・高さを `0..1000` に正規化した整数値です。
- `mask`: ポリゴン頂点列 `[x1, y1, x2, y2, ...]` のフラット配列であり、同様に `0..1000` スケールです。
- VisionStudio AI や Web 標準の `[0, 1]` 座標系へ変換する際は、`Math.min(1, Math.max(0, val / 1000))` による正規化とクランプが必要です。

### 2.3 構造化フォーム・選択肢判定 API: `vision.extract(image, form, **options)`

任意の画像に対して、指定したスキーマに沿った選択肢分類、個数カウント、テキスト抽出をゼロショットで実行します。

```python
form_schema = {
    "scene_domain": {
        "enum": [
            "application_ui_screenshot",
            "camera_person_capture",
            "document_or_form"
        ]
    },
    "person_count": {
        "type": "integer",
        "minimum": 0,
        "maximum": 5
    },
    "contains_readable_text": {
        "enum": ["yes", "no"]
    }
}

# フォーム抽出のみを行う場合は options={"sections": []} を指定して高速化
response = vision.extract("image.png", form_schema, options={"sections": []})
form_output = response["form"]
```

#### 【重要】判定保留（Abstention `"-"`）と `null` 値の挙動
- 選択肢（`enum`）フィールドでは、指定した候補（例: `["yes", "no"]`）のほかに、モデル内部で **判定保留（`"-"`）の確率** が自動的に算出されます。
- `"-"` の確率が最大となった場合、Mirage は `"value": null` を返却します。
  ```json
  "contains_readable_text": {
    "value": null,
    "probabilities": {
      "yes": 0.12,
      "no": 0.18,
      "-": 0.70
    }
  }
  ```
- この `"-"` の確率は、**ハルシネーションの検知やクラウド VLM へのエスカレーション判定フラグとして極めて有用** です。

---

## 3. 実機検証で判明したエッジケースと実運用の注意点

### 3.1 C エンジン (`libcactus.so` v4.1.0) の `-nan` 出力とサニタイズ
- **事象**: 縦長モバイル画面（`1080x2424`）の画面上端（`y=0..20` 付近）にある微小テキスト・記号領域を推論した際、`libcactus.so` の内部バッファに `"score":-nan` という文字列が書き込まれるケースが確認されました。
- **影響**: Python 標準の `json.loads()` は小文字の `-nan` をパースできず、`json.decoder.JSONDecodeError: Expecting value: line 1 column ...` が発生します。
- **回避策**: `scripts/mirage/test_mirage_smoke.py` および `scripts/mirage/run_benchmark.py` に以下の正規表現サニタイザを組み込みました：
  ```python
  import re, json

  _NAN_TOKEN_RE = re.compile(r"(:\s*)-?nan\b", re.IGNORECASE)

  def parse_mirage_c_buffer(vision):
      raw_text = vision._buffer.value.decode("utf-8", errors="replace")
      sanitized = _NAN_TOKEN_RE.sub(r"\g<1>0.0", raw_text)
      return json.loads(sanitized)
  ```

### 3.2 日本語 OCR（文字認識）の特性
- **英数字・記号**: 認識率・精度ともに極めて高く（ベンチマーク 8 画像平均のトークン Recall **89.17%**、5 画像で 100%）、`VisionStudio AI`, `YOLO-SEG`, `24 FPS`, `73.4ms`, `INV-2026-0842` などを正確に bbox 付きで認識します。
- **日本語（かな・漢字）**: 現行の `Mirage` 重みはラテン文字中心に学習されているため、日本語文字列（例: `人物 (カメラ被写体) 94%` → `"A (xSt) 94%"`、`テーブル・机 88%` → `"7-IL·M88%"`）はアルファベットや記号に近似されます。
- **運用指針**: 日本語文字列の厳密な読み取りが必要なドメインでは、Mirage の OCR 信頼度スコアをトリガーとしてクラウド VLM（Gemini Flash 等）へフォールバックさせる運用が推奨されます。

### 3.3 複合画面（UI 内の小さなカメラ枠）での注意点
- アプリ画面全体（1080x2424）のスクリーンショットをそのまま入力すると、画面全体を占める `button`, `smartphone`, `status bar` が優位に検出され、内部の小さなカメラ枠内の人物の優先度が下がる傾向があります。
- カメラ映像内の人物や物体を高精度に検出したい場合は、**カメラビューポート領域（例: 640x480）をクロップして入力** することで、`man`, `woman`, 服装などを 100% 確実に捉えることができます。

---

## 4. 既存パイプラインとの比較ベンチマーク結果（代表8画像）

### 4.1 総合比較表

| モデル名 | ランタイム | 重み容量 (MB) | 初期化時間 (ms) | 1画像あたり推論時間 (ms) | ピークメモリ増加 (MB) | 物体検出 | ポリゴンマスク | OCR | 空間関係 | ゼロショット判定 |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: | :---: | :---: | :---: | :---: |
| **Selfie Segmenter** | MediaPipe (WASM/GPU) | 0.24 MB | ~185 ms | **~14.2 ms** | ~68 MB | 人物のみ | ◎ (画素単位) | × | × | × |
| **EfficientDet-Lite0** | MediaPipe (WASM/GPU) | 7.04 MB | ~220 ms | **~24.8 ms** | ~85 MB | COCO 80種 | × | × | × | × |
| **YOLOv8n-seg** | ONNX Runtime Web | 13.70 MB | ~340 ms | **~78.5 ms** | ~145 MB | COCO 80種 | ◎ (160x160) | × | × | × |
| **YOLOv8s-seg** | ONNX Runtime Web | 45.88 MB | ~490 ms | ~168.0 ms | ~210 MB | COCO 80種 | ◎ (160x160) | × | × | × |
| **SegFormer-B0** | ONNX Runtime Web | 4.30 MB | ~280 ms | ~92.4 ms | ~130 MB | 領域外接 | ◎ (ADE20K 150種) | × | × | × |
| **EfficientNet-Lite0** | MediaPipe (Classifier) | 17.75 MB | ~214 ms | **~11.5 ms** | ~92 MB | × (全体分類) | × | × | × | × |
| **Mirage (Full Describe)** | Cactus Native (`libcactus.so`) | **20.61 MB** | **312.6 ms** | **約 6.8〜7.4 秒 (CPU)** *(M4 Pro等: ~350 ms)* | **+174.4 MB** | **◎ (オープン語彙)** | **◎ (頂点列)** | **◎ (89.2% Recall)** | **◎ (グラフ)** | **◎ (同時出力)** |
| **Mirage (Fast Triage)** | Cactus Native (`libcactus.so`) | **20.61 MB** | **312.6 ms** | **約 6.6〜8.0 秒 (CPU)** | **+174.4 MB** | **◎ (上位10件)** | **◎ (頂点列)** | **◎ (上位20行)** | 省略 | 省略 |
| **Mirage (Decisions Only)** | Cactus Native (`libcactus.so`) | **20.61 MB** | **312.6 ms** | **約 7.9〜9.0 秒 (CPU)** *(M4 Pro等: ~120 ms)* | **+174.4 MB** | ー | ー | ー | ー | **◎ (保留確率付き)** |

---

## 5. 推奨アーキテクチャ: ハイブリッド・エッジ・トリアージ

### 5.1 役割分担の原則
- **毎フレームのライブ描画（30〜60 FPS）**: 既存の WebGPU/WASM モデル（`Selfie Segmenter` / `EfficientDet-Lite0` / `YOLOv8n-seg`）を使用し、遅延のないリアルタイムオーバーレイを維持します。
- **オンデマンド知覚 / スナップショット解析 / OCR / 音声対話**: ユーザーが操作したタイミング、または 1〜2 秒間隔のキーフレームで `MirageVisionAdapter` を非同期実行します。

### 5.2 3段階ハイブリッドルーティング (`evaluateHybridTriage`)

```text
[入力画像 / カメラキーフレーム]
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│ Step 1: ローカル一次知覚 (MirageVisionAdapter)           │
│  - 20.61 MB 量子化モデルで構造化 JSON & Decisions を取得 │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│ Step 2: トリアージ評価 (evaluateHybridTriage)            │
│  ① Decisions の保留確率 ("-") < 0.40 か？                │
│  ② 主たるオブジェクトの信頼度 >= 0.45 か？              │
│  ③ OCR 平均信頼度 >= 0.55 か？                           │
│  ④ 自由な多段階推論・世界知識が不要か？                 │
└───────┬──────────────────────────┬───────────────────────┘
        │ YES (質問なし)            │ YES (質問あり)                  │ NO (低信頼 / 保留高 / 複雑推論)
        ▼                          ▼                                ▼
┌────────────────────┐   ┌────────────────────────────┐   ┌────────────────────────────────┐
│ Route A:           │   │ Route B:                   │   │ Route C:                       │
│ local-mirage-only  │   │ mirage-json-to-text-llm    │   │ cloud-vlm-fallback             │
│ エッジ完結 (費用0) │   │ 構造化JSONのみを軽量LLMへ  │   │ 画像+Mirage事前情報をクラウド  │
│                    │   │ 送信 (画像トークン不要)    │   │ VLM (Gemini Flash等)へ転送     │
└────────────────────┘   └────────────────────────────┘   └────────────────────────────────┘
```

#### VQAv2 アブレーション知見のプロンプト設計への反映
Cactus Compute の検証において、Mirage の内部 `reasoning` トレースを含めず、構造化された `scene`, `objects`, `text`, `relations`, `form` のみを後段 LLM に入力した方が回答精度が高い（**71.1% vs 67.3%**）ことが判明しています。`src/services/mirageVisionAdapter.ts` の `buildDownstreamLlmPayload` ではデフォルトで `includeReasoningTrace: false` としており、トークン消費の削減と精度の最大化を両立しています。

---

## 7. ブラウザ WebAssembly (WASM CPU / Web Worker) 実装と動作仕様

### 7.1 WebGPU 非対応と Cactus WASM (CPU) ランタイムの採用
- Mirage の重みファイル（`mirage.cact`, 20.61 MB）は Cactus 独自の 2-bit / 4-bit 量子化アーカイブ形式であるため、`onnxruntime-web` や `@mediapipe/tasks-vision` の WebGPU デリゲートには直接読み込めません。
- 一方で、Cactus Compute 公式の Emscripten ビルド（`cactus.js` + `cactus.wasm`, 約 1.2 MB）を用いることで、**ブラウザ内の WebAssembly (CPU) のみで `mirage.cact` をロードしローカル推論を実行することが可能**です。

### 7.2 Web Worker 非同期連続推論アーキテクチャ
CPU WASM 推論は 1 フレームあたり数百ミリ秒〜数秒を要するため、メインスレッド（UI・カメラ映像描画）をブロックしないよう以下の構成で統合しています：

1. **Web Worker 分離 (`src/services/mirageWasmWorker.ts`)**:
   - `public/wasm/cactus/cactus.js` と `cactus.wasm` を Worker 内で初期化し、`/api/model-cache/mirage/mirage.cact`（未キャッシュ時は Hugging Face 公式 URL へ自動フォールバック）から重みを WASM ヒープへロード（`_cactus_load(ptr, BigInt(len))`）。
   - `_cactus_describe` が稀に出力する `-nan` スコアを JSON パース前に `sanitizeCactusJson()` で `0.0` に正規化。
2. **Latest-Frame-Wins 非同期スケジューリング (`src/services/mirageWasmService.ts` + `VisionEngine`)**:
   - カメラ映像は 60fps で滑らかにプレビューし続け、Worker 推論が完了するたびに最新の検出物体・ポリゴン輪郭マスク・OCR テキスト領域・空間位置ラベル（左・中央・右）を更新し、日本語音声読み上げ（TTS）を行います。
3. **高速モード（320px）とフル解析モード（512px）の切り替え (`ControlPanel.tsx`)**:
   - **高速モード（デフォルト）**: 入力画像を長辺 320px に縮小し、`sections: ['objects', 'text']`, `fields: ['bbox', 'score', 'mask', 'position']` に絞ることで WASM CPU 推論時間を大幅に短縮。
   - **フル解析モード**: 入力画像を長辺 512px とし、`scene`, `regions`, `relations`, `description` を含む全構造化項目を抽出。
   - **OCR テキスト領域の表示・読み上げトグル**: 検出物体・輪郭マスクに加えて画像内の文字領域（OCR）も枠表示・音声発話するかどうかをリアルタイムに切り替え可能。

---

## 8. リポジトリ内関連ファイル一覧

- `docs/mirage-integration.md`: 本ドキュメント（技術知見・WASM統合リファレンス）
- `docs/verification/mirage-evaluation-report.md`: 包括的評価・比較検証レポート
- `docs/verification/mirage-benchmark-results.json`: 代表 8 画像の実機ベンチマーク生データ（レイテンシ、メモリ、Recall）
- `public/wasm/cactus/`: Cactus Compute 公式 WebAssembly ランタイム（`cactus.js`, `cactus.wasm`, `cactus.h`）
- `src/services/mirageWasmWorker.ts`: ブラウザ Web Worker 上で `_cactus_load` / `_cactus_describe` を実行する WASM ワーカー
- `src/services/mirageWasmService.ts`: フレーム縮小・RGB パッキング・高速/フル解析モード制御・`VisionEngine` タスク実装
- `src/services/mirageVisionAdapter.ts`: アダプタ層・座標正規化・要約・ハイブリッドルータ実装
- `src/services/mirageVisionAdapter.test.ts`: アダプタ層および WASM Worker 統合の Vitest 単体テスト（10 tests, 100% PASS）
- `pyproject.toml` / `requirements-mirage.txt`: `cactus-compute==4.1.0` 依存定義
- `scripts/mirage/test_mirage_smoke.py`: 単体推論・CLI ブリッジスクリプト（`npm run mirage:smoke`）
- `scripts/mirage/run_benchmark.py`: 8画像比較ベンチマーク実行スクリプト（`npm run mirage:benchmark`）
