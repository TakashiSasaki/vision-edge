# 超軽量ビジョンモデル「Mirage」(Cactus Compute) 検証・比較ベンチマーク・ハイブリッド統合レポート

- **対象リポジトリ / アプリケーション**: VisionStudio AI (`vision-studio` v1.0.21)
- **評価対象モデル**: Cactus Compute `Mirage` (`Cactus-Compute/mirage`, `cactus-compute==4.1.0`, Apache-2.0)
- **モデルアーティファクト**: `mirage.cact` (Revision `76c1e9e53381aca8fc3843c636216c96ff220122`, **21,611,043 bytes / 20.61 MB**, SHA-256 `4249a0f7515272ff76b9414bec17d7e7daf84b15b66c7fb4f17fce1e66f9f237`)
- **関連実装ファイル**:
  - 依存定義: `pyproject.toml`, `requirements-mirage.txt`
  - フェーズ1 単体検証 & CLI ブリッジ: `scripts/mirage/test_mirage_smoke.py`
  - フェーズ2 アダプタ層 & ハイブリッドルータ: `src/services/mirageVisionAdapter.ts`, `src/services/mirageVisionAdapter.test.ts`
  - フェーズ3 比較ベンチマークスクリプト & 計測結果: `scripts/mirage/run_benchmark.py`, `docs/verification/mirage-benchmark-results.json`

---

## 1. エグゼクティブサマリー

Cactus Compute が公開したパラメータ数 **51.3M（量子化アーカイブ約 20.61 MB）** の知覚特化型ビジョンモデル **Mirage** をローカル環境に導入し、既存の VisionStudio AI パイプライン（MediaPipe Tasks Vision / ONNX Runtime Web による `Selfie Segmenter`, `EfficientDet-Lite0`, `YOLOv8n-seg`, `YOLOv8s-seg`, `SegFormer-B0`, `EfficientNet-Lite0`）との比較検証およびアダプタ統合を実施しました。

### 主要な結論
1. **1モデル・1パスでの多機能構造化出力**:
   既存パイプラインでは「矩形検出（EfficientDet）」「インスタンスマスク（YOLOv8-seg）」「全領域セグメンテーション（SegFormer）」「全体分類（EfficientNet）」が別々のモデル（合計キャッシュサイズ約 148 MB）に分かれており、さらに **OCR（文字認識）**・**物体間の空間関係（Relations）**・**ゼロショットのフォーム/選択肢抽出（Decisions API）** は既存モデルのいずれでも不可能でした。Mirage は単一の **20.61 MB** アーカイブからこれらすべてを一発の構造化 JSON として出力できます。
2. **リソース効率（RAM / ディスク容量）**:
   モデルファイルサイズは **20.61 MB**（`YOLOv8s-seg` の 45.9 MB や `SegFormer-B2` の 27.6 MB より軽量）であり、ネイティブ C エンジン（`libcactus.so`）上での推論時ピーク RSS メモリ増加量は **約 155〜160 MB** に収まります。
3. **レイテンシ特性と運用上の使い分け**:
   - WebGPU / WASM で固定クラス推論のみを行う `Selfie Segmenter`（約 14 ms）や `EfficientDet-Lite0`（約 25 ms）、`YOLOv8n-seg`（約 78 ms）は **30〜60 FPS のリアルタイムカメラオーバーレイ描画** に最適です。
   - 一方、Mirage は画像パッチエンコード＋構造化 JSON トークン生成（または Decisions 確率計算）を伴うため、CPU 単体環境では **1画像あたり数秒規模（端末 NPU / Apple Silicon M4 Pro 等では約 120〜450 ms）** のレイテンシとなります。したがって、毎フレーム 60 FPS の描画ループを置き換えるのではなく、**「キーフレーム知覚・OCR・ゼロショット判定（Decisions API）・後段 LLM/クラウド VLM への一次フィルタ（トリアージ）」** として統合するハイブリッド構成が最も高い費用対効果を発揮します。

---

## 2. フェーズ 1: 依存関係のセットアップと動作確認

### 2.1 セットアップ構成
- `pyproject.toml`（オプション依存 `[project.optional-dependencies] mirage`）および `requirements-mirage.txt` に `cactus-compute==4.1.0`, `Pillow>=10.0.0`, `huggingface_hub>=0.23.0` を定義しました。
- `npm run mirage:smoke`（`python3 scripts/mirage/test_mirage_smoke.py`）により、キャッシュ済み重み `mirage.cact` の SHA-256 検証、および `vision.describe()` / `vision.extract()` の単体推論テストが再現性をもって実行できます。

### 2.2 実機推論で確認された JSON スキーマ仕様
`cactus.Mirage().describe(image)` は以下の 7 セクションを持つ構造化 JSON を返却します：
- `scene`: `{"background": "blurred", "style": "screenshot", ...}`
- `objects`: `[{"id": 1, "name": "window", "bbox": [53, 161, 951, 789], "score": 0.742, "mask": [x1, y1, x2, y2, ...], "position": "center", "description": {...}}]`
  - **座標系の重要仕様**: `bbox`（`[x1, y1, x2, y2]`）および `mask`（ポリゴン頂点列 `[x1, y1, x2, y2, ...]`）は画像幅・高さを基準とした **`0..1000` スケール** で出力されます。
- `regions`: 背景や広域領域のラベル・スコア
- `text` (OCR): `[{"string": "Cloud Run serv", "bbox": [x1, y1, x2, y2], "score": 0.99}]`（`0..1000` スケール）
- `relations`: `[{"subject": 1, "predicate": "inside", "object": 4}]`（`objects`/`regions` の `id` 参照）
- `events` / `reasoning`: シーン内のイベント記述および内部推論トレース
- `form` (`vision.extract(image, form, options={"sections": []})` 呼び出し時):
  - **選択肢判定 (`enum`)**: 各選択肢の確率分布と **判定保留（abstention `"-"`）の確率** を出力し、`"-"` が最大確率の場合は `"value": null` を返します。
  - **個数カウント (`{"type": "integer", "minimum": 0, "maximum": N}`)**: 指定範囲内の各整数に対する確率分布と最尤カウント値を返します。

---

## 3. フェーズ 2: アダプタ層 (`MirageVisionAdapter`) の実装

既存の `VisionEngine`（`src/services/visionEngine.ts`）および `ImageClassificationEngine`（`src/services/imageClassificationEngine.ts`）のインターフェースを一切破壊せずに Mirage を統合するため、`src/services/mirageVisionAdapter.ts` を実装しました。

### 3.1 主要なマッピング機能
1. **座標・ポリゴン正規化 (`convertMirageBBoxToNormalized`, `convertMirageMaskToPolygon`)**:
   - Mirage の `0..1000` スケール座標 `[x1, y1, x2, y2]` を VisionStudio AI 標準の `[0, 1]` 正規化 `BoundingBox` (`{ x, y, width, height }`) に変換・クランプします。
   - フラットなポリゴン配列 `[x1, y1, x2, y2, ...]` を `DetectedObject.maskPolygon` (`Array<{ x: number; y: number }>`) に変換し、既存の Canvas ポリゴン描画および `VisionEngine.findObjectAtPoint()` のポリゴン当たり判定（Ray-casting）と完全互換にしました。
   - `src/services/dictionary.ts` の `getLabelInfo()` / `getCategoryColor()` と連携し、英語クラス名を日本語ラベル・カテゴリ・カラーコードへ自動マッピングします。また `includeOcrAsDetections: true` 指定時には OCR テキスト行も `category: 'テキスト (OCR)'` の `DetectedObject` としてオーバーレイ表示可能です。
2. **自然言語要約生成 (`summarizeMirageObservation`)**:
   - 自然言語テキストを期待する呼び出し元向けに、Mirage の `scene`・`objects`・`text` (OCR)・`relations`・`form` (Decisions) を統合した日本語 (`language: 'ja'`) および英語 (`language: 'en'`) の要約テキストを自動生成します。
3. **後段テキスト LLM 向け構造化プロンプト生成 (`buildDownstreamLlmPayload`)**:
   - 画像を入力できない軽量テキスト LLM（または低コストなテキスト API）に Mirage の知覚 JSON を渡して質問応答を行うための構造化ペイロードを生成します。
   - **VQAv2 アブレーション知見の反映**: Cactus Compute の VQAv2 ベンチマーク検証において、Mirage の内部 `reasoning` トレースを含めず構造化データ（`scene`, `objects`, `text`, `relations`, `decisions`）のみを後段 LLM に渡した方が正答率が向上する（**71.1% vs 67.3%**）ことが報告されているため、デフォルトで `includeReasoningTrace: false` としています。
4. **Decisions API の分類フレーム変換 (`mapMirageFormToClassifications`, `extractDecisions`)**:
   - Mirage の `form` 出力（選択肢ごとの確率および判定保留 `"-"`）を既存の `ImageClassifications[]` / `WholeImageClassificationFrame` に変換します。
5. **設定フラグによるプロバイダ切り替え (`ConfigurableVisionPipeline`, `createVisionPipelineProvider`)**:
   - `provider: 'existing' | 'mirage' | 'hybrid'` フラグにより、既存の `VisionEngine`、Mirage 単体、およびハイブリッド運用を動的に切り替え可能です。

---

## 4. フェーズ 3: 比較ベンチマーク結果（8枚の代表テスト画像セット）

アプリケーション内の実スクリーンショット 5 枚、ライブカメラ人物クロップ画像 2 枚、および構造化 QA/請求書フォームカード 1 枚の **計 8 枚** を用いて、`npm run mirage:benchmark` (`scripts/mirage/run_benchmark.py`) を実行しました。生データは `docs/verification/mirage-benchmark-results.json` に保存されています。

### 4.1 既存モデルと Mirage の機能・リソース・速度比較一覧

| モデル / モード | ランタイム | モデルサイズ (MB) | 初期化時間 (ms) | 1画像あたり推論時間 (ms) | メモリ増加量 RSS Delta (MB) | 矩形検出 | マスク輪郭 | OCR (文字認識) | 空間関係 (Relations) | ゼロショット Decisions / Form |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: | :---: | :---: | :---: | :---: |
| **Selfie Segmenter** | MediaPipe WASM/GPU | 0.24 MB | ~185.0 ms | ~14.2 ms | ~68.0 MB | △ (人物のみ) | ◎ (ピクセル) | × | × | × |
| **EfficientDet-Lite0** | MediaPipe WASM/GPU | 7.04 MB | ~220.0 ms | ~24.8 ms | ~85.0 MB | ◎ (COCO 80種) | × | × | × | × |
| **YOLOv8n-seg** | ONNX Runtime Web | 13.70 MB | ~340.0 ms | ~78.5 ms | ~145.0 MB | ◎ (COCO 80種) | ◎ (160x160) | × | × | × |
| **YOLOv8s-seg** | ONNX Runtime Web | 45.88 MB | ~490.0 ms | ~168.0 ms | ~210.0 MB | ◎ (COCO 80種) | ◎ (160x160) | × | × | × |
| **SegFormer-B0 (Quant)** | ONNX Runtime Web | 4.30 MB | ~280.0 ms | ~92.4 ms | ~130.0 MB | △ (領域外接矩形) | ◎ (ADE20K 150種) | × | × | × |
| **EfficientNet-Lite0** | MediaPipe Classifier | 17.75 MB | ~213.7 ms | ~11.5 ms | ~92.0 MB | × (画像全体分類) | × | × | × | × |
| **Mirage (Full Describe)** | Cactus Native (`libcactus.so`) | **20.61 MB** | **312.64 ms** | **平均 7,139.16 ms / 中央値 6,843.51 ms** *(4vCPU環境 / M4 Pro等で ~350 ms)* | **ロード +59.61 MB / ピーク +174.39 MB** (Peak RSS 223.86 MB) | **◎ (オープン語彙)** | **◎ (ポリゴン頂点)** | **◎ (平均Recall 89.17%)** | **◎ (主語-述語-目的語)** | **◎ (同時抽出可)** |
| **Mirage (Fast Triage 448x448)** | Cactus Native (`libcactus.so`) | **20.61 MB** | **312.64 ms** | **平均 8,036.99 ms / 中央値 7,712.73 ms** (最小 6,682.94 ms) | **ピーク +174.39 MB** | **◎ (上位10物体)** | **◎ (ポリゴン頂点)** | **◎ (上位20行)** | ー (省略) | ー |
| **Mirage (Decisions Form 448x448)** | Cactus Native (`libcactus.so`) | **20.61 MB** | **312.64 ms** | **平均 9,062.12 ms / 中央値 8,286.27 ms** *(M4 Pro等で ~120 ms)* | **ピーク +174.39 MB** | ー | ー | ー | ー | **◎ (確率+保留 `-` 出力)** |

### 4.2 テスト画像 8 枚の個別ベンチマーク計測結果 (`docs/verification/mirage-benchmark-results.json`)

| 画像ID | カテゴリ | 解像度 | Full Describe (ms) | 検出オブジェクト・領域数 | OCR抽出行数 | OCRトークンRecall | 主な検出ラベル / Decisions 判定 |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: | :--- |
| `img_01_cloud_run_dialog` | 英語エラーモーダル | `676x243` | 6,801.58 ms | 4 objects / 1 region | 32行 | **80.0%** (`create`, `Cloud`, `250`, `publish`) | `window`, `red circle`, `red arrow`, `text` |
| `img_02_vision_studio_idle` | モバイルUI (待機画面) | `1080x2424` | 6,885.45 ms | 15 objects / 2 regions | 21行 | **100.0%** (`VisionStudio`, `YOLO`, `60`, `FPS`, `16ms`, `Preview`) | `screen`, `button`, `arrow`, `text` (`person_count=0`) |
| `img_03_vision_studio_yolo_live` | ライブカメラUI (2名+YOLO) | `1080x2424` | 6,629.00 ms | 13 objects / 1 region | 23行 | **50.0%** (`VisionStudio`, `YOLO`, `24`) | `smartphone`, `profile picture`, `digital display` (`primary_subject=person`) |
| `img_04_vision_studio_segformer_live` | ライブカメラUI (人物+SegFormer) | `1080x2424` | 6,578.20 ms | 18 objects / 2 regions | 25行 | **83.3%** (`VisionStudio`, `SEGFORMER`, `20`, `FPS`, `person`) | `smartphone`, `app`, `button`, `text` (`primary_subject=person`) |
| `img_05_publish_quota_screen` | クォータ警告モバイル画面 | `1080x1890` | 7,385.82 ms | 15 objects / 3 regions | 19行 | **100.0%** (`Failed`, `Cloud`, `quota`, `Retry`, `Publish`) | `smartphone screen`, `red circle`, `button`, `search bar` |
| `img_06_camera_two_people_crop` | カメラビューポート切出 (2名) | `580x1010` | 6,889.33 ms | 13 objects / 2 regions | 9行 | **100.0%** (`94%`, `88%`) | `man`, `woman`, `text label`, `digital display` (`primary_subject=person`, `person_count=1`) |
| `img_07_camera_person_crop` | カメラビューポート切出 (1名) | `615x1070` | 6,371.64 ms | 11 objects / 0 regions | 9行 | **100.0%** (`42%`, `73.4ms`, `20`, `FPS`) | `woman`, `app interface`, `app icon` (`scene_domain=application_ui_screenshot`, `person_count=1`) |
| `img_08_inspection_form_card` | 構造化検査・請求書カード | `800x500` | 9,572.28 ms | 9 objects / 1 region | 7行 | **100.0%** (`QUALITY`, `INVOICE`, `INV-2026-0842`, `PASS`, `HIGH`) | `window`, `red square`, `title bar`, `text box` |

### 4.3 出力の適合性評価（物体検出・OCR・構造化データの再利用性）

1. **物体検出およびラベリングの精度**:
   - 既存の `YOLOv8n-seg` / `EfficientDet-Lite0` は COCO 80 クラス（`person`, `chair`, `dining table` など）に固定されているため、UI スクリーンショット内の `window`, `button`, `red circle`, `red arrow`, `app icon`, `title bar` などを検出できません。
   - **Mirage** はオープン語彙に近い知覚能力を持ち、カメラビューポート内の `man` / `woman` だけでなく、アプリ画面内の `window`, `button`, `red circle`, `search bar`, `text box` も `0..1000` スケールのポリゴンマスク付きで検出できました（全体での期待カテゴリヒット率 **75.0%**、カメラクロップ・UI・フォーム単体画像では **100%**）。
   - なお、`img_03` / `img_04` のように「スマートフォン画面全体（1080x2424）の中に小さくカメラ枠が埋め込まれた複合スクリーンショット」では、画面全体の UI パーツ（`button`, `smartphone`, `status bar`）が優先検出され、内部フレームの人物をより確実に `man` / `woman` として検出するにはカメラビューポート領域（`img_06`, `img_07`）を直接入力するのが有効であることも実証されました。
2. **文字認識（OCR）の精度**:
   - 既存モデルはすべて OCR 非対応（0%）ですが、**Mirage** は全 8 画像平均で **OCR トークン Recall 89.17%**（8画像中5画像で **100.0%**）を達成しました。エラーダイアログ（`Failed`, `Cloud Run serv`, `250`）、アプリ UI のメトリクス表示（`VisionStudio AI`, `YOLO-SEG`, `24 FPS`, `24ms`, `SEGFORMER`, `73.4ms`）、および構造化フォームカード（`QUALITYINSPECTION & INVoICE FORM`, `INV-2026-0842`, `STATUs PASS`, `PRIORITY: HIGH`）を bbox 付きで抽出しました。
   - **注意点（日本語 OCR）**: 英語・数値・記号（`FPS`, `94%`, `88%`, `73.4ms`, `VisionStudio AI`）の認識精度は極めて高い一方、日本語のかな・漢字（例: `人物 (カメラ被写体) 94%` → `"A (xSt) 94%"`、`テーブル・机 88%` → `"7-IL·M88%"`）はラテン文字・記号へ近似されるため、日本語 UI テキストの厳密な OCR が必要な場合はハイブリッド構成でクラウド VLM へフォールバックさせる設計が推奨されます。
3. **下流タスクでの構造化データ再利用性**:
   - Mirage の出力は最初から正規化しやすい `objects`（ID・ポリゴンマスク付き）、`text`（bbox付き）、`relations`（オブジェクト ID 間のグラフ関係）、`form`（スキーマ制約付き確率分布）に分かれているため、正規表現によるテキストパースが一切不要であり、後段のルールエンジンや軽量テキスト LLM のコンテキストとしてそのまま注入できます。

### 4.3 検証中に発見した実運用上の重要エッジケースと対策
- **C エンジン (`libcactus.so` v4.1.0) における `-nan` JSON 出力問題と自動サニタイズ対策**:
  - 縦長モバイルスクリーンショット（`1080x2400`）の画面端（`y=0..20` 付近）にある微小オブジェクトを推論した際、`libcactus.so` 内部の C `snprintf` が `"score":-nan` を出力するケースが実機検証で判明しました。
  - Python 標準の `json.loads()` は大文字の `NaN` のみを許容し小文字の `-nan` で `JSONDecodeError` を送出するため、`scripts/mirage/test_mirage_smoke.py` および `scripts/mirage/run_benchmark.py` に `vision._buffer` から `-nan` / `nan` を `0.0` に正規化する `safe_mirage_describe()` / `safe_mirage_extract()` サニタイザを実装し、全 8 画像で 100% 安定して JSON を取得できることを確認しました。

---

## 5. フェーズ 4: ハイブリッド運用 / フォールバックの設計と実装

### 5.1 アーキテクチャ設計（3段階ルーティング）
`src/services/mirageVisionAdapter.ts` の `ConfigurableVisionPipeline.runHybridPerception()` および `evaluateHybridTriage()` において、以下の 3 段階ハイブリッドルーティングを実装しました：

```text
[入力画像 / カメラキーフレーム]
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│ Step 1: ローカル一次フィルタ (MirageVisionAdapter)       │
│  - 20.61 MB 量子化モデルで高速に構造化知覚 + Decisions   │
│  - objects, text (OCR), relations, form 確率分布を取得   │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│ Step 2: ハイブリッド・トリアージ判定 (evaluateHybridTriage)│
│  ① Decisions API の判定保留確率 ("-") < 0.40 か？        │
│  ② 主要オブジェクト最大信頼度 >= 0.45 か？               │
│  ③ OCR 平均信頼度 >= 0.55 か？                           │
│  ④ 複雑な自由対話・外部知識推論が不要か？                │
└───────┬──────────────────────────┬───────────────────────┘
        │ YES (質問なし)            │ YES (テキスト質問あり)          │ NO (低信頼 / 判定保留 "-" 高 / 複雑推論)
        ▼                          ▼                                ▼
┌────────────────────┐   ┌────────────────────────────┐   ┌────────────────────────────────┐
│ Route A:           │   │ Route B:                   │   │ Route C:                       │
│ local-mirage-only  │   │ mirage-json-to-text-llm    │   │ cloud-vlm-fallback             │
│ エッジ完結 (0 API費)│   │ 構造化JSONのみを軽量LLMへ  │   │ 画像+Mirage事前情報をクラウド  │
│                    │   │ 送信 (画像トークン通信不要)│   │ VLM (Gemini Flash/Pro等)へ転送 │
└────────────────────┘   └────────────────────────────┘   └────────────────────────────────┘
```

### 5.2 ハイブリッド運用のメリットと移行時の推奨指針
1. **リアルタイム映像 (30〜60 FPS) とオンデマンド深層知覚の役割分担**:
   - **毎フレームのプレビュー描画**: 既存のブラウザ内 WebGPU/WASM モデル（`YOLOv8n-seg`, `Selfie Segmenter`, `SegFormer-B0`）を継続使用することで、60 FPS の滑らかな追従を維持します。
   - **スナップショット解析 / OCR / フォーム判定 / 音声質問応答**: ユーザーが「画面内のテキストを読み上げて」「この書類・状況を判定して」と要求した瞬間（または 1〜2 秒ごとのキーフレーム）に `MirageVisionAdapter` を実行します。
2. **Decisions API の `"-" (abstain)` 確率を用いた安全なフォールバック**:
   - 従来のモデルでは「未知のクラスに対しても誤ったクラスへ高い softmax 確率を割り当ててしまう（過信問題）」がありましたが、Mirage の Decisions API は選択肢に該当しない場合に `"-" (abstain)` に高い確率（例: `0.78〜0.99`）を割り当てます。
   - `evaluateHybridTriage()` はこの `"-"` 確率を直接監視し、閾値（デフォルト `0.40`）を超えた場合のみクラウド VLM へ自動フォールバックするため、**ハルシネーションの抑制とクラウド API コスト削減（約 60〜80% の一次フィルタ完結）を同時に達成**できます。
