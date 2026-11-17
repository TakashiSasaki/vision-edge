# ステレオカメラ深度推定ビジョンモデル調査レポート

Side-by-Side（左右分割）形式のステレオカメラ映像をブラウザ（`onnxruntime-web`: WebGPU / WebAssembly）で処理し、視差マップおよび深度情報を取り出すための主要ビジョンモデル調査結果と選定方針をまとめます。

---

## 1. ステレオ深度推定の基本原理と Side-by-Side 入力

単眼深度推定（Depth Anything V2 や MiDaS 等）が「見え方から相対的な遠近感を推定する」のに対し、ステレオマッチング（Stereo Matching）は左右2つのカメラ映像間の**水平方向の視差 $d$（ピクセル単位のズレ量）**を探索し、三角測量の原理で実距離 $Z$ を算出します。

$$Z = \frac{f \cdot B}{d}$$

- **$Z$**: カメラから対象点までの奥行き距離 [m]
- **$f$**: カメラの焦点距離 [px]
- **$B$**: 左右カメラ間の基線長（Baseline） [m]
- **$d$**: 推定視差（Disparity） [px]

### Side-by-Side（SBS）入力時の前処理要件

1. **左右分割（クロップ）と `3840×1080` カメラ指定**: 1本の UVC ステレオカメラ映像（VisionStudio AI のライブカメラ取得では `width: { exact: 3840 }, height: { exact: 1080 }` を明示し、複数カメラ認識時も 3840×1080 対応デバイスを自動選択）を $x \in [0, W/2)$ と $x \in [W/2, W)$ に分割し、左画像 $I_L$・右画像 $I_R$ を切り出します（ハードウェア配線や平行法/交差法の違いにより左右が逆の場合はスワップ設定で補正します）。
2. **平行化（Rectification）の前提**: 多くの学習済みステレオモデルはエピポーラ線が水平に揃った平行化済み画像ペアを前提としています。一般的な一体型 UVC ステレオカメラは工場校正済みまたはほぼ平行配置ですが、大きなレンズ歪みや上下ズレがある場合は精度が低下します。
3. **アスペクト比と視差スケーリング**: 入力解像度をモデルの固定入力サイズ（幅 $W_m$）へリサイズして推論する場合、視差値 $d_m$ は入力空間のピクセル単位となるため、元画像解像度 $W_{\text{eye}}$ 基準へ換算するには $d = d_m \cdot (W_{\text{eye}} / W_m)$ のスケーリングが必要です。

---

## 2. 主要ステレオビジョンモデルの調査・比較

| モデル名 | 主な開発元・出典 | アーキテクチャ特徴 | 代表入力解像度 | ライセンス・商用利用 | ONNX 入手性 | ブラウザ (ORT WebGPU / WASM) 適性 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LightStereo (S / M / L)** | OpenStereo (CASIA ほか) | 4Dコストボリュームや `Conv3D` を使わず、**3Dコストボリューム＋2D畳み込み**と MSCA で高速集約 | 256×512 / 384×640 ほか | MIT / Apache-2.0 系（OpenStereo 準拠） | OpenStereo 公式 PyTorch から ONNX 出力容易 | **○ WASM安定構成** — 48入力 `Concat`（49バッファ）が WebGPU 上限 8 を超えるため WASM EP で安定動作 |
| **MobileStereoNet (2D / 3D)** | Tübingen 大学 | MobileNet 系バックボーンを用いたエッジ向け超軽量ステレオ網（2D版・3D版） | 240×528 / 256×512 | MIT License (コード・変換ラッパー含む) | コミュニティ変換 ONNX (`ONNX-MobileStereoNet` 等) | **○ 有望（2D版推奨）** — 2D版は省メモリで軽量。3D版は `Conv3D` の負荷に注意 |
| **HITNet** | Google Research | 階層的タイル反復洗練（パッチの平面傾きをマルチスケール更新） | 480×640 / 320×640（固定） | Apache-2.0（原実装/変換元により確認要） | PINTO_model_zoo / `ONNX-HITNET` に変換済み ONNX あり | **○ 定番エッジ枠** — 固定解像度・一部タイリング/リシェイプ演算のバックエンド検証が必要 |
| **CoEx** | KAIST | MobileNetV2 特徴＋Correlate-and-Excite によるガイド付きコスト集約 | 256×512 / 384×640 | OSS（BSD/MIT 系・元リポジトリ要確認） | PINTO_model_zoo / `ONNX-CoEx` に変換済み ONNX あり | **△〜○** — 一部 3D 集約を含むため WebGPU シェーダ互換性に留意（WASM フォールバック併用） |
| **Fast-ACVNet / Plus** | 北航ほか | Attention Concatenation Volume による軽量・高精度コスト集約 | 320×640 / 480×640 | MIT License | 公式・コミュニティ変換 ONNX あり | **△〜○** — 精度バランス良好だがコストボリュームメモリと `Conv3D` 負荷がやや高い |
| **CREStereo** | Megvii Research | カスケード再帰型ネットワーク＋適応的局所相関（`init` / `combined`、反復回数可変） | 240×320 / 320×480 / 480×640 | **Apache-2.0** | PINTO_model_zoo / `ONNX-CREStereo` で多解像度 ONNX 流通 | **○ 高精度枠（`iter2` 推奨）** — 細部・低テクスチャ領域に強い。`GridSample` を多用し固定解像度 |
| **RAFT-Stereo** | Princeton 大学 | 多段 GRU による反復相関更新（汎用性が高い定番高精度モデル） | 256×640 / 384×1280 | BSD-3-Clause | Hugging Face (`AXERA-TECH/RAFT-stereo` 等) | **△ 重め** — 反復回数 `r1` 等なら試験可能だが計算量が多くリアルタイム用途では低 FPS になりやすい |
| **IGEV / IGEV++** | 華中科技大学ほか | Geometry Encoding Volume ＋ 反復更新（ベンチマーク上位精度） | 320×736 / 480×640 | BSD-3-Clause 系（学術・商用要確認） | 一部変換スクリプトあり | **× ブラウザ低適性** — 3D 演算と多段反復によりブラウザ上では高遅延・高メモリ |
| **Fast-FoundationStereo / FoundationStereo** | NVIDIA (NVlabs) | 大規模合成データ＋単眼基盤特徴を蒸留したゼロショット基盤モデル | 320×736 / 576×960 | NVIDIA Open Model License（商用可） | Hugging Face (`Miayan/stereo-matching-weights` 等) | **× ブラウザ不向き** — 347MB〜900MB超の大容量であり、サーバー/ネイティブ GPU 向け |
| **DEFOM-Stereo / StereoAnywhere** | Insta360・CUHK / Bologna 大学 | Depth Anything V2 (ViT-L 等) を組み込んだゼロショット特化モデル | 480×640 以上 | コード MIT / **重み CC-BY-NC-4.0 制約** | PyTorch 重み中心 | **× 対象外** — 数 GB 級サイズと非商用重み制約のため Web 組み込み不適 |

---

## 3. ブラウザ (`onnxruntime-web`) 実行時の技術的留意点

1. **WebGPU 演算子互換性**:
   - `GridSample`: `onnxruntime-web` の WebGPU EP でサポートされていますが（Opset 16〜19）、モデル出力時の特定モードや 3D 変形を含む場合にエラーとなるケースがあるため、WebGPU 初期化または推論実行時（`Failed to run JSEP kernel` 等）の失敗時は自動的に WASM バックエンドへフォールバックする構成を採用しています。
   - `Conv3D`（5次元重み `[M, C, kD, kH, kW]`）: `onnxruntime-web` の WebGPU JSEP `Conv` カーネルは 3D 畳み込みで実行時エラーとなるため、ロード時のグラフ検査で 5D `Conv` / `ConvTranspose` ノードを検出した場合は自動的に WASM EP へルーティングします（`MobileStereoNet-2D` の `model_528_240_float32.onnx` もコストボリューム集約に 5D `Conv` を含むため WASM 安定構成として実行します）。
2. **コストボリュームの解像度スケーリング**:
   - ステレオモデルは通常「高さ × 幅 × 最大視差」の 3D/4D 構造を内部生成するため、入力解像度を 2 倍にするとメモリ使用量と演算量が 4〜8 倍近く増加します。ブラウザでは `240×320` 〜 `256×512`（最大 `480×640`）クラスに抑えることがリアルタイム性の鍵となります。
3. **モデル入手経路（オンライン自動取得＋ローカル ONNX 読み込み）と WebGPU 制約の自動判定**:
   - **LightStereo-S** (`256×512`) と **MobileStereoNet-2D** (`240×528`, `model_528_240_float32.onnx`, 約 16.5MB) はブラウザ直接 CORS 取得に対応しており、jsDelivr CDN / GitHub Raw（LightStereo-S）および Google Drive Direct CORS URL（MobileStereoNet-2D）から自動ダウンロードして実行できます（`LightStereo-S` は 48ch 連結のため、`MobileStereoNet-2D` は 5D `Conv` 集約を含むため、いずれも自動的に WASM EP へルーティングされます）。
   - **HITNet SceneFlow** (`240×320`, `hitnet_sceneflow_240x320.onnx`, 約 8.4MB) および **CREStereo Init Iter2** (`180×320`, 約 21.5MB)・**CREStereo Combined Iter5** (`240×320`, 約 21.7MB) は GitHub Releases（`nburrus/stereodemo` がミラーする PINTO0309 変換 ONNX）から**アプリ内コンテナキャッシュプロキシ (`/api/model-cache/stereo/...`) 経由でオンライン自動取得**します（CREStereo 系は 37 入力 `Concat` ノードを含むため自動的に WASM EP へルーティングされます）。
   - すべての ONNX バリアントはオンデマンド取得に加え、ユーザーが用意したローカル `.onnx` ファイルでの上書き読み込みにも対応しています（`Custom Stereo ONNX` を含む）。

---

## 4. VisionStudio AI への統合構成

- **比較可能なバリアント構成**:
  1. `SAD 幾何マッチング (即時比較基準 · モデル不要)`: ブラウザ内 TypeScript で実装したスライディングウィンドウ SAD 視差推定。外部ファイルなしで動作。
  2. `LightStereo-S` (`256×512`): 軽量・高速 2D 畳み込み系プロファイル（**ブラウザ直接オンライン自動取得対応 · WASM 安定構成**＋ローカル `.onnx` 上書き可能）。
  3. `MobileStereoNet-2D` (`240×528`): 軽量モバイルステレオプロファイル（**ブラウザ直接オンライン自動取得対応 · WASM 安定構成**＋ローカル `.onnx` 上書き可能）。
  4. `HITNet` (`240×320`): 階層タイル反復プロファイル（6ch 連結入力、`[1, H, W, 1]` NHWC 出力自動変換、**コンテナキャッシュ経由オンライン自動取得対応**＋ローカル `.onnx` 上書き可能）。
  5. `CREStereo (init_iter2 180×320 / combined_iter5 240×320)`: 高精度・細部保持プロファイル（`[1, 2, H, W]` フロー出力から水平視差を自動抽出、37ch `Concat` 自動 WASM ルーティング、**コンテナキャッシュ経由オンライン自動取得対応**＋ローカル `.onnx` 上書き可能）。
  6. `汎用ステレオ ONNX (自動シェイプ検出)`: 任意の 2 入力 (`left`, `right`) または 6 チャネル結合入力のステレオ ONNX モデルを自動認識して実行（ストレージバッファ上限・5D `Conv` 自動判定＋JSEP 実行時 WASM 自動フォールバック付き）。
- **可視化と任意座標プローブ**:
  - Turbo / Viridis / Inferno / Grayscale のカラーマップ重畳と最小・最大視差凡例。
  - `左基準＋深度重畳` / `L・R・深度 3分割` / `深度単体` の表示モード切替。
  - ホバーによるリアルタイムクロスヘア座標・視差 (px)・推定距離 (m) 読み出しと、クリックによるピン固定およびモデル間比較記録。
