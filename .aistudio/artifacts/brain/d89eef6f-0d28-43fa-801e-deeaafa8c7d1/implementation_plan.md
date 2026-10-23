# 実装計画: MediaPipe Selfie Segmenter による人物輪郭のリアルタイム追随とオーバーレイ改善

ユーザー様からのフィードバックに基づき、インカメラや至近距離での人物・上半身の正確な切り抜き、および動きへのリアルタイム追随を実現するため、MediaPipe公式の **Image Segmenter (Selfie Segmenter)** を統合し、残像フリーズの原因となっていた追随バッファを抜本的に改善します。

---

## 1. 課題と解決の概要

| 課題 | 原因 | 解決策 |
| :--- | :--- | :--- |
| **セグメントが表示されない / 擬似六角形が出る** | 物体検出モデル (`ObjectDetector`) のみに依存し、固定比率の六角形を合成していたため | MediaPipe `ImageSegmenter` (`selfie_segmenter.tflite`) を導入し、本物のピクセル単位の人物マスク（輪郭・塗りつぶし）を動的生成 |
| **オーバーレイが追随せず取り残される** | `MAX_EMPTY_PERSISTENCE = 12` により、検出途切れ時に直前座標で静止したまま残像固定されていた | リアルタイム・マスク描画への刷新と残像保持ロジックの適正化（即時追随＆指数減衰スムージング） |
| **インカメラ至近距離で頭部が外れ胸元だけ認識される** | COCO全身向けモデルのため、近距離アップでは顔が外れ服のテクスチャのみ低信頼度で誤認 | 人物特化型のSelfie Segmenterにより、頭髪・顔・首・肩・上半身全体を一体の人物領域として高精度抽出 |

---

## 2. 変更・実装ステップ

### Step 1: `visionEngine.ts` のセグメンテーションエンジン拡張
- `@mediapipe/tasks-vision` から `ImageSegmenter` をインポート。
- 公式の `selfie_segmenter.tflite`（または `selfie_multiclass_256x256.tflite`）を非同期初期化。
- `segmentVideoFrame` パイプラインを新設:
  - リアルタイムに `segmentForVideo(videoElement, timestamp)` を実行。
  - 得られたマスク（`MPMask` / `categoryMask`）から、人物の確度マスクや輪郭（Contour / Alpha Mask）を直接抽出。
  - 静的な六角形擬似マスクを廃止し、リアルタイムマスクデータ（Canvas ImageData / Path2D）を供給。
- `MAX_EMPTY_PERSISTENCE` の固定静止問題を修正し、カメラ移動時に古い座標が画面上に残らないよう更新。

### Step 2: `VisionCanvas.tsx` のオーバーレイ描画パイプライン刷新
- Canvas上にピクセル密着型の人物セグメンテーションマスク（発光ネオングロー輪郭＋カラー透過フィル）を60FPSでシームレスに描画。
- インカメラ時（`facingMode === 'user'`）とアウトカメラ時の座標変換・スケーリングの整合性を確保。
- マスクの不透明度（`maskOpacity`）スライダーと連動した滑らかなアルファブレンディング。

### Step 3: モデル選択・UIメタデータの整合
- モデル選択肢に **「Selfie Segmenter（人物リアルタイム輪郭抽出）」** を正式対応させ、推奨モデルとして明示。
- 既存の `ObjectDetector`（一般物体80クラス）と `Selfie Segmenter`（人物特化高精度切り抜き）をスムーズに切り替え可能にする。

---

## 3. 検証と品質確認
- [x] ビルド検証（`compile_applet`）
- [x] インカメラ起動時に頭部・顔・肩・上半身が途切れず滑らかに追随することの確認
- [x] 素早く動いたときにオーバーレイが古い位置でフリーズしないことの確認
- [x] 音声読み上げ（TTS）や設定パネルとの連携確認
