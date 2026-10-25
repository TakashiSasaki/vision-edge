---
name: reduce-test-output
description: テスト・オフライン CI・検証ログが長く、コーディングエージェントの入力トークンを浪費するときに使う。生ログと終了状態を保持したまま成功出力を要約し、失敗診断だけを段階的に読む。reporter、ラッパー、CI artifact、GitHub の読取出力を改善する作業にも使う。
---

# Reduce test output

1. 実行前に対象 HEAD/tree、dirty 状態、既存規約を確認。古い証拠を新結果として使わない。
2. 出力を先に測る。代表的な成功・失敗・BLOCKED の bytes/lines と発生源を記録する。bytes 減少率を実トークン数や実行速度の改善と呼ばない。
3. このリポジトリでは `npm run --silent verify:offline` を優先。依存関係と利用可能なブラウザは事前に用意する。これ自体は導入・モデル取得を行わない。対象が狭いときは `npm test` / `npm run test:foundation` / `npm run test:output` / `npm run test:e2e` を選ぶ。
4. 成功時は task、exit、件数（pass/fail/skip/cancelled/todo 等）、artifact の場所だけ表示する。生 stdout/stderr と reporter の JSON/HTML/trace は一意の ignored 出力先へ保存する。失敗時だけ短い原因と上限付き抜粋を表示する。
5. 続報は `npm run --silent test:logs -- <printed-artifact-directory> [literal-search]`。まず要約、次に該当 task/失敗 test、必要な場合だけ raw artifact。失敗を診断するために既に完了した全テストを再実行しない。
6. GitHub/MCP の大きな返答はツール層で保存し、HEAD、状態、件数、URL、必要な行だけをモデルへ返す。PR 本文・レビュー履歴・CI 全ログ・hash 一覧を毎回展開しない。対象 SHA と checkout 行の照合は省かない。
7. 次を実測で守る：非ゼロ exit を伝搬、レポート欠落/破損/矛盾を緑にしない、exit2 はその意味を定義した asset/real-SDK task のみ BLOCKED、skip を成功件数に加えない。起動障害・timeout・signal・log 上限到達は明示。保存前に `>/dev/null`、`|| true`、終了コード未確認の pipe、warning 全消去を使わない。
8. reporter/表示/保存先を改善しても assertion、待機時間、retry、検証対象、GPU/model/input を変更しない。期待される故障 control の子 exit1 と全体失敗を区別する。HTTP503 診断・checker-only データを実推論成功にしない。拒否済み download は再試行・迂回しない。
9. 完全ログは CI artifact に含める。通常の試験は `testInfo.outputPath` へ保存し、歴史的 screenshot/証拠を上書きしない。大量 raw データの新規コミットも避ける。
10. ログラッパー変更は最低限、成功・非ゼロ・欠損/不正 report・大量出力・signal/timeout・保存先衝突を試す。件数・終了状態・診断到達性を旧動作と比較し、測定条件と未実行を明記する。

個別 reporter や完全なストリームを調べるときは既存の直接 CLI、`npm run test:raw`、または `TEST_OUTPUT=verbose` を明示的に使う。既定の入力へ巨大な JSON を戻さない。絞り込み実行は全件検証と区別する。Node は filter で除外したケースを skip として数えない場合があるため、存在しない skip 件数を補わない。
