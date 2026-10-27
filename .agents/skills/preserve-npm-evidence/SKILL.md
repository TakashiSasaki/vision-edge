---
name: preserve-npm-evidence
description: npmロック・ランタイム契約や保護済み検証ファイルの欠落、不整合、外部エディタからの同期を調査・復旧するときに使う。最新機能を保持して原因を切り分け、原本ハッシュと現在の依存定義に基づいて復旧する。
---

# npm契約と検証原本を保持する

[AGENTS.md](../../../AGENTS.md) と
[同期手順](../../../scripts/testing/ai-studio-sync.md) を読み、作業範囲を確認する。
このスキル自体は変更・取得・マージの許可を与えない。

1. 現在のbranch/main、作業HEAD/tree、dirty状態、並行作業を記録する。専用branch/worktreeを使い、最新mainとの差分を読む。外部エディタの動作原因を観測だけで断定しない。
2. 正しい値を [.nvmrc](../../../.nvmrc)、[package.json](../../../package.json)、[preflight](../../../scripts/testing/preflight.mjs) から読む。版・件数・依存グラフを古い報告から固定値としてコピーしない。
3. キャッシュ／install前に `npm run --silent preflight` を実行する。失敗の完全な構造化出力を保存し、lock欠落、不正JSON、root定義不一致、競合lock、Node/npm不一致、証拠欠落／改変を別々に扱う。セットアップ失敗や未実行を、preflightの意図した失敗に数えない。
4. 復旧範囲をファイル単位に限定する。現行のsource・UI・依存宣言を残し、コミット全体のrevertや古いツリーへの置換を避ける。依存宣言が変わっていれば古いlockの丸写しでは直らない。
5. [保護台帳](../../../scripts/testing/protected-evidence.json) の各path/SHA256を読み、既存ファイルも照合する。欠落分だけを履歴から取り出し、書き込む前にハッシュ一致を確認する。原本を発見できなければ未解決として報告し、再生成・台帳削除・別画像で穴埋めしない。ignoredログの追加は照合したpathだけを明示する。
6. 互換な履歴lockをseedにできるか現行packageと比較し、必要なnpm指定とlock rootをcanonical npmで整合させる。宣言外の依存追加や無関係な更新を避け、resolved URL・全依存グラフを検証する。版変更が必要ならAGENTSのコマンドでpackage/lockを同期し、タグを自動生成しない。
7. `npm run --silent preflight`、`npm run --silent test:preflight`、`npm run --silent ci:install` の順で確認する。isolated missing-lock検証は [専用control](../../../scripts/testing/missing-lock-control.mjs) を使い、欠落以外も壊れたcheckoutを成功負例にしない。通常のcheckoutからlockを消して試さない。
8. 前提が揃ったら `npm run --silent verify:offline` を実行する。アプリ起動やモデルprefetchを契約検証の代用にしない。previewが必要なら [AGENTSのモデル不要検証](../../../AGENTS.md#model-free-verification) の環境条件を先に確認する。検証後も台帳の全ハッシュを照合し、push前にbranch/mainと他者の未保存変更を再確認する。予期しない更新では影響する書込を止め、依頼内で安全に調整するか範囲を確認して新候補を再検証する。必要な生成物比較では同じ依存を使い、意図した版表示の差を明示する。

完了報告にはbase/HEAD/tree、復旧元と照合結果、保持した最新変更、lock変更の理由、実行コマンド・exit・件数・未実行を含める。
ログの保管・絞込は [reduce-test-output](../reduce-test-output/SKILL.md) に従う。
