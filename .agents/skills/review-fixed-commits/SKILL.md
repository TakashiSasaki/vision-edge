---
name: review-fixed-commits
description: PRの独立レビュー、指摘修正後の再レビュー、stacked PRの累積統合を行うときに使う。固定したHEAD・tree・baseと実行証拠を対応させ、レビュー中のmain更新や古いCIの流用を検出する。
---

# 固定コミットでレビューと統合を進める

[AGENTS.md](../../../AGENTS.md) と依頼範囲を確認する。
このスキルはレビュー公開・マージ・デプロイ等の権限を追加しない。
ログ出力の扱いは [reduce-test-output](../reduce-test-output/SKILL.md) を参照する。

1. 実装担当と独立レビュー担当を区別し、既存のレビュー／実行中作業を確認する。進捗が見えないだけで他者がidleとみなさず、既存依頼の進行を読まずに重複タスクや重複投稿を作らない。
2. 実refs、PRのhead/base、HEAD/tree、merge-base、対象差分、コメント・レビュー・CIを取得して対象を固定する。GitHubのPR表示値と現時点のmain branch refは別々に確認する。ソース未取得や読取失敗なら、古い値で最新確認済みと言わない。
3. 固定差分と関連source/test/configを読み、再現可能な指摘ID・重大度・実行手順・期待／実測を残す。実行済み、FAIL、BLOCKED、NOT RUNを区別し、mockや意図したHTTP故障を実モデル成功と扱わない。修正が必要なら別担当へ返し、新HEADを改めてレビューする。
4. [CI定義](../../../.github/workflows/ci.yml) と [実行コマンド](../../../package.json) を読み、run metadata、raw checkout SHA、実際のstep・exit・件数を固定HEADと照合する。成功ラベルだけで判断せず、skip、continue-on-error、起動障害、成果物公開の有無を確認する。artifact metadataだけでZIP内容を検証済みとしない。
5. 公開する判定を対象SHA・範囲・未検証事項・証拠リンクに結び付け、書込後は読戻す。同一アカウントのCOMMENTを別人の正式Approveと呼ばない。結果が不明な投稿は既存状態を確認してから扱い、二重投稿しない。
6. push前と統合準備では現在のbranch・未コミット変更とmain/headを再取得し、レビュー時点からの差分・競合・チェックを確認する。予期しない更新で範囲が変わったら影響する書込を止め、依頼内の安全な調整と範囲外の変更を区別する。mainの実質変更や新sourceを古いPASSで承認済みと扱わない。最新機能を保持して調整し、新候補・テスト・独立再レビューを揃える。他者の変更を上書きして元のtreeを再現しない。
7. stacked PRでは親子関係と累積treeを確認する。親単独では壊れ、累積候補だけが成立する場合は、main向けの累積差分を明示してレビュー・検証する。retarget後のbaseとCI実行contextを確認し、古いbaseのrunを新contextの検証に流用しない。競合解消や版／lock変更後は新HEADに固定し直す。
8. マージが依頼範囲なら、公開済みレビューの対象・範囲、最新base、必要チェック、mergeabilityを再確認し、期待HEADを指定した通常のmergeを使う。保護回避・force push・自動mergeの追加で条件を飛ばさない。結果が不明ならPR/mainを読んでから次へ進む。
9. 統合後はmainのcommit/tree・親子関係を実際の統合候補と照合し、そのmain SHAをcheckoutしたpush CIを終端まで確認する。PR CIを統合後CIと呼ばない。次のstacked PRはbase・累積差分・版・チェックを再確認する。source/evidence枝と過去FAILを保持し、失敗時は原因と未完了範囲を報告する。自動デプロイが観測された場合も手動操作と分けて記録する。

報告を固定対象、結論、実行証拠、残る前提に絞る。npm／証拠の復旧が必要なら
[preserve-npm-evidence](../preserve-npm-evidence/SKILL.md) を使い、製品修正と混同しない。
