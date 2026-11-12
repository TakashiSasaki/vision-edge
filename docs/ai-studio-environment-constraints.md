# Google AI Studio Build Runtime Environment & Operational Constraints

本書は、Google AI Studio Build（および類似のコンテナ化エージェント作業環境）における特有の制約、注意点、および他エージェント（OpenAI Codex 等）との共同作業時の共有プロトコルを記録したものです。

---

## 1. ランタイム環境・パッケージマネージャー制約

1. **Node.js / npm の厳格な固定 (Standard Runtime Policy)**
   - 本リポジトリの標準・正準ランタイムは **Node.js `v24.22.1`** および **npm `11.19.1`** です。
   - コンテナ初期状態では古い Node/npm や競合ツールが含まれている場合があるため、`preflight` による契約検査（`NODE_CONTRACT`, `NPM_CONTRACT`）が実行されます。
   - **`bun` / `bunx` の完全禁止**:
     - `bun` はビルド、テスト、依存関係解決、実行に一切使用しません。
     - 一部の自動化タスクや環境初期化が `bun.lock` を副次的に再生成することがあります。`preflight` で `CONFLICTING_LOCK` 違反（exit 1）が発生した場合は、直ちに `rm -f bun.lock` で削除してください。
     - 単一の信頼できるロックファイルは `package-lock.json`（331 entries、`fsevents` 等の全プラットフォーム任意依存を含む）です。

2. **バージョン管理規則 (One Authoritative Version & Monotonicity)**
   - アプリケーションの正準バージョンはルート `package.json` の `version` のみです。
   - `metadata.json` やマニフェスト、コード内に独立したバージョン番号をハードコードしてはなりません。
   - エージェントによる機能・コード変更は、変更セットごとに **PATCH バンプ**（`npm version patch --no-git-tag-version`）が必須です。
   - 人間の明示的な指示がない限り、MINOR/MAJOR バンプを推論・適用してはなりません。また git タグを自動生成してはなりません。

---

## 2. Playwright ブラウザ実行環境の制約

1. **ヘッドレスシェルのバージョン一致性**
   - Playwright (v1.64.x) は `/ms-playwright/chromium_headless_shell-1254/chrome-headless-shell-linux64/chrome-headless-shell` を探索します。
   - コンテナ環境に旧バージョン（例: `chromium_headless_shell-1247`）のみが存在する場合、ブラウザ起動が `Executable doesn't exist` で失敗します。
   - 解決策: `/ms-playwright/chromium_headless_shell-1254/` に既存のヘッドレスシェルバイナリを同期配置することでオフライン検証（`test:e2e:controls` 等）を通過させます。

2. **WebGPU アダプターとフォールバックの識別**
   - ヘッドレス環境・コンテナ環境では物理 GPU アダプター（`navigator.gpu`）が存在しないか、SwiftShader 等のソフトウェアフォールバックアダプター（`isFallbackAdapter === true`）になる場合があります。
   - WebGPU の検証では、**「API の存在」「アダプターの取得」「セッション生成」「実推論カーネル実行」** を分離して判定し、silent WASM fallback（暗黙的な WASM へのフォールバック）と真の GPU 実行を混同しないように設計する必要があります。

---

## 3. テスト・検証およびモデルフリー実行規則

1. **`verify:offline` のバックグラウンド実行タイムアウト**
   - 全体テストスイート `npm run --silent verify:offline` は多数のタスク（preflight, unit, foundation, output, controls, offline-check）を包含するため、バックグラウンド実行時のタイムアウト（ツール側のキャンセル）に注意が必要です。
   - 問題の局所化や個別検証では、以下のサブタスクを個別・同期的に実行することを推奨します：
     - `npm run --silent preflight`
     - `npm run --silent test:preflight`
     - `npm run --silent typecheck`
     - `npm run --silent lint`
     - `npm test`
     - `npm run --silent test:foundation`
     - `npm run --silent test:output`
     - `npm run --silent test:e2e:controls`

2. **モデルダウンロードおよびプレビューサーバーの制約**
   - `CI` または `TEST_LOG_DIR` 環境変数が定義されている場合、プレビューサーバーは不要なモデル自動取得・プリフェッチを抑制します。
   - モデル非依存の E2E・CSS コントロールテスト（`test:e2e:controls`）では、HTTP 404 や未配信の API エンドポイントへの不要なリクエストが failure と判定されるため、テレメトリやログ送信は `__CONTAINER_MODEL_CACHE_ENABLED__` で厳格にガードされています。

3. **保護対象検証記録の不変性 (Protected Historical Evidence)**
   - `docs/verification/**` に含まれる 702 件の過去の検証記録・ログ・結果 JSON は `preflight` の SHA-256 改竄防止台帳に登録されています。
   - これらの過去ファイルを改変・削除してはならず、新しい検証結果はテスト実行ごとの `.cache/test-runs/` または明示的な新規レポートファイルに出力します。

---

## 4. 他エージェントとの協調・PR運用規則

- タスク着手時およびプッシュ／PR 作成前に、リモートの最新ブランチ、PR、未コミット変更を確認すること。
- 他エージェントの正当な変更を上書きせず、分離されたブランチ（`feat/*`）で作業を進めること。
- CI の 403 制限（モデルダウンロード禁止）をバイパスしようとしないこと。
