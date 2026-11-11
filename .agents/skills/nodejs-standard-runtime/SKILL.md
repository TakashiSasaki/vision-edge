---
name: nodejs-standard-runtime
description: 本リポジトリの開発・依存関係管理・ビルド・テストにおいて、作業環境標準の Node.js (node, npm) のみを使用し、Bun の使用を厳格に禁止する運用指示。
---

# Node.js 標準開発環境の利用方針（Bun使用の完全禁止）

本アプリケーションの開発・検証・運用作業においては、作業環境に最初から含まれている標準の **Node.js 開発環境（`node`, `npm`）** のみを使用します。

## 厳格な運用ルール

1. **Bun の実行禁止**:
   - `bun`, `bunx`, `bun install`, `bun add`, `bun run`, `bun test` 等のコマンドは一切実行してはなりません。
   - パッケージのインストール、ビルド、テスト、スクリプト実行には必ず `npm` または `node` を使用してください。
2. **ロックファイルの単一性**:
   - プロジェクトの正本ロックファイルは **`package-lock.json`** のみです。
   - `bun.lock` や `bun.lockb` 等の競合ロックファイルが生成された場合は、即座に削除し、`package-lock.json` と整合させてください。
3. **ビルド・デプロイへの混入防止**:
   - `.gitignore`, `.gcloudignore`, `.dockerignore` にて `bun.lock` および `bun.lockb` を恒久的に除外対象としています。
   - Cloud Run デプロイ時や CI パイプラインに Bun 関連ファイルを含めないよう維持してください。
