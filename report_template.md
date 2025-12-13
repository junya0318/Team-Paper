# 期末レポート雛形（このまま埋めてOK）

## 1. 対象コード（出典）
- 対象: DA-MUSIC / DR-MUSIC_ICASSP23（Deep Root-MUSIC; DoA推定）
- 参照URL: （GitHub）
- 何をするコードか（2〜4行）：
  - Root-MUSICを深層学習で拡張し、コヒーレント源・低SNR・少スナップショットで劣化する課題への対処を狙う。

## 2. 改善方針（本講義で学んだことの適用）
### 2.1 パッケージ管理（uv / pyproject.toml）
- 目的: 再現性・依存関係の明示・実行手順の簡易化
- 実施:
  - pyproject.toml (PEP 621) を追加
  - `uv pip install -e ".[dev]"` で開発依存も含めて構築

### 2.2 型ヒント・静的解析（typing / ruff）
- 目的: インタフェースを明確化し、改修時の事故を減らす
- 実施:
  - 主要関数に型ヒント・docstring を付与
  - ruff によるlint（今回は設定のみ）

### 2.3 pytest による単体テスト
- 目的: 数値処理（対角和/根計算）が壊れていないことを自動で確認
- 実施:
  - `sum_of_diagonals` が NumPy のtraceと一致するテスト
  - `companion_eig_roots` が NumPy roots と一致するテスト

### 2.4 再現性（Seed固定）
- 目的: 実験結果の再現を容易にする
- 実施:
  - Python/NumPy/PyTorch(CUDA含む) のseedをまとめて設定する `set_seed`

## 3. 実装内容（差分の説明）
- 変更点サマリ（箇条書き）
- 主要ファイル:
  - src/dr_music/utils.py
  - src/dr_music/model.py
  - src/dr_music/seed.py
  - tests/

## 4. 実行方法
- 環境構築
- テスト実行
- デモ実行

## 5. 考察（何が良くなったか）
- 再現性
- 保守性（型・テスト）
- 拡張性（srcレイアウト）

## 6. 今後の改善
- 元リポジトリ全体をsrcレイアウトへ移行（相対import化）
- CLIの引数を増やし、config(YAML)化
- CI（GitHub Actions）でpytestを自動化
