# Sarashina2.2-OCR 性能確認デモ

[sbintuitions/sarashina2.2-ocr](https://huggingface.co/sbintuitions/sarashina2.2-ocr) の動作確認用 Gradio アプリです。

## セットアップ

### 1. プロジェクト作成・依存関係インストール

```bash
# リポジトリに移動後
uv sync
```

### 2. CUDA 環境の場合（推奨）

`pyproject.toml` の `[tool.uv.sources]` と `[[tool.uv.index]]` のコメントを解除してから:

```bash
uv sync
```

または手動で CUDA 版 PyTorch を上書きインストール:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. 起動

```bash
uv run app.py
```

ブラウザで http://localhost:7860 が自動的に開きます。

## 必要スペック

| 項目 | 最低 | 推奨 |
|------|------|------|
| GPU VRAM | - | 8GB 以上（BF16） |
| RAM | 16GB | 32GB |
| Python | 3.10+ | 3.11+ |

> CPU のみの場合でも動作しますが、1枚の推論に数分〜数十分かかります。

## 画面構成

```
┌──────────────────────────────────────────┐
│ 画像アップロード エリア     │ OCR 実行ボタン │
├────────────────────┬─────────────────────┤
│    元画像          │  バウンディングボックス│
│                    │  付き画像            │
├────────────────────┴─────────────────────┤
│           検出テキスト (Markdown)         │
└──────────────────────────────────────────┘
```

## バウンディングボックスについて

モデルは図・グラフ・画像などの視覚的要素を検出し、  
`<bbox>[(x1, y1), (x2, y2)]</bbox>` 形式で座標を出力します（0–1000 の正規化整数）。  
右パネルではこれを実ピクセル座標に変換して色付きの矩形で描画します。