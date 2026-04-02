"""
Sarashina2.2-OCR 性能確認デモ
- 左: 元画像
- 右: バウンディングボックス付き画像（図・グラフ等の検出位置）
- 下: 検出テキスト（Markdown 形式）

起動方法:
  uv run app.py
"""

import os

# MPS (Mac GPU) のメモリ上限を無効化して OOM を防ぐ
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
# MPS で未対応のオペレーション（Conv3Dなど）を CPU にフォールバックさせる
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


import re
import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM, AutoProcessor, set_seed

# ---------------------------------------------------------------------------
# モデル設定
# ---------------------------------------------------------------------------
MODEL_PATH = "sbintuitions/sarashina2.2-ocr"

import platform

print("[INFO] デバイス検出中...")
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available() and platform.machine() == "arm64":
    # MPS は Apple Silicon (M1/M2/M3/M4) でのみ使用
    # Intel Mac の MPS は Conv3D 非対応など制約が多いため CPU にフォールバック
    device = "mps"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32
print(f"[INFO] 使用デバイス: {device}  dtype: {dtype}")

print(f"[INFO] プロセッサ読み込み中: {MODEL_PATH}")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

print(f"[INFO] モデル読み込み中（初回は数分かかります）: {MODEL_PATH}")
if device == "cuda":
    # CUDA は device_map="cuda" で accelerate によるマルチ GPU 対応
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cuda",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
elif device == "mps":
    # MPS (Apple Silicon) は一旦 CPU でロードしてから転送
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cpu",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = model.to(dtype=dtype).to(device)
else:
    # CPU（Intel Mac 含む）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cpu",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
model.eval()
print("[INFO] モデル準備完了")

# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------
BBOX_PATTERN = re.compile(
    r"<bbox>\[\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)\]</bbox>"
)


def parse_bboxes(text: str) -> list[tuple[int, int, int, int]]:
    """
    モデル出力から <bbox>[(x1, y1), (x2, y2)]</bbox> を抽出する。
    座標は 0–1000 の正規化整数。
    """
    return [
        (int(x1), int(y1), int(x2), int(y2))
        for x1, y1, x2, y2 in BBOX_PATTERN.findall(text)
    ]


def draw_bboxes(image: Image.Image, bboxes: list[tuple[int, int, int, int]]) -> Image.Image:
    """
    正規化座標 (0–1000) のバウンディングボックスを実ピクセル座標に変換して描画する。
    """
    img = image.copy()
    if not bboxes:
        return img

    draw = ImageDraw.Draw(img)
    w, h = img.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    colors = ["#FF4444", "#44AAFF", "#44FF88", "#FFAA44", "#FF44FF"]

    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        px1 = int(x1 / 1000 * w)
        py1 = int(y1 / 1000 * h)
        px2 = int(x2 / 1000 * w)
        py2 = int(y2 / 1000 * h)
        color = colors[i % len(colors)]

        # ボックス描画（太さ 3px）
        draw.rectangle([px1, py1, px2, py2], outline=color, width=3)

        # ラベル背景
        label = f"図{i + 1}"
        bbox_text = draw.textbbox((px1, py1 - 22), label, font=font)
        draw.rectangle(bbox_text, fill=color)
        draw.text((px1, py1 - 22), label, fill="white", font=font)

    return img


def format_output_text(raw_text: str) -> str:
    """検出テキストを表示用に整形する（bbox タグを除去して Markdown として返す）。"""
    cleaned = BBOX_PATTERN.sub("[図 検出]", raw_text)
    return cleaned


# ---------------------------------------------------------------------------
# 推論関数
# ---------------------------------------------------------------------------
def run_ocr(image):
    if image is None:
        gr.Warning("画像をアップロードしてください。")
        return None, None, ""

    pil_image = Image.fromarray(image).convert("RGB")

    set_seed(42)
    message = [
        {
            "role": "user",
            "content": [{"type": "image", "image": pil_image}],
        }
    ]

    raw_inputs = processor.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    # MPS / CUDA / CPU いずれも対応: Tensor のみデバイス転送
    inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in raw_inputs.items()
    }

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=6000,
            temperature=0.0,
            top_p=0.95,
            repetition_penalty=1.2,
            use_cache=True,
        )

    # 入力トークンを除いた生成部分だけをデコード
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    raw_text = processor.decode(generated_ids, skip_special_tokens=True)

    # バウンディングボックスの描画
    bboxes = parse_bboxes(raw_text)
    bbox_count = len(bboxes)
    img_with_boxes = draw_bboxes(pil_image, bboxes)

    # テキスト整形
    display_text = format_output_text(raw_text)
    summary = f"**検出された図・グラフ数:** {bbox_count} 件\n\n---\n\n{display_text}"

    return pil_image, img_with_boxes, summary


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
CSS = """
#title { text-align: center; }
#upload-col { max-width: 600px; margin: 0 auto; }
.result-image img { object-fit: contain; max-height: 500px; }
#text-output { font-size: 14px; line-height: 1.7; }
"""

with gr.Blocks(title="Sarashina2.2-OCR デモ", css=CSS, theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 📄 Sarashina2.2-OCR 性能確認デモ
        **SB Intuitions** が開発した日本語・英語ドキュメント向け OCR モデルのテストツールです。
        画像をアップロードして「OCR 実行」を押してください。
        > ⚠️ モデルは 3B パラメータ（BF16 約 6GB VRAM）です。GPU 環境を推奨します。
        """,
        elem_id="title",
    )

    with gr.Row():
        with gr.Column(scale=3):
            image_input = gr.Image(
                label="📁 画像をアップロード",
                type="numpy",
                sources=["upload", "clipboard"],
                height=320,
            )
        with gr.Column(scale=1, min_width=160):
            run_btn = gr.Button("▶ OCR 実行", variant="primary", size="lg")
            gr.Markdown(
                """
                **出力の見方**
                - 左: 元画像
                - 右: 図・グラフの検出位置（バウンディングボックス）
                - 下: 抽出テキスト（Markdown）
                """
            )

    gr.Markdown("### 🔍 比較ビュー")

    with gr.Row(equal_height=True):
        orig_out = gr.Image(
            label="元画像",
            interactive=False,
            elem_classes=["result-image"],
        )
        bbox_out = gr.Image(
            label="バウンディングボックス付き",
            interactive=False,
            elem_classes=["result-image"],
        )

    gr.Markdown("### 📝 検出テキスト")
    text_out = gr.Markdown(
        value="*OCR 実行後にここに結果が表示されます*",
        elem_id="text-output",
    )

    # サンプル（HuggingFace のサンプル画像 URL）
    gr.Examples(
        examples=[
            ["https://huggingface.co/sbintuitions/sarashina2.2-ocr/resolve/main/assets/sample1.jpeg"],
        ],
        inputs=[image_input],
        label="サンプル画像（クリックで読み込み）",
        cache_examples=False,
    )

    run_btn.click(
        fn=run_ocr,
        inputs=[image_input],
        outputs=[orig_out, bbox_out, text_out],
        api_name="ocr",
    )

    gr.Markdown(
        """
        ---
        **モデル:** [sbintuitions/sarashina2.2-ocr](https://huggingface.co/sbintuitions/sarashina2.2-ocr) |
        **ライセンス:** MIT |
        バウンディングボックスは図・グラフ・画像などの視覚的要素の検出位置を示します。
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )