# src/demo_gradio.py
# -*- coding: utf-8 -*-
"""
Gradio Image Captioning Demo (BLIP-2 + optional LoRA)
"""

import os
os.environ["GRADIO_UI_LANGUAGE"] = "en"

import argparse
from typing import Optional
import gradio as gr
import torch
from PIL import Image
from transformers import (
    Blip2ForConditionalGeneration,
    BlipImageProcessor,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model


def freeze_all_params(model):
    for p in model.parameters():
        p.requires_grad = False


def inject_lora_for_blip2(model, r=16, alpha=32, dropout=0.05):
    """Inject LoRA into language_model (T5) + qformer.bert."""
    lcfg_lm = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type="SEQ_2_SEQ_LM",
        target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    )
    model.language_model = get_peft_model(model.language_model, lcfg_lm)

    if hasattr(model.qformer, "bert"):
        lcfg_qf = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            bias="none", task_type="FEATURE_EXTRACTION",
            target_modules=[
                "query", "key", "value",
                "dense", "output.dense", "intermediate.dense"
            ],
        )
        model.qformer.bert = get_peft_model(model.qformer.bert, lcfg_qf)
    return model


def try_load_lora_from_dir_or_file(model, ckpt_dir: Optional[str], ckpt_file: Optional[str]) -> bool:
    loaded = False
    if ckpt_dir and os.path.isdir(ckpt_dir):
        lm_path = os.path.join(ckpt_dir, "lora_lm")
        qf_path = os.path.join(ckpt_dir, "lora_qformer")
        try:
            if os.path.isdir(lm_path):
                model.language_model.load_adapter(lm_path)
                loaded = True
        except Exception:
            pass
        try:
            if hasattr(model.qformer, "bert") and os.path.isdir(qf_path):
                model.qformer.bert.load_adapter(qf_path)
                loaded = True
        except Exception:
            pass
        if loaded:
            print(f"[INFO] Loaded LoRA adapters from dir: {ckpt_dir}")
            return True

    if ckpt_file and os.path.isfile(ckpt_file):
        state = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded state_dict from file: {ckpt_file}")
        return True
    return loaded


class Captioner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(0)[0] >= 8
        )
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        print(f"[INFO] Using dtype={self.dtype}  device={self.device}")

        print(f"[INFO] Loading base model: {args.model_name}")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            args.model_name, torch_dtype=self.dtype
        )
        self.processor = BlipImageProcessor.from_pretrained(args.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        freeze_all_params(self.model)

        if not args.no_lora:
            self.model = inject_lora_for_blip2(
                self.model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
            )
            ok = try_load_lora_from_dir_or_file(self.model, args.ckpt_dir, args.ckpt_file)
            if not ok:
                print("[WARN] No LoRA weights found; using base BLIP-2.")

        self.model.eval().to(self.device)
        self.eos_id = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id or 1
        self.pad_id = self.tokenizer.pad_token_id or self.eos_id

    @torch.inference_mode()
    def caption(self, image: Image.Image, prompt: str, num_beams: int, max_new_tokens: int):
        try:
            if image is None:
                return "Please upload an image first."
            img = image.convert("RGB")
            if self.args.image_size and self.args.image_size > 0:
                img = img.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
            pix = self.processor(images=img, return_tensors="pt").pixel_values.to(self.device)

            tok = self.tokenizer(prompt, return_tensors="pt")
            inp_ids = tok.input_ids.to(self.device)
            attn = tok.attention_mask.to(self.device)

            autocast_ok = torch.cuda.is_available()
            with torch.cuda.amp.autocast(enabled=autocast_ok, dtype=self.dtype if autocast_ok else None):
                gen_ids = self.model.generate(
                    pixel_values=pix,
                    input_ids=inp_ids,
                    attention_mask=attn,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    length_penalty=1.15,
                    no_repeat_ngram_size=3,
                    min_new_tokens=20,
                    pad_token_id=self.pad_id,
                    eos_token_id=self.eos_id,
                )
            text = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
            return text if text else "(empty output)"
        except Exception as e:
            return f"[ERROR] {type(e).__name__}: {e}"


def build_demo(captioner: Captioner, default_prompt: str):
    css = """
    .gradio-container { max-width: 1400px; }
    .gradio-container, .gradio-container * {
        font-family: Inter, Roboto, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif !important;
        font-size: 18px !important;
        line-height: 1.5;
    }
    h1, h2, h3 { font-weight: 700 !important; }
    .gradio-container .prose h2 { font-size: 28px !important; }
    button { padding: 12px 18px !important; border-radius: 10px !important; font-weight: 600 !important; }

    /* 完全上下左右居中 + 加粗黑字 + 浅蓝背景 + 字体放大 */
    .big-caption textarea {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        font-size: 26px !important;        /* 字体更大 */
        font-weight: 700 !important;       /* 加粗 */
        color: #000000 !important;         /* 黑色文字 */
        background-color: #DAE8FC !important; /* 浅蓝背景 */
        line-height: 1.6 !important;
        height: 340px !important;
        border-color: #BCD0F5 !important;
        border-radius: 10px !important;
        resize: none !important;
        overflow: hidden !important;
        white-space: pre-wrap !important;
    }
    """

    with gr.Blocks(css=css) as demo:
        gr.Markdown("## 🖼️ Image Captioning Demo")
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="pil", label="Upload Image")
                max_toks = gr.Slider(8, 40, step=1, value=40, label="Max New Tokens")
                const_prompt = gr.State(default_prompt)
                const_beams = gr.State(1)
                btn = gr.Button("Generate Caption", variant="primary")

            with gr.Column(scale=1):
                out = gr.Textbox(
                    label="Caption Output",
                    lines=12,
                    elem_classes=["big-caption"]
                )

        btn.click(
            fn=captioner.caption,
            inputs=[img, const_prompt, const_beams, max_toks],
            outputs=[out]
        )
    return demo


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Salesforce/blip2-flan-t5-xl")
    ap.add_argument("--no_lora", action="store_true")
    ap.add_argument("--ckpt_dir", type=str, default="")
    ap.add_argument("--ckpt_file", type=str, default="")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--prompt", type=str, default="caption")
    ap.add_argument("--image_size", type=int, default=336)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    cap = Captioner(args)
    demo = build_demo(cap, args.prompt)
    demo.queue(max_size=128).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
