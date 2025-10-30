#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
inference_blip2.py

功能：
- 对 VizWiz 风格 processed_*.json（含 "image" 字段）做批量推理，输出 {"id","caption"} 列表 JSON
- 支持 Hugging Face "Salesforce/blip2-flan-t5-*" 基座模型
- 支持 LoRA 权重注入（目录 or .pt 文件）
- tqdm 进度条

示例：
python src/inference_blip2.py \
  --model_name "Salesforce/blip2-flan-t5-xl" \
  --ckpt outputs/vizwiz_au_lora_sec/best.pt \
  --data_file data/vizwiz/processed_val.json \
  --out_file result/au_au_blip2_lora_vizwiz_1017/preds_val.json \
  --image_size 336 \
  --batch_size 128 \
  --max_new_tokens 40 \
  --num_beams 1
"""

import os
import json
import argparse
from typing import List, Tuple

import torch
from tqdm import tqdm
from PIL import Image

from transformers import Blip2ForConditionalGeneration, AutoProcessor
from peft import PeftModel
from peft.utils import set_peft_model_state_dict

# ----------------------------
# 数据加载
# ----------------------------
def load_vizwiz_like(path: str) -> List[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for it in data:
        img_path = it.get("image", "")
        base = os.path.splitext(os.path.basename(img_path))[0]
        items.append((base, img_path))
    return items


# ----------------------------
# 模型加载 + LoRA 支持
# ----------------------------
def load_model_with_lora(model_name: str, ckpt_path: str, precision="auto", device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if precision == "auto":
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    else:
        if precision.lower() in ["fp16", "half"]:
            dtype = torch.float16
        elif precision.lower() == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

    print(f"[INFO] Loading base model: {model_name} (dtype={dtype}, device={device})")
    processor = AutoProcessor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)

    # ✅ LoRA 注入逻辑
    if ckpt_path and os.path.exists(ckpt_path):

        # === 目录方式：PeftModel.from_pretrained ===
        if os.path.isdir(ckpt_path):
            print(f"[INFO] Loading LoRA adapter directory: {ckpt_path}")
            model = PeftModel.from_pretrained(model, ckpt_path, torch_dtype=dtype)

        # === 单 .pt 文件：手动注入 LoRA ===
        else:
            print(f"[INFO] Loading LoRA .pt: {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu")

            if isinstance(state, dict):
                if "model" in state and isinstance(state["model"], dict):
                    state = state["model"]
                elif "state_dict" in state and isinstance(state["state_dict"], dict):
                    state = state["state_dict"]

            # ✅ dummy config for FLAN-T5 BLIP2
            from peft import LoraConfig, get_peft_model
            print("[INFO] Injecting LoRA weights via dummy config...")

            dummy_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=["q", "k", "v", "o"],  # ✅ T5 correct target
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )


            model = get_peft_model(model, dummy_config)
            missing, unexpected = set_peft_model_state_dict(model, state)
            print(f"[INFO] LoRA Loaded. missing={len(missing)}, unexpected={len(unexpected)}")

    model = model.to(device)
    model.eval()
    return model, processor, device, dtype


# ----------------------------
# 推理
# ----------------------------
@torch.no_grad()
def infer_captions(model, processor, device, dtype, pairs, batch_size, image_size, max_new_tokens, num_beams, prompt):

    results = []
    cached = []

    for _id, img_path in tqdm(pairs, desc="Reading images"):
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print(f"[WARN] Failed to open: {_id}, using blank image.")
            img = Image.new("RGB", (image_size, image_size), (255,255,255))
        cached.append((_id, img))

    for i in tqdm(range(0, len(cached), batch_size), desc="Inference"):
        chunk = cached[i : i + batch_size]
        ids = [x[0] for x in chunk]
        imgs = [x[1] for x in chunk]
        texts = [prompt] * len(imgs)

        inputs = processor(images=imgs, text=texts, return_tensors="pt").to(device)

        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams
        )

        caps = processor.batch_decode(generated, skip_special_tokens=True)

        for _id, cap in zip(ids, caps):
            results.append({"id": _id, "caption": cap.strip()})

    return results


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--data_file", type=str, required=True)
    p.add_argument("--out_file", type=str, required=True)
    p.add_argument("--image_size", type=int, default=336)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=40)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--prompt", type=str, default="Describe the image in one sentence.")
    p.add_argument("--precision", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Loading dataset: {args.data_file}")
    pairs = load_vizwiz_like(args.data_file)
    print(f"[INFO] Found {len(pairs)} samples.")

    model, processor, device, dtype = load_model_with_lora(
        args.model_name, args.ckpt, args.precision
    )

    preds = infer_captions(
        model, processor, device, dtype, pairs,
        args.batch_size, args.image_size,
        args.max_new_tokens, args.num_beams,
        args.prompt
    )

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)

    print(f"[INFO] ✅ Saved predictions to: {args.out_file}")


if __name__ == "__main__":
    main()
