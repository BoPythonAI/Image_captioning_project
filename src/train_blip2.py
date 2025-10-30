# --- auto-patch coco-caption for Python3 ---
import os, io, re, sys, json, argparse, random
from pathlib import Path

CIDER_ROOT = "/root/autodl-tmp/image_captioning_project/coco-caption"


def _patch_file_py3(fp: Path):
    if not fp.exists():
        return
    txt = fp.read_text(encoding="utf-8")
    new = txt
    # 典型 Python2 兼容替换
    new = new.replace("xrange", "range")
    new = new.replace(".iteritems()", ".items()")
    new = new.replace("unicode", "str")
    # 有些文件写成 for (ngram,count) in ref.iteritems():
    new = re.sub(r"\.iteritems\(\)", ".items()", new)
    # 仅在内容发生变化时写回
    if new != txt:
        fp.write_text(new, encoding="utf-8")


def patch_coco_caption_py3(root: str):
    rootp = Path(root)
    # 主要有用的 cider/bleu scorer 做兼容
    targets = [
        rootp / "pycocoevalcap/cider/cider_scorer.py",
        rootp / "pycocoevalcap/cider/cider.py",
        rootp / "pycocoevalcap/bleu/bleu_scorer.py",
        rootp / "pycocoevalcap/bleu/bleu.py",
        rootp / "pycocoevalcap/meteor/meteor.py",
        rootp / "pycocoevalcap/rouge/rouge.py",
        rootp / "pycocoevalcap/spice/spice.py",
    ]
    for fp in targets:
        _patch_file_py3(fp)


# 先修补再加到 sys.path 并导入
patch_coco_caption_py3(CIDER_ROOT)
sys.path.insert(0, CIDER_ROOT)
sys.path.insert(0, str(Path(CIDER_ROOT) / "pycocoevalcap" / "cider"))

# ---------------- std imports ----------------
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    Blip2ForConditionalGeneration,
    BlipImageProcessor,
)
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from tqdm import tqdm
from PIL import Image

# ========== Try import COCO-Caption CIDEr ==========
USE_CIDER = False
try:
    # 再做一次兜底：xrange/unicode
    import builtins
    if not hasattr(builtins, "xrange"):
        builtins.xrange = range
    if not hasattr(builtins, "unicode"):
        builtins.unicode = str

    from pycocoevalcap.cider.cider import Cider  # noqa: E402
    USE_CIDER = True
except Exception as e:
    from evaluate import load as load_metric  # noqa: E402
    print(f"[WARN] CIDEr scorer not available ({e}). Will fall back to SacreBLEU for validation selection.)")


# -------------------- utils --------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_ddp(model):
    return model.module if hasattr(model, "module") else model


def _human_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return str(n)


def print_trainable_params(accelerator: Accelerator, model, head: int = 40):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = (trainable / total * 100.0) if total > 0 else 0.0
    accelerator.print(
        f">> Parameters: total={_human_count(total)} "
        f"| trainable={_human_count(trainable)} ({ratio:.4f}%)"
    )
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if names:
        accelerator.print(">> Trainable tensors (head):")
        for n in names[:head]:
            accelerator.print(f"   - {n}")
        if len(names) > head:
            accelerator.print(f"   ... (+{len(names)-head} more)")


def freeze_all_params(model):
    for p in model.parameters():
        p.requires_grad = False


# -------------------- Datasets --------------------
BLIP2_PROMPT = "caption:"  # 也可用 "Describe the image in one short sentence."

class VizWizDatasetBLIP2(torch.utils.data.Dataset):
    def __init__(self, ann_file, image_processor, tokenizer, max_len=40, image_size=None, prompt: str = BLIP2_PROMPT):
        self.data = json.load(open(ann_file, "r", encoding="utf-8"))
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.image_size = image_size
        self.prompt = prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["image"]
        img = Image.open(img_path).convert("RGB")
        if self.image_size is not None:
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)

        caption = random.choice(item["captions"])  # 目标

        pix = self.image_processor(images=img, return_tensors="pt")

        tok_inp = self.tokenizer(
            self.prompt, return_tensors="pt",
            padding="max_length", truncation=True, max_length=16
        )

        tok_lbl = self.tokenizer(
            caption, return_tensors="pt",
            padding="max_length", truncation=True, max_length=self.max_len
        )

        labels = tok_lbl["input_ids"].clone()
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        labels[labels == pad_token_id] = -100

        return {
            "pixel_values": pix["pixel_values"].squeeze(0),
            "input_ids": tok_inp["input_ids"].squeeze(0),
            "attention_mask": tok_inp["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "image_id": os.path.splitext(os.path.basename(img_path))[0],
            "idx": torch.tensor(idx, dtype=torch.long),
        }


class VizWizDatasetGeneric(torch.utils.data.Dataset):
    def __init__(self, ann_file, processor, max_len=40, image_size=None):
        self.data = json.load(open(ann_file, "r", encoding="utf-8"))
        self.processor = processor
        self.max_len = max_len
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["image"]
        img = Image.open(img_path).convert("RGB")
        if self.image_size is not None:
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        caption = random.choice(item["captions"])
        enc = self.processor(
            images=img, text=caption, return_tensors="pt",
            padding="max_length", truncation=True, max_length=self.max_len
        )
        if "input_ids" in enc:
            enc["labels"] = enc["input_ids"].clone()
            maybe_tok = getattr(self.processor, "tokenizer", None)
            pad_token_id = getattr(maybe_tok, "pad_token_id", 0) if maybe_tok is not None else 0
            enc["labels"][enc["labels"] == pad_token_id] = -100

        sample = {k: v.squeeze(0) for k, v in enc.items()}
        sample["image_id"] = os.path.splitext(os.path.basename(img_path))[0]
        sample["idx"] = torch.tensor(idx, dtype=torch.long)
        return sample


# -------------------- Validation --------------------
@torch.no_grad()
def evaluate_on_val(model, val_loader, decode_fn, accelerator: Accelerator,
                    max_new_tokens=40, pad_id: int = 0):
    model.eval()
    raw_model = unwrap_ddp(model)

    all_pred_ids, all_ref_ids, all_indices = [], [], []

    for batch in tqdm(val_loader, desc="Eval", leave=False,
                      disable=not accelerator.is_main_process):
        input_keys = {"pixel_values", "input_ids", "attention_mask", "labels"}
        model_inp = {k: v for k, v in batch.items() if k in input_keys}

        # 生成时：pixel + prompt
        if "pixel_values" in model_inp:
            gen_out = raw_model.generate(
                pixel_values=model_inp["pixel_values"],
                input_ids=model_inp.get("input_ids"),
                attention_mask=model_inp.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id
            )
        else:
            gen_out = raw_model.generate(
                input_ids=model_inp.get("input_ids"),
                attention_mask=model_inp.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id
            )

        gen_out = accelerator.pad_across_processes(gen_out, dim=1, pad_index=pad_id)
        gen_out = accelerator.gather(gen_out).cpu()

        labels = accelerator.gather(model_inp["labels"]).cpu()
        idxs = accelerator.gather(batch["idx"]).cpu()

        pred_txt = decode_fn(gen_out)

        labels_for_decode = labels.clone()
        labels_for_decode[labels_for_decode == -100] = pad_id
        ref_txt = [decode_fn(l.unsqueeze(0))[0] for l in labels_for_decode]

        all_pred_ids.extend(pred_txt)
        all_ref_ids.extend(ref_txt)
        all_indices.extend(idxs.tolist())

    merged = list(zip(all_indices, all_pred_ids, all_ref_ids))
    merged.sort(key=lambda x: x[0])

    res = {str(idx): [pred] for idx, pred, _ in merged}
    gts = {str(idx): [ref] for idx, _, ref in merged}

    if USE_CIDER:
        cider_scorer = Cider()
        score, _ = cider_scorer.compute_score(gts=gts, res=res)
        return float(score), "cider"
    else:
        from evaluate import load as load_metric
        metric_bleu = load_metric("sacrebleu")
        bleu = metric_bleu.compute(
            predictions=[p for _, p, _ in merged],
            references=[[r] for _, _, r in merged]
        )["score"]
        return float(bleu), "sacrebleu"


# -------------------- Train --------------------
def train(args):
    accelerator = Accelerator()
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.print(f"Loading model: {args.model_name}")
    is_blip2 = "blip2" in args.model_name.lower()

    if is_blip2:
        _to_dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else None
        if _to_dtype is not None:
            model = Blip2ForConditionalGeneration.from_pretrained(
                args.model_name,
                torch_dtype=_to_dtype,
            )
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(args.model_name)

        # >>> 开启 gradient checkpointing（强烈建议）
        try:
            model.gradient_checkpointing_enable()
            accelerator.print(">> Gradient Checkpointing ENABLED.")
        except Exception:
            accelerator.print("!! WARN: gradient_checkpointing_enable() not available; skip.")

        image_processor = BlipImageProcessor.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

        train_ds = VizWizDatasetBLIP2(
            args.train_file, image_processor, tokenizer,
            image_size=args.image_size, max_len=40, prompt=BLIP2_PROMPT
        )
        val_ds = VizWizDatasetBLIP2(
            args.val_file, image_processor, tokenizer,
            image_size=args.image_size, max_len=40, prompt=BLIP2_PROMPT
        )

        def decode_fn(token_ids_batch):
            return tokenizer.batch_decode(token_ids_batch, skip_special_tokens=True)

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)

        try:
            model.gradient_checkpointing_enable()
            accelerator.print(">> Gradient Checkpointing ENABLED.")
        except Exception:
            accelerator.print("!! WARN: gradient_checkpointing_enable() not available; skip.")

        processor = AutoProcessor.from_pretrained(args.model_name)

        train_ds = VizWizDatasetGeneric(args.train_file, processor, image_size=args.image_size)
        val_ds = VizWizDatasetGeneric(args.val_file, processor, image_size=args.image_size)

        def decode_fn(token_ids_batch):
            return processor.batch_decode(token_ids_batch, skip_special_tokens=True)

        maybe_tok = getattr(processor, "tokenizer", None)
        pad_id = getattr(maybe_tok, "pad_token_id", 0) if maybe_tok is not None else 0

    # 默认不训练 query_tokens（如需训练可改 True）
    if hasattr(model, "query_tokens") and isinstance(model.query_tokens, torch.nn.Parameter):
        model.query_tokens.requires_grad = False

    # ===== 训练策略 =====
    if args.use_lora:
        # 用 LoRA：先冻结全部，再只训练 LoRA 模块
        freeze_all_params(model)
        if is_blip2:
            # 1) 语言模型 (T5) LoRA
            lcfg_lm = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                task_type="SEQ_2_SEQ_LM",
                target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
            )
            model.language_model = get_peft_model(model.language_model, lcfg_lm)
            accelerator.print(">> LoRA enabled on BLIP-2 language_model (T5).")

            # 2) Q-Former LoRA（若存在）
            if hasattr(model.qformer, "bert"):
                lcfg_qf = LoraConfig(
                    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                    task_type="FEATURE_EXTRACTION",
                    target_modules=["query", "key", "value", "dense", "output.dense", "intermediate.dense"],
                )
                model.qformer.bert = get_peft_model(model.qformer.bert, lcfg_qf)
                accelerator.print(">> LoRA ALSO enabled on BLIP-2 Q-Former (bert).")
            else:
                accelerator.print("!! WARN: model.qformer has no attribute 'bert'; skip Q-Former LoRA.")
        else:
            lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
            model = get_peft_model(model, lcfg)
            accelerator.print(">> LoRA enabled. Trainable parameters set.")
    else:
        # 不用 LoRA：**部分微调** —— 冻结全局，只放开 language_model
        freeze_all_params(model)

        if hasattr(model, "language_model"):
            if args.lm_last2_only:
                # 仅解冻 T5 解码器最后两层 + lm_head
                for n, p in model.named_parameters():
                    if n.startswith("language_model.decoder.block.22.") or \
                       n.startswith("language_model.decoder.block.23.") or \
                       n.startswith("language_model.lm_head"):
                        p.requires_grad = True
                accelerator.print(">> Partial-FT: Unfreezing T5 decoder last 2 layers + lm_head.")
            else:
                # 解冻整个 language_model（T5 全部）
                for n, p in model.named_parameters():
                    if n.startswith("language_model."):
                        p.requires_grad = True
                accelerator.print(">> Partial-FT: Unfreezing ALL language_model (T5).")
        else:
            # 兜底：如果不是 BLIP-2 结构，至少放开最终 LM 头
            for n, p in model.named_parameters():
                if "lm_head" in n or "classifier" in n:
                    p.requires_grad = True
            accelerator.print("!! WARN: language_model not found; only unfreezing lm_head/classifier.")

    # ======= 打印参数数量（prepare 之前，避免 DDP 包裹影响统计） =======
    print_trainable_params(accelerator, model, head=40)
    if is_blip2 and args.use_lora:
        accelerator.print(">> NOTE: With BLIP-2 + LoRA, trainables should appear under language_model.* and qformer.bert.*")

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Optimizer（只会拿到 requires_grad=True 的权重）
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # DDP 准备 - 包含 optimizer
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    # 训练 & 早停
    os.makedirs(args.save_dir, exist_ok=True)
    best_score = -1.0
    best_path = os.path.join(args.save_dir, "best.pt")
    no_improve = 0
    metric_name = "cider" if USE_CIDER else "sacrebleu"

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not accelerator.is_main_process)
        for step, batch in enumerate(pbar, start=1):
            input_keys = {"pixel_values", "input_ids", "attention_mask", "labels"}
            model_inp = {k: v for k, v in batch.items() if k in input_keys}
            out = model(**model_inp)
            loss = out.loss / args.grad_accum
            accelerator.backward(loss)
            if step % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

        # 处理最后不完整的梯度累积
        if step % args.grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad()

        score, used_metric = evaluate_on_val(
            model, val_loader, decode_fn, accelerator,
            max_new_tokens=40, pad_id=pad_id
        )
        accelerator.print(f"[Val] Epoch {epoch}: {used_metric.upper()} = {score:.3f}")

        if score - best_score > args.min_delta:
            best_score = score
            no_improve = 0
            if accelerator.is_main_process:
                torch.save(unwrap_ddp(model).state_dict(), best_path)
                with open(os.path.join(args.save_dir, "best_meta.json"), "w", encoding="utf-8") as f:
                    json.dump({"epoch": epoch, used_metric: best_score}, f, indent=2, ensure_ascii=False)
            accelerator.print(f"  ↳ New best! Saved to {best_path}")
        else:
            no_improve += 1
            accelerator.print(f"  ↳ No improve ({no_improve}/{args.early_stop_patience})")

        if no_improve >= args.early_stop_patience:
            accelerator.print(
                f"[Early Stop] patience={args.early_stop_patience} reached. Best {used_metric.upper()}={best_score:.3f}")
            break

    accelerator.print(f"[Done] Best {metric_name.upper()}={best_score:.3f} | ckpt={best_path}")


# -------------------- CLI --------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--val_file", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--early_stop_patience", type=int, default=2)
    ap.add_argument("--min_delta", type=float, default=1e-3)
    # 新增：仅解冻 T5 解码器最后两层 + lm_head（不用 LoRA 时更省显存）
    ap.add_argument("--lm_last2_only", action="store_true",
                    help="When NOT using LoRA, unfreeze only T5 decoder last 2 layers + lm_head.")
    args = ap.parse_args()
    train(args)
