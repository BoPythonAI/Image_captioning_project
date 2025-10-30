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
    # 再兜底一次
    new = re.sub(r"\.iteritems\(\)", ".items()", new)
    if new != txt:
        fp.write_text(new, encoding="utf-8")


def patch_coco_caption_py3(root: str):
    rootp = Path(root)
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

# ====== 图像与文本增强 ======
from torchvision import transforms

# ========== Try import COCO-Caption CIDEr ==========
USE_CIDER = False
try:
    import builtins
    if not hasattr(builtins, "xrange"):
        builtins.xrange = range
    if not hasattr(builtins, "unicode"):
        builtins.unicode = str

    from pycocoevalcap.cider.cider import Cider  # noqa: E402
    USE_CIDER = True
except Exception as e:
    from evaluate import load as load_metric  # noqa: E402
    print(f"[WARN] CIDEr scorer not available ({e}). Will fall back to SacreBLEU for validation and disable SCST.")


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
        f">> Parameters: total={_human_count(total)} | trainable={_human_count(trainable)} ({ratio:.4f}%)"
    )
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if names:
        accelerator.print(
            ">> Trainable tensors (head):\n" + "\n".join([f"   - {n}" for n in names[:head]])
        )
        if len(names) > head:
            accelerator.print(f"   ... (+{len(names)-head} more)")


def freeze_all_params(model):
    for p in model.parameters():
        p.requires_grad = False


# -------------------- 文本增强（仅训练集使用） --------------------
SYNONYMS = {
    "dog": ["puppy", "canine"],
    "cat": ["kitten", "feline"],
    "man": ["person", "gentleman"],
    "woman": ["lady", "person"],
    "photo": ["picture", "image"],
    "phone": ["cellphone", "mobile phone"],
    "car": ["vehicle", "automobile"],
    "cup": ["mug", "cup"],
    "bottle": ["flask", "bottle"],
}

def augment_caption(caption: str, syn_prob=0.2, shuffle_sent_prob=0.2) -> str:
    # 同义词替换
    if syn_prob > 0:
        words = caption.split()
        new_words = []
        for w in words:
            wl = w.lower().strip(",.!?")
            if wl in SYNONYMS and random.random() < syn_prob:
                new_words.append(random.choice(SYNONYMS[wl]))
            else:
                new_words.append(w)
        caption = " ".join(new_words)
    # 句子级打乱
    if shuffle_sent_prob > 0 and random.random() < shuffle_sent_prob and "." in caption:
        sents = [s.strip() for s in caption.split(".") if s.strip()]
        if len(sents) > 1:
            random.shuffle(sents)
            caption = ". ".join(sents)
    return caption


# -------------------- Datasets --------------------
BLIP2_PROMPT = "caption:"

class VizWizDatasetBLIP2(torch.utils.data.Dataset):
    def __init__(
        self,
        ann_file,
        image_processor,
        tokenizer,
        max_len=40,
        image_size=None,
        prompt: str = BLIP2_PROMPT,
        use_aug: bool = False,
        syn_prob: float = 0.2,
        shuffle_sent_prob: float = 0.2,
    ):
        self.data = json.load(open(ann_file, "r", encoding="utf-8"))
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.image_size = image_size
        self.prompt = prompt
        self.use_aug = use_aug
        self.syn_prob = syn_prob
        self.shuffle_sent_prob = shuffle_sent_prob

        self.img_aug = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomGrayscale(p=0.1),
        ]) if (use_aug and image_size is not None) else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["image"]
        img = Image.open(img_path).convert("RGB")

        if self.img_aug is not None:
            img = self.img_aug(img)
        elif self.image_size is not None:
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)

        caption = random.choice(item["captions"])  # 目标
        if self.use_aug:
            caption = augment_caption(caption, syn_prob=self.syn_prob, shuffle_sent_prob=self.shuffle_sent_prob)

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
    def __init__(
        self,
        ann_file,
        processor,
        max_len=40,
        image_size=None,
        use_aug: bool = False,
        syn_prob: float = 0.2,
        shuffle_sent_prob: float = 0.2,
    ):
        self.data = json.load(open(ann_file, "r", encoding="utf-8"))
        self.processor = processor
        self.max_len = max_len
        self.image_size = image_size
        self.use_aug = use_aug
        self.syn_prob = syn_prob
        self.shuffle_sent_prob = shuffle_sent_prob

        self.img_aug = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomGrayscale(p=0.1),
        ]) if (use_aug and image_size is not None) else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["image"]
        img = Image.open(img_path).convert("RGB")

        if self.img_aug is not None:
            img = self.img_aug(img)
        elif self.image_size is not None:
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)

        caption = random.choice(item["captions"])
        if self.use_aug:
            caption = augment_caption(caption, syn_prob=self.syn_prob, shuffle_sent_prob=self.shuffle_sent_prob)

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


# -------------------- 生成工具（SCST会用到） --------------------
def generate_with_scores(model, decode_fn, batch, pad_id, max_new_tokens=40,
                         sample=False, temperature=1.0, top_p=1.0, top_k=0, num_beams=1):
    """
    生成候选序列并返回：token 序列、解码文本、平均 token log-prob。
    
    修复要点：
    - 对于 encoder-decoder（如 BLIP-2 + T5），用 labels=seq 计算 logits，
      然后对 (logits[:, :-1], seq[:, 1:]) 对齐取 log-prob。
    - 对 causal LM：按 prompt_len 切，但在极端情况下做安全兜底。
    """
    raw_model = unwrap_ddp(model)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=pad_id)

    # 组装基础输入
    if "pixel_values" in batch:
        base_kwargs = dict(pixel_values=batch["pixel_values"],
                           input_ids=batch.get("input_ids"),
                           attention_mask=batch.get("attention_mask"))
    else:
        base_kwargs = dict(input_ids=batch.get("input_ids"),
                           attention_mask=batch.get("attention_mask"))

    # 采样/束搜索
    if sample:
        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        if top_k > 0:
            gen_kwargs["top_k"] = top_k
    else:
        gen_kwargs["do_sample"] = False
        if num_beams and num_beams > 1:
            gen_kwargs["num_beams"] = num_beams

    # 生成
    seq = raw_model.generate(**base_kwargs, **gen_kwargs)

    # 判定是否为 encoder-decoder
    is_enc_dec = bool(getattr(getattr(raw_model, "config", None), "is_encoder_decoder", False))

    if is_enc_dec:
        # —— Encoder-Decoder：直接用 labels=seq 前向；logits 与 seq 对齐，忽略第一个 token ——
        out = raw_model(**base_kwargs, labels=seq)
        logits = out.logits  # [B, T, V]
        if logits.size(1) == 0 or seq.size(1) == 0:
            # 安全兜底：返回零 log-prob
            token_logprob = torch.zeros(seq.size(0), dtype=logits.dtype, device=logits.device)
        else:
            target = seq[:, 1:].contiguous()               # [B, T-1]
            step_logits = logits[:, :-1, :].contiguous()   # [B, T-1, V]
            log_probs = torch.log_softmax(step_logits, dim=-1)
            gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
            mask = (target != pad_id).float()
            token_logprob = (gathered * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
    else:
        # —— Causal LM：仅评估续写部分（去掉 prompt） ——
        prompt_len = base_kwargs["input_ids"].shape[1] if base_kwargs.get("input_ids") is not None else 0
        # 避免切片为空：若生成长度 <= prompt_len，则至少保留最后 1 个 token 参与计算
        start = min(prompt_len, max(seq.size(1) - 1, 0))
        labels_slice = seq[:, start:].contiguous()
        if labels_slice.numel() == 0:
            token_logprob = torch.zeros(seq.size(0), dtype=torch.float32, device=seq.device)
        else:
            out = raw_model(**base_kwargs, labels=labels_slice)
            logits = out.logits  # [B, T, V]
            log_probs = torch.log_softmax(logits, dim=-1)
            gathered = log_probs.gather(-1, labels_slice.unsqueeze(-1)).squeeze(-1)  # [B, T]
            mask = (labels_slice != pad_id).float()
            token_logprob = (gathered * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)

    texts = decode_fn(seq)
    return seq, texts, token_logprob


# -------------------- Validation --------------------
@torch.no_grad()
def evaluate_on_val(model, val_loader, decode_fn, accelerator: Accelerator,
                    max_new_tokens=40, pad_id: int = 0, num_beams: int = 1):
    model.eval()
    raw_model = unwrap_ddp(model)

    all_pred_ids, all_ref_ids, all_indices = [], [], []

    for batch in tqdm(val_loader, desc="Eval", leave=False,
                      disable=not accelerator.is_main_process):
        input_keys = {"pixel_values", "input_ids", "attention_mask", "labels"}
        model_inp = {k: v for k, v in batch.items() if k in input_keys}

        # 生成：pixel + prompt
        if "pixel_values" in model_inp:
            gen_out = raw_model.generate(
                pixel_values=model_inp["pixel_values"],
                input_ids=model_inp.get("input_ids"),
                attention_mask=model_inp.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
                num_beams=num_beams
            )
        else:
            gen_out = raw_model.generate(
                input_ids=model_inp.get("input_ids"),
                attention_mask=model_inp.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
                num_beams=num_beams
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


# -------------------- RL 奖励（修复：每个样本单独计算） --------------------
def compute_cider_rewards(cider_scorer, preds, refs):
    """
    为每个样本单独计算 CIDEr 奖励
    
    修复要点：
    - CIDEr.compute_score 返回 (average_score, scores_array)
    - scores_array 是每个样本的分数，这才是我们需要的
    """
    res = {str(i): [p] for i, p in enumerate(preds)}
    gts = {str(i): [r] for i, r in enumerate(refs)}
    
    # CIDEr.compute_score 会返回 (average_score, scores_array)
    avg_score, scores = cider_scorer.compute_score(gts=gts, res=res)
    
    # scores 是每个样本的分数列表或数组
    if isinstance(scores, np.ndarray):
        return torch.tensor(scores, dtype=torch.float32)
    elif isinstance(scores, (list, tuple)):
        return torch.tensor(list(scores), dtype=torch.float32)
    else:
        # 兜底：如果只返回标量，则广播（但这不应该发生）
        print(f"[WARN] CIDEr returned scalar score, broadcasting to batch")
        return torch.tensor([float(avg_score)] * len(preds), dtype=torch.float32)


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

        image_processor = BlipImageProcessor.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

        train_ds = VizWizDatasetBLIP2(
            args.train_file, image_processor, tokenizer,
            image_size=args.image_size, max_len=40, prompt=BLIP2_PROMPT,
            use_aug=args.use_aug, syn_prob=args.syn_prob, shuffle_sent_prob=args.shuffle_sent_prob
        )
        val_ds = VizWizDatasetBLIP2(
            args.val_file, image_processor, tokenizer,
            image_size=args.image_size, max_len=40, prompt=BLIP2_PROMPT,
            use_aug=False
        )

        def decode_fn(token_ids_batch):
            return tokenizer.batch_decode(token_ids_batch, skip_special_tokens=True)

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(args.model_name)

        train_ds = VizWizDatasetGeneric(
            args.train_file, processor, image_size=args.image_size,
            use_aug=args.use_aug, syn_prob=args.syn_prob, shuffle_sent_prob=args.shuffle_sent_prob
        )
        val_ds = VizWizDatasetGeneric(
            args.val_file, processor, image_size=args.image_size,
            use_aug=False
        )

        def decode_fn(token_ids_batch):
            return processor.batch_decode(token_ids_batch, skip_special_tokens=True)

        maybe_tok = getattr(processor, "tokenizer", None)
        pad_id = getattr(maybe_tok, "pad_token_id", 0) if maybe_tok is not None else 0

    # ====== 冻结所有参数，之后只训LoRA ======
    freeze_all_params(model)
    if hasattr(model, "query_tokens") and isinstance(model.query_tokens, torch.nn.Parameter):
        model.query_tokens.requires_grad = False

    # ====== 注入 LoRA ======
    if args.use_lora:
        if is_blip2:
            lcfg_lm = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                task_type="SEQ_2_SEQ_LM",
                target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
            )
            model.language_model = get_peft_model(model.language_model, lcfg_lm)
            accelerator.print(">> LoRA enabled on BLIP-2 language_model (T5).")
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

    print_trainable_params(accelerator, model, head=40)
    if is_blip2 and args.use_lora:
        accelerator.print(">> NOTE: With BLIP-2 + LoRA, trainables should appear under language_model.* and qformer.bert.*")

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Optimizer（只会拿到 requires_grad=True 的 LoRA 权重）
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # DDP 准备
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    # 训练 & 早停
    os.makedirs(args.save_dir, exist_ok=True)
    best_score = -1.0
    best_path = os.path.join(args.save_dir, "best.pt")
    no_improve = 0
    metric_name = "cider" if USE_CIDER else "sacrebleu"

    # 全局 CIDEr scorer（SCST 用）
    cider_scorer_global = Cider() if USE_CIDER else None

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        use_rl = (args.use_scst and USE_CIDER and (epoch > args.ce_warmup_epochs))
        
        # 动态调整数据增强：SCST 阶段关闭增强
        if use_rl and hasattr(train_ds, 'use_aug'):
            original_aug = train_ds.use_aug
            train_ds.use_aug = False
            accelerator.print(f"[Epoch {epoch}] SCST mode: disabled data augmentation")
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not accelerator.is_main_process)

        for step, batch in enumerate(pbar, start=1):
            input_keys = {"pixel_values", "input_ids", "attention_mask", "labels"}
            model_inp = {k: v for k, v in batch.items() if k in input_keys}

            if not use_rl:
                # -------- 交叉熵阶段 --------
                out = model(**model_inp)
                loss = out.loss / args.grad_accum
            else:
                # -------- SCST 阶段（混合 CE + RL） --------
                # 1. 计算 CE loss（保留监督信号）
                ce_out = model(**model_inp)
                ce_loss = ce_out.loss
                
                # 2. baseline: 贪心/beam
                with torch.no_grad():
                    _, greedy_txt, _ = generate_with_scores(
                        model, decode_fn, model_inp, pad_id,
                        max_new_tokens=args.scst_max_new_tokens, sample=False, num_beams=args.decode_beam_size
                    )
                
                # 3. sample: 采样（需带梯度）
                _, sample_txt, sample_logp = generate_with_scores(
                    model, decode_fn, model_inp, pad_id,
                    max_new_tokens=args.scst_max_new_tokens, sample=True, temperature=1.0, top_p=0.9
                )

                # 4. 参考文本（把 -100 还原为 pad）
                labels_for_decode = model_inp["labels"].clone()
                labels_for_decode[labels_for_decode == -100] = pad_id
                refs = [decode_fn(l.unsqueeze(0))[0] for l in labels_for_decode]

                # 5. 计算奖励（每个样本独立）
                with torch.no_grad():
                    r_greedy = compute_cider_rewards(cider_scorer_global, greedy_txt, refs)
                    r_sample = compute_cider_rewards(cider_scorer_global, sample_txt, refs)

                # 6. 计算优势函数和 RL loss
                advantage = (r_sample - r_greedy).to(sample_logp.device)
                rl_loss = -(advantage * sample_logp).mean()
                
                # 7. 混合损失（保留部分 CE 信号以稳定训练）
                loss = (args.ce_weight * ce_loss + args.rl_weight * rl_loss) / args.grad_accum

            accelerator.backward(loss)
            if step % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

        # 处理最后不完整的梯度累积
        if step % args.grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # 恢复增强设置
        if use_rl and hasattr(train_ds, 'use_aug'):
            train_ds.use_aug = original_aug

        score, used_metric = evaluate_on_val(
            model, val_loader, decode_fn, accelerator,
            max_new_tokens=40, pad_id=pad_id, num_beams=args.decode_beam_size
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

    # 数据增强（仅CE阶段建议开启）
    ap.add_argument("--use_aug", action="store_true",
                    help="Enable image & text augmentation on training set only")
    ap.add_argument("--syn_prob", type=float, default=0.2,
                    help="Synonym replacement probability for caption words")
    ap.add_argument("--shuffle_sent_prob", type=float, default=0.2,
                    help="Probability to shuffle sentence order in multi-sentence captions")

    # ====== SCST 相关参数 ======
    ap.add_argument("--use_scst", action="store_true",
                    help="Enable Self-Critical Sequence Training after ce_warmup_epochs (requires CIDEr)")
    ap.add_argument("--ce_warmup_epochs", type=int, default=3,
                    help="Number of CE-only warmup epochs before switching to SCST (建议至少3-5)")
    ap.add_argument("--ce_weight", type=float, default=0.3,
                    help="Weight for CE loss in SCST phase (0-1, 建议0.2-0.4保留监督信号)")
    ap.add_argument("--rl_weight", type=float, default=0.7,
                    help="Weight for RL loss in SCST phase (0-1)")
    ap.add_argument("--scst_sample_n", type=int, default=1,
                    help="How many sampled captions per image for SCST (current impl uses 1)")
    ap.add_argument("--scst_max_new_tokens", type=int, default=40)
    ap.add_argument("--decode_beam_size", type=int, default=1,
                    help=">1 to use beam search in eval and greedy baseline")

    args = ap.parse_args()
    
    # 参数验证
    if args.use_scst and not USE_CIDER:
        print("[ERROR] SCST requires CIDEr scorer, but it's not available. Disabling SCST.")
        args.use_scst = False
    
    if args.use_scst:
        print(f"[INFO] SCST enabled with:")
        print(f"  - CE warmup epochs: {args.ce_warmup_epochs}")
        print(f"  - CE weight in SCST: {args.ce_weight}")
        print(f"  - RL weight in SCST: {args.rl_weight}")
        print(f"  - Data augmentation will be disabled during SCST phase")
    
    train(args)