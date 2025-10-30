#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_utils.py

功能：
- 兼容 VizWiz processed_*.json（从 `image` 提取文件名作为 id）
- 计算 BLEU-1~4（pycocoevalcap）、CIDEr（pycocoevalcap）、METEOR（HuggingFace evaluate 版）、SPICE（可选）
- 带 tqdm 进度条

用法：
python -m src.eval_utils \
  --preds outputs/vizwiz_full_lora/preds_val.json \
  --refs  data/vizwiz/processed_val.json \
  [--no_spice]
"""

import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm

# ----------------------------
# 尝试导入 pycocoevalcap 指标
# ----------------------------
try:
    from pycocoevalcap.bleu.bleu import Bleu
    _HAVE_COCO_BLEU = True
except Exception:
    _HAVE_COCO_BLEU = False

try:
    from pycocoevalcap.cider.cider import Cider
    _HAVE_CIDER = True
except Exception:
    _HAVE_CIDER = False

# SPICE 仍使用 coco-caption（需要 Java，失败会优雅跳过）
try:
    from pycocoevalcap.spice.spice import Spice
    _HAVE_SPICE = True
except Exception:
    _HAVE_SPICE = False

# ----------------------------
# METEOR 改为使用 HuggingFace evaluate
# ----------------------------
try:
    import evaluate  # pip install evaluate
    _HAVE_EVAL = True
except Exception:
    _HAVE_EVAL = False


# ===========================================================
# 数据加载
# ===========================================================

def _load_preds_file(path: str):
    """
    预测文件支持：
      A) [{"id": "...", "caption": "..."}...]
      B) {"id1": "caption1", ...} 或 {"id1": {"caption": "..."}, ...}
    """
    data = json.load(open(path, "r", encoding="utf-8"))
    pairs = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict) and "caption" in v:
                pairs.append((str(k), str(v["caption"])))
            else:
                pairs.append((str(k), str(v)))
    elif isinstance(data, list):
        for it in data:
            _id = str(it.get("id", ""))
            _cap = str(it.get("caption", ""))
            pairs.append((_id, _cap))
    else:
        raise ValueError("Unsupported preds JSON format.")
    return pairs


def _load_refs_file(path: str):
    """
    参考文件支持：
      A) VizWiz processed_*.json：[{ "image": ".../VizWiz_val_00000000.jpg", "captions": [...] }, ...]
      B) [{"id":"...", "captions":[...]} ...]
      C) {"id1":[...], "id2":[...], ...}
    """
    data = json.load(open(path, "r", encoding="utf-8"))
    pairs = []

    # VizWiz 兼容分支
    if isinstance(data, list) and data and "image" in data[0]:
        for it in tqdm(data, desc="Loading reference captions"):
            img_path = it.get("image", "")
            base = os.path.splitext(os.path.basename(img_path))[0]  # e.g., VizWiz_val_00000000
            caps = it.get("captions", [])
            caps = [str(c) for c in caps]
            pairs.append((base, caps))
        return pairs

    # 通用分支
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list):
                pairs.append((str(k), [str(x) for x in v]))
            else:
                pairs.append((str(k), [str(v)]))
    elif isinstance(data, list):
        for it in tqdm(data, desc="Loading reference captions"):
            _id = str(it.get("id", ""))
            _caps = it.get("captions", [])
            if not isinstance(_caps, list):
                _caps = [str(_caps)]
            pairs.append((_id, [str(x) for x in _caps]))
    else:
        raise ValueError("Unsupported refs JSON format.")

    return pairs


# ===========================================================
# 计算指标
# ===========================================================

def _ensure_gts_res_format(preds, refs):
    """整理为 coco-caption 期望的 gts/res 结构。"""
    gts = defaultdict(list)
    res = defaultdict(list)
    for pid, pred in tqdm(preds, desc="Preparing predictions"):
        res[pid] = [pred]
    for rid, ref_list in tqdm(refs, desc="Preparing references"):
        gts[rid] = ref_list
    return gts, res


def compute_bleu_all(preds, refs):
    print("\n[INFO] Computing BLEU scores...")
    gts, res = _ensure_gts_res_format(preds, refs)
    if _HAVE_COCO_BLEU:
        scorer = Bleu(n=4)
        scores, _ = scorer.compute_score(gts, res)
        return {
            "BLEU-1": float(scores[0]),
            "BLEU-2": float(scores[1]),
            "BLEU-3": float(scores[2]),
            "BLEU-4": float(scores[3]),
        }
    return {"BLEU-1": float("nan"), "BLEU-2": float("nan"),
            "BLEU-3": float("nan"), "BLEU-4": float("nan")}


def compute_cider(preds, refs):
    print("[INFO] Computing CIDEr...")
    gts, res = _ensure_gts_res_format(preds, refs)
    if _HAVE_CIDER:
        scorer = Cider()
        score, _ = scorer.compute_score(gts, res)
        return {"CIDEr": float(score)}
    return {"CIDEr": float("nan")}


def compute_meteor(preds, refs):
    """
    使用 HuggingFace evaluate 版 METEOR（无需 Java / meteor-1.5.jar）
    """
    print("[INFO] Computing METEOR (HuggingFace evaluate)...")
    gts, res = _ensure_gts_res_format(preds, refs)
    if not _HAVE_EVAL:
        print("[WARN] `evaluate` 未安装，METEOR 跳过。")
        return {"METEOR": float("nan")}
    try:
        # predictions 和 references 要一一对应
        ids = list(gts.keys())
        predictions = [res[i][0] if i in res and len(res[i]) > 0 else "" for i in ids]
        references = [[r for r in gts[i]] for i in ids]  # list[list[str]]

        meteor = evaluate.load("meteor")
        result = meteor.compute(predictions=predictions, references=references)
        # HF 返回 {"meteor": score}
        return {"METEOR": float(result.get("meteor", float("nan")))}
    except Exception as e:
        print(f"[WARN] METEOR 计算失败：{e}，已返回 NaN。")
        return {"METEOR": float("nan")}


def compute_spice(preds, refs):
    print("[INFO] Computing SPICE...")
    gts, res = _ensure_gts_res_format(preds, refs)
    if _HAVE_SPICE:
        try:
            scorer = Spice()
            score, _ = scorer.compute_score(gts, res)
            # 有的实现返回 dict({"All": {"f": ...}})，有的返回数值
            if isinstance(score, dict) and "All" in score and "f" in score["All"]:
                return {"SPICE": float(score["All"]["f"])}
            if isinstance(score, (int, float)):
                return {"SPICE": float(score)}
        except Exception as e:
            print(f"[WARN] SPICE 计算失败：{e}，已返回 NaN。")
            return {"SPICE": float("nan")}
    return {"SPICE": float("nan")}


def compute_all_metrics(preds, refs, with_spice=False):
    metrics = {}
    metrics.update(compute_bleu_all(preds, refs))
    metrics.update(compute_cider(preds, refs))
    metrics.update(compute_meteor(preds, refs))
    if with_spice:
        metrics.update(compute_spice(preds, refs))
    return metrics


# ===========================================================
# 主程序入口
# ===========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True, help="预测文件路径")
    parser.add_argument("--refs", required=True, help="参考文件路径")
    parser.add_argument("--no_spice", action="store_true", help="不计算 SPICE")
    args = parser.parse_args()

    print(f"Loading predictions from: {args.preds}")
    preds = _load_preds_file(args.preds)
    print(f"Loading references from: {args.refs}")
    refs = _load_refs_file(args.refs)

    # 仅对公共 id 进行评测
    pred_ids = {pid for pid, _ in preds}
    ref_ids = {rid for rid, _ in refs}
    common_ids = pred_ids & ref_ids

    preds_aligned = [(pid, p) for pid, p in preds if pid in common_ids]
    refs_aligned = [(rid, r) for rid, r in refs if rid in common_ids]

    print(f"Found {len(common_ids)} common IDs for evaluation.")

    metrics = compute_all_metrics(preds_aligned, refs_aligned, with_spice=not args.no_spice)

    print("\n===== Evaluation Results =====")
    for k, v in metrics.items():
        try:
            print(f"{k}: {v:.4f}")
        except Exception:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
