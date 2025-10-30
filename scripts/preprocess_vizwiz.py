import json, argparse, os

def load_json(p):
    with open(p,'r',encoding='utf-8') as f:
        return json.load(f)

def build_split(split, ann, img_root):
    # ann: COCO风格，含 images[] 和 annotations[]
    images = {im['id']: im for im in ann['images']}
    cap_map = {}
    for a in ann['annotations']:
        if a.get('is_rejected', False):   # 与官方评测一致：剔除 rejected
            continue
        if a.get('is_precanned', False):  # 剔除套话式描述
            continue
        cap_map.setdefault(a['image_id'], []).append(a['caption'])

    out = []
    for img_id, im in images.items():
        caps = cap_map.get(img_id, [])
        if not caps:      # 没有可用caption就跳过
            continue
        out.append({
            "image": os.path.join(img_root, split, im['file_name']),
            "image_id": img_id,
            "captions": caps,
            "text_detected": im.get('text_detected', False),
            "split": split
        })
    return out

def main(args):
    ann_train = load_json(os.path.join(args.ann_dir, 'train.json'))
    ann_val   = load_json(os.path.join(args.ann_dir, 'val.json'))

    train = build_split('train', ann_train, args.img_root)
    val   = build_split('val',   ann_val,   args.img_root)

    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    with open(args.out_train,'w',encoding='utf-8') as f: json.dump(train, f, ensure_ascii=False, indent=2)
    with open(args.out_val,'w',encoding='utf-8') as f:   json.dump(val,   f, ensure_ascii=False, indent=2)
    print(f"[OK] train:{len(train)}  val:{len(val)}  (expected ~23431/~7750, 允许少量差异)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann_dir', required=True)
    ap.add_argument('--img_root', required=True)
    ap.add_argument('--out_train', required=True)
    ap.add_argument('--out_val', required=True)
    args = ap.parse_args()
    main(args)
