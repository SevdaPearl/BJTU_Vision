from pathlib import Path
import json
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
from tqdm import tqdm

LANDMARKS = ["fhy", "jx", "kx", "mh", "nm", "sjz", "sy", "tsg", "ty", "yf", "yk", "zx"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def get_landmark_label(path):
    name = Path(path).stem.lower()
    for label in LANDMARKS:
        if name == label or name.startswith(label + "_") or name.startswith(label + "-"):
            return label
    return "irrelevant"


def list_images(root):
    root = Path(root)
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]) if root.exists() else []


def load_labelme_json(json_path):
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_labelme_index(data_dir):
    data_dir = Path(data_dir)
    index = {}
    if not data_dir.exists():
        return index
    image_paths = {p.stem.lower(): p for p in list_images(data_dir)}

    for jp in sorted(data_dir.rglob("*.json")):
        try:
            data = load_labelme_json(jp)
        except Exception:
            continue
        candidates = []
        if data.get("imagePath"):
            candidates.append(Path(data["imagePath"]).stem.lower())
        candidates.append(jp.stem.lower())
        matched_img = None
        for stem in candidates:
            if stem in image_paths:
                matched_img = image_paths[stem]
                break
        if matched_img is None and data.get("imagePath"):
            p = jp.parent / data["imagePath"]
            if p.exists():
                matched_img = p
        key = candidates[0]
        index[key] = {"json_path": jp, "image_path": matched_img, "data": data}
        index[jp.stem.lower()] = {"json_path": jp, "image_path": matched_img, "data": data}
    return index


def find_annotation_for_image(image_path, annotation_index):
    stem = Path(image_path).stem.lower()
    return annotation_index.get(stem)


def shape_to_bbox(shape):
    pts = np.array(shape.get("points", []), dtype=np.float32)
    if pts.size == 0:
        return None
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    return [float(x1), float(y1), float(x2), float(y2)]


def load_image_any(path):
    try:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def draw_labelme_boxes(image_path, annotation, out_path=None):
    image_path = Path(image_path)
    img = load_image_any(image_path)
    if img is None:
        return None, []
    h, w = img.shape[:2]
    boxes = []
    pil = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(pil)

    shapes = annotation.get("data", annotation).get("shapes", [])
    for shape in shapes:
        bbox = shape_to_bbox(shape)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
        label = str(shape.get("label", "text"))
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=max(2, w // 300))
        draw.text((x1 + 2, max(0, y1 - 14)), label, fill=(255, 0, 0))
        boxes.append({"label": label, "bbox": [x1, y1, x2, y2]})
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(out_path)
    return pil, boxes


def make_detection_panel(items, annotation_index, out_path, max_items=11):
    """Create one visual group: query + TopK retrieved images with LabelMe boxes if matched."""
    thumb_size = (260, 180)
    title_h = 52

    items = items[:max_items]

    if len(items) == 0:
        return []

    # Top10 时一共有 Query + 10 = 11 张图。每行放 4 张，自动换行。
    cols = 4
    rows = math.ceil(len(items) / cols)

    canvas = Image.new(
        "RGB",
        (cols * thumb_size[0], rows * (thumb_size[1] + title_h)),
        "white"
    )

    draw = ImageDraw.Draw(canvas)
    records = []

    for i, item in enumerate(items):
        role = item.get("role", "image")
        p = Path(item["path"])

        ann = find_annotation_for_image(p, annotation_index)
        shown_path = p
        has_ann = ann is not None
        box_count = 0

        if has_ann and ann.get("image_path") is not None and Path(ann["image_path"]).exists():
            # 使用 LabelMe json 对应的原图，避免尺寸不一致导致框偏移
            shown_path = Path(ann["image_path"])
            pil_boxed, boxes = draw_labelme_boxes(shown_path, ann)
            box_count = len(boxes)
        else:
            img = load_image_any(p)
            pil_boxed = Image.fromarray(img).convert("RGB") if img is not None else Image.new("RGB", thumb_size, "white")

        pil_boxed = ImageOps.contain(pil_boxed, thumb_size)

        tile = Image.new("RGB", thumb_size, "white")
        tile.paste(
            pil_boxed,
            ((thumb_size[0] - pil_boxed.width) // 2, (thumb_size[1] - pil_boxed.height) // 2)
        )

        col = i % cols
        row = i // cols

        x = col * thumb_size[0]
        y = row * (thumb_size[1] + title_h)

        canvas.paste(tile, (x, y + title_h))

        label = get_landmark_label(p)
        status = f"boxes={box_count}" if has_ann else "NO JSON"
        color = (0, 0, 0) if has_ann else (180, 0, 0)

        draw.text((x + 5, y + 4), f"{role} | {label} | {status}", fill=color)
        draw.text((x + 5, y + 22), p.name[:34], fill=(0, 0, 0))

        records.append({
            "role": role,
            "original_path": str(p),
            "shown_path": str(shown_path),
            "annotation_found": bool(has_ann),
            "box_count": int(box_count),
            "json_path": str(ann["json_path"]) if has_ann else ""
        })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

    return records


def generate_24_detection_groups(topk_df, data_dir, out_dir, cases_per_label=2, retrieved_per_case=10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = build_labelme_index(data_dir)
    review_rows = []
    for label in LANDMARKS:
        q_paths = topk_df[topk_df["query_label"] == label]["query_path"].drop_duplicates().tolist()
        for case_idx, q in enumerate(q_paths[:cases_per_label], start=1):
            sub = topk_df[topk_df["query_path"] == q].sort_values("rank")
            # Prefer relevant retrieved images, but fall back to top-ranked images if not enough.
            rel = sub[sub["relevant"] == True]["base_path"].tolist()
            top = sub["base_path"].tolist()
            chosen = []
            for p in rel + top:
                if p not in chosen:
                    chosen.append(p)
                if len(chosen) >= retrieved_per_case:
                    break
            items = [{"role": "Query", "path": q}]
            for j, p in enumerate(chosen, start=1):
                items.append({"role": f"Top{j}", "path": p})
            out_path = out_dir / f"{label}_case{case_idx:02d}_top{retrieved_per_case}_retrieval_detection.png"
            recs = make_detection_panel(items, index, out_path, max_items=1 + retrieved_per_case)
            for rec in recs:
                review_rows.append({
                    "case_id": f"{label}_case{case_idx:02d}",
                    "label": label,
                    "visualization_path": str(out_path),
                    **rec,
                    "manual_result": "",  # Fill manually: OK / missed_text / false_box / unclear
                    "manual_note": ""
                })
    review_df = pd.DataFrame(review_rows)
    review_csv = out_dir.parent / "manual_review.csv"
    review_df.to_csv(review_csv, index=False, encoding="utf-8-sig")
    return review_df, review_csv
