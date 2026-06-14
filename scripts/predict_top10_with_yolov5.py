from pathlib import Path
import argparse
import sys
import math

import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
from tqdm import tqdm


LANDMARKS = ["fhy", "jx", "kx", "mh", "nm", "sjz", "sy", "tsg", "ty", "yf", "yk", "zx"]


def get_landmark_label(path):
    name = Path(path).stem.lower()
    for label in LANDMARKS:
        if name == label or name.startswith(label + "_") or name.startswith(label + "-"):
            return label
    return "irrelevant"


def read_image_rgb(path):
    path = Path(path)
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception:
        return None


def load_yolov5_model(yolov5_dir, weights, device=""):
    yolov5_dir = Path(yolov5_dir)
    weights = Path(weights)

    if not yolov5_dir.exists():
        raise FileNotFoundError(f"YOLOv5 dir not found: {yolov5_dir}")
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    # 加入 YOLOv5 源码路径，方便 torch.hub 本地加载
    sys.path.insert(0, str(yolov5_dir))

    if device:
        model = torch.hub.load(
            str(yolov5_dir),
            "custom",
            path=str(weights),
            source="local",
            device=device
        )
    else:
        model = torch.hub.load(
            str(yolov5_dir),
            "custom",
            path=str(weights),
            source="local"
        )

    model.conf = 0.15       # 置信度阈值，调低一点，避免框不显示
    model.iou = 0.45        # NMS IoU 阈值
    model.max_det = 100     # 每张图最多检测框数量
    return model


def predict_and_draw(model, image_path, conf_thres=0.15):
    """
    返回画好预测框的 PIL 图像和检测框数量。
    """
    image_path = Path(image_path)
    img = read_image_rgb(image_path)

    if img is None:
        pil = Image.new("RGB", (320, 240), "white")
        draw = ImageDraw.Draw(pil)
        draw.text((10, 10), "image read error", fill=(255, 0, 0))
        return pil, 0

    # YOLOv5 model 可以直接吃 numpy RGB 图
    results = model(img)

    # results.xyxy[0]: x1, y1, x2, y2, conf, cls
    pred = results.xyxy[0].detach().cpu().numpy()

    pil = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(pil)

    box_count = 0

    for row in pred:
        x1, y1, x2, y2, conf, cls_id = row[:6]

        if conf < conf_thres:
            continue

        box_count += 1

        x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])

        # 红框画文字检测结果
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text(
            (x1 + 2, max(0, y1 - 16)),
            f"text {conf:.2f}",
            fill=(255, 0, 0)
        )

    return pil, box_count


def make_top10_detection_grid(model, query_path, retrieved_rows, out_path, topk=10, conf_thres=0.15):
    """
    生成一张 Query + Top10 的检测可视化图。
    """
    items = [{"role": "Query", "path": query_path}]

    for _, row in retrieved_rows.head(topk).iterrows():
        items.append({
            "role": f"Top{int(row['rank'])}",
            "path": row["base_path"],
            "base_label": row.get("base_label", ""),
            "relevant": bool(row.get("relevant", False))
        })

    thumb_size = (260, 180)
    title_h = 54

    # Query + Top10 = 11 张图，每行 4 张，三行展示
    cols = 4
    rows = math.ceil(len(items) / cols)

    canvas = Image.new(
        "RGB",
        (cols * thumb_size[0], rows * (thumb_size[1] + title_h)),
        "white"
    )
    draw = ImageDraw.Draw(canvas)

    record_rows = []

    for i, item in enumerate(items):
        role = item["role"]
        p = Path(item["path"])

        boxed_img, box_count = predict_and_draw(model, p, conf_thres=conf_thres)
        boxed_img = ImageOps.contain(boxed_img, thumb_size)

        tile = Image.new("RGB", thumb_size, "white")
        tile.paste(
            boxed_img,
            ((thumb_size[0] - boxed_img.width) // 2, (thumb_size[1] - boxed_img.height) // 2)
        )

        col = i % cols
        row = i // cols

        x = col * thumb_size[0]
        y = row * (thumb_size[1] + title_h)

        canvas.paste(tile, (x, y + title_h))

        label = get_landmark_label(p)

        if role == "Query":
            title_color = (0, 0, 0)
            relevant = ""
        else:
            relevant_bool = item.get("relevant", False)
            title_color = (0, 128, 0) if relevant_bool else (180, 0, 0)
            relevant = "same" if relevant_bool else "diff"

        draw.text((x + 5, y + 4), f"{role} | {label} | boxes={box_count}", fill=title_color)
        draw.text((x + 5, y + 23), p.name[:34], fill=(0, 0, 0))

        record_rows.append({
            "role": role,
            "image_path": str(p),
            "label": label,
            "box_count": box_count,
            "visualization_path": str(out_path),
            "retrieval_check": relevant,
            "manual_result": "",
            "manual_note": ""
        })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

    return record_rows


def generate_24_groups(model, topk_df, out_dir, cases_per_label=2, topk=10, conf_thres=0.15):
    out_dir = Path(out_dir)
    visual_dir = out_dir / "visual_groups"
    visual_dir.mkdir(parents=True, exist_ok=True)

    review_rows = []

    for label in LANDMARKS:
        q_paths = (
            topk_df[topk_df["query_label"] == label]["query_path"]
            .drop_duplicates()
            .tolist()
        )

        for case_idx, q in enumerate(q_paths[:cases_per_label], start=1):
            sub = topk_df[topk_df["query_path"] == q].sort_values("rank")

            out_path = visual_dir / f"{label}_case{case_idx:02d}_top{topk}_yolov5_detection.png"

            recs = make_top10_detection_grid(
                model=model,
                query_path=q,
                retrieved_rows=sub,
                out_path=out_path,
                topk=topk,
                conf_thres=conf_thres
            )

            for rec in recs:
                rec["case_id"] = f"{label}_case{case_idx:02d}"
                rec["query_path"] = str(q)
                rec["query_label"] = label
                review_rows.append(rec)

    review_df = pd.DataFrame(review_rows)
    review_csv = out_dir / "manual_review_yolov5.csv"
    review_df.to_csv(review_csv, index=False, encoding="utf-8-sig")

    return review_df, review_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolov5-dir", required=True, help="YOLOv5 source code directory")
    parser.add_argument("--weights", required=True, help="trained best.pt path")
    parser.add_argument("--retrieval-csv", required=True, help="retrieval_topk.csv path")
    parser.add_argument("--out-dir", default="outputs/yolov5_text_detection_groups")
    parser.add_argument("--cases-per-label", type=int, default=2)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--device", default="", help="0 for GPU, cpu for CPU, empty for auto")
    args = parser.parse_args()

    topk_df = pd.read_csv(args.retrieval_csv)

    print("Loading YOLOv5 model...")
    model = load_yolov5_model(
        yolov5_dir=args.yolov5_dir,
        weights=args.weights,
        device=args.device
    )

    print("Generating Query + Top10 detection visualizations...")
    review_df, review_csv = generate_24_groups(
        model=model,
        topk_df=topk_df,
        out_dir=args.out_dir,
        cases_per_label=args.cases_per_label,
        topk=args.topk,
        conf_thres=args.conf
    )

    print("Generated visual rows:", len(review_df))
    print("Manual review CSV:", review_csv)
    print("Visual groups:", Path(args.out_dir) / "visual_groups")


if __name__ == "__main__":
    main()