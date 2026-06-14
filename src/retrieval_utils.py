from pathlib import Path
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
from tqdm import tqdm

LANDMARKS = ["fhy", "jx", "kx", "mh", "nm", "sjz", "sy", "tsg", "ty", "yf", "yk", "zx"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def get_landmark_label(path):
    """Extract label only for evaluation. Do not call this before retrieval ranking."""
    name = Path(path).stem.lower()
    for label in LANDMARKS:
        if name == label or name.startswith(label + "_") or name.startswith(label + "-"):
            return label
    return "irrelevant"


def is_relevant(query_path, base_path):
    q = get_landmark_label(query_path)
    b = get_landmark_label(base_path)
    return q == b and q in LANDMARKS


def read_image_cv(path, size=224):
    """Read image robustly. Return None for broken/empty/non-readable files."""
    path = Path(path)
    try:
        if not path.exists() or not path.is_file():
            return None
        if path.stat().st_size == 0:
            return None

        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None

        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return img

    except Exception:
        return None


def extract_global_feature(path, size=224):
    """Fast handcrafted feature: HSV histogram + grayscale thumbnail + edge histogram."""
    img = read_image_cv(path, size=size)
    if img is None:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [16, 8, 4],
        [0, 180, 0, 256, 0, 256]
    ).flatten()
    hist = hist / (np.linalg.norm(hist) + 1e-8)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    small = small.astype(np.float32).flatten() / 255.0
    small = small - small.mean()
    small = small / (np.linalg.norm(small) + 1e-8)

    edges = cv2.Canny(gray, 80, 180)
    edge_small = cv2.resize(edges, (32, 32), interpolation=cv2.INTER_AREA)
    edge_small = edge_small.astype(np.float32).flatten() / 255.0
    edge_small = edge_small - edge_small.mean()
    edge_small = edge_small / (np.linalg.norm(edge_small) + 1e-8)

    feat = np.concatenate([
        hist * 1.5,
        small * 0.8,
        edge_small * 0.8
    ]).astype(np.float32)

    feat = feat / (np.linalg.norm(feat) + 1e-8)
    return feat


def build_feature_matrix(image_paths, cache_path=None, force_rebuild=False):
    cache_path = Path(cache_path) if cache_path else None
    image_paths = [Path(p) for p in image_paths]

    if cache_path and cache_path.exists() and not force_rebuild:
        data = np.load(cache_path, allow_pickle=True)
        cached_paths = [Path(p) for p in data["paths"].tolist()]

        if [str(p) for p in cached_paths] == [str(p) for p in image_paths]:
            return cached_paths, data["features"].astype(np.float32)

    valid_paths = []
    skipped_paths = []
    feats = []

    for p in tqdm(image_paths, desc="Extracting image features"):
        f = extract_global_feature(p)

        if f is not None:
            valid_paths.append(p)
            feats.append(f)
        else:
            skipped_paths.append(p)

    if not feats:
        raise RuntimeError("No valid image features were extracted. Please check image paths.")

    mat = np.vstack(feats).astype(np.float32)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            cache_path,
            paths=np.array([str(p) for p in valid_paths], dtype=object),
            features=mat
        )

        if skipped_paths:
            skipped_csv = cache_path.parent / (cache_path.stem + "_skipped_images.csv")
            pd.DataFrame({
                "skipped_path": [str(p) for p in skipped_paths]
            }).to_csv(skipped_csv, index=False, encoding="utf-8-sig")

            print(f"Skipped {len(skipped_paths)} unreadable images. See: {skipped_csv}")

    return valid_paths, mat


def retrieve_topk(query_paths, query_features, base_paths, base_features, topk=60):
    rows = []
    base_T = base_features.T

    for q_path, q_feat in tqdm(list(zip(query_paths, query_features)), desc="Retrieving"):
        sims = q_feat @ base_T
        order = np.argsort(-sims)[:topk]
        q_label = get_landmark_label(q_path)

        for rank, idx in enumerate(order, start=1):
            b_path = base_paths[idx]
            b_label = get_landmark_label(b_path)

            rows.append({
                "query_path": str(q_path),
                "query_name": Path(q_path).name,
                "query_label": q_label,
                "rank": rank,
                "base_path": str(b_path),
                "base_name": Path(b_path).name,
                "base_label": b_label,
                "similarity": float(sims[idx]),
                "relevant": bool(q_label == b_label and q_label in LANDMARKS)
            })

    return pd.DataFrame(rows)


def compute_precision_tables(topk_df, ks=(20, 40, 60)):
    query_rows = []

    for q, group in topk_df.groupby("query_path", sort=False):
        group = group.sort_values("rank")

        row = {
            "query_path": q,
            "query_name": group["query_name"].iloc[0],
            "query_label": group["query_label"].iloc[0]
        }

        for k in ks:
            top = group[group["rank"] <= k]
            row[f"P@{k}"] = float(top["relevant"].mean()) if len(top) else 0.0

        query_rows.append(row)

    per_query = pd.DataFrame(query_rows)

    label_rows = []

    for label in LANDMARKS:
        sub = per_query[per_query["query_label"] == label]

        row = {
            "label": label,
            "num_queries": int(len(sub))
        }

        for k in ks:
            row[f"P@{k}"] = float(sub[f"P@{k}"].mean()) if len(sub) else 0.0

        label_rows.append(row)

    per_label = pd.DataFrame(label_rows)
    return per_query, per_label


def safe_open_pil(path, thumb_size=(180, 140)):
    try:
        img = Image.open(path).convert("RGB")
        img = ImageOps.contain(img, thumb_size)

        canvas = Image.new("RGB", thumb_size, "white")
        x = (thumb_size[0] - img.width) // 2
        y = (thumb_size[1] - img.height) // 2
        canvas.paste(img, (x, y))

        return canvas

    except Exception:
        canvas = Image.new("RGB", thumb_size, "white")
        d = ImageDraw.Draw(canvas)
        d.text((10, 10), "image error", fill=(255, 0, 0))
        return canvas


def make_topk_grid(topk_df, query_path, out_path, k=10):
    query_path = Path(query_path)
    group = topk_df[topk_df["query_path"] == str(query_path)].sort_values("rank").head(k)

    thumb_size = (180, 140)
    title_h = 38

    cols = min(k + 1, 6)
    rows = math.ceil((k + 1) / cols)

    canvas = Image.new(
        "RGB",
        (cols * thumb_size[0], rows * (thumb_size[1] + title_h)),
        "white"
    )
    draw = ImageDraw.Draw(canvas)

    items = [("Query", query_path, get_landmark_label(query_path), True)]

    for _, r in group.iterrows():
        items.append((
            f"Top{int(r['rank'])}",
            Path(r["base_path"]),
            r["base_label"],
            bool(r["relevant"])
        ))

    for i, (rank_txt, p, label, ok) in enumerate(items):
        c = i % cols
        r = i // cols

        x = c * thumb_size[0]
        y = r * (thumb_size[1] + title_h)

        img = safe_open_pil(p, thumb_size)
        canvas.paste(img, (x, y + title_h))

        color = (0, 128, 0) if ok else (180, 0, 0)
        if rank_txt == "Query":
            color = (0, 0, 0)

        draw.text((x + 5, y + 4), f"{rank_txt}: {label}", fill=color)
        draw.text((x + 5, y + 20), Path(p).name[:28], fill=(0, 0, 0))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def save_all_topk_grids(topk_df, out_dir, k=10, max_per_label=2):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = []

    for label in LANDMARKS:
        qs = topk_df[topk_df["query_label"] == label]["query_path"].drop_duplicates().tolist()

        for q in qs[:max_per_label]:
            selected.append(q)

    for q in tqdm(selected, desc="Saving TopK grids"):
        label = get_landmark_label(q)
        stem = Path(q).stem
        make_topk_grid(topk_df, q, out_dir / f"{label}_{stem}_top{k}.png", k=k)

    return selected


def save_pk_plots(per_query, per_label, out_dir, ks=(20, 40, 60)):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    made = []

    for label in LANDMARKS:
        sub = per_query[per_query["query_label"] == label]

        means = []

        for k in ks:
            means.append(float(sub[f"P@{k}"].mean()) if len(sub) else 0.0)

        plt.figure(figsize=(5, 3.2))

        x_labels = [f"P@{k}" for k in ks]
        x = list(range(len(ks)))

        plt.plot(x, means, marker="o", linewidth=1.5)
        plt.scatter(x, means, s=40)

        plt.xticks(x, x_labels)
        plt.ylim(0, 1)
        plt.title(f"{label} Precision@K")
        plt.xlabel("K")
        plt.ylabel("Precision")

        for i, v in enumerate(means):
            plt.text(i, min(v + 0.03, 0.98), f"{v:.3f}", ha="center", fontsize=9)

        plt.tight_layout()

        path = out_dir / f"{label}_pk.png"
        plt.savefig(path, dpi=160)
        plt.close()

        made.append(path)

    return made

def read_gray_for_orb(path, max_side=900):
    """Read image as grayscale for ORB local feature matching."""
    path = Path(path)
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None

        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None

        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        h, w = img.shape[:2]
        scale = max_side / max(h, w)

        if scale < 1:
            img = cv2.resize(
                img,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA
            )

        return img

    except Exception:
        return None


def compute_orb_descriptors(image_paths, nfeatures=1200):
    """Compute ORB descriptors for a list of images."""
    orb = cv2.ORB_create(nfeatures=nfeatures)
    desc_dict = {}

    for p in tqdm(image_paths, desc="Computing ORB descriptors"):
        img = read_gray_for_orb(p)
        if img is None:
            desc_dict[str(p)] = None
            continue

        kp, desc = orb.detectAndCompute(img, None)
        desc_dict[str(p)] = desc

    return desc_dict


def orb_similarity(desc1, desc2):
    """Compute a normalized ORB matching score between two images."""
    if desc1 is None or desc2 is None:
        return 0.0

    if len(desc1) < 2 or len(desc2) < 2:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    try:
        matches = bf.knnMatch(desc1, desc2, k=2)
    except Exception:
        return 0.0

    good = []

    for pair in matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good.append(m)

    # 80 是一个经验归一化值，不用标签，只是把匹配数量压到 0~1 范围
    score = min(len(good) / 80.0, 1.0)

    return float(score)


def retrieve_topk_rerank(
    query_paths,
    query_features,
    base_paths,
    base_features,
    first_stage_k=200,
    final_k=60,
    alpha=0.65
):
    """
    Two-stage retrieval:
    1. Global handcrafted feature retrieves first_stage_k candidates.
    2. ORB local feature reranks candidates.
    
    alpha controls the weight of global similarity.
    final_score = alpha * global_similarity + (1 - alpha) * orb_similarity
    """
    rows = []

    base_T = base_features.T

    print("Preparing ORB descriptors...")
    query_orb = compute_orb_descriptors(query_paths)
    base_orb = compute_orb_descriptors(base_paths)

    for q_path, q_feat in tqdm(list(zip(query_paths, query_features)), desc="Retrieving with ORB rerank"):
        sims = q_feat @ base_T

        candidate_order = np.argsort(-sims)[:first_stage_k]

        q_desc = query_orb.get(str(q_path), None)
        q_label = get_landmark_label(q_path)

        candidate_rows = []

        for idx in candidate_order:
            b_path = base_paths[idx]
            b_desc = base_orb.get(str(b_path), None)

            global_score = float(sims[idx])
            local_score = orb_similarity(q_desc, b_desc)

            final_score = alpha * global_score + (1 - alpha) * local_score

            b_label = get_landmark_label(b_path)

            candidate_rows.append({
                "query_path": str(q_path),
                "query_name": Path(q_path).name,
                "query_label": q_label,
                "base_path": str(b_path),
                "base_name": Path(b_path).name,
                "base_label": b_label,
                "global_similarity": global_score,
                "orb_similarity": local_score,
                "similarity": final_score,
                "relevant": bool(q_label == b_label and q_label in LANDMARKS)
            })

        candidate_rows = sorted(
            candidate_rows,
            key=lambda x: x["similarity"],
            reverse=True
        )[:final_k]

        for rank, item in enumerate(candidate_rows, start=1):
            item["rank"] = rank
            rows.append(item)

    return pd.DataFrame(rows)