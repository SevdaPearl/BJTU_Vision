from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
paths = {
    "image_retrieval/base/BJTU": ROOT / "image_retrieval" / "base" / "BJTU",
    "image_retrieval/base/util_pic": ROOT / "image_retrieval" / "base" / "util_pic",
    "image_retrieval/query": ROOT / "image_retrieval" / "query",
    "image_retrieval/base_statistics.json": ROOT / "image_retrieval" / "base_statistics.json",
    "image_retrieval/query_statistics.json": ROOT / "image_retrieval" / "query_statistics.json",
    "object_detection/data": ROOT / "object_detection" / "data",
}
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
for name, path in paths.items():
    print(f"{name}: {'OK' if path.exists() else 'MISSING'}")
    if path.is_dir():
        n_img = sum(1 for p in path.rglob("*") if p.suffix.lower() in image_exts)
        n_json = sum(1 for p in path.rglob("*.json"))
        print(f"  images={n_img}, json={n_json}")
