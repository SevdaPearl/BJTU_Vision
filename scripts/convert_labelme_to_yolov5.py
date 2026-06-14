from pathlib import Path
import argparse
import json
import random
import shutil
from PIL import Image
import yaml
from tqdm import tqdm

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def list_images(root: Path):
    return sorted([p for p in root.rglob('*') if p.suffix.lower() in IMAGE_EXTS])


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_image_for_json(json_path: Path, data: dict, image_map: dict):
    candidates = []
    if data.get('imagePath'):
        candidates.append(Path(data['imagePath']).stem.lower())
    candidates.append(json_path.stem.lower())

    for stem in candidates:
        if stem in image_map:
            return image_map[stem]

    if data.get('imagePath'):
        p = json_path.parent / data['imagePath']
        if p.exists():
            return p

    return None


def shape_to_yolo_bbox(shape: dict, image_w: int, image_h: int):
    points = shape.get('points', [])
    if not points:
        return None

    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]

    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(image_w - 1), max(xs))
    y2 = min(float(image_h - 1), max(ys))

    if x2 <= x1 or y2 <= y1:
        return None

    cx = ((x1 + x2) / 2.0) / image_w
    cy = ((y1 + y2) / 2.0) / image_h
    bw = (x2 - x1) / image_w
    bh = (y2 - y1) / image_h

    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)

    return 0, cx, cy, bw, bh


def unique_target_path(dst_dir: Path, src_path: Path):
    target = dst_dir / src_path.name
    if not target.exists():
        return target

    stem = src_path.stem
    suffix = src_path.suffix
    i = 1
    while True:
        target = dst_dir / f'{stem}_{i}{suffix}'
        if not target.exists():
            return target
        i += 1


def convert(data_dir: Path, out_dir: Path, val_ratio: float, seed: int):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)

    image_map = {p.stem.lower(): p for p in list_images(data_dir)}
    json_paths = sorted(data_dir.rglob('*.json'))

    samples = []
    skipped = []

    for jp in tqdm(json_paths, desc='Reading LabelMe JSON'):
        try:
            data = load_json(jp)
        except Exception as e:
            skipped.append({'json_path': str(jp), 'reason': f'json_read_error: {e}'})
            continue

        img_path = find_image_for_json(jp, data, image_map)
        if img_path is None:
            skipped.append({'json_path': str(jp), 'reason': 'matched_image_not_found'})
            continue

        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as e:
            skipped.append({'json_path': str(jp), 'image_path': str(img_path), 'reason': f'image_read_error: {e}'})
            continue

        lines = []
        for shape in data.get('shapes', []):
            bbox = shape_to_yolo_bbox(shape, w, h)
            if bbox is None:
                continue
            cls_id, cx, cy, bw, bh = bbox
            lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')

        if not lines:
            skipped.append({'json_path': str(jp), 'image_path': str(img_path), 'reason': 'no_valid_boxes'})
            continue

        samples.append({'image_path': img_path, 'json_path': jp, 'label_lines': lines})

    if not samples:
        raise RuntimeError('No valid LabelMe samples were converted. Please check object_detection/data.')

    random.seed(seed)
    random.shuffle(samples)

    n_val = max(1, int(len(samples) * val_ratio)) if len(samples) >= 5 else 1
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    if not train_samples:
        train_samples, val_samples = samples, samples[:1]

    for split in ['train', 'val']:
        (out_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    def write_split(split_name, split_samples):
        rows = []
        for s in tqdm(split_samples, desc=f'Writing {split_name}'):
            img_src = Path(s['image_path'])
            img_dst = unique_target_path(out_dir / 'images' / split_name, img_src)
            shutil.copy2(img_src, img_dst)

            label_dst = out_dir / 'labels' / split_name / f'{img_dst.stem}.txt'
            label_dst.write_text('\n'.join(s['label_lines']) + '\n', encoding='utf-8')

            rows.append({
                'split': split_name,
                'image_path': str(img_dst),
                'label_path': str(label_dst),
                'source_image': str(img_src),
                'source_json': str(s['json_path']),
                'num_boxes': len(s['label_lines'])
            })
        return rows

    rows = []
    rows.extend(write_split('train', train_samples))
    rows.extend(write_split('val', val_samples))

    data_yaml = {
        'path': str(out_dir.resolve()).replace('\\', '/'),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['text']
    }

    yaml_path = out_dir / 'text_yolov5.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out_dir / 'converted_samples.csv', index=False, encoding='utf-8-sig')
        if skipped:
            pd.DataFrame(skipped).to_csv(out_dir / 'skipped_labelme_files.csv', index=False, encoding='utf-8-sig')
    except Exception:
        pass

    print('Done.')
    print(f'Train samples: {len(train_samples)}')
    print(f'Val samples: {len(val_samples)}')
    print(f'Data YAML: {yaml_path}')
    print(f'Skipped files: {len(skipped)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='object_detection/data')
    parser.add_argument('--out-dir', type=str, default='datasets/text_yolo')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    convert(Path(args.data_dir), Path(args.out_dir), args.val_ratio, args.seed)
