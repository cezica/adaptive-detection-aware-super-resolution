import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = Path(r"C:\Users\cezica\Downloads\coco2017\coco2017")

TRAIN_IMAGES_DIR = DATA_DIR / "train2017"
VAL_IMAGES_DIR = DATA_DIR / "val2017"

TRAIN_ANN_PATH = DATA_DIR / "annotations" / "instances_train2017.json"
VAL_ANN_PATH = DATA_DIR / "annotations" / "instances_val2017.json"

OUT_TRAIN_IMG = DATA_DIR / "coco_da_train" / "images"
OUT_TRAIN_LBL = DATA_DIR / "coco_da_train" / "labels"

OUT_VAL_IMG = DATA_DIR / "coco_da_val" / "images"
OUT_VAL_LBL = DATA_DIR / "coco_da_val" / "labels"

NUM_TRAIN_IMAGES = 1000
NUM_VAL_IMAGES = 300

SEED = 42

# COCO category_id -> YOLO COCO80 class_id
COCO91_TO_COCO80 = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17,
    20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25,
    31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33,
    39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
    48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57,
    64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
    76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73,
    85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}

# clase utile pentru detecție unde SR poate conta
# person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign,
# bird, cat, dog, horse, sheep, cow, boat
PREFERRED_COCO80_CLASSES = {
    0, 1, 2, 3, 5, 7, 9, 11, 14, 15, 16, 17, 18, 19, 8
}


def reset_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    for p in path.iterdir():
        if p.is_file():
            p.unlink()


def coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox

    if w <= 1 or h <= 1:
        return None

    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    bw = w / img_w
    bh = h / img_h

    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)

    if bw <= 0 or bh <= 0:
        return None

    return xc, yc, bw, bh


def prepare_split(
    images_dir: Path,
    ann_path: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    num_images: int,
    split_name: str,
):
    print(f"\n=== Preparing {split_name} ===")
    print("Images:", images_dir)
    print("Annotations:", ann_path)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    reset_dir(out_img_dir)
    reset_dir(out_lbl_dir)

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    anns_by_img = defaultdict(list)

    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue

        cat_id = ann["category_id"]
        if cat_id not in COCO91_TO_COCO80:
            continue

        yolo_cls = COCO91_TO_COCO80[cat_id]

        # păstrăm mai ales clase relevante pentru detection
        if yolo_cls not in PREFERRED_COCO80_CLASSES:
            continue

        img_id = ann["image_id"]
        if img_id not in images:
            continue

        img_info = images[img_id]
        img_w = img_info["width"]
        img_h = img_info["height"]

        converted = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
        if converted is None:
            continue

        anns_by_img[img_id].append((yolo_cls, *converted))

    # păstrăm doar imagini care au cel puțin o adnotare relevantă
    candidate_ids = list(anns_by_img.keys())

    # sortăm preferând imagini cu obiecte mici/medii
    # fiindcă acolo SR are șanse să conteze
    def score_image(img_id):
        img_info = images[img_id]
        img_area = img_info["width"] * img_info["height"]

        anns = anns_by_img[img_id]
        small_medium = 0
        total = len(anns)

        for _, _, _, bw, bh in anns:
            box_area_ratio = bw * bh
            if box_area_ratio < 0.08:
                small_medium += 1

        return (small_medium, total)

    candidate_ids = sorted(candidate_ids, key=score_image, reverse=True)

    random.seed(SEED)
    top_pool = candidate_ids[: max(num_images * 3, num_images)]
    random.shuffle(top_pool)

    selected_ids = top_pool[:num_images]

    copied = 0
    total_labels = 0

    for img_id in selected_ids:
        img_info = images[img_id]
        file_name = img_info["file_name"]

        src_img = images_dir / file_name
        if not src_img.exists():
            continue

        dst_img = out_img_dir / file_name
        dst_lbl = out_lbl_dir / (Path(file_name).stem + ".txt")

        shutil.copy2(src_img, dst_img)

        lines = []
        for cls_id, xc, yc, bw, bh in anns_by_img[img_id]:
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        copied += 1
        total_labels += len(lines)

    print(f"Selected images requested: {num_images}")
    print(f"Copied images: {copied}")
    print(f"Total labels: {total_labels}")
    print(f"Output images: {out_img_dir}")
    print(f"Output labels: {out_lbl_dir}")


def main():
    prepare_split(
        images_dir=TRAIN_IMAGES_DIR,
        ann_path=TRAIN_ANN_PATH,
        out_img_dir=OUT_TRAIN_IMG,
        out_lbl_dir=OUT_TRAIN_LBL,
        num_images=NUM_TRAIN_IMAGES,
        split_name="COCO train2017 subset",
    )

    prepare_split(
        images_dir=VAL_IMAGES_DIR,
        ann_path=VAL_ANN_PATH,
        out_img_dir=OUT_VAL_IMG,
        out_lbl_dir=OUT_VAL_LBL,
        num_images=NUM_VAL_IMAGES,
        split_name="COCO val2017 subset",
    )

    print("\nDone.")
    print("Use these paths in detection-aware training:")
    print("train_hr_dir = data/coco_da_train/images")
    print("train_label_dir = data/coco_da_train/labels")
    print("val_hr_dir = data/coco_da_val/images")
    print("val_label_dir = data/coco_da_val/labels")


if __name__ == "__main__":
    main()