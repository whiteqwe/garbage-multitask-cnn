"""
数据集准备 — 拆分 GD 为 train/val/test + 整合瓶子照片 + 生成液体标注

用法: python setup_data.py

原始格式: data/Garbage_Dataset/standardized_256/{class_name}/*.jpg
目标格式: data/Garbage_Dataset/{train,val,test}/{class_name}/*.jpg
照片来源: 照片/{多,中,少,空}/*.jpg
"""

import os
import random
import shutil
import csv

from config import DATA_DIR, PHOTO_DIR, CLASS_NAMES

SOURCE_DIR = DATA_DIR / "Garbage_Dataset" / "standardized_256"
TARGET_DIR = DATA_DIR / "Garbage_Dataset"
LIQUID_CSV = DATA_DIR / "liquid_annotation.csv"

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
LIQUID_MAP = {"多": 1, "中": 1, "少": 1, "空": 0}

random.seed(42)


def setup_garbage_dataset():
    """拆分 GD 数据集为 train/val/test"""
    print("=" * 50)
    print("1. 拆分 Garbage Dataset 为 train/val/test")
    print("=" * 50)

    for split in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            os.makedirs(TARGET_DIR / split / cls, exist_ok=True)

    for cls in CLASS_NAMES:
        src_cls_dir = SOURCE_DIR / cls
        if not src_cls_dir.is_dir():
            print(f"  警告: {cls} 目录不存在 ({src_cls_dir})")
            continue

        images = sorted([f for f in os.listdir(str(src_cls_dir))
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }
        for split_name, split_images in splits.items():
            dst_dir = TARGET_DIR / split_name / cls
            for img in split_images:
                shutil.copy2(str(src_cls_dir / img), str(dst_dir / img))

        print(f"  {cls}: {n}张 -> train={len(splits['train'])}, "
              f"val={len(splits['val'])}, test={len(splits['test'])}")

    total = sum(len(os.listdir(str(TARGET_DIR / "train" / cls))) for cls in CLASS_NAMES)
    print(f"\n  [OK] 训练集总计: {total} 张")


def setup_liquid_data():
    """将瓶子照片复制到主数据集，生成液体标注 CSV"""
    print("\n" + "=" * 50)
    print("2. 整合瓶子照片 + 生成液体标注")
    print("=" * 50)

    if not PHOTO_DIR.is_dir():
        print(f"  跳过: 照片目录不存在 ({PHOTO_DIR})")
        return

    photos = []
    for folder_name, liquid_label in LIQUID_MAP.items():
        folder_path = PHOTO_DIR / folder_name
        if not folder_path.is_dir():
            continue
        for fname in sorted(os.listdir(str(folder_path))):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                photos.append((folder_name, fname, liquid_label))

    if not photos:
        print("  没有找到瓶子照片")
        return

    print(f"  瓶子照片: {len(photos)}张 (有液体={sum(1 for p in photos if p[2]==1)}, "
          f"空瓶={sum(1 for p in photos if p[2]==0)})")

    liquid_photos = [p for p in photos if p[2] == 1]
    empty_photos = [p for p in photos if p[2] == 0]

    def split_group(group):
        random.shuffle(group)
        n = len(group)
        n1, n2 = int(n * 0.7), int(n * 0.2)
        return group[:n1], group[n1:n1+n2], group[n1+n2:]

    csv_rows = []
    for split_name, liq_part, emp_part in zip(
        ["train", "val", "test"],
        split_group(liquid_photos),
        split_group(empty_photos),
    ):
        combined = list(liq_part) + list(emp_part)
        random.shuffle(combined)

        for folder_name, fname, liquid_label in combined:
            dst_dir = TARGET_DIR / split_name / "plastic"
            os.makedirs(str(dst_dir), exist_ok=True)
            new_fname = f"bottle_{folder_name}_{fname}"
            shutil.copy2(str(PHOTO_DIR / folder_name / fname),
                        str(dst_dir / new_fname))
            csv_rows.append({
                "filename": f"{split_name}/plastic/{new_fname}",
                "liquid": liquid_label,
            })

    os.makedirs(str(DATA_DIR), exist_ok=True)
    with open(str(LIQUID_CSV), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "liquid"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"  [OK] 已复制到 {TARGET_DIR}/{{train,val,test}}/plastic/")
    print(f"  [OK] 标注文件: {LIQUID_CSV} ({len(csv_rows)}条)")


def verify():
    """验证最终数据结构"""
    print("\n" + "=" * 50)
    print("3. 验证最终数据")
    print("=" * 50)

    ok = True
    for split in ["train", "val", "test"]:
        split_dir = TARGET_DIR / split
        if not split_dir.is_dir():
            print(f"  [FAIL] {split} 目录不存在")
            ok = False
            continue
        for cls in CLASS_NAMES:
            cls_dir = split_dir / cls
            if not cls_dir.is_dir():
                print(f"  [WARN] {split}/{cls}: 空!")
                ok = False
                continue
            count = len([f for f in os.listdir(str(cls_dir))
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            if count == 0:
                print(f"  [WARN] {split}/{cls}: 0张!")
                ok = False
            elif split == "train":
                print(f"  [OK] {split}/{cls}: {count}张")

    total_train = sum(
        len([f for f in os.listdir(str(TARGET_DIR / "train" / cls))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        for cls in CLASS_NAMES if (TARGET_DIR / "train" / cls).is_dir()
    )

    if LIQUID_CSV.exists():
        csv_count = sum(1 for _ in open(str(LIQUID_CSV))) - 1
        print(f"  [OK] 数据集就绪: train={total_train}张, liquid标注={csv_count}条")
    if not ok:
        print("\n  [WARN] 有异常，请检查数据集")


if __name__ == "__main__":
    print("慧眼 — 数据准备\n")
    setup_garbage_dataset()
    setup_liquid_data()
    verify()
    print("\n" + "=" * 50)
    print("[OK] 全部完成!")
    print("=" * 50)
    print("\n  python train.py --mode single  # 单任务训练")
    print("  python train.py --mode multi --liquid_csv data/liquid_annotation.csv  # 多任务")
    print("  python app.py  # 启动Web界面")
