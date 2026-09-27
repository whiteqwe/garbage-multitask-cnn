"""
将瓶子照片整合到数据集中，并生成液体标注 CSV

用法:
    python utils/prepare_liquid_data.py

说明:
    将 "照片/" 下的4类瓶子照片复制到 data/liquid_images/
    并按 7:2:1 划分 train/val/test，生成 liquid_annotation.csv
"""

import os
import sys
import random
import shutil
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PHOTO_DIR

SOURCE_DIR = str(PHOTO_DIR)

# 目标目录（在项目内）
LIQUID_DIR = "data/liquid_images"
CSV_PATH = "data/liquid_annotation.csv"

# 液体标签映射: 有空=0, 有液体=1
LIQUID_MAP = {
    "多": 1,   # 水很多 → 有液体
    "中": 1,   # 中等   → 有液体
    "少": 1,   # 少量   → 有液体
    "空": 0,   # 空瓶   → 无液体
}

# 划分比例
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.2, 0.1


def collect_photos():
    """收集所有照片"""
    photos = []  # [(类别文件夹名, 文件名, 液体标签), ...]
    for folder_name, liquid_label in LIQUID_MAP.items():
        folder_path = os.path.join(SOURCE_DIR, folder_name)
        if not os.path.exists(folder_path):
            print(f"警告: 目录不存在 {folder_path}")
            continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                photos.append((folder_name, fname, liquid_label))
    return photos


def split_photos(photos):
    """按比例划分 train/val/test"""
    random.seed(42)
    # 按液体标签分层采样
    liquid_photos = [p for p in photos if p[2] == 1]
    empty_photos = [p for p in photos if p[2] == 0]

    def split_group(group):
        random.shuffle(group)
        n = len(group)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        train = group[:n_train]
        val = group[n_train:n_train + n_val]
        test = group[n_train + n_val:]
        return train, val, test

    train_l, val_l, test_l = split_group(liquid_photos)
    train_e, val_e, test_e = split_group(empty_photos)

    splits = {
        "train": train_l + train_e,
        "val": val_l + val_e,
        "test": test_l + test_e,
    }
    for split_name in splits:
        random.shuffle(splits[split_name])

    print(f"划分结果: train={len(splits['train'])}, "
          f"val={len(splits['val'])}, test={len(splits['test'])}")
    return splits


def copy_photos(splits, liquid_dir):
    """复制照片到目标目录"""
    # 清理旧目录
    if os.path.exists(liquid_dir):
        shutil.rmtree(liquid_dir)

    csv_rows = []
    for split_name, photo_list in splits.items():
        for folder_name, fname, liquid_label in photo_list:
            # 保存到 liquid_images/{split}/plastic/
            dest_dir = os.path.join(liquid_dir, split_name, "plastic")
            os.makedirs(dest_dir, exist_ok=True)

            src_path = os.path.join(SOURCE_DIR, folder_name, fname)
            # 重命名避免冲突
            new_fname = f"bottle_{folder_name}_{fname}"
            dest_path = os.path.join(dest_dir, new_fname)

            shutil.copy2(src_path, dest_path)

            # CSV 中记录相对路径
            rel_path = os.path.join(split_name, "plastic", new_fname).replace("\\", "/")
            csv_rows.append({"filename": rel_path, "liquid": liquid_label})

    return csv_rows


def main():
    print("=" * 50)
    print("准备液体检测数据")
    print("=" * 50)

    # 1. 收集照片
    photos = collect_photos()
    print(f"共找到 {len(photos)} 张瓶子照片")
    liquid_count = sum(1 for p in photos if p[2] == 1)
    empty_count = sum(1 for p in photos if p[2] == 0)
    print(f"  有液体: {liquid_count}张, 空瓶: {empty_count}张")

    # 2. 划分
    splits = split_photos(photos)

    # 3. 复制
    csv_rows = copy_photos(splits, LIQUID_DIR)

    # 4. 写 CSV
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "liquid"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"标注文件已生成: {CSV_PATH} (共 {len(csv_rows)} 条)")

    print("\n✅ 完成! 现在可以运行多任务训练了:")
    print("  python train.py --mode multi --liquid_csv data/liquid_annotation.csv --epochs 50")


if __name__ == "__main__":
    main()
