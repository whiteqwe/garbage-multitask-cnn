"""
数据加载器 — 支持 Garbage Dataset (10类) + 多属性标注
"""

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import CLASS_NAMES, RECYCLING_MAP, CATEGORY_CN, \
    TASK_COLUMN_NAMES, NORM_MEAN, NORM_STD


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def get_no_augment_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


class GarbageDataset(Dataset):
    """Garbage Dataset 加载器 — 支持多属性标注"""

    def __init__(self, root_dir, split="train", transform=None,
                 annotation_csvs=None, liquid_csv=None, multi_task=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or get_val_transform()

        # 向后兼容
        if annotation_csvs is None:
            annotation_csvs = {}
        if liquid_csv is not None and multi_task:
            annotation_csvs.setdefault("liquid", liquid_csv)
        self.annotation_csvs = annotation_csvs
        self.task_names = list(annotation_csvs.keys())

        split_dir = os.path.join(root_dir, split)
        self.samples = []
        self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

        for class_name in CLASS_NAMES:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((
                        os.path.join(class_dir, fname),
                        self.class_to_idx[class_name],
                    ))

        self.task_labels = {}
        for task_name, csv_path in self.annotation_csvs.items():
            self.task_labels[task_name] = {}
            if csv_path and os.path.exists(csv_path):
                col_name = TASK_COLUMN_NAMES.get(task_name, task_name)
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    self.task_labels[task_name][row["filename"]] = int(row[col_name])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls_label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception:
            image = torch.zeros(3, 128, 128)
            cls_label = 0

        task_labels = {}
        if self.task_names:
            rel_path = os.path.relpath(img_path, self.root_dir).replace("\\", "/")
            for task_name in self.task_names:
                task_labels[task_name] = self.task_labels[task_name].get(rel_path, -1)

        return image, cls_label, task_labels


def task_labels_collate(batch):
    images = torch.stack([item[0] for item in batch])
    cls_labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

    task_labels_dict = {}
    first_task_labels = batch[0][2]
    if first_task_labels:
        for task_name in first_task_labels.keys():
            task_labels_dict[task_name] = torch.tensor(
                [item[2][task_name] for item in batch], dtype=torch.long
            )
    return images, cls_labels, task_labels_dict


def create_dataloaders(data_root, batch_size=32, num_workers=0,
                       annotation_csvs=None, liquid_csv=None,
                       multi_task=False, use_augmentation=True):
    if annotation_csvs is None:
        annotation_csvs = {}
    if liquid_csv is not None and multi_task:
        annotation_csvs.setdefault("liquid", liquid_csv)

    train_transform = get_train_transform() if use_augmentation else get_no_augment_transform()
    val_transform = get_val_transform()
    csvs = annotation_csvs if annotation_csvs else None
    collate_fn = task_labels_collate if csvs else None

    train_dataset = GarbageDataset(data_root, "train", transform=train_transform, annotation_csvs=csvs)
    val_dataset   = GarbageDataset(data_root, "val",   transform=val_transform,  annotation_csvs=csvs)
    test_dataset  = GarbageDataset(data_root, "test",  transform=val_transform,  annotation_csvs=csvs)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,  num_workers=num_workers, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    data_root = "data/Garbage_Dataset"
    csvs = {"liquid": "data/liquid_annotation.csv"}
    loader, _, _ = create_dataloaders(data_root, batch_size=4, annotation_csvs=csvs)
    images, cls_labels, task_labels = next(iter(loader))
    print(f"Batch 图像形状: {images.shape}")
    print(f"分类标签: {cls_labels}")
    print(f"属性标签: {task_labels}")
