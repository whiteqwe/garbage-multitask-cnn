"""
YOLOv8 分类模型训练脚本

用法:
    python models/yolo_train.py --data_root data/Garbage_Dataset --epochs 20
"""

import os
import sys
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from ultralytics import YOLO
from config import DATA_DIR, CHECKPOINT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 classification")
    parser.add_argument("--data_root", default=str(DATA_DIR / "Garbage_Dataset"),
                        help="数据集根目录（需包含 train/val/test 子目录）")
    parser.add_argument("--model", default="yolov8n-cls.pt",
                        help="YOLOv8 预训练模型")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default=None,
                        help="设备: cuda / cpu / cuda:0 (默认自动检测)")
    parser.add_argument("--save_dir", default=str(CHECKPOINT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"YOLOv8 分类训练")
    print(f"模型: {args.model} | 数据集: {args.data_root}")
    print(f"设备: {args.device} | Epochs: {args.epochs}")
    print("=" * 60)

    # 加载预训练模型
    model = YOLO(args.model)

    # 训练
    results = model.train(
        data=args.data_root,
        task="classify",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        lr0=args.lr,
        device=args.device,
        project=args.save_dir,
        name="yolov8_cls",
        exist_ok=True,
        verbose=True,
        mosaic=0.0,        # 禁掉，分类任务不需要
        auto_augment="",   # 禁掉，太激进
    )

    print(f"\n训练完成! 模型已保存至 {args.save_dir}/yolov8_cls/")

    # 验证
    val_results = model.val()
    print(f"\n验证集准确率: Top-1 = {val_results.top1:.2f}%")

    # 导出为 ONNX（可选）
    model.export(format="onnx", imgsz=args.imgsz)
    print(f"模型已导出为 ONNX 格式")


if __name__ == "__main__":
    main()
