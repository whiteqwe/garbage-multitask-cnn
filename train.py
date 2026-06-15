"""
主训练脚本 — 支持单任务和多任务训练

用法:
    python train.py --mode single --epochs 50 --batch_size 32
    python train.py --mode multi --liquid_csv data/liquid_annotation.csv
    python train.py --mode multi --tasks liquid flatten spread \\
        --annotation_csvs liquid=data/liquid_annotation.csv flatten=data/flatten_annotation.csv
"""

import os
import argparse
import time
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, CLASS_NAMES
from models.cnn_model import MultiTaskCNN, count_parameters
from utils.dataset import (
    GarbageDataset, get_train_transform, get_val_transform,
    get_no_augment_transform, task_labels_collate,
)
from utils.visualize import plot_training_curves, plot_confusion_matrix, plot_task_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train garbage classification CNN")
    parser.add_argument("--data_root", default=str(DATA_DIR / "Garbage_Dataset"))
    parser.add_argument("--mode", choices=["single", "multi"], default="multi")
    parser.add_argument("--liquid_csv", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--annotation_csvs", nargs="*", default=None)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--fc_dim", type=int, default=512)
    parser.add_argument("--backbone", choices=["cnn", "resnet18"], default="resnet18",
                        help="骨干网络: cnn(自建) / resnet18(预训练)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--task_loss_weight", type=float, default=0.2,
                        help="任务分支损失权重 (0=禁用多任务)")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal Loss gamma (0=标准交叉熵, >0加重难样本)")
    parser.add_argument("--mixup_warmup", type=int, default=5,
                        help="MixUp 延迟开启的 epoch 数 (0=从头开启)")
    parser.add_argument("--class_boost", nargs="*", default=["paper:1.5", "glass:1.3"],
                        help="加大指定类别的权重，如 \"paper:2.0\" \"glass:1.5\"")
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--save_dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--results_dir", default=str(RESULTS_DIR))
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None,
                        help="从 checkpoint 恢复训练 (路径)")
    parser.add_argument("--device", default="auto",
                        help="训练设备: auto / cuda / cpu / cuda:0")
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    """Focal Loss — 对易分样本降权，让模型更关注难分类的样本

    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    gamma=0 → 标准交叉熵；gamma=2 是论文推荐值
    """
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha,
                                               reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_loss_and_metrics(model, images, cls_labels, task_labels_dict,
                             criterion_cls, task_criteria, device,
                             task_loss_weight=0.3):
    """公共前向 + 损失计算  (train/validate 共用)"""
    images = images.to(device)
    cls_labels = cls_labels.to(device)
    logits, task_logits = model(images)

    loss = criterion_cls(logits, cls_labels)
    _, preds = logits.max(1)

    metrics = {
        "cls_correct": preds.eq(cls_labels).sum().item(),
        "total": cls_labels.size(0),
        "preds": preds,
    }

    for task_name, t_logits in task_logits.items():
        t_labels = task_labels_dict[task_name].to(device)
        mask = t_labels != -1
        if mask.sum() > 0:
            t_loss = task_criteria[task_name](t_logits[mask], t_labels[mask])
            loss = loss + task_loss_weight * t_loss
            _, t_preds = t_logits.max(1)
            metrics[f"{task_name}_correct"] = t_preds[mask].eq(t_labels[mask]).sum().item()
            metrics[f"{task_name}_total"] = mask.sum().item()

    return loss, logits, metrics


def mixup_data(x, alpha=0.2):
    """MixUp: 返回混合后的图像和混合比例"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], index, lam


def train_one_epoch(model, loader, criterion_cls, task_criteria, optimizer, device,
                    epoch=1, task_loss_weight=0.2, mixup_alpha=0.2, mixup_warmup=5):
    model.train()
    total_loss = 0.0
    total_samples = 0
    cls_correct = 0
    task_correct = {t: 0 for t in task_criteria}
    task_total = {t: 0 for t in task_criteria}

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, cls_labels, task_labels_dict in pbar:
        images = images.to(device)
        cls_labels = cls_labels.to(device)

        optimizer.zero_grad()

        # MixUp: 前 mixup_warmup 个 epoch 不开启，之后 50% 概率
        do_mixup = mixup_alpha > 0 and epoch > mixup_warmup and torch.rand(1).item() < 0.5
        if do_mixup:
            mixed, idx, lam = mixup_data(images, mixup_alpha)
            logits, _ = model(mixed)
            loss = lam * criterion_cls(logits, cls_labels) + (1 - lam) * criterion_cls(logits, cls_labels[idx])
            _, preds = logits.max(1)
            cls_correct += (lam * preds.eq(cls_labels).float() +
                           (1 - lam) * preds.eq(cls_labels[idx]).float()).sum().item()
            total_samples += images.size(0)
        else:
            loss, logits, metrics = compute_loss_and_metrics(
                model, images, cls_labels, task_labels_dict,
                criterion_cls, task_criteria, device, task_loss_weight,
            )
            cls_correct += metrics["cls_correct"]
            total_samples += metrics["total"]
            for t in task_criteria:
                kc, kt = f"{t}_correct", f"{t}_total"
                if kc in metrics:
                    task_correct[t] += metrics[kc]
                    task_total[t] += metrics[kt]

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    cls_acc = 100.0 * cls_correct / total_samples if total_samples > 0 else 0.0
    task_accs = {}
    for t in task_criteria:
        task_accs[t] = 100.0 * task_correct[t] / task_total[t] if task_total[t] > 0 else 0.0
    return avg_loss, cls_acc, task_accs


@torch.no_grad()
def validate(model, loader, criterion_cls, task_criteria, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    cls_correct = 0
    task_correct = {t: 0 for t in task_criteria}
    task_total = {t: 0 for t in task_criteria}
    all_preds, all_labels = [], []

    for images, cls_labels, task_labels_dict in tqdm(loader, desc="Val", leave=False):
        loss, logits, metrics = compute_loss_and_metrics(
            model, images, cls_labels, task_labels_dict,
            criterion_cls, task_criteria, device, task_loss_weight=0.3,
        )
        total_loss += loss.item() * metrics["total"]
        total_samples += metrics["total"]
        cls_correct += metrics["cls_correct"]
        all_preds.extend(metrics["preds"].cpu().numpy())
        all_labels.extend(cls_labels.cpu().numpy())
        for t in task_criteria:
            kc = f"{t}_correct"
            kt = f"{t}_total"
            if kc in metrics:
                task_correct[t] += metrics[kc]
                task_total[t] += metrics[kt]

    avg_loss = total_loss / total_samples
    cls_acc = 100.0 * cls_correct / total_samples
    task_accs = {}
    for t in task_criteria:
        task_accs[t] = 100.0 * task_correct[t] / task_total[t] if task_total[t] > 0 else 0.0
    return avg_loss, cls_acc, task_accs, np.array(all_preds), np.array(all_labels)


def main():
    args = parse_args()
    set_seed(args.seed)

    # 设备选择
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"设备: {device}")

    # 解析任务
    tasks, annotation_csvs = [], {}
    if args.annotation_csvs:
        for pair in args.annotation_csvs:
            task_name, csv_path = pair.split("=", 1)
            annotation_csvs[task_name] = csv_path
            tasks.append(task_name)
    if args.mode == "multi" and args.liquid_csv:
        annotation_csvs.setdefault("liquid", args.liquid_csv)
        if "liquid" not in tasks:
            tasks.append("liquid")
    if args.tasks is not None:
        tasks = args.tasks
    # 默认全任务: 没指定任何参数时自动加载四个CSV
    if not tasks and not annotation_csvs and args.mode == "multi":
        tasks = ["is_bottle", "liquid", "flatten", "spread"]
        annotation_csvs = {
            "is_bottle": str(DATA_DIR / "is_bottle_annotation.csv"),
            "liquid": str(DATA_DIR / "liquid_annotation.csv"),
            "flatten": str(DATA_DIR / "flatten_annotation.csv"),
            "spread": str(DATA_DIR / "spread_annotation.csv"),
        }

    multi_task = len(tasks) > 0
    task_suffix = "_".join(tasks) if tasks else ""
    use_aug = not args.no_augmentation
    exp_name = args.exp_name or f"kernel{args.kernel_size}_drop{args.dropout}_{args.mode}"
    if task_suffix:
        exp_name = f"{exp_name}_{task_suffix}"
    print(f"实验: {exp_name} | 模式: {'多任务('+','.join(tasks)+')' if multi_task else '单任务'}")

    # 数据加载
    train_transform = get_train_transform() if use_aug else get_no_augment_transform()
    val_transform = get_val_transform()
    csvs = annotation_csvs if annotation_csvs else None
    collate_fn = task_labels_collate if csvs else None

    train_set = GarbageDataset(args.data_root, "train", transform=train_transform, annotation_csvs=csvs)
    val_set   = GarbageDataset(args.data_root, "val",   transform=val_transform,  annotation_csvs=csvs)
    test_set  = GarbageDataset(args.data_root, "test",  transform=val_transform,  annotation_csvs=csvs)

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True,  num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader  = DataLoader(test_set,  args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(f"训练集: {len(train_set)} | 验证集: {len(val_set)} | 测试集: {len(test_set)}")

    # 类别权重 (直接从 samples 取标签，不加载图片)
    label_counts = Counter(s[1] for s in train_set.samples)
    print(f"各类别样本数: {dict(sorted(label_counts.items()))}")
    total = sum(label_counts.values())
    num_cls = args.num_classes
    class_weights = torch.tensor(
        [total / (num_cls * max(label_counts.get(i, 1), 1)) for i in range(num_cls)],
        dtype=torch.float,
    )

    # 手动提升指定类别的补偿值
    if args.class_boost:
        from config import CLASS_NAMES as CN
        name_to_idx = {n: i for i, n in enumerate(CN)}
        for spec in args.class_boost:
            name, factor = spec.split(":")
            idx = name_to_idx.get(name)
            if idx is not None:
                old = class_weights[idx].item()
                class_weights[idx] = old * float(factor)
                print(f"  类别 '{name}' 权重: {old:.4f} → {class_weights[idx].item():.4f} (x{factor})")

    class_weights = class_weights.to(device)

    # 模型
    model = MultiTaskCNN(
        num_classes=args.num_classes, kernel_size=args.kernel_size,
        dropout=args.dropout, fc_dim=args.fc_dim,
        multi_task=multi_task, tasks=tasks if tasks else None,
        backbone=args.backbone, pretrained=True,
    ).to(device)
    print(f"参数量: {count_parameters(model):,}")

    criterion_cls = FocalLoss(gamma=args.focal_gamma, alpha=class_weights)

    # 任务分支也加 class_weight + Focal Loss（flatten 正负比 1:8）
    task_criteria = nn.ModuleDict()
    task_focal_gamma = min(args.focal_gamma * 0.75, 1.5)  # 任务分支用稍低的 gamma
    if tasks and hasattr(train_set, 'task_labels') and train_set.task_labels:
        for t in tasks:
            t_label_counts = Counter(v for v in train_set.task_labels.get(t, {}).values() if v in (0, 1))
            if len(t_label_counts) == 2:
                total_t = sum(t_label_counts.values())
                t_weights = torch.tensor(
                    [total_t / (2 * max(t_label_counts.get(i, 1), 1)) for i in range(2)],
                    dtype=torch.float,
                ).to(device)
                task_criteria[t] = FocalLoss(gamma=task_focal_gamma, alpha=t_weights)
            else:
                task_criteria[t] = FocalLoss(gamma=task_focal_gamma)
    else:
        for t in tasks:
            task_criteria[t] = FocalLoss(gamma=task_focal_gamma)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=8, factor=0.5,
    )

    # 断点续训
    start_epoch = 1
    best_val_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"从 {args.resume} 恢复训练 (epoch {start_epoch})")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_task_history = {t: [] for t in tasks}
    val_task_history = {t: [] for t in tasks}

    print(f"\n开始训练 {args.epochs} 个 epoch...")
    start_time = time.time()
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_cls_acc, train_task_accs = train_one_epoch(
            model, train_loader, criterion_cls, task_criteria, optimizer, device,
            epoch=epoch, task_loss_weight=args.task_loss_weight,
            mixup_warmup=args.mixup_warmup,
        )
        val_loss, val_cls_acc, val_task_accs, _, _ = validate(
            model, val_loader, criterion_cls, task_criteria, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_cls_acc)
        val_accs.append(val_cls_acc)
        for t in tasks:
            train_task_history[t].append(train_task_accs.get(t, 0))
            val_task_history[t].append(val_task_accs.get(t, 0))

        task_info = ""
        if multi_task:
            parts = [f"{t}: {train_task_accs.get(t,0):.1f}/{val_task_accs.get(t,0):.1f}" for t in tasks]
            task_info = f" | {' '.join(parts)}"

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
              f"TrainAcc: {train_cls_acc:.2f}% | ValAcc: {val_cls_acc:.2f}%{task_info} | "
              f"LR: {current_lr:.6f}")

        if val_cls_acc > best_val_acc:
            best_val_acc = val_cls_acc
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
            }
            torch.save(ckpt, os.path.join(args.save_dir, f"{exp_name}_best.pth"))
            print(f"  — 最佳模型已保存 (ValAcc: {best_val_acc:.2f}%)")

        # 每 10 epoch 保存恢复点
        if epoch % 10 == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
            }, os.path.join(args.save_dir, f"{exp_name}_latest.pth"))

        if current_lr < 1e-6:
            print("学习率过低，停止训练")
            break

    elapsed = time.time() - start_time
    print(f"\n训练完成! 耗时: {elapsed:.0f}s | 最佳验证准确率: {best_val_acc:.2f}%")

    # 保存曲线
    os.makedirs(args.results_dir, exist_ok=True)
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(args.results_dir, f"{exp_name}_curves.png"),
        title=exp_name,
    )

    # 测试集评估
    print("\n测试集评估...")
    _, test_acc, test_task_accs, test_preds, test_labels = validate(
        model, test_loader, criterion_cls, task_criteria, device)
    print(f"测试集准确率: {test_acc:.2f}%")
    for t in tasks:
        print(f"  {t} 检测准确率: {test_task_accs.get(t, 0):.2f}%")

    # 混淆矩阵
    plot_confusion_matrix(
        test_labels, test_preds, CLASS_NAMES,
        save_path=os.path.join(args.results_dir, f"{exp_name}_cm.png"),
        title=exp_name,
    )
    print(f"\n所有结果已保存至 {args.results_dir}/")


if __name__ == "__main__":
    main()
