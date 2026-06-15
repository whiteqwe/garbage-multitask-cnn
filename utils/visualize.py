"""
可视化工具 — 绘制训练曲线、混淆矩阵、对比图表
"""

import os
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，兼容服务器
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config import RESULTS_DIR


# 中文字体设置（尝试常见中文字体）
for font_name in ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]:
    try:
        plt.rcParams["font.sans-serif"] = [font_name]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue


def plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         save_path=str(RESULTS_DIR / "training_curves.png"), title=None):
    """绘制训练/验证 Loss 和 Accuracy 曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    # Loss 曲线
    axes[0].plot(epochs, train_losses, "b-", label="Train Loss", linewidth=1.5)
    axes[0].plot(epochs, val_losses, "r-", label="Val Loss", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve" if not title else f"{title} - Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy 曲线
    axes[1].plot(epochs, train_accs, "b-", label="Train Acc", linewidth=1.5)
    axes[1].plot(epochs, val_accs, "r-", label="Val Acc", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy Curve" if not title else f"{title} - Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 训练曲线已保存至 {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names,
                          save_path=str(RESULTS_DIR / "confusion_matrix.png"), title=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title or "Confusion Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 混淆矩阵已保存至 {save_path}")

    # 计算各类别精度
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)
    class_accs = cm_norm.diagonal()
    print("\n各类别准确率:")
    for name, acc in zip(class_names, class_accs):
        print(f"  {name}: {acc:.2%}")


def plot_experiment_comparison(results, metric_name="Test Accuracy (%)",
                                save_path="results/experiment_comparison.png"):
    """
    绘制对比实验结果（柱状图）

    参数:
        results: dict, {实验名称: 指标值}
        metric_name: y轴标签
    """
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    values = list(results.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    bars = plt.bar(names, values, color=colors, width=0.6)
    plt.ylabel(metric_name)
    plt.title(f"Experiment Comparison — {metric_name}")
    plt.xticks(rotation=30, ha="right")

    # 在柱子上标注数值
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 对比实验图已保存至 {save_path}")


def plot_liquid_confusion(y_true, y_pred, save_path="results/liquid_cm.png"):
    """绘制液体检测的混淆矩阵（二分类）"""
    plot_confusion_matrix(
        y_true, y_pred,
        class_names=["Empty", "Has Liquid"],
        save_path=save_path,
        title="Liquid Detection Confusion Matrix",
    )


def plot_multi_loss_curves(all_curves, save_path="results/multi_curves.png", title=None):
    """在一个图上绘制多条 Loss 曲线（用于对比多个实验）"""
    plt.figure(figsize=(10, 6))
    for label, losses in all_curves.items():
        plt.plot(range(1, len(losses) + 1), losses, label=label, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title or "Loss Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 多条曲线对比图已保存至 {save_path}")


def plot_task_curves(train_task_hist, val_task_hist,
                     save_path=str(RESULTS_DIR / "task_curves.png"), title=None):
    """绘制各任务分支的验证准确率曲线"""
    tasks = list(train_task_hist.keys())
    if not tasks:
        return
    n = len(tasks)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, task_name in enumerate(tasks):
        epochs = range(1, len(train_task_hist[task_name]) + 1)
        axes[i].plot(epochs, train_task_hist[task_name], "b-", label="Train", linewidth=1.5)
        axes[i].plot(epochs, val_task_hist[task_name], "r-", label="Val", linewidth=1.5)
        axes[i].set_title(task_name)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Accuracy (%)")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title or "Task Branch Accuracy")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 任务分支曲线已保存至 {save_path}")


def plot_model_comparison(cnn_acc, yolo_acc, cnn_param, yolo_param,
                          save_path=str(RESULTS_DIR / "model_comparison.png")):
    """YOLOv8 vs 自建CNN 模型对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 准确率对比
    models = ["自建CNN", "YOLOv8"]
    accs = [cnn_acc, yolo_acc]
    colors = ["#2196F3", "#FF9800"]
    bars = axes[0].bar(models, accs, color=colors, width=0.5)
    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{acc:.1f}%", ha="center", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Test Accuracy")
    axes[0].set_ylim(0, 105)

    # 参数量对比
    params = [cnn_param / 1e4, yolo_param / 1e4]
    bars2 = axes[1].bar(models, params, color=colors, width=0.5)
    for bar, p in zip(bars2, params):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f"{p:.0f}万", ha="center", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Parameters (万)")
    axes[1].set_title("Model Size")

    plt.suptitle("自建CNN vs YOLOv8 对比")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 模型对比图已保存至 {save_path}")


if __name__ == "__main__":
    plot_training_curves(
        [1.2, 0.8, 0.5, 0.3], [1.3, 0.9, 0.6, 0.4],
        [50, 70, 82, 88], [48, 68, 80, 86],
    )
    plot_experiment_comparison(
        {"kernel=3": 85.2, "kernel=5": 83.1, "kernel=7": 79.8},
    )
    plot_task_curves(
        {"liquid": [60, 70, 75], "flatten": [65, 70, 72], "spread": [70, 75, 80], "is_bottle": [80, 88, 92]},
        {"liquid": [58, 68, 72], "flatten": [60, 67, 70], "spread": [68, 73, 78], "is_bottle": [78, 85, 90]},
    )
    plot_model_comparison(75.0, 90.6, 51e4, 270e4)
    print("可视化测试完成")
