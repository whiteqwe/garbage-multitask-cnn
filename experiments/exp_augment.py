"""
实验3: 数据增强对比 (Exp3)
比较 有数据增强 vs 无数据增强 对泛化能力的影响
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import subprocess
from utils.visualize import plot_experiment_comparison, plot_multi_loss_curves


EXP_NAME = "exp3_augment"
RESULTS_DIR = "results"


def run_augment_experiment(use_augmentation):
    mode = "augment" if use_augmentation else "no_augment"
    exp_name = f"{EXP_NAME}_{mode}"
    print(f"\n{'='*50}")
    print(f"运行实验: {mode}")
    print(f"{'='*50}")

    cmd = [
        "python", "train.py",
        "--mode", "single",
        "--backbone", "cnn",
        "--epochs", "10",
        "--batch_size", "32",
        "--exp_name", exp_name,
    ]
    if not use_augmentation:
        cmd.append("--no_augmentation")

    print(f"启动训练: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    print(f"训练完成, 返回码: {result.returncode}")

    if result.returncode != 0:
        print(f"实验失败 ({mode})")
        return None, None

    import torch
    ckpt_path = f"checkpoints/{exp_name}_best.pth"
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        val_acc = ckpt.get("best_val_acc", None)
        return None, val_acc
    except Exception as e:
        print(f"  读取checkpoint失败: {e}")
    return None, None


def main():
    print("=" * 60)
    print("实验3: 数据增强对比 (有增强 vs 无增强)")
    print("=" * 60)

    test_results = {}
    val_results = {}

    for use_aug in [True, False]:
        test_acc, val_acc = run_augment_experiment(use_aug)
        if test_acc is not None:
            label = "With Augmentation" if use_aug else "Without Augmentation"
            test_results[label] = test_acc
            if val_acc:
                val_results[label] = val_acc

    if test_results:
        print(f"\n测试集准确率: {test_results}")
        plot_experiment_comparison(
            test_results,
            metric_name="Test Accuracy (%)",
            save_path=f"{RESULTS_DIR}/{EXP_NAME}_comparison.png",
        )

        best = max(test_results, key=test_results.get)
        print(f"\n结论: {'有数据增强' if 'With' in best else '无数据增强'} 表现更好 "
              f"(准确率: {test_results[best]:.2f}%)")
    else:
        print("所有实验均失败")


if __name__ == "__main__":
    main()
