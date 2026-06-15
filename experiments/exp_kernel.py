"""
实验1: 卷积核大小对比 (Exp1)
比较 kernel_size = {3, 5, 7} 对模型性能的影响
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import subprocess
from utils.visualize import plot_experiment_comparison


KERNEL_SIZES = [3, 5, 7]
EXP_NAME = "exp1_kernel"
RESULTS_DIR = "results"


def run_kernel_experiment(kernel_size):
    """训练指定卷积核大小的模型并返回测试准确率"""
    exp_name = f"{EXP_NAME}_k{kernel_size}"
    print(f"\n{'='*50}")
    print(f"运行实验: kernel_size = {kernel_size}")
    print(f"{'='*50}")

    cmd = [
        "python", "train.py",
        "--mode", "single",
        "--backbone", "cnn",
        "--kernel_size", str(kernel_size),
        "--epochs", "10",
        "--batch_size", "32",
        "--exp_name", exp_name,
    ]
    print(f"启动训练: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    print(f"训练完成, 返回码: {result.returncode}")

    if result.returncode != 0:
        print(f"实验失败 (kernel={kernel_size})")
        return None

    # 从 checkpoint 读取最佳验证准确率
    import torch
    ckpt_path = f"checkpoints/{exp_name}_best.pth"
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        acc = ckpt.get("best_val_acc", None)
        if acc is not None:
            return acc
    except Exception as e:
        print(f"  读取checkpoint失败: {e}")
    return None


def main():
    print("=" * 60)
    print("实验1: 卷积核大小对比 (kernel_size = 3, 5, 7)")
    print("=" * 60)

    results = {}
    for k in KERNEL_SIZES:
        acc = run_kernel_experiment(k)
        if acc is not None:
            results[f"kernel={k}"] = acc

    if results:
        print(f"\n实验结果汇总: {results}")
        plot_experiment_comparison(
            results,
            metric_name="Test Accuracy (%)",
            save_path=f"{RESULTS_DIR}/{EXP_NAME}_comparison.png",
        )

        best_k = max(results, key=results.get)
        print(f"\n最佳卷积核大小: {best_k} (准确率: {results[best_k]:.2f}%)")
    else:
        print("所有实验均失败")


if __name__ == "__main__":
    main()
