"""
实验2: Dropout 比例对比 (Exp2)
比较 dropout = {0, 0.3, 0.5, 0.7} 对过拟合的影响
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import subprocess
from utils.visualize import plot_experiment_comparison


DROPOUT_RATES = [0.0, 0.3, 0.5, 0.7]
EXP_NAME = "exp2_dropout"
RESULTS_DIR = "results"


def run_dropout_experiment(dropout):
    exp_name = f"{EXP_NAME}_drop{dropout}"
    print(f"\n{'='*50}")
    print(f"运行实验: dropout = {dropout}")
    print(f"{'='*50}")

    cmd = [
        "python", "train.py",
        "--mode", "single",
        "--backbone", "cnn",
        "--dropout", str(dropout),
        "--epochs", "10",
        "--batch_size", "32",
        "--exp_name", exp_name,
    ]
    print(f"启动训练: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    print(f"训练完成, 返回码: {result.returncode}")

    if result.returncode != 0:
        print(f"实验失败 (dropout={dropout})")
        return None

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
    print("实验2: Dropout 比例对比 (dropout = 0, 0.3, 0.5, 0.7)")
    print("=" * 60)

    results = {}
    for d in DROPOUT_RATES:
        acc = run_dropout_experiment(d)
        if acc is not None:
            results[f"dropout={d}"] = acc

    if results:
        print(f"\n实验结果汇总: {results}")
        plot_experiment_comparison(
            results,
            metric_name="Test Accuracy (%)",
            save_path=f"{RESULTS_DIR}/{EXP_NAME}_comparison.png",
        )

        best_d = max(results, key=results.get)
        print(f"\n最佳 Dropout: {best_d} (准确率: {results[best_d]:.2f}%)")
    else:
        print("所有实验均失败")


if __name__ == "__main__":
    main()
