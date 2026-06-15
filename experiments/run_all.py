"""
一键运行所有对比实验 + 生成综合报告

用法:
    python experiments/run_all.py
"""

import os
import sys
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RESULTS_DIR

EXPERIMENTS = [
    ("卷积核大小对比", "python experiments/exp_kernel.py"),
    ("Dropout比例对比", "python experiments/exp_dropout.py"),
    ("数据增强对比",   "python experiments/exp_augment.py"),
]


def main():
    print("=" * 60)
    print("慧眼 — 一键对比实验")
    print("=" * 60)

    results = {}
    for name, cmd in EXPERIMENTS:
        print(f"\n{'─' * 50}")
        print(f"▶ {name}")
        print(f"{'─' * 50}")
        r = subprocess.run(cmd.split())
        if r.returncode != 0:
            print(f"[FAIL] {name} 返回码: {r.returncode}")
        else:
            results[name] = "✓"

    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    print(f"\n{'─' * 50}")
    print("实验结果汇总:")
    for name, status in results.items():
        print(f"  {status} {name}")
    print(f"\n图表已保存至: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
