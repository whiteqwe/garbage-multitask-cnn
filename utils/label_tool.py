"""
数据标注工具 — 为新增属性生成标注CSV

用法:
    # 交互模式: 逐张显示图片，按 0/1/s 标注
    python utils/label_tool.py interactive --image_dir data/Garbage_Dataset/train/plastic --task flatten

    # 批量模式: 文件夹名对应标签
    python utils/label_tool.py batch --image_dir 照片/压扁 --task flatten --label 1
    python utils/label_tool.py batch --image_dir 照片/正常 --task flatten --label 0

    # 合并多个CSV
    python utils/label_tool.py merge --output data/flatten_annotation.csv --inputs part1.csv part2.csv
"""

import os
import csv
import argparse
import sys

# 加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.dataset import TASK_COLUMN_NAMES


def interactive_mode(image_dir, output, task, data_root):
    """
    交互模式：显示图片路径，让用户输入 0/1/s

    注意：在无GUI环境（如远程服务器）无法显示图片，只能通过路径名判断
    """
    rows = []
    image_files = []
    for root, _, files in os.walk(image_dir):
        for fname in sorted(files):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, fname))

    print(f"共找到 {len(image_files)} 张图片")
    print(f"任务: {task} (CSV列名: {TASK_COLUMN_NAMES.get(task, task)})")
    print("输入: 0=否, 1=是, s=跳过, q=退出\n")

    for idx, img_path in enumerate(image_files):
        rel_path = os.path.relpath(img_path, data_root).replace("\\", "/")

        print(f"[{idx+1}/{len(image_files)}] {rel_path}")
        val = input("  标签 (0/1/s/q): ").strip()

        if val == 'q':
            break
        elif val == 's':
            continue
        elif val in ('0', '1'):
            rows.append({"filename": rel_path, TASK_COLUMN_NAMES.get(task, task): int(val)})
            print(f"  → 已标注: {val}")
        else:
            print("  跳过（无效输入）")

    # 写入CSV
    write_csv(output, rows, task)
    print(f"\n已标注 {len(rows)} 张图片，保存至 {output}")


def batch_mode(image_dir, output, task, label, data_root):
    """
    批量模式：将目录下所有图片标注为指定标签
    """
    rows = []
    for root, _, files in os.walk(image_dir):
        for fname in sorted(files):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, data_root).replace("\\", "/")
                rows.append({"filename": rel_path, TASK_COLUMN_NAMES.get(task, task): label})

    write_csv(output, rows, task)
    print(f"批量标注完成: 共 {len(rows)} 张, 标签={label}, 已保存至 {output}")


def merge_mode(output, inputs):
    """合并多个CSV文件"""
    all_rows = {}
    for csv_path in inputs:
        if not os.path.exists(csv_path):
            print(f"警告: {csv_path} 不存在，跳过")
            continue
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fn = row["filename"]
                if fn not in all_rows:
                    all_rows[fn] = row
                else:
                    all_rows[fn].update(row)

    if not all_rows:
        print("没有可合并的数据")
        return

    fieldnames = ["filename"]
    for row in all_rows.values():
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows.values())

    print(f"合并完成: {len(all_rows)} 行, 已保存至 {output}")
    print(f"列: {fieldnames}")


def write_csv(output, rows, task):
    """写入CSV"""
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    col_name = TASK_COLUMN_NAMES.get(task, task)
    with open(output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", col_name])
        writer.writeheader()
        writer.writerows(rows)


def list_images(image_dir):
    """列出目录下的图片文件"""
    files = []
    for root, _, fnames in os.walk(image_dir):
        for fname in sorted(fnames):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                files.append(os.path.join(root, fname))
    return files


def main():
    parser = argparse.ArgumentParser(description="属性标注工具")
    subparsers = parser.add_subparsers(dest="mode", help="模式")

    # 交互模式
    p_interactive = subparsers.add_parser("interactive", help="交互式标注")
    p_interactive.add_argument("--image_dir", required=True, help="图片目录")
    p_interactive.add_argument("--output", default=None, help="输出CSV路径")
    p_interactive.add_argument("--task", required=True, choices=["flatten", "spread", "liquid", "is_bottle"], help="标注任务")
    p_interactive.add_argument("--data_root", default="data/Garbage_Dataset", help="数据集根目录（用于计算相对路径）")

    # 批量模式
    p_batch = subparsers.add_parser("batch", help="批量标注")
    p_batch.add_argument("--image_dir", required=True, help="图片目录")
    p_batch.add_argument("--output", default=None, help="输出CSV路径")
    p_batch.add_argument("--task", required=True, choices=["flatten", "spread", "liquid", "is_bottle"], help="标注任务")
    p_batch.add_argument("--label", type=int, required=True, choices=[0, 1], help="标签值")
    p_batch.add_argument("--data_root", default="data/Garbage_Dataset", help="数据集根目录")

    # 合并模式
    p_merge = subparsers.add_parser("merge", help="合并多个CSV")
    p_merge.add_argument("--output", required=True, help="输出CSV路径")
    p_merge.add_argument("--inputs", nargs="+", required=True, help="输入CSV文件列表")

    args = parser.parse_args()

    if args.mode == "interactive":
        output = args.output or f"data/{args.task}_annotation.csv"
        interactive_mode(args.image_dir, output, args.task, args.data_root)
    elif args.mode == "batch":
        output = args.output or f"data/{args.task}_annotation.csv"
        batch_mode(args.image_dir, output, args.task, args.label, args.data_root)
    elif args.mode == "merge":
        merge_mode(args.output, args.inputs)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
