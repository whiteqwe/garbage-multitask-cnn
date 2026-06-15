"""全局配置文件 — 路径与常量集中管理"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
PHOTO_DIR = BASE_DIR / "照片"

CLASS_NAMES = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash",
]

RECYCLING_MAP = {
    "battery":    "harmful",
    "biological": "kitchen",
    "cardboard":  "recyclable",
    "clothes":    "recyclable",
    "glass":      "recyclable",
    "metal":      "recyclable",
    "paper":      "recyclable",
    "plastic":    "recyclable",
    "shoes":      "other",
    "trash":      "other",
}

CATEGORY_CN = {
    "recyclable": "可回收物",
    "kitchen":    "厨余垃圾",
    "harmful":    "有害垃圾",
    "other":      "其他垃圾",
}

CAT_COLORS = {
    "recyclable": "#2E7D32",
    "kitchen":    "#E65100",
    "harmful":    "#C62828",
    "other":      "#546E7A",
}

TASK_COLUMN_NAMES = {
    "liquid":   "liquid",
    "flatten":  "flattened",
    "spread":   "spread",
    "is_bottle":"is_bottle",
}

# 预训练/checkpoint 默认路径
SINGLE_MODEL = CHECKPOINT_DIR / "kernel3_drop0.5_single_best.pth"
MULTI_MODEL  = CHECKPOINT_DIR / "kernel3_drop0.5_multi_best.pth"
FULL_MODEL   = CHECKPOINT_DIR / "kernel3_drop0.5_multi_is_bottle_liquid_flatten_spread_best.pth"
YOLO_MODEL   = BASE_DIR / "runs/classify/checkpoints/yolov8_cls/weights/best.pt"

# 图片归一化参数 (ImageNet)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]
