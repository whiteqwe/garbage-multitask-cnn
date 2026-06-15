"""
Gradio Web 界面 — 智能垃圾分类与回收建议系统
用法: python app.py
"""

import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import subprocess
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import gradio as gr

from config import (
    CLASS_NAMES, RECYCLING_MAP, CATEGORY_CN, CAT_COLORS,
    FULL_MODEL, YOLO_MODEL, DATA_DIR,
)
from models.cnn_model import MultiTaskCNN, remap_old_checkpoint
from utils.dataset import get_val_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10

CLASS_CN = {
    "battery":    "电池",
    "biological": "厨余",
    "cardboard":  "纸板",
    "clothes":    "衣物",
    "glass":      "玻璃",
    "metal":      "金属",
    "paper":      "纸张",
    "plastic":    "塑料",
    "shoes":      "鞋子",
    "trash":      "其他垃圾",
}


# ============ 回收建议 ============

def get_recycling_advice(class_name, task_preds=None):
    task_preds = task_preds or {}
    category = RECYCLING_MAP.get(class_name, "other")
    category_name = CATEGORY_CN.get(category, "其他垃圾")
    pre_check, steps, tips = [], [], []
    is_bottle = task_preds.get("is_bottle")
    liquid_pred = task_preds.get("liquid")
    flatten_pred = task_preds.get("flatten")
    spread_pred = task_preds.get("spread")

    if class_name == "plastic" and is_bottle == 1:
        if liquid_pred == 1:
            pre_check.append(("❌", "瓶内有液体 → 请先倒空"))
        elif liquid_pred == 0:
            pre_check.append(("✅", "瓶内已空"))
        if flatten_pred == 1:
            pre_check.append(("✅", "已压扁"))
        else:
            pre_check.append(("❌", "未压扁 → 请压扁后投放"))
        steps = [
            "① 倒空瓶内残留液体" if liquid_pred != 0 else "① 瓶内已空 ✓",
            "② 拧下瓶盖（PP材质，单独投入可回收物桶）",
            "③ 将瓶身压扁以节省空间" if flatten_pred != 1 else "③ 瓶身已压扁 ✓",
            "④ 投入可回收物桶",
        ]
        tips = ["瓶盖和瓶身材质不同（PP vs PET），建议分开回收", "压扁后投放可减少80%空间占用"]

    elif class_name == "plastic" and is_bottle == 0:
        steps = [
            "① 确认塑料制品保持清洁", "② 去除胶带、标签等非塑料部分",
            "③ 软塑料袋请打结后再投放", "④ 投入可回收物桶",
        ]
        tips = ["受污染的塑料（如油污餐盒）不属于可回收物"]

    elif class_name == "cardboard":
        if spread_pred == 1:
            pre_check.append(("✅", "纸箱已展平"))
        else:
            pre_check.append(("❌", "纸箱未展平 → 请拆开压平"))
        steps = [
            "① 拆开纸箱并压平" if spread_pred != 1 else "① 纸箱已展平 ✓",
            "② 去除胶带、塑料封皮、金属订书钉",
            "③ 保持纸箱干燥清洁", "④ 投入可回收物桶",
        ]
        tips = ["油污严重的纸箱（如披萨盒）不可回收"]

    elif class_name == "glass":
        steps = [
            "① 倒空瓶内液体",
            "② 去掉瓶盖和金属环", "③ 玻璃制品轻拿轻放，避免破碎",
            "④ 投入可回收物桶",
        ]
        tips = ["碎玻璃应用厚纸包好再投放，避免伤到清洁人员"]

    elif class_name == "metal":
        if flatten_pred == 0:
            pre_check.append(("❌", "未压扁 → 请压扁后投放"))
        elif flatten_pred == 1:
            pre_check.append(("✅", "已压扁"))
        steps = [
            "① 倒空残留液体",
            "② 将易拉罐压扁" if flatten_pred != 1 else "② 已压扁 ✓",
            "③ 投入可回收物桶",
        ]
        tips = ["铝制易拉罐可以100%循环再利用"]

    elif class_name == "paper":
        steps = ["① 保持纸张干燥清洁", "② 去除胶带、订书钉等非纸杂质",
                 "③ 平整叠放或捆扎好", "④ 投入可回收物桶"]
        tips = ["纸巾、湿巾、卫生纸不属于可回收纸张"]

    elif class_name == "battery":
        steps = ["① 确认电池外壳完好无破损", "② 用绝缘胶带包裹正负极（可选）",
                 "③ 单独装入塑料袋，避免与其他垃圾混放",
                 "④ 投入有害垃圾桶或专用回收箱"]
        tips = ["一粒纽扣电池可污染60万升水"]

    elif class_name == "biological":
        steps = ["① 沥干多余水分", "② 去除包装袋、橡皮筋等非厨余部分",
                 "③ 用报纸或可降解垃圾袋包好", "④ 投入厨余垃圾桶"]
        tips = ["大棒骨、贝壳等硬物不属于厨余垃圾"]

    elif class_name == "clothes":
        steps = ["① 干净衣物整理折叠后投入纺织物回收箱",
                 "② 破损严重的作为其他垃圾处理"]
        tips = ["旧衣物也可捐赠给公益组织"]

    elif class_name == "shoes":
        steps = ["① 用袋子装好投入回收桶", "② 完好可穿的鞋子可捐赠"]
        tips = ["建议清洗后再投放"]

    else:
        steps = ["① 直接投入其他垃圾桶"]

    return category_name, pre_check, steps, tips


# ============ Grad-CAM 可视化 ============

class GradCAMHook:
    def __init__(self, module):
        self.grad = None
        self.act = None
        self.hook_grad = module.register_full_backward_hook(self._backward_hook)
        self.hook_fwd  = module.register_forward_hook(self._forward_hook)

    def _backward_hook(self, _m, _g_in, grad_out):
        self.grad = grad_out[0].detach()

    def _forward_hook(self, _m, _in, output):
        self.act = output.detach()

    def remove(self):
        self.hook_grad.remove()
        self.hook_fwd.remove()


def generate_gradcam(model, image_tensor, target_layer):
    """生成 Grad-CAM 热力图"""
    hook = GradCAMHook(target_layer)
    model.zero_grad()

    logits, _ = model(image_tensor)
    pred = logits.max(1)[1].item()
    logits[0, pred].backward()

    weights = hook.grad.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    cam = (weights * hook.act).sum(dim=1, keepdim=True)   # [1, 1, H, W]
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    hook.remove()
    return cam.squeeze().cpu().numpy(), pred


def cam_overlay(image, cam):
    """将 Grad-CAM 热力图叠加到原图上 (返回 RGB)，热力图缩放到原图尺寸"""
    import cv2
    from PIL import Image as PILImage
    w, h = image.size
    # 将小尺寸热力图放大到原图分辨率
    cam_img = PILImage.fromarray(np.uint8(255 * cam)).resize((w, h), PILImage.BICUBIC)
    cam_np = np.array(cam_img)
    heatmap = cv2.applyColorMap(cam_np, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_np = np.array(image)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_rgb, 0.4, 0)
    return overlay


# ============ 模型加载 ============

def load_cnn_model(model_path, tasks=None):
    path = str(model_path)
    if not os.path.exists(path):
        print(f"警告: 模型文件不存在 ({path})，该模型不可用")
        return None
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    state_dict = ckpt.get("model", ckpt)

    # 自动检测 backbone: fc.weight 形状决定
    fc_w = state_dict.get("fc.weight")
    backbone = "cnn"
    if fc_w is not None and fc_w.dim() == 2:
        in_f, out_f = fc_w.size(1), fc_w.size(0)
        backbone = "resnet18" if (in_f == 512 and out_f == 512) else "cnn"

    model = MultiTaskCNN(num_classes=NUM_CLASSES, tasks=tasks, backbone=backbone)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception:
        state_dict = remap_old_checkpoint(state_dict)
        model.load_state_dict(state_dict, strict=False)
    print(f"模型已加载: {path} (backbone={backbone})")
    model.to(DEVICE).eval()
    return model


def load_yolo_model(model_path):
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        # 跑一次空图验证模型可用
        import numpy as np
        _ = model(np.zeros((224, 224, 3), dtype=np.uint8))
        print(f"YOLOv8 模型已加载: {model_path}")
        return model
    except Exception as e:
        print(f"YOLOv8 加载失败: {e}")
        return None


# ============ 推理函数 ============

def predict_cnn(image, model):
    transform = get_val_transform()
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, task_logits = model(img_tensor)
        cls_probs = F.softmax(logits, dim=1)[0]
        cls_pred = cls_probs.argmax().item()
        cls_conf = cls_probs[cls_pred].item()
        task_preds, task_confs = {}, {}
        for task_name, t_logits in task_logits.items():
            t_probs = F.softmax(t_logits, dim=1)[0]
            t_pred = t_probs.argmax().item()
            task_preds[task_name] = t_pred
            task_confs[task_name] = t_probs[t_pred].item()
    return cls_pred, cls_conf, task_preds, task_confs, cls_probs.cpu().numpy()


def predict_yolo(image, model):
    try:
        results = model(image)
        probs = results[0].probs
        cls_pred = probs.top1
        cls_conf = probs.top1conf.item()
        cls_probs = probs.data.cpu().numpy()
        return cls_pred, cls_conf, cls_probs
    except Exception as e:
        print(f"YOLOv8 推理失败: {e}")
        return 0, 0.0, np.zeros(NUM_CLASSES)


# ============ 端口管理 (跨平台) ============

def free_port(port):
    """查找并释放占用指定端口的进程"""
    import psutil
    freed = False
    try:
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.status == "LISTEN":
                pid = conn.pid
                proc = psutil.Process(pid)
                print(f"  [{port}] 发现进程 {proc.name()} (PID {pid})，正在停止...")
                proc.kill()
                proc.wait(timeout=3)
                freed = True
                print(f"  [{port}] 已释放")
    except Exception as e:
        print(f"  [{port}] psutil 不可用，尝试备用方案 ({e})")
        freed = _free_port_fallback(port)

    if not freed:
        print(f"  [{port}] 端口空闲")


def _free_port_fallback(port):
    """备用方案：通过命令行工具释放端口"""
    import platform
    try:
        if platform.system() == "Windows":
            # netstat -ano | findstr :PORT
            r = subprocess.run(
                f'netstat -ano | findstr ":{port}"',
                capture_output=True, text=True, timeout=5, shell=True,
            )
            for line in r.stdout.splitlines():
                if "LISTENING" in line:
                    parts = line.strip().split()
                    pid = parts[-1]
                    subprocess.run(['taskkill', '/PID', pid, '/F'],
                                 capture_output=True, timeout=5)
                    print(f"  [{port}] 已释放 (PID {pid})")
                    time.sleep(0.5)
                    return True
        else:
            r = subprocess.run(['lsof', '-ti', f':{port}'],
                             capture_output=True, text=True, timeout=5)
            pids = r.stdout.strip().split()
            for pid in pids:
                subprocess.run(['kill', '-9', pid], capture_output=True, timeout=5)
            if pids:
                print(f"  [{port}] 已释放")
                time.sleep(0.5)
                return True
    except Exception:
        pass
    return False


# ============ 前端样式 ============

CSS = """
.gradio-container {max-width: 1100px !important; margin: auto !important;
  font-family: -apple-system, 'Segoe UI', sans-serif !important;}
.result-card {background: white; border-radius: 12px; padding: 20px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 12px;}
.check-item {display: flex; align-items: center; gap: 8px; padding: 6px 0;
  font-size: 15px;}
.check-pass {color: #2E7D32;}
.check-fail {color: #C62828;}
.step-item {padding: 4px 0; font-size: 14px; color: #333;}
.tip-item {padding: 3px 0; font-size: 13px; color: #666;}
.category-badge {display: inline-block; padding: 3px 12px; border-radius: 20px;
  color: white; font-size: 13px; font-weight: 600;}
.status-ok {background: #E8F5E9; color: #2E7D32; padding: 6px 14px;
  border-radius: 20px; font-size: 13px; font-weight: 600;
  display: inline-block;}
.status-warn {background: #FFF3E0; color: #E65100; padding: 6px 14px;
  border-radius: 20px; font-size: 13px; font-weight: 600;
  display: inline-block;}
"""


# ============ 主推理解析入口 ============

def no_model_html(model_name):
    """统一的无模型错误提示"""
    msg = (
        f"<div style='text-align:center;padding:60px 20px;color:#C62828'>"
        f"<span style='font-size:32px'>⚠️</span><br><br>"
        f"<b style='font-size:18px'>模型「{model_name}」未训练</b><br><br>"
        f"<span style='color:#999;font-size:14px'>请先运行训练脚本：</span><br>"
        f"<code style='background:#FFF3E0;padding:4px 12px;border-radius:4px;font-size:12px'>"
        f"python train.py --mode multi --tasks is_bottle liquid flatten spread ...</code>"
        f"</div>"
    )
    return msg, gr.Image(value=None, visible=False), (
        "<span class='status-warn'>❌ 模型未训练</span>"
    )


def classify_image(image, model_choice, show_gradcam):
    if image is None:
        return (
            "<div style='text-align:center;padding:40px;color:#999'>请上传图片后点击「开始识别」</div>",
            gr.Image(value=None, visible=False),
            "<span class='status-warn'>⚠️ 请先上传图片</span>",
        )

    try:
        image_pil = Image.fromarray(image.astype("uint8"), "RGB")
    except Exception:
        return (
            "<div style='text-align:center;padding:40px;color:#C62828'>图片格式错误</div>",
            gr.Image(value=None, visible=False),
            "<span class='status-warn'>❌ 图片格式错误</span>",
        )

    task_preds, task_confs = {}, {}
    gradcam_img = None

    if "YOLOv8" in model_choice:
        if yolo_model is None:
            return no_model_html("YOLOv8")
        cls_pred, cls_conf, cls_probs = predict_yolo(image_pil, yolo_model)
    else:
        if full_model is None:
            return no_model_html("自建CNN (全任务)")
        cls_pred, cls_conf, task_preds, task_confs, cls_probs = predict_cnn(image_pil, full_model)

    class_name = CLASS_NAMES[cls_pred]
    is_bottle_pred = task_preds.get("is_bottle")

    display_name = CLASS_CN.get(class_name, class_name)
    if class_name == "plastic" and is_bottle_pred == 1:
        display_name = "塑料瓶"
    elif class_name == "plastic" and is_bottle_pred == 0:
        display_name = "塑料制品"

    category_name, pre_check, steps, tips = get_recycling_advice(class_name, task_preds)
    cat_color = CAT_COLORS.get(RECYCLING_MAP.get(class_name, "other"), "#546E7A")

    # --- 构建 HTML ---
    parts = []
    parts.append(
        f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:16px'>"
        f"<span style='font-size:32px'>🔄</span>"
        f"<span style='font-size:22px;font-weight:700'>{display_name}</span>"
        f"<span class='category-badge' style='background:{cat_color}'>{category_name}</span>"
        f"<span style='color:#999;font-size:14px;margin-left:auto'>置信度 {cls_conf:.1%}</span>"
        f"</div>"
    )

    # 属性检测 — 严格按训练数据覆盖范围
    # is_bottle: plastic/glass/metal, liquid: 仅plastic瓶, flatten: plastic瓶+metal, spread: 仅cardboard
    attr_items = []

    if "is_bottle" in task_preds and class_name in ("plastic", "glass", "metal"):
        pred = task_preds["is_bottle"]
        conf = task_confs["is_bottle"]
        names = {"plastic": ("塑料制品", "塑料瓶"), "glass": ("碎玻璃/玻璃板", "玻璃瓶"), "metal": ("金属件", "易拉罐")}
        text = names.get(class_name, ("非瓶", "瓶子"))[pred]
        attr_items.append(("📦", text, conf, "pass"))

    if "liquid" in task_preds and class_name == "plastic" and is_bottle_pred == 1:
        pred = task_preds["liquid"]
        conf = task_confs["liquid"]
        text = "空瓶" if pred == 0 else "有液体"
        state = "fail" if pred == 1 else "pass"
        attr_items.append(("💧", text, conf, state))

    flatten_relevant = (class_name == "plastic" and is_bottle_pred == 1) or class_name == "metal"
    if "flatten" in task_preds and flatten_relevant:
        pred = task_preds["flatten"]
        conf = task_confs["flatten"]
        text = "已压扁" if pred == 1 else "未压扁"
        state = "pass" if pred == 1 else "fail"
        attr_items.append(("📌", text, conf, state))

    if "spread" in task_preds and class_name == "cardboard":
        pred = task_preds["spread"]
        conf = task_confs["spread"]
        text = "已展平" if pred == 1 else "未展平"
        state = "pass" if pred == 1 else "fail"
        attr_items.append(("📄", text, conf, state))

    if attr_items:
        attr_html = "<div class='result-card'><div style='font-weight:600;margin-bottom:8px;color:#555'>🔍 属性检测</div>"
        for icon, text, conf, state in attr_items:
            color = "#2E7D32" if state == "pass" else "#C62828"
            mark = "✓" if state == "pass" else "✗"
            attr_html += (
                f"<div class='check-item'>"
                f"<span style='color:{color};font-weight:bold'>{mark}</span>"
                f"<span>{icon} {text}</span>"
                f"<span style='color:#999;font-size:12px'>({conf:.0%})</span>"
                f"</div>"
            )
        attr_html += "</div>"
        parts.append(attr_html)

    # 预处理检查
    if pre_check:
        check_html = "<div class='result-card'><div style='font-weight:600;margin-bottom:8px;color:#555'>📋 预处理检查</div>"
        for icon, text in pre_check:
            cls = "check-pass" if icon == "✅" else "check-fail"
            check_html += f"<div class='check-item {cls}'>{icon} <span>{text}</span></div>"
        check_html += "</div>"
        parts.append(check_html)

    # 投放步骤
    if steps:
        s = "<div class='result-card'><div style='font-weight:600;margin-bottom:8px;color:#555'>♻️ 投放步骤</div>"
        s += "".join(f"<div class='step-item'>{x}</div>" for x in steps) + "</div>"
        parts.append(s)

    # 小贴士
    if tips:
        t = "<div class='result-card' style='background:#FAFAFA'><div style='font-weight:600;margin-bottom:6px;color:#555'>💡 小贴士</div>"
        t += "".join(f"<div class='tip-item'>• {x}</div>" for x in tips) + "</div>"
        parts.append(t)

    result_html = "".join(parts)

    # Grad-CAM (仅自建CNN支持)
    if show_gradcam and "YOLOv8" not in model_choice:
        try:
            model_for_cam = full_model
            target_layer = model_for_cam._gradcam_layer
            transform = get_val_transform()
            img_t = transform(image_pil).unsqueeze(0).to(DEVICE)
            cam, _ = generate_gradcam(model_for_cam, img_t, target_layer)
            overlay = cam_overlay(image_pil, cam)
            gradcam_img = overlay
            status_html = "<span class='status-ok'>✅ 识别完成 + 热力图已生成</span>"
        except Exception as e:
            print(f"Grad-CAM 生成失败: {e}")
            status_html = f"<span class='status-warn'>⚠️ 识别完成 (热力图生成失败)</span>"
    elif show_gradcam and "YOLOv8" in model_choice:
        status_html = "<span class='status-warn'>⚠️ YOLOv8 不支持热力图</span>"
    else:
        status_html = "<span class='status-ok'>✅ 识别完成</span>"

    gradcam_update = gr.update(value=gradcam_img, visible=True) if show_gradcam else gr.update(value=None, visible=False)
    return result_html, gradcam_update, status_html


# ============ Gradio 界面 ============

def gradio_interface():
    model_status = {}
    model_status["full"] = os.path.exists(str(FULL_MODEL))
    model_status["yolo"] = yolo_model is not None

    available_models = []
    if model_status["full"]:
        available_models.append("自建CNN (全任务)")
    if model_status["yolo"]:
        available_models.append("YOLOv8")

    default_model = "自建CNN (全任务)" if model_status["full"] else available_models[-1] if available_models else "自建CNN (全任务)"

    missing_models = []
    if not model_status["full"]: missing_models.append("自建CNN (全任务)")
    if not model_status["yolo"]: missing_models.append("YOLOv8")

    warning_html = ""
    if missing_models:
        warning_html = (
            "<div style='background:#FFF3E0;border:1px solid #FFB74D;border-radius:8px;"
            "padding:8px 16px;margin:0 auto 12px;max-width:600px;text-align:center;color:#E65100;font-size:13px'>"
            f"⚠️ 以下模型未训练或文件缺失：{'、'.join(missing_models)}。请先运行训练脚本。</div>"
        )

    # 预置演示图片
    demo_images = [
        ("", "👆 选择演示图片..."),
    ]
    for cls_dir, cls_name in [
        ("plastic", "塑料瓶"),
        ("cardboard", "纸箱"),
        ("metal",    "易拉罐"),
        ("glass",    "玻璃瓶"),
        ("paper",    "纸张"),
    ]:
        img_dir = os.path.join(str(DATA_DIR), "Garbage_Dataset", "test", cls_dir)
        if os.path.isdir(img_dir):
            files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
            if files:
                demo_images.append((os.path.join(img_dir, files[0]), cls_name))
    import numpy as np
    from PIL import Image as PILImage

    def load_demo_image(path):
        if not path:
            return None
        try:
            return np.array(PILImage.open(path).convert("RGB"))
        except Exception:
            return None

    with gr.Blocks(title="慧眼 — 智能垃圾分类系统") as demo:
        gr.HTML(
            "<div style='text-align:center;padding:20px 0 8px'>"
            "<span style='font-size:40px'>♻️</span>"
            "<h1 style='margin:6px 0;font-size:26px;font-weight:700;color:#1a1a1a'>慧眼</h1>"
            "<p style='margin:0;color:#666;font-size:15px'>智能垃圾分类与回收建议系统</p>"
            "</div>"
        )
        gr.HTML(warning_html) if warning_html else None

        with gr.Tabs():
            with gr.TabItem("🔍 智能识别"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=280):
                        image_input = gr.Image(
                            label="上传垃圾图片", type="numpy",
                            sources=["upload", "webcam", "clipboard"],
                        )
                        if len(demo_images) > 1:
                            demo_dropdown = gr.Dropdown(
                                choices=[p for _, p in demo_images],
                                value="👆 选择演示图片...",
                                label="📸 快速演示（预置图片）",
                            )
                            def pick_demo(path):
                                for img_path, label in demo_images:
                                    if label == path and img_path:
                                        return load_demo_image(img_path)
                                return None
                        model_choice = gr.Dropdown(
                            choices=available_models,
                            value=default_model,
                            label="选择模型",
                        )
                        show_gradcam = gr.Checkbox(label="显示 Grad-CAM 热力图", value=False)
                        submit_btn = gr.Button("🔍 开始识别", variant="primary", size="lg")
                        status_text = gr.HTML(value="")

                    with gr.Column(scale=3, min_width=400):
                        result_text = gr.HTML(
                            value="<div style='text-align:center;padding:60px 20px;color:#bbb;font-size:16px'>"
                                  "上传图片后点击「开始识别」查看结果</div>"
                        )
                        gradcam_output = gr.Image(label="Grad-CAM 热力图")

                submit_btn.click(
                    fn=classify_image,
                    inputs=[image_input, model_choice, show_gradcam],
                    outputs=[result_text, gradcam_output, status_text],
                )

                if len(demo_images) > 1:
                    demo_dropdown.change(
                        fn=lambda path: pick_demo(path),
                        inputs=[demo_dropdown],
                        outputs=[image_input],
                    )

                cat_names = list(CAT_COLORS.values())
                gr.HTML(
                    f"<div style='display:flex;flex-wrap:wrap;gap:6px;justify-content:center;padding:8px 0'>"
                    f"<span class='category-badge' style='background:{cat_names[0]}'>可回收物</span>"
                    f"<span class='category-badge' style='background:{cat_names[1]}'>厨余垃圾</span>"
                    f"<span class='category-badge' style='background:{cat_names[2]}'>有害垃圾</span>"
                    f"<span class='category-badge' style='background:{cat_names[3]}'>其他垃圾</span>"
                    f"</div>"
                )

            # --- 模型对比 Tab ---
            with gr.TabItem("📊 模型对比"):
                gr.HTML("""
<div style='overflow-x:auto;max-width:900px;margin:12px auto'>
<table style='width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.1)'>
<thead><tr style='background:#37474F;color:white'>
<th style='padding:10px 14px;font-size:13px'>模型</th>
<th style='padding:10px 14px;font-size:13px'>参数量</th>
<th style='padding:10px 14px;font-size:13px'>属性检测</th>
</tr></thead><tbody>
<tr style='border-bottom:1px solid #eee'>
<td style='padding:10px 14px;font-weight:600'>自建CNN (全任务)</td>
<td style='padding:10px 14px;text-align:center'>51.1万 / 1120万</td>
<td style='padding:10px 14px;text-align:center'>瓶子 · 液体 · 压扁 · 展平</td>
</tr><tr>
<td style='padding:10px 14px;font-weight:600'>YOLOv8n-cls</td>
<td style='padding:10px 14px;text-align:center'>270万</td>
<td style='padding:10px 14px;text-align:center'>—</td>
</tr></tbody></table>
<p style='text-align:center;color:#999;font-size:12px;margin:8px 0 0'>
10类识别：电池 · 厨余 · 纸板 · 衣物 · 玻璃 · 金属 · 纸张 · 塑料 · 鞋子 · 其他<br>
4项属性：is_bottle(瓶子) · liquid(液体) · flatten(压扁) · spread(展平)</p>
</div>
                """)


    return demo


# ============ 主程序 ============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Gradio web interface")
    parser.add_argument("--full_model",  default=str(FULL_MODEL))
    parser.add_argument("--yolo_model",  default=str(YOLO_MODEL))
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=True,
                        help="Generate public link (set --no-share to disable)")
    args = parser.parse_args()

    print(f"设备: {DEVICE}")

    print("加载自建CNN (全任务)...")
    full_model = load_cnn_model(args.full_model, tasks=["liquid", "flatten", "spread", "is_bottle"])
    print("加载YOLOv8...")
    yolo_model = load_yolo_model(args.yolo_model)

    # 启动
    port = args.port
    free_port(port)
    print(f"\n启动 Gradio 界面: http://127.0.0.1:{port}")
    demo = gradio_interface()
    demo.launch(server_port=port, share=args.share, css=CSS, theme=gr.themes.Soft())
