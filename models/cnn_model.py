"""
自建 CNN 模型 — 支持单任务（分类）和多任务（分类 + 属性检测）

使用方式:
    model = MultiTaskCNN(num_classes=10)                        # 仅分类
    model = MultiTaskCNN(num_classes=10, tasks=["liquid"])      # 分类+液体
    model = MultiTaskCNN(num_classes=10, tasks=["liquid", "flatten", "spread"])
    logits, task_logits = model(images)  # task_logits = {"liquid": Tensor, ...}
"""

import os
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """卷积块: Conv2d → BN → ReLU → MaxPool"""
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size),
        )

    def forward(self, x):
        return self.block(x)


class MultiTaskCNN(nn.Module):
    """
    多任务 CNN: 共享骨干 → 分类分支 + 多属性检测分支

    参数:
        num_classes: 分类类别数（默认10）
        kernel_size: 卷积核大小（3/5/7，仅 backbone=cnn 时生效）
        dropout: Dropout 比例
        fc_dim: 全连接层维度
        multi_task: 向后兼容，等价于 tasks=["liquid"]
        tasks: 属性任务列表
        backbone: "cnn" (自建, 默认) 或 "resnet18" (预训练)
        pretrained: backbone="resnet18" 时是否加载预训练权重
    """
    def __init__(self, num_classes=10, kernel_size=3, dropout=0.5, fc_dim=512,
                 multi_task=True, tasks=None, backbone="cnn", pretrained=True):
        super().__init__()

        if tasks is None:
            self.active_tasks = ["liquid"] if multi_task else []
        else:
            self.active_tasks = tasks

        self.backbone_type = backbone

        if backbone == "resnet18":
            try:
                from torchvision import models as tv_models
                local_weights = os.path.join(os.path.dirname(__file__), "..", "resnet18-f37072fd.pth")
                if os.path.exists(local_weights):
                    resnet = tv_models.resnet18(weights=None)
                    resnet.load_state_dict(torch.load(local_weights, map_location="cpu", weights_only=True))
                    print(f"  预训练权重已加载: {local_weights}")
                elif pretrained:
                    resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
                else:
                    resnet = tv_models.resnet18(weights=None)
                self.backbone = nn.Sequential(*list(resnet.children())[:-1])
                backbone_dim = 512
                self._gradcam_layer = resnet.layer4[-1].conv2
            except Exception:
                print("  ResNet18 加载失败，回退至自建CNN")
                backbone = "cnn"
                self.backbone_type = "cnn"
                self.backbone = nn.Sequential(
                    ConvBlock(3, 64, kernel_size),
                    ConvBlock(64, 128, kernel_size),
                    ConvBlock(128, 256, kernel_size),
                )
                backbone_dim = 256
                self._gradcam_layer = self.backbone[-1].block[0]
        if backbone == "cnn":
            self.backbone = nn.Sequential(
                ConvBlock(3, 64, kernel_size),
                ConvBlock(64, 128, kernel_size),
                ConvBlock(128, 256, kernel_size),
            )
            backbone_dim = 256
            self._gradcam_layer = self.backbone[-1].block[0]

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(backbone_dim, fc_dim)
        self.relu = nn.ReLU(inplace=True)

        self.classifier = nn.Linear(fc_dim, num_classes)

        self.task_classifiers = nn.ModuleDict()
        for task_name in self.active_tasks:
            self.task_classifiers[task_name] = nn.Linear(fc_dim, 2)

    def forward(self, x):
        feat = self.backbone(x)            # CNN: [B,256,H/8,W/8]  ResNet: [B,512,1,1]
        if self.backbone_type == "cnn":
            feat = self.global_pool(feat)
        feat = feat.view(feat.size(0), -1)
        feat = self.dropout(feat)
        feat = self.fc(feat)
        feat = self.relu(feat)

        logits = self.classifier(feat)
        task_logits = {}
        for task_name, classifier in self.task_classifiers.items():
            task_logits[task_name] = classifier(feat)

        return logits, task_logits

    def predict(self, x):
        """推理用: 返回分类和所有属性的softmax概率"""
        logits, task_logits = self.forward(x)
        cls_probs = torch.softmax(logits, dim=1)
        task_probs = {}
        for task_name, t_logits in task_logits.items():
            task_probs[task_name] = torch.softmax(t_logits, dim=1)
        return cls_probs, task_probs

    def load_old_checkpoint(self, state_dict):
        """加载旧格式checkpoint（liquid_classifier → task_classifiers.liquid）"""
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("liquid_classifier"):
                k = k.replace("liquid_classifier", "task_classifiers.liquid")
            remapped[k] = v
        return self.load_state_dict(remapped, strict=False)


class SingleTaskCNN(MultiTaskCNN):
    """单任务 CNN（仅分类）"""
    def __init__(self, num_classes=10, kernel_size=3, dropout=0.5, fc_dim=512,
                 backbone="cnn", pretrained=True):
        super().__init__(
            num_classes=num_classes,
            kernel_size=kernel_size,
            dropout=dropout,
            fc_dim=fc_dim,
            multi_task=False,
            tasks=[],
            backbone=backbone,
            pretrained=pretrained,
        )


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def remap_old_checkpoint(state_dict):
    """将旧格式checkpoint的键映射为新格式"""
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("liquid_classifier"):
            k = k.replace("liquid_classifier", "task_classifiers.liquid")
        remapped[k] = v
    return remapped


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 3, 128, 128).to(device)

    # 测试全任务模型
    model = MultiTaskCNN(num_classes=10, tasks=["liquid", "flatten", "spread"]).to(device)
    logits, task_logits = model(x)
    print(f"全任务模型参数量: {count_parameters(model):,}")
    print(f"分类输出形状: {logits.shape}")
    for name, t in task_logits.items():
        print(f"  {name} 输出形状: {t.shape}")

    # 测试单任务模型
    model_s = SingleTaskCNN(num_classes=10).to(device)
    logits_s, task_logits_s = model_s(x)
    print(f"\n单任务模型参数量: {count_parameters(model_s):,}")
    print(f"分类输出形状: {logits_s.shape}, task_logits: {task_logits_s}")

    # 测试predict
    cls_probs, task_probs = model.predict(x)
    print(f"\npredict 输出: cls_probs={cls_probs.shape}, tasks={list(task_probs.keys())}")

    # 测试旧checkpoint加载
    old_state = {"liquid_classifier.weight": torch.randn(2, 512), "liquid_classifier.bias": torch.zeros(2)}
    model2 = MultiTaskCNN(num_classes=10, tasks=["liquid", "flatten", "spread"])
    model2.load_old_checkpoint(old_state)
    print("\n旧checkpoint加载: OK")
