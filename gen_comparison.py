"""生成 CNN vs YOLOv8 对比图（CNN/YOLO 都训练完后运行一次）"""
import os, sys, torch
sys.path.insert(0, '.')
from utils.visualize import plot_model_comparison
from models.cnn_model import MultiTaskCNN

# CNN 参数量
model = MultiTaskCNN(num_classes=10, tasks=["liquid","flatten","spread","is_bottle"], backbone='resnet18')
cnn_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

# YOLOv8 参数量
yolo_param = 2_709_000  # yolov8n-cls

# 准确率——训练完替换为实际值
cnn_acc = 95.50   # 更新为实际训练结果（100 epoch ResNet18 多任务）
yolo_acc = 99.1   # 30 epoch 完整训练结果

print(f'CNN(ResNet18): {cnn_param/1e4:.0f}万参数, acc={cnn_acc}%')
print(f'YOLOv8n-cls:   {yolo_param/1e4:.0f}万参数, acc={yolo_acc}%')

plot_model_comparison(cnn_acc, yolo_acc, cnn_param, yolo_param)
print('CNN vs YOLOv8 对比图已生成')
