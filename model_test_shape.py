# model_test_shape.py
import torch
from model import MultiScale1DCNN


# ------------------------------
# 创建模型
# ------------------------------
model = MultiScale1DCNN(num_classes=4)

# ------------------------------
# 构造一个虚拟输入
# 输入尺寸必须是 [batch, 1, 1024]
# ------------------------------
dummy_input = torch.randn(8, 1, 1024)   # batch_size = 8

# ------------------------------
# 前向推理（不计算梯度）
# ------------------------------
with torch.no_grad():
    output = model(dummy_input)

# ------------------------------
# 打印形状
# ------------------------------
print("输入 shape:", dummy_input.shape)
print("输出 shape:", output.shape)
print("模型参数量:", sum(p.numel() for p in model.parameters()))
