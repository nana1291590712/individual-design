#baseline_test_shape.py
import torch
from baseline_model import Baseline1DCNN

# --------------------------
# 1. 创建模型（4 类输出）
# --------------------------
model = Baseline1DCNN(num_classes=4)

# --------------------------
# 2. 构造假数据（用于 shape 测试）
#    batch = 2, 通道=1, 长度=1024
# --------------------------
x = torch.randn(2, 1, 1024)

# --------------------------
# 3. 前向推理（forward）
# --------------------------
output = model(x)

# --------------------------
# 4. 打印输出 shape
# --------------------------
print("输入形状：", x.shape)
print("输出形状：", output.shape)
