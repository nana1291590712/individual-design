# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 改进模型：多尺度卷积 MultiScale1DCNN
# 功能：
#   - 并行三尺度卷积（16 / 32 / 64）
#   - BN + Dropout 防止过拟合
#   - GlobalAveragePooling 提升跨负载泛化
# 输入：  [batch, 1, 1024]
# 输出：  [batch, num_classes]
# =========================================================
class MultiScale1DCNN(nn.Module):
    """
    改进版 1D-CNN：
        1) 第一卷积块加入三支路多尺度卷积
        2) 卷积层结构：Conv → BN → Tanh → Dropout
        3) 使用全局平均池化替代 flatten
    """

    def __init__(self,
                 num_classes: int = 4,
                 input_channels: int = 1,
                 branch_out_channels: int = 32):
        super().__init__()

        # ------------------------------
        # 多尺度卷积块：三支路并行
        # ------------------------------
        self.branch1 = nn.Sequential(
            nn.Conv1d(input_channels, branch_out_channels,
                      kernel_size=16, padding=16 // 2),
            nn.BatchNorm1d(branch_out_channels),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.30)
        )

        self.branch2 = nn.Sequential(
            nn.Conv1d(input_channels, branch_out_channels,
                      kernel_size=32, padding=32 // 2),
            nn.BatchNorm1d(branch_out_channels),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.30)
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(input_channels, branch_out_channels,
                      kernel_size=64, padding=64 // 2),
            nn.BatchNorm1d(branch_out_channels),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.30)
        )

        # 多尺度输出通道
        multi_out = branch_out_channels * 3

        # ------------------------------
        # 后续卷积块（保持与 baseline 类似，但加入 BN + Dropout）
        # ------------------------------
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(multi_out, 64, kernel_size=8, padding=8 // 2),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.25)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=4, padding=4 // 2),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.25)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=4, padding=4 // 2),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.25)
        )

        # ------------------------------
        # 全局平均池化 + 分类层
        # ------------------------------
        self.gap = nn.AdaptiveAvgPool1d(1)  # 输出 [B, C, 1]
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        # x: [B, 1, 1024]

        # 多尺度三支路
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        # 通道拼接
        x = torch.cat([b1, b2, b3], dim=1)  # [B, 32*3, L]

        # 后续卷积
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        # GAP
        x = self.gap(x).squeeze(-1)  # [B, 16]

        # 分类
        x = self.classifier(x)       # [B, num_classes]
        return x
