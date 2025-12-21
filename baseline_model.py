# baseline_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline1DCNN(nn.Module):
    """
    复现论文结构的 baseline 1D-CNN
    输入形状： [batch, 1, 1024]
    """

    def __init__(self, num_classes=4):
        super(Baseline1DCNN, self).__init__()

        # ------------------------------
        # Conv1: 128 filters, kernel=16
        # ------------------------------
        self.conv1 = nn.Conv1d(1, 128, kernel_size=16, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.3)

        # ------------------------------
        # Conv2: 64 filters, kernel=8
        # ------------------------------
        self.conv2 = nn.Conv1d(128, 64, kernel_size=8, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.3)

        # ------------------------------
        # Conv3: 32 filters, kernel=4
        # ------------------------------
        self.conv3 = nn.Conv1d(64, 32, kernel_size=4, stride=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(0.25)

        # ------------------------------
        # Conv4: 16 filters, kernel=4
        # ------------------------------
        self.conv4 = nn.Conv1d(32, 16, kernel_size=4, stride=1)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        # 无 dropout（论文写明）

        # ------------------------------
        # Conv5: 8 filters, kernel=4
        # ------------------------------
        self.conv5 = nn.Conv1d(16, 8, kernel_size=4, stride=1)
        self.drop5 = nn.Dropout(0.25)

        # ----------------------------------------------------------
        # 最终特征长度计算：1024 经 4 次池化后约 56
        # Flatten = 56 * 8 = 448
        # ----------------------------------------------------------
        self.fc = nn.Linear(448, num_classes)

    def forward(self, x):
        # 输入 [B, 1, 1024]

        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = torch.tanh(x)
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.conv4(x)
        x = torch.tanh(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = torch.tanh(x)
        x = self.drop5(x)

        x = x.reshape(x.size(0), -1)  # [B, 448]

        out = self.fc(x)
        return out
