# model_load_test.py
"""
对比实验（改进模型 MultiScale1DCNN）：
1. 单负载训练（如 0HP）→ 其它负载测试（1/2/3HP）
2. 多负载训练（0/1/2HP）→ 未见负载测试（3HP）

本脚本训练 20 轮，并使用你的 MultiScale1DCNN 改进模型。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from load_dataset import load_dataset
from preprocess import preprocess_dataset
from model import MultiScale1DCNN      # ← 使用你的改进模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# numpy → tensor
# ============================================================
def to_tensor(x, y):
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # [N,1,1024]
    y = torch.tensor(y, dtype=torch.long)
    return x, y


# ============================================================
# 训练 1 epoch
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    correct = 0
    total = 0
    total_loss = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, pred = torch.max(out, 1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


# ============================================================
# 测试
# ============================================================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


# ============================================================
# 实验 A：单负载训练 → 跨负载测试
# ============================================================
def experiment_single_load(input_dataset, train_load=0):
    print(f"\n========== 实验 A：单负载训练 {train_load}HP ==========")

    single_load_data = [d for d in input_dataset if d["load"] == train_load]

    x_all, y_all, loads_all = preprocess_dataset(single_load_data)

    N = len(x_all)
    idx = np.arange(N)
    np.random.shuffle(idx)

    train_idx = idx[:int(0.8 * N)]
    val_idx = idx[int(0.8 * N):]

    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]

    train_loader = DataLoader(TensorDataset(*to_tensor(x_train, y_train)), batch_size=64, shuffle=True)
    val_loader   = DataLoader(TensorDataset(*to_tensor(x_val, y_val)),   batch_size=64, shuffle=False)

    model = MultiScale1DCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 训练 20 轮
    for epoch in range(20):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch[{epoch+1}/20] TrainAcc={acc:.4f}, ValAcc={val_acc:.4f}")

    # 跨负载测试
    test_loads = [1, 2, 3]
    results = {}

    for tl in test_loads:
        test_data = [d for d in input_dataset if d["load"] == tl]
        if len(test_data) == 0:
            continue

        x_t, y_t, _ = preprocess_dataset(test_data)
        test_loader = DataLoader(TensorDataset(*to_tensor(x_t, y_t)), batch_size=64, shuffle=False)
        acc = evaluate(model, test_loader)
        results[tl] = acc
        print(f"单负载训练 {train_load}HP → 测试 {tl}HP = {acc:.4f}")

    return results


# ============================================================
# 实验 B：多负载训练 → 3HP
# ============================================================
def experiment_multi_load(input_dataset):
    print("\n========== 实验 B：多负载训练 0/1/2HP → 3HP ==========")

    train_data = [d for d in input_dataset if d["load"] in [0, 1, 2]]
    test_data  = [d for d in input_dataset if d["load"] == 3]

    x_train, y_train, _ = preprocess_dataset(train_data)
    x_test,  y_test,  _ = preprocess_dataset(test_data)

    N = len(x_train)
    idx = np.arange(N)
    np.random.shuffle(idx)

    train_idx = idx[:int(0.85 * N)]
    val_idx   = idx[int(0.85 * N):]

    x_tn, y_tn = x_train[train_idx], y_train[train_idx]
    x_val, y_val = x_train[val_idx], y_train[val_idx]

    train_loader = DataLoader(TensorDataset(*to_tensor(x_tn, y_tn)), batch_size=64, shuffle=True)
    val_loader   = DataLoader(TensorDataset(*to_tensor(x_val, y_val)), batch_size=64, shuffle=False)
    test_loader  = DataLoader(TensorDataset(*to_tensor(x_test, y_test)), batch_size=64, shuffle=False)

    model = MultiScale1DCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 训练 20 轮
    for epoch in range(20):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch[{epoch+1}/20] TrainAcc={acc:.4f}, ValAcc={val_acc:.4f}")

    final_acc = evaluate(model, test_loader)
    print(f"\n多负载训练 0/1/2HP → 测试 3HP = {final_acc:.4f}")

    return final_acc


# ============================================================
# 主程序入口
# ============================================================
if __name__ == "__main__":
    print("加载 CWRU 数据集中...")
    dataset = load_dataset("D:/design/data/")   # ← 修改为你的数据路径

    result_single_0HP = experiment_single_load(dataset, train_load=0)
    result_multi = experiment_multi_load(dataset)

    print("\n===== 最终结果统计 =====")
    print("单负载训练(0HP) → 各负载测试：", result_single_0HP)
    print("多负载训练(0/1/2HP) → 未见负载3HP：", result_multi)
