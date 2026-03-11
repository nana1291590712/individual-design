# lolo_experiment.py
"""
LOLO 跨负载实验（第一阶段）：
    - Baseline vs MultiScale（Fault only）
    - 返回 y_true / y_pred
    - 直接生成混淆矩阵 & 识别率曲线
"""

import os
import torch
import torch.nn as nn
import numpy as np
import csv

from torch.utils.data import DataLoader, TensorDataset

from load_dataset import load_dataset
from baseline_model import Baseline1DCNN
from model import MultiScale1DCNN
from dataset_split import split_dataset_by_leave_one_load

# ⭐ 引入 fault 评估模块
from fault_result import evaluate_fault_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# 工具：统一取 fault logits（兼容 Baseline / MultiScale）
# =====================================================
def get_fault_logits(model_outputs):
    if isinstance(model_outputs, tuple):
        return model_outputs[0]
    return model_outputs


# =====================================================
# 本地训练 1 epoch
# =====================================================
def train_one_epoch_local(model, loader, criterion, optimizer):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)

        optimizer.zero_grad()
        outputs = model(bx)
        fault_logits = get_fault_logits(outputs)

        loss = criterion(fault_logits, by)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * bx.size(0)
        preds = torch.argmax(fault_logits, dim=1)
        correct += (preds == by).sum().item()
        total += by.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)


# =====================================================
# 验证 / 测试
# =====================================================
def eval_local(model, loader, criterion, return_preds=False):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)

            outputs = model(bx)
            fault_logits = get_fault_logits(outputs)

            loss = criterion(fault_logits, by)
            loss_sum += loss.item() * bx.size(0)

            preds = torch.argmax(fault_logits, dim=1)
            correct += (preds == by).sum().item()
            total += by.size(0)

            if return_preds:
                all_preds.append(preds.cpu().numpy())
                all_labels.append(by.cpu().numpy())

    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)

    if return_preds:
        return (
            avg_loss,
            acc,
            np.concatenate(all_labels),
            np.concatenate(all_preds),
        )

    return avg_loss, acc


# =====================================================
# 通用训练 + 测试（fault-only）
# =====================================================
def train_and_test(model_class, train_data, val_data, test_data,
                   lr=1e-3, epochs=20):

    x_train, y_train = train_data
    x_val,   y_val   = val_data
    x_test,  y_test  = test_data

    # tensor 化
    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_val   = torch.tensor(x_val,   dtype=torch.float32).unsqueeze(1)
    y_val   = torch.tensor(y_val,   dtype=torch.long)
    x_test  = torch.tensor(x_test,  dtype=torch.float32).unsqueeze(1)
    y_test  = torch.tensor(y_test,  dtype=torch.long)

    train_loader = DataLoader(TensorDataset(x_train, y_train), 64, shuffle=True)
    val_loader   = DataLoader(TensorDataset(x_val, y_val),     64, shuffle=False)
    test_loader  = DataLoader(TensorDataset(x_test, y_test),   64, shuffle=False)

    # 初始化模型
    if model_class is Baseline1DCNN:
        model = Baseline1DCNN(num_classes=4).to(device)
    else:
        model = MultiScale1DCNN(
            num_fault_classes=4,
            num_severity_classes=3
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    # 训练
    for _ in range(epochs):
        train_one_epoch_local(model, train_loader, criterion, optimizer)
        val_loss, _ = eval_local(model, val_loader, criterion)
        scheduler.step(val_loss)

    # 测试
    _, test_acc, y_true, y_pred = eval_local(
        model, test_loader, criterion, return_preds=True
    )

    return test_acc, y_true, y_pred


# =====================================================
# 主程序：LOLO × Baseline vs MultiScale
# =====================================================
def run_lolo():

    print("Loading CWRU dataset...")
    raw_data = load_dataset("data/CWRU")
    print("Loaded:", len(raw_data))

    loads = [0, 1, 2, 3]
    summary = []

    class_names = ["Normal", "Ball", "Inner", "Outer"]

    for leave_out in loads:
        print(f"\n================ LOLO-{leave_out}HP ================")

        (x_train, y_f_train, _), \
        (x_val,   y_f_val,   _), \
        (x_test,  y_f_test,  _), \
        train_loads, test_load = split_dataset_by_leave_one_load(
            raw_data, leave_out
        )

        train_data = (x_train, y_f_train)
        val_data   = (x_val,   y_f_val)
        test_data  = (x_test,  y_f_test)

        # -------- Baseline --------
        print("Training Baseline...")
        acc_b, y_tb, y_pb = train_and_test(
            Baseline1DCNN, train_data, val_data, test_data
        )

        # -------- MultiScale --------
        print("Training MultiScale...")
        acc_m, y_tm, y_pm = train_and_test(
            MultiScale1DCNN, train_data, val_data, test_data
        )

        delta = acc_m - acc_b
        summary.append([f"LOLO-{leave_out}HP", acc_b, acc_m, delta])

        print(f"Baseline  Acc = {acc_b*100:.2f}%")
        print(f"MultiScale Acc = {acc_m*100:.2f}%")
        print(f"Δ Improvement = {delta*100:.2f}%")

        # -------- ⭐ 生成结果图 --------
        base_dir = f"results/fault/LOLO-{leave_out}HP"

        evaluate_fault_results(
            y_tb, y_pb, class_names,
            save_dir=os.path.join(base_dir, "baseline")
        )

        evaluate_fault_results(
            y_tm, y_pm, class_names,
            save_dir=os.path.join(base_dir, "multiscale")
        )

    # -------- 汇总表 --------
    print("\n================ FINAL SUMMARY ================")
    for tag, b, m, d in summary:
        print(f"{tag:12s} | {b*100:6.2f}% | {m*100:6.2f}% | {d*100:6.2f}%")

    os.makedirs("results/summary", exist_ok=True)
    with open("results/summary/lolo_fault_summary.csv",
              "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Load-out", "Baseline Acc", "MultiScale Acc", "Delta"])
        writer.writerows(summary)

    print("\nAll LOLO fault results saved.")


if __name__ == "__main__":
    run_lolo()
