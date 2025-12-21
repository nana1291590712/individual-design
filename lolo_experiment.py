# lolo_experiment.py
"""
一次运行完成 LOLO 四组跨负载泛化实验，并对比：
    模型A：Baseline1DCNN
    模型B：MultiScale1DCNN

含优化点：
    - stratified validation split（解决 LOLO-3HP 反转问题）
    - 完整四组负载实验
    - 自动表格 + CSV 输出

论文章节：“跨负载泛化能力验证（Leave-One-Load-Out）”
"""

import torch
import torch.nn as nn
import numpy as np
import csv

from torch.utils.data import DataLoader, TensorDataset

from load_dataset import load_dataset
from baseline_model import Baseline1DCNN
from model import MultiScale1DCNN
from dataset_split import split_dataset_by_leave_one_load
from model_train import to_tensor, train_one_epoch, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# 通用训练 + 测试函数（适用于两个模型）
# =====================================================
def train_and_test(model_class, train_data, val_data, test_data,
                   lr=1e-3, epochs=15):

    x_train, y_train = to_tensor(*train_data)
    x_val, y_val = to_tensor(*val_data)
    x_test, y_test = to_tensor(*test_data)

    train_loader = DataLoader(TensorDataset(x_train, y_train),
                              batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val),
                            batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test),
                             batch_size=64, shuffle=False)

    model = model_class(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

    _, test_acc = evaluate(model, test_loader, criterion)
    return test_acc


# =====================================================
# 主程序：四组 LOLO + Baseline vs MultiScale
# =====================================================
def run_lolo():

    print("Loading CWRU dataset...")
    raw_data = load_dataset("data/CWRU")
    print("Loaded:", len(raw_data))

    # 四个负载
    loads = [0, 1, 2, 3]
    results = []

    for leave_out in loads:

        print(f"\n================ LOLO-{leave_out}HP ================")

        # 数据划分
        (x_train, y_train), (x_val, y_val), (x_test, y_test), \
            train_loads, test_load = split_dataset_by_leave_one_load(
                raw_data, leave_out
            )

        print(f"Train loads : {train_loads}")
        print(f"Test load  : {test_load}")

        train_data = (x_train, y_train)
        val_data = (x_val, y_val)
        test_data = (x_test, y_test)

        # 模型 A：Baseline
        print("Training Baseline...")
        acc_base = train_and_test(Baseline1DCNN,
                                  train_data, val_data, test_data)

        # 模型 B：MultiScale
        print("Training MultiScale...")
        acc_multi = train_and_test(MultiScale1DCNN,
                                   train_data, val_data, test_data)

        delta = acc_multi - acc_base

        print(f"Baseline  Acc = {acc_base * 100:.2f}%")
        print(f"MultiScale Acc = {acc_multi * 100:.2f}%")
        print(f"Δ Improvement = {delta * 100:.2f}%")

        results.append([
            f"LOLO-{leave_out}HP",
            acc_base,
            acc_multi,
            delta
        ])

    # 总表
    print("\n================ FINAL SUMMARY ================")
    print("Load-out | Baseline Acc | MultiScale Acc | Δ")

    for tag, b, m, d in results:
        print(f"{tag:12s} | {b*100:12.2f}% | {m*100:14.2f}% | {d*100:8.2f}%")

    # CSV 保存
    with open("lolo_results_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Load-out", "Baseline Acc", "MultiScale Acc", "Delta"])
        for row in results:
            writer.writerow(row)

    print("\nSaved to lolo_results_summary.csv")


if __name__ == "__main__":
    run_lolo()
