# lolo_severity_experiment.py
"""
LOLO 跨负载实验（第二阶段）：
    - 仅使用 MultiScale1DCNN
    - 评估 Severity（Low / Medium / High）
    - 只统计 fault != Normal 的样本
"""

import os
import csv
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report

from load_dataset import load_dataset
from dataset_split import split_dataset_by_leave_one_load
from model import MultiScale1DCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# 评估 severity（仅 fault != Normal）
# =====================================================
def evaluate_severity(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for bx, y_fault, y_sev in loader:
            bx = bx.to(device)
            y_fault = y_fault.to(device)
            y_sev = y_sev.to(device)

            _, sev_logits = model(bx)
            sev_preds = torch.argmax(sev_logits, dim=1)

            for i in range(len(y_fault)):
                # 只统计 fault != Normal
                if y_fault[i].item() != 0:
                    all_preds.append(sev_preds[i].item())
                    all_labels.append(y_sev[i].item())

    return np.array(all_labels), np.array(all_preds)


# =====================================================
# 绘制 severity 混淆矩阵
# =====================================================
def plot_severity_confusion(y_true, y_pred, save_dir):
    class_names = ["Low", "Medium", "High"]
    cm = confusion_matrix(y_true, y_pred)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Severity Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names)
    plt.yticks(ticks, class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "severity_confusion_matrix.png"))
    plt.close()


# =====================================================
# 主程序：LOLO Severity
# =====================================================
def run_lolo_severity():

    print("Loading CWRU dataset...")
    raw_data = load_dataset("data/CWRU")
    print("Loaded:", len(raw_data))

    loads = [0, 1, 2, 3]
    summary = []

    for leave_out in loads:
        print(f"\n================ Severity LOLO-{leave_out}HP ================")

        (x_train, y_f_train, y_s_train), \
        (x_val,   y_f_val,   y_s_val), \
        (x_test,  y_f_test,  y_s_test), \
        train_loads, test_load = split_dataset_by_leave_one_load(
            raw_data, leave_out
        )

        print(f"Train loads : {train_loads}")
        print(f"Test load  : {test_load}")

        # tensor 化（test only）
        x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
        y_f_test = torch.tensor(y_f_test, dtype=torch.long)
        y_s_test = torch.tensor(y_s_test, dtype=torch.long)

        test_loader = DataLoader(
            TensorDataset(x_test, y_f_test, y_s_test),
            batch_size=64,
            shuffle=False
        )

        # -------------------------
        # 加载已训练好的 MultiScale 模型
        # （这里默认：你在 fault 阶段训练得到的模型）
        # 如果你想每个 LOLO 单独训练 severity，也可以扩展
        # -------------------------
        model = MultiScale1DCNN(
            num_fault_classes=4,
            num_severity_classes=3
        ).to(device)

        # ⚠️ 注意：
        # 这里假设你已经保存了对应 LOLO 的 multiscale 权重
        # 比如：results/models/LOLO-3HP/multiscale.pth
        model_path = f"results/models/LOLO-{leave_out}HP/multiscale.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))

        print("Evaluating severity...")
        y_true, y_pred = evaluate_severity(model, test_loader)

        report = classification_report(
            y_true, y_pred,
            target_names=["Low", "Medium", "High"],
            digits=4
        )

        print(report)

        save_dir = f"results/severity/LOLO-{leave_out}HP"
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "severity_report.txt"), "w") as f:
            f.write(report)

        plot_severity_confusion(y_true, y_pred, save_dir)

        # summary（可选：macro F1）
        summary.append([f"LOLO-{leave_out}HP", len(y_true)])

    # -------------------------
    # 汇总表（可扩展）
    # -------------------------
    os.makedirs("results/severity/summary", exist_ok=True)
    with open("results/severity/summary/lolo_severity_summary.csv",
              "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Load-out", "Num Severity Samples"])
        writer.writerows(summary)

    print("\nAll LOLO severity results saved.")


if __name__ == "__main__":
    run_lolo_severity()
