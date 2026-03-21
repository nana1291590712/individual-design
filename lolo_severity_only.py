# lolo_severity_only.py
"""
LOLO severity-only experiment
- Model: MultiScale1DCNN only
- Task: severity regression only
- Evaluation: severity interval accuracy / MAE / RMSE
"""

import os
import csv
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from load_dataset import load_dataset
from dataset_split import split_dataset_by_leave_one_load
from model import MultiScale1DCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------
# 参数
# --------------------------------------------------------
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 30
MAX_DIAMETER = 0.028

RESULT_ROOT = "results/severity_lolo"

SEVERITY_CLASS_NAMES = ["Low", "Medium", "High", "Very Severe"]


# --------------------------------------------------------
# 连续直径 -> 类别
# --------------------------------------------------------
def diameter_to_class(d):
    d = float(d)
    if d < 0.0105:
        return 0
    elif d < 0.0175:
        return 1
    elif d < 0.0245:
        return 2
    else:
        return 3


def normalized_diameter_to_class(d):
    d = float(d)
    if d < 0.375:
        return 0
    elif d < 0.625:
        return 1
    elif d < 0.875:
        return 2
    else:
        return 3


# --------------------------------------------------------
# 只保留故障样本
# fault=0 为 Normal，severity-only 实验中去掉
# --------------------------------------------------------
def keep_fault_only(x, y_fault, y_sev):
    mask = (y_fault != 0)
    return x[mask], y_fault[mask], y_sev[mask]


# --------------------------------------------------------
# numpy -> tensor
# severity 归一化到 [0,1]
# --------------------------------------------------------
def to_tensor(x, y_sev):
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_sev = torch.tensor(y_sev / MAX_DIAMETER, dtype=torch.float32)
    return x, y_sev


# --------------------------------------------------------
# 训练 1 epoch
# 只优化 severity loss
# --------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    total_abs_error = 0.0
    total_interval_correct = 0
    total_samples = 0

    for batch_x, batch_sev in loader:
        batch_x = batch_x.to(device)
        batch_sev = batch_sev.to(device)

        optimizer.zero_grad()

        _, sev_out = model(batch_x)
        sev_out = torch.clamp(sev_out, min=0.0, max=1.0)

        loss = criterion(sev_out, batch_sev)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        total_abs_error += torch.abs(sev_out - batch_sev).sum().item()

        pred_cls = np.array([normalized_diameter_to_class(v) for v in sev_out.detach().cpu().numpy()])
        true_cls = np.array([normalized_diameter_to_class(v) for v in batch_sev.detach().cpu().numpy()])
        total_interval_correct += (pred_cls == true_cls).sum()

        total_samples += batch_x.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    mae = total_abs_error / max(total_samples, 1)
    interval_acc = total_interval_correct / max(total_samples, 1)

    return avg_loss, mae, interval_acc


# --------------------------------------------------------
# 验证 / 测试
# --------------------------------------------------------
def evaluate(model, loader, criterion, return_preds=False):
    model.eval()

    total_loss = 0.0
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_interval_correct = 0
    total_samples = 0

    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch_x, batch_sev in loader:
            batch_x = batch_x.to(device)
            batch_sev = batch_sev.to(device)

            _, sev_out = model(batch_x)
            sev_out = torch.clamp(sev_out, min=0.0, max=1.0)

            loss = criterion(sev_out, batch_sev)

            total_loss += loss.item() * batch_x.size(0)

            diff = sev_out - batch_sev
            total_abs_error += torch.abs(diff).sum().item()
            total_sq_error += (diff ** 2).sum().item()

            pred_cls = np.array([normalized_diameter_to_class(v) for v in sev_out.detach().cpu().numpy()])
            true_cls = np.array([normalized_diameter_to_class(v) for v in batch_sev.detach().cpu().numpy()])
            total_interval_correct += (pred_cls == true_cls).sum()

            total_samples += batch_x.size(0)

            if return_preds:
                all_true.append(batch_sev.cpu().numpy())
                all_pred.append(sev_out.cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    mae = total_abs_error / max(total_samples, 1)
    rmse = np.sqrt(total_sq_error / max(total_samples, 1))
    interval_acc = total_interval_correct / max(total_samples, 1)

    if return_preds:
        return (
            avg_loss,
            mae,
            rmse,
            interval_acc,
            np.concatenate(all_true),
            np.concatenate(all_pred)
        )

    return avg_loss, mae, rmse, interval_acc


# --------------------------------------------------------
# 结果图
# --------------------------------------------------------
def plot_confusion_matrix(y_true_cls, y_pred_cls, save_path):
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1, 2, 3])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Severity Confusion Matrix")
    plt.colorbar(shrink=0.8)

    ticks = np.arange(4)
    plt.xticks(ticks, SEVERITY_CLASS_NAMES, rotation=20)
    plt.yticks(ticks, SEVERITY_CLASS_NAMES)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_scatter(true_d, pred_d, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(true_d, pred_d, alpha=0.5)
    plt.plot([0.0, MAX_DIAMETER], [0.0, MAX_DIAMETER], linestyle="--")
    plt.xlabel("True Diameter")
    plt.ylabel("Predicted Diameter")
    plt.title("Severity Diameter Prediction")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# --------------------------------------------------------
# 单个 LOLO 训练与测试
# --------------------------------------------------------
def train_and_test_severity_only(train_data, val_data, test_data, lr=LR, epochs=EPOCHS):
    x_train, _, y_s_train = train_data
    x_val, _, y_s_val = val_data
    x_test, _, y_s_test = test_data

    x_train, y_s_train = to_tensor(x_train, y_s_train)
    x_val, y_s_val = to_tensor(x_val, y_s_val)
    x_test, y_s_test = to_tensor(x_test, y_s_test)

    train_loader = DataLoader(TensorDataset(x_train, y_s_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_s_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_s_test), batch_size=BATCH_SIZE, shuffle=False)

    model = MultiScale1DCNN().to(device)
    criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    best_state = None
    best_score = -1.0

    for epoch in range(epochs):
        train_loss, train_mae, train_interval_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_mae, val_rmse, val_interval_acc = evaluate(
            model, val_loader, criterion
        )

        scheduler.step(val_loss)

        if val_interval_acc > best_score:
            best_score = val_interval_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch [{epoch+1:02d}/{epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.6f}, Train Interval Acc: {train_interval_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.6f}, Val RMSE: {val_rmse:.6f}, Val Interval Acc: {val_interval_acc*100:.2f}%"
        )

    model.load_state_dict(best_state)

    test_loss, test_mae, test_rmse, test_interval_acc, y_true_norm, y_pred_norm = evaluate(
        model, test_loader, criterion, return_preds=True
    )

    return {
        "model": model,
        "test_loss": test_loss,
        "test_mae_norm": test_mae,
        "test_rmse_norm": test_rmse,
        "test_interval_acc": test_interval_acc,
        "y_true_norm": y_true_norm,
        "y_pred_norm": y_pred_norm
    }


# --------------------------------------------------------
# 保存 severity 结果
# --------------------------------------------------------
def save_severity_outputs(y_true_norm, y_pred_norm, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    true_d = np.clip(y_true_norm * MAX_DIAMETER, 0.0, MAX_DIAMETER)
    pred_d = np.clip(y_pred_norm * MAX_DIAMETER, 0.0, MAX_DIAMETER)

    y_true_cls = np.array([diameter_to_class(v) for v in true_d])
    y_pred_cls = np.array([diameter_to_class(v) for v in pred_d])

    report = classification_report(
        y_true_cls,
        y_pred_cls,
        labels=[0, 1, 2, 3],
        target_names=SEVERITY_CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    with open(os.path.join(save_dir, "severity_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    plot_confusion_matrix(
        y_true_cls, y_pred_cls,
        os.path.join(save_dir, "severity_confusion_matrix.png")
    )

    plot_scatter(
        true_d, pred_d,
        os.path.join(save_dir, "severity_scatter.png")
    )

    detail_csv = os.path.join(save_dir, "severity_details.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "true_diameter",
            "pred_diameter",
            "true_class",
            "pred_class",
            "abs_error",
            "correct"
        ])
        for i in range(len(true_d)):
            writer.writerow([
                i + 1,
                round(float(true_d[i]), 6),
                round(float(pred_d[i]), 6),
                int(y_true_cls[i]),
                int(y_pred_cls[i]),
                round(abs(float(pred_d[i]) - float(true_d[i])), 6),
                int(y_true_cls[i] == y_pred_cls[i])
            ])


# --------------------------------------------------------
# 主程序
# --------------------------------------------------------
def run_lolo_severity_only():
    print("Loading dataset...")

    fault_root = r"D:\design\data\CWRU\12kDriveEndFault"
    normal_root = r"D:\design\data\CWRU\NormalBaseline"

    fault_dataset = load_dataset(fault_root)
    normal_dataset = load_dataset(normal_root)
    raw_data = fault_dataset + normal_dataset

    loads = [0, 1, 2, 3]
    summary = []

    os.makedirs(RESULT_ROOT, exist_ok=True)

    for leave_out in loads:
        print(f"\n================ LOLO-{leave_out}HP ================")

        train_data, val_data, test_data, train_loads, test_load = split_dataset_by_leave_one_load(
            raw_data, leave_out
        )

        # 只保留故障样本
        train_data = keep_fault_only(*train_data)
        val_data = keep_fault_only(*val_data)
        test_data = keep_fault_only(*test_data)

        print(f"Train loads: {train_loads}, Test load: {test_load}")
        print(f"Train size: {len(train_data[0])}, Val size: {len(val_data[0])}, Test size: {len(test_data[0])}")

        result = train_and_test_severity_only(train_data, val_data, test_data)

        test_mae_real = result["test_mae_norm"] * MAX_DIAMETER
        test_rmse_real = result["test_rmse_norm"] * MAX_DIAMETER

        print(f"Test Interval Acc = {result['test_interval_acc']*100:.2f}%")
        print(f"Test MAE          = {test_mae_real:.6f}")
        print(f"Test RMSE         = {test_rmse_real:.6f}")

        save_dir = os.path.join(RESULT_ROOT, f"LOLO-{leave_out}HP")
        save_severity_outputs(
            result["y_true_norm"],
            result["y_pred_norm"],
            save_dir
        )

        # 保存该折最佳模型
        torch.save(result["model"].state_dict(), os.path.join(save_dir, "multiscale_severity_best.pth"))

        summary.append([
            f"LOLO-{leave_out}HP",
            result["test_interval_acc"],
            test_mae_real,
            test_rmse_real
        ])

    print("\n================ FINAL SUMMARY ================")
    for row in summary:
        tag, acc, mae, rmse = row
        print(f"{tag:12s} | Interval Acc: {acc*100:6.2f}% | MAE: {mae:.6f} | RMSE: {rmse:.6f}")

    with open(os.path.join(RESULT_ROOT, "severity_lolo_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Load-out", "Severity Interval Acc", "Severity MAE", "Severity RMSE"])
        writer.writerows(summary)

    print("\nAll severity-only LOLO results saved.")


if __name__ == "__main__":
    run_lolo_severity_only()