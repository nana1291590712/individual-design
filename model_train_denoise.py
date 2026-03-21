# model_train_denoise.py
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model import MultiScale1DCNN
from preprocess_denoise import preprocess_dataset
from load_dataset import load_dataset


# --------------------------------------------------------
# 参数
# --------------------------------------------------------
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 40
SEVERITY_LOSS_WEIGHT = 10.0
MAX_DIAMETER = 0.028
RESULT_DIR = "denoise_train_results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------
# numpy -> tensor
# --------------------------------------------------------
def to_tensor(x, y_fault, y_severity):
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_fault = torch.tensor(y_fault, dtype=torch.long)
    y_severity = torch.tensor(y_severity / MAX_DIAMETER, dtype=torch.float32)
    return x, y_fault, y_severity


# --------------------------------------------------------
# 归一化直径 -> 严重程度区间类别
# 0: 0~0.0105
# 1: 0.0105~0.0175
# 2: 0.0175~0.0245
# 3: 0.0245~0.028
# --------------------------------------------------------
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
# 训练 1 epoch
# --------------------------------------------------------
def train_one_epoch(model, loader, criterion_fault, criterion_severity, optimizer):
    model.train()
    total_loss = 0.0
    correct_fault = 0
    total_fault = 0
    total_severity_abs_error = 0.0
    total_interval_correct = 0
    total_samples = 0

    for batch_x, batch_fault, batch_severity in loader:
        batch_x = batch_x.to(device)
        batch_fault = batch_fault.to(device)
        batch_severity = batch_severity.to(device)

        optimizer.zero_grad()

        fault_out, severity_out = model(batch_x)
        severity_out = torch.clamp(severity_out.view(-1), 0.0, 1.0)

        loss_fault = criterion_fault(fault_out, batch_fault)
        loss_severity = criterion_severity(severity_out, batch_severity)
        loss = loss_fault + SEVERITY_LOSS_WEIGHT * loss_severity

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

        _, preds = fault_out.max(1)
        correct_fault += preds.eq(batch_fault).sum().item()
        total_fault += batch_fault.size(0)

        total_severity_abs_error += torch.abs(severity_out - batch_severity).sum().item()

        pred_cls = np.array([normalized_diameter_to_class(v) for v in severity_out.detach().cpu().numpy()])
        true_cls = np.array([normalized_diameter_to_class(v) for v in batch_severity.detach().cpu().numpy()])
        total_interval_correct += (pred_cls == true_cls).sum()

        total_samples += batch_x.size(0)

    avg_loss = total_loss / total_fault
    fault_acc = correct_fault / total_fault
    severity_mae = total_severity_abs_error / total_samples
    severity_interval_acc = total_interval_correct / total_samples

    return avg_loss, fault_acc, severity_mae, severity_interval_acc


# --------------------------------------------------------
# 验证 / 测试
# --------------------------------------------------------
def evaluate(model, loader, criterion_fault, criterion_severity):
    model.eval()

    total_loss = 0.0
    correct_fault = 0
    total_fault = 0
    total_severity_abs_error = 0.0
    total_severity_sq_error = 0.0
    total_interval_correct = 0
    total_samples = 0

    all_fault_true = []
    all_fault_pred = []
    all_sev_true_cls = []
    all_sev_pred_cls = []

    with torch.no_grad():
        for batch_x, batch_fault, batch_severity in loader:
            batch_x = batch_x.to(device)
            batch_fault = batch_fault.to(device)
            batch_severity = batch_severity.to(device)

            fault_out, severity_out = model(batch_x)
            severity_out = torch.clamp(severity_out.view(-1), 0.0, 1.0)

            loss_fault = criterion_fault(fault_out, batch_fault)
            loss_severity = criterion_severity(severity_out, batch_severity)
            loss = loss_fault + SEVERITY_LOSS_WEIGHT * loss_severity
            total_loss += loss.item() * batch_x.size(0)

            _, fault_pred = fault_out.max(1)
            correct_fault += fault_pred.eq(batch_fault).sum().item()
            total_fault += batch_fault.size(0)

            diff = severity_out - batch_severity
            total_severity_abs_error += torch.abs(diff).sum().item()
            total_severity_sq_error += (diff ** 2).sum().item()

            sev_pred_cls = np.array([normalized_diameter_to_class(v) for v in severity_out.detach().cpu().numpy()])
            sev_true_cls = np.array([normalized_diameter_to_class(v) for v in batch_severity.detach().cpu().numpy()])

            total_interval_correct += (sev_pred_cls == sev_true_cls).sum()
            total_samples += batch_x.size(0)

            all_fault_true.extend(batch_fault.detach().cpu().numpy())
            all_fault_pred.extend(fault_pred.detach().cpu().numpy())
            all_sev_true_cls.extend(sev_true_cls)
            all_sev_pred_cls.extend(sev_pred_cls)

    avg_loss = total_loss / total_fault
    fault_acc = correct_fault / total_fault
    severity_mae = total_severity_abs_error / total_samples
    severity_rmse = np.sqrt(total_severity_sq_error / total_samples)
    severity_interval_acc = total_interval_correct / total_samples

    return (
        avg_loss, fault_acc, severity_mae, severity_rmse, severity_interval_acc,
        np.array(all_fault_true), np.array(all_fault_pred),
        np.array(all_sev_true_cls), np.array(all_sev_pred_cls)
    )


# --------------------------------------------------------
# 保存训练日志 CSV
# --------------------------------------------------------
def save_training_log(log_rows, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_fault_acc", "train_sev_mae", "train_sev_interval_acc",
            "val_loss", "val_fault_acc", "val_sev_mae", "val_sev_rmse", "val_sev_interval_acc"
        ])
        writer.writerows(log_rows)


# --------------------------------------------------------
# 画训练曲线
# --------------------------------------------------------
def plot_curve(values1, values2, ylabel, title, save_path, label1="Train", label2="Val"):
    plt.figure(figsize=(8, 5))
    plt.plot(values1, label=label1)
    plt.plot(values2, label=label2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# --------------------------------------------------------
# 保存混淆矩阵
# --------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, labels, display_labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# --------------------------------------------------------
# 主程序
# --------------------------------------------------------
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("Loading dataset with denoise preprocessing...")

    fault_root = r"D:\design\data\CWRU\12kDriveEndFault"
    normal_root = r"D:\design\data\CWRU\NormalBaseline"

    fault_dataset = load_dataset(fault_root)
    normal_dataset = load_dataset(normal_root)
    dataset = fault_dataset + normal_dataset

    print("Preprocessing dataset (denoise + normalize + window)...")
    x, y_fault, y_sev, loads = preprocess_dataset(dataset)

    # train / val / test = 70 / 15 / 15
    x_train, x_temp, y_fault_train, y_fault_temp, y_sev_train, y_sev_temp = train_test_split(
        x, y_fault, y_sev,
        test_size=0.3,
        random_state=42,
        stratify=y_fault
    )

    x_val, x_test, y_fault_val, y_fault_test, y_sev_val, y_sev_test = train_test_split(
        x_temp, y_fault_temp, y_sev_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_fault_temp
    )

    x_train, y_fault_train, y_sev_train = to_tensor(x_train, y_fault_train, y_sev_train)
    x_val, y_fault_val, y_sev_val = to_tensor(x_val, y_fault_val, y_sev_val)
    x_test, y_fault_test, y_sev_test = to_tensor(x_test, y_fault_test, y_sev_test)

    train_loader = DataLoader(
        TensorDataset(x_train, y_fault_train, y_sev_train),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_fault_val, y_sev_val),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_fault_test, y_sev_test),
        batch_size=BATCH_SIZE, shuffle=False
    )

    model = MultiScale1DCNN().to(device)

    criterion_fault = nn.CrossEntropyLoss()
    criterion_severity = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    best_score = -1.0
    best_model_path = os.path.join(RESULT_DIR, "multiscale_best.pth")

    log_rows = []

    train_losses = []
    val_losses = []
    train_fault_accs = []
    val_fault_accs = []
    train_sev_interval_accs = []
    val_sev_interval_accs = []

    print("Start training model with denoised input...")

    for epoch in range(EPOCHS):
        train_loss, train_acc, train_sev_mae, train_sev_interval_acc = train_one_epoch(
            model, train_loader, criterion_fault, criterion_severity, optimizer
        )

        val_result = evaluate(model, val_loader, criterion_fault, criterion_severity)
        val_loss, val_acc, val_sev_mae, val_sev_rmse, val_sev_interval_acc, _, _, _, _ = val_result

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
            f"Train Sev MAE: {train_sev_mae:.6f}, Train Sev Interval Acc: {train_sev_interval_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, "
            f"Val Sev MAE: {val_sev_mae:.6f}, Val Sev RMSE: {val_sev_rmse:.6f}, "
            f"Val Sev Interval Acc: {val_sev_interval_acc*100:.2f}%"
        )

        log_rows.append([
            epoch + 1,
            train_loss, train_acc, train_sev_mae, train_sev_interval_acc,
            val_loss, val_acc, val_sev_mae, val_sev_rmse, val_sev_interval_acc
        ])

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_fault_accs.append(train_acc)
        val_fault_accs.append(val_acc)
        train_sev_interval_accs.append(train_sev_interval_acc)
        val_sev_interval_accs.append(val_sev_interval_acc)

        # 按验证集 severity interval acc 保存最佳模型
        if val_sev_interval_acc > best_score:
            best_score = val_sev_interval_acc
            torch.save(model.state_dict(), best_model_path)

    print("Training finished.")
    print(f"Best Validation Severity Interval Accuracy: {best_score*100:.2f}%")

    # 保存训练日志
    save_training_log(log_rows, os.path.join(RESULT_DIR, "training_log.csv"))

    # 保存训练曲线
    plot_curve(
        train_losses, val_losses,
        ylabel="Loss",
        title="Training and Validation Loss",
        save_path=os.path.join(RESULT_DIR, "loss_curve.png")
    )

    plot_curve(
        train_fault_accs, val_fault_accs,
        ylabel="Fault Accuracy",
        title="Training and Validation Fault Accuracy",
        save_path=os.path.join(RESULT_DIR, "fault_acc_curve.png")
    )

    plot_curve(
        train_sev_interval_accs, val_sev_interval_accs,
        ylabel="Severity Interval Accuracy",
        title="Training and Validation Severity Interval Accuracy",
        save_path=os.path.join(RESULT_DIR, "severity_interval_acc_curve.png")
    )

    print("Evaluating on test set...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_result = evaluate(model, test_loader, criterion_fault, criterion_severity)
    (
        test_loss, test_acc, test_sev_mae, test_sev_rmse, test_sev_interval_acc,
        fault_true, fault_pred, sev_true_cls, sev_pred_cls
    ) = test_result

    print(
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_acc*100:.2f}% | "
        f"Test Sev MAE: {test_sev_mae:.6f} | "
        f"Test Sev RMSE: {test_sev_rmse:.6f} | "
        f"Test Sev Interval Acc: {test_sev_interval_acc*100:.2f}%"
    )

    # 保存测试指标
    with open(os.path.join(RESULT_DIR, "test_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Test Fault Accuracy: {test_acc*100:.2f}%\n")
        f.write(f"Test Severity MAE: {test_sev_mae:.6f}\n")
        f.write(f"Test Severity RMSE: {test_sev_rmse:.6f}\n")
        f.write(f"Test Severity Interval Accuracy: {test_sev_interval_acc*100:.2f}%\n")

    # 保存故障混淆矩阵
    save_confusion_matrix(
        fault_true, fault_pred,
        labels=[0, 1, 2, 3],
        display_labels=["Normal", "Ball", "Inner", "Outer"],
        title="Fault Confusion Matrix",
        save_path=os.path.join(RESULT_DIR, "fault_confusion_matrix.png")
    )

    # 保存严重程度混淆矩阵
    save_confusion_matrix(
        sev_true_cls, sev_pred_cls,
        labels=[0, 1, 2, 3],
        display_labels=["Low", "Medium", "High", "Very High"],
        title="Severity Confusion Matrix",
        save_path=os.path.join(RESULT_DIR, "severity_confusion_matrix.png")
    )

    print(f"All results have been saved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()