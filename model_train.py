# model_train.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from load_dataset import load_dataset
from dataset_split import split_dataset
from model import MultiScale1DCNN


# --------------------------------------------------------
# 参数
# --------------------------------------------------------
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 60

# 关键：提高 severity 分支权重
SEVERITY_LOSS_WEIGHT = 10.0

# 关键：将直径归一化到 [0,1]
MAX_DIAMETER = 0.028

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------
# numpy → tensor
# --------------------------------------------------------
def to_tensor(x, y_fault, y_severity):
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_fault = torch.tensor(y_fault, dtype=torch.long)

    # severity 归一化到 [0,1]
    y_severity = torch.tensor(y_severity / MAX_DIAMETER, dtype=torch.float32)

    return x, y_fault, y_severity


# --------------------------------------------------------
# 计算 interval accuracy
# 输入和输出都应为归一化后的 [0,1] 直径
# --------------------------------------------------------
def normalized_diameter_to_class(d):
    d = float(d)

    # 原始中点阈值：0.0105, 0.0175, 0.0245
    # 归一化后：
    # 0.0105 / 0.028 = 0.375
    # 0.0175 / 0.028 = 0.625
    # 0.0245 / 0.028 = 0.875
    if d < 0.375:
        return 0
    elif d < 0.625:
        return 1
    elif d < 0.875:
        return 2
    else:
        return 3


def compute_interval_accuracy(pred, target):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    pred_cls = np.array([normalized_diameter_to_class(p) for p in pred])
    true_cls = np.array([normalized_diameter_to_class(t) for t in target])

    return (pred_cls == true_cls).mean()


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

        # 裁剪到 [0,1]，避免回归值明显越界
        severity_out = torch.clamp(severity_out, min=0.0, max=1.0)

        loss_fault = criterion_fault(fault_out, batch_fault)
        loss_severity = criterion_severity(severity_out, batch_severity)

        loss = loss_fault + SEVERITY_LOSS_WEIGHT * loss_severity
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

        # fault accuracy
        _, preds = fault_out.max(1)
        correct_fault += preds.eq(batch_fault).sum().item()
        total_fault += batch_fault.size(0)

        # severity MAE
        total_severity_abs_error += torch.abs(severity_out - batch_severity).sum().item()

        # severity interval acc
        pred_cls = torch.tensor(
            [normalized_diameter_to_class(v) for v in severity_out.detach().cpu().numpy()]
        )
        true_cls = torch.tensor(
            [normalized_diameter_to_class(v) for v in batch_severity.detach().cpu().numpy()]
        )
        total_interval_correct += (pred_cls == true_cls).sum().item()

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

    with torch.no_grad():
        for batch_x, batch_fault, batch_severity in loader:
            batch_x = batch_x.to(device)
            batch_fault = batch_fault.to(device)
            batch_severity = batch_severity.to(device)

            fault_out, severity_out = model(batch_x)

            severity_out = torch.clamp(severity_out, min=0.0, max=1.0)

            loss_fault = criterion_fault(fault_out, batch_fault)
            loss_severity = criterion_severity(severity_out, batch_severity)

            loss = loss_fault + SEVERITY_LOSS_WEIGHT * loss_severity
            total_loss += loss.item() * batch_x.size(0)

            # fault accuracy
            _, preds = fault_out.max(1)
            correct_fault += preds.eq(batch_fault).sum().item()
            total_fault += batch_fault.size(0)

            # severity errors
            diff = severity_out - batch_severity
            total_severity_abs_error += torch.abs(diff).sum().item()
            total_severity_sq_error += (diff ** 2).sum().item()

            pred_cls = torch.tensor(
                [normalized_diameter_to_class(v) for v in severity_out.detach().cpu().numpy()]
            )
            true_cls = torch.tensor(
                [normalized_diameter_to_class(v) for v in batch_severity.detach().cpu().numpy()]
            )
            total_interval_correct += (pred_cls == true_cls).sum().item()

            total_samples += batch_x.size(0)

    avg_loss = total_loss / total_fault
    fault_acc = correct_fault / total_fault
    severity_mae = total_severity_abs_error / total_samples
    severity_rmse = np.sqrt(total_severity_sq_error / total_samples)
    severity_interval_acc = total_interval_correct / total_samples

    return avg_loss, fault_acc, severity_mae, severity_rmse, severity_interval_acc


# --------------------------------------------------------
# 主程序
# --------------------------------------------------------
def main():
    print("Loading raw CWRU dataset...")

    fault_root = r"D:\design\data\CWRU\12kDriveEndFault"
    normal_root = r"D:\design\data\CWRU\NormalBaseline"

    fault_dataset = load_dataset(fault_root)
    normal_dataset = load_dataset(normal_root)
    data = fault_dataset + normal_dataset

    print("Loaded items:", len(data))
    print("Example item:", data[0])

    print("Dataset loaded. Now splitting...")

    x_train, x_val, x_test, \
    y_fault_train, y_fault_val, y_fault_test, \
    y_sev_train, y_sev_val, y_sev_test = split_dataset(input_dataset=data)

    print("Train severity range:", float(np.min(y_sev_train)), "to", float(np.max(y_sev_train)))
    print("Val severity range:", float(np.min(y_sev_val)), "to", float(np.max(y_sev_val)))
    print("Test severity range:", float(np.min(y_sev_test)), "to", float(np.max(y_sev_test)))

    # ----------------------------------------------------
    # numpy → tensor
    # ----------------------------------------------------
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

    print("Start training improved MultiScale1DCNN...")

    # 不再只按 fault acc 保存
    best_score = -1.0

    for epoch in range(EPOCHS):
        train_loss, train_acc, train_sev_mae, train_sev_interval_acc = train_one_epoch(
            model, train_loader,
            criterion_fault, criterion_severity,
            optimizer
        )

        val_loss, val_acc, val_sev_mae, val_sev_rmse, val_sev_interval_acc = evaluate(
            model, val_loader,
            criterion_fault, criterion_severity
        )

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
            f"Train Sev MAE: {train_sev_mae:.6f}, Train Sev Interval Acc: {train_sev_interval_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, "
            f"Val Sev MAE: {val_sev_mae:.6f}, Val Sev RMSE: {val_sev_rmse:.6f}, "
            f"Val Sev Interval Acc: {val_sev_interval_acc*100:.2f}%"
        )

        # 关键：以 severity interval acc 为主保存模型
        score = val_sev_interval_acc
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "multiscale_best.pth")

    print("Training finished.")
    print(f"Best Validation Severity Interval Accuracy: {best_score*100:.2f}%")

    print("Evaluating on test set...")
    model.load_state_dict(torch.load("multiscale_best.pth", map_location=device))

    test_loss, test_acc, test_sev_mae, test_sev_rmse, test_sev_interval_acc = evaluate(
        model, test_loader,
        criterion_fault, criterion_severity
    )

    print(
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_acc*100:.2f}% | "
        f"Test Sev MAE: {test_sev_mae:.6f} | "
        f"Test Sev RMSE: {test_sev_rmse:.6f} | "
        f"Test Sev Interval Acc: {test_sev_interval_acc*100:.2f}%"
    )


if __name__ == "__main__":
    main()