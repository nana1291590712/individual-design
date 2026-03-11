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

# ✅ 建议提高一些（mask 后更安全）
SEVERITY_LOSS_WEIGHT = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------
# numpy → tensor
# --------------------------------------------------------
def to_tensor(x, y_fault, y_severity):
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_fault = torch.tensor(y_fault, dtype=torch.long)
    y_severity = torch.tensor(y_severity, dtype=torch.long)
    return x, y_fault, y_severity


# --------------------------------------------------------
# 训练 1 epoch
# --------------------------------------------------------
def train_one_epoch(model, loader, criterion_fault, criterion_severity, optimizer):
    model.train()
    total_loss = 0.0

    correct_fault = 0
    total_fault = 0

    for batch_x, batch_fault, batch_severity in loader:
        batch_x = batch_x.to(device)
        batch_fault = batch_fault.to(device)
        batch_severity = batch_severity.to(device)

        optimizer.zero_grad()

        fault_out, severity_out = model(batch_x)

        loss_fault = criterion_fault(fault_out, batch_fault)

        # ✅ severity 只对 severity!=-1 的样本计算
        mask = (batch_severity != -1)
        if mask.any():
            loss_severity = criterion_severity(severity_out[mask], batch_severity[mask])
        else:
            loss_severity = torch.tensor(0.0, device=device)

        loss = loss_fault + SEVERITY_LOSS_WEIGHT * loss_severity
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

        # fault accuracy
        _, preds = fault_out.max(1)
        correct_fault += preds.eq(batch_fault).sum().item()
        total_fault += batch_fault.size(0)

    return total_loss / total_fault, correct_fault / total_fault


# --------------------------------------------------------
# 验证
# --------------------------------------------------------
def evaluate(model, loader, criterion_fault, criterion_severity):
    model.eval()
    total_loss = 0.0

    correct_fault = 0
    total_fault = 0

    with torch.no_grad():
        for batch_x, batch_fault, batch_severity in loader:
            batch_x = batch_x.to(device)
            batch_fault = batch_fault.to(device)
            batch_severity = batch_severity.to(device)

            fault_out, severity_out = model(batch_x)

            loss_fault = criterion_fault(fault_out, batch_fault)

            mask = (batch_severity != -1)
            if mask.any():
                loss_severity = criterion_severity(severity_out[mask], batch_severity[mask])
            else:
                loss_severity = torch.tensor(0.0, device=device)

            loss = loss_fault + SEVERITY_LOSS_WEIGHT * loss_severity
            total_loss += loss.item() * batch_x.size(0)

            _, preds = fault_out.max(1)
            correct_fault += preds.eq(batch_fault).sum().item()
            total_fault += batch_fault.size(0)

    return total_loss / total_fault, correct_fault / total_fault


# --------------------------------------------------------
# 主程序
# --------------------------------------------------------
def main():
    print("Loading raw CWRU dataset...")
    data = load_dataset("data/CWRU")
    print("Loaded items:", len(data))
    print("Example item:", data[0])

    print("Dataset loaded. Now splitting...")

    x_train, x_val, x_test, \
    y_fault_train, y_fault_val, y_fault_test, \
    y_sev_train, y_sev_val, y_sev_test = split_dataset(input_dataset=data)

    # ----------------------------------------------------
    # ✅ 统计 severity 类别分布（只统计故障 severity: 0/1/2）
    # ----------------------------------------------------
    sev_train_valid = y_sev_train[y_sev_train != -1]
    if len(sev_train_valid) == 0:
        raise RuntimeError("No valid severity labels found in training set (all -1).")

    counts = np.bincount(sev_train_valid, minlength=3)  # [low, med, high]
    print("Train severity counts (fault-only):", counts.tolist())

    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.mean()  # normalize around 1
    sev_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print("Severity class weights:", weights.tolist())

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

    # ✅ ignore_index=-1 + class weights
    criterion_severity = nn.CrossEntropyLoss(
        weight=sev_weights,
        ignore_index=-1
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    print("Start training improved MultiScale1DCNN...")
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader,
            criterion_fault, criterion_severity,
            optimizer
        )

        val_loss, val_acc = evaluate(
            model, val_loader,
            criterion_fault, criterion_severity
        )

        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "multiscale_best.pth")

    print("Training finished.")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")

    print("Evaluating on test set...")
    model.load_state_dict(torch.load("multiscale_best.pth", map_location=device))
    test_loss, test_acc = evaluate(
        model, test_loader,
        criterion_fault, criterion_severity
    )

    print(f"Test Accuracy: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()