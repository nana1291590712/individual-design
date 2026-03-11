"""
optimizer_curves.py
-------------------------------------------------
Purpose:
    Visualize optimizer and scheduler behavior for Adam +
    ReduceLROnPlateau based on model_train.py configuration.

Outputs:
    - Learning rate vs epoch
    - Training / validation loss vs epoch
    - Training / validation accuracy vs epoch

This script does NOT save models and does NOT
affect existing experiments.
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from load_dataset import load_dataset
from dataset_split import split_dataset
from model import MultiScale1DCNN


# -------------------------------------------------
# Hyper-parameters (keep consistent with model_train.py)
# -------------------------------------------------
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 40
SEVERITY_LOSS_WEIGHT = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# numpy → tensor
# -------------------------------------------------
def to_tensor(x, y_fault, y_severity):
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_fault = torch.tensor(y_fault, dtype=torch.long)
    y_severity = torch.tensor(y_severity, dtype=torch.long)
    return x, y_fault, y_severity


# -------------------------------------------------
# Train one epoch
# -------------------------------------------------
def train_one_epoch(model, loader,
                    criterion_fault, criterion_severity,
                    optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y_fault, y_sev in loader:
        x = x.to(device)
        y_fault = y_fault.to(device)
        y_sev = y_sev.to(device)

        optimizer.zero_grad()

        fault_out, sev_out = model(x)
        loss_fault = criterion_fault(fault_out, y_fault)
        loss_sev = criterion_severity(sev_out, y_sev)

        loss = loss_fault + SEVERITY_LOSS_WEIGHT * loss_sev
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, preds = fault_out.max(1)
        correct += preds.eq(y_fault).sum().item()
        total += y_fault.size(0)

    return total_loss / total, correct / total


# -------------------------------------------------
# Validation
# -------------------------------------------------
def evaluate(model, loader,
             criterion_fault, criterion_severity):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y_fault, y_sev in loader:
            x = x.to(device)
            y_fault = y_fault.to(device)
            y_sev = y_sev.to(device)

            fault_out, sev_out = model(x)
            loss_fault = criterion_fault(fault_out, y_fault)
            loss_sev = criterion_severity(sev_out, y_sev)

            loss = loss_fault + SEVERITY_LOSS_WEIGHT * loss_sev

            total_loss += loss.item() * x.size(0)
            _, preds = fault_out.max(1)
            correct += preds.eq(y_fault).sum().item()
            total += y_fault.size(0)

    return total_loss / total, correct / total


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():

    print("Loading dataset...")
    data = load_dataset("data/CWRU")

    x_train, x_val, _, \
    y_fault_train, y_fault_val, _, \
    y_sev_train, y_sev_val, _ = split_dataset(data)

    x_train, y_fault_train, y_sev_train = to_tensor(
        x_train, y_fault_train, y_sev_train
    )
    x_val, y_fault_val, y_sev_val = to_tensor(
        x_val, y_fault_val, y_sev_val
    )

    train_loader = DataLoader(
        TensorDataset(x_train, y_fault_train, y_sev_train),
        batch_size=BATCH_SIZE, shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(x_val, y_fault_val, y_sev_val),
        batch_size=BATCH_SIZE, shuffle=False
    )

    model = MultiScale1DCNN().to(device)

    criterion_fault = nn.CrossEntropyLoss()
    criterion_severity = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    # -------------------------------------------------
    # History containers
    # -------------------------------------------------
    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []
    lr_hist = []

    print("Start training for optimizer analysis...")

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

        current_lr = optimizer.param_groups[0]["lr"]

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        lr_hist.append(current_lr)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    # -------------------------------------------------
    # Plotting
    # -------------------------------------------------
    save_dir = "results/optimizer_analysis"
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, EPOCHS + 1)

    # ---- Loss curve ----
    plt.figure()
    plt.plot(epochs, train_loss_hist, label="Train Loss")
    plt.plot(epochs, val_loss_hist, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # ---- Accuracy curve ----
    plt.figure()
    plt.plot(epochs, train_acc_hist, label="Train Accuracy")
    plt.plot(epochs, val_acc_hist, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=300)
    plt.close()

    # ---- Learning rate curve ----
    plt.figure()
    plt.plot(epochs, lr_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lr_curve.png"), dpi=300)
    plt.close()

    print("\nOptimizer curves saved to:", save_dir)


if __name__ == "__main__":
    main()
