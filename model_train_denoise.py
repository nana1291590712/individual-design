# model_train_denoise_cm.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

from load_dataset import load_dataset
from preprocess_denoise import preprocess_dataset_denoise
from model import MultiScale1DCNN


# --------------------------------------------------------
# 参数
# --------------------------------------------------------
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 40
SEVERITY_LOSS_WEIGHT = 0.3
BEST_MODEL_PATH = "multiscale_best_denoise.pth"

VAL_RATIO = 0.2
RANDOM_SEED = 42
TEST_LOAD = 3

DATA_ROOT = r"data/CWRU"
RESULT_DIR = "denoise_train_results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULT_DIR, exist_ok=True)


# --------------------------------------------------------
# numpy → tensor
# --------------------------------------------------------
def to_tensor(x, y_fault, y_severity):
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_fault = torch.tensor(y_fault, dtype=torch.long)
    y_severity = torch.tensor(y_severity, dtype=torch.long)
    return x, y_fault, y_severity


# --------------------------------------------------------
# 固定测试负载划分：Train/Val=0/1/2HP, Test=3HP
# --------------------------------------------------------
def split_by_load(x, y_fault, y_sev, loads,
                  test_load=TEST_LOAD, val_ratio=VAL_RATIO, seed=RANDOM_SEED):
    x = np.asarray(x)
    y_fault = np.asarray(y_fault)
    y_sev = np.asarray(y_sev)
    loads = np.asarray(loads)

    idx_test = np.where(loads == test_load)[0]
    idx_trainval = np.where(loads != test_load)[0]

    if len(idx_test) == 0:
        raise ValueError(f"No samples found for test_load={test_load}. Check parse_load / dataset path.")

    x_trainval = x[idx_trainval]
    y_fault_trainval = y_fault[idx_trainval]
    y_sev_trainval = y_sev[idx_trainval]

    x_test = x[idx_test]
    y_fault_test = y_fault[idx_test]
    y_sev_test = y_sev[idx_test]

    # 分层按 fault label（兜底：失败就不分层）
    try:
        x_train, x_val, y_fault_train, y_fault_val, y_sev_train, y_sev_val = train_test_split(
            x_trainval, y_fault_trainval, y_sev_trainval,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
            stratify=y_fault_trainval
        )
    except ValueError:
        x_train, x_val, y_fault_train, y_fault_val, y_sev_train, y_sev_val = train_test_split(
            x_trainval, y_fault_trainval, y_sev_trainval,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True
        )

    return (x_train, x_val, x_test,
            y_fault_train, y_fault_val, y_fault_test,
            y_sev_train, y_sev_val, y_sev_test)


# --------------------------------------------------------
# 训练 1 epoch
# --------------------------------------------------------
def train_one_epoch(model, loader, criterion_fault, criterion_severity, optimizer):
    model.train()
    total_loss = 0.0
    correct_fault = 0
    total = 0

    for batch_x, batch_fault, batch_severity in loader:
        batch_x = batch_x.to(device)
        batch_fault = batch_fault.to(device)
        batch_severity = batch_severity.to(device)

        optimizer.zero_grad()

        fault_out, severity_out = model(batch_x)

        loss_fault = criterion_fault(fault_out, batch_fault)
        loss_severity = criterion_severity(severity_out, batch_severity)
        loss = loss_fault + SEVERITY_LOSS_WEIGHT * loss_severity

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

        _, preds_fault = fault_out.max(1)
        correct_fault += preds_fault.eq(batch_fault).sum().item()
        total += batch_fault.size(0)

    return total_loss / total, correct_fault / total


# --------------------------------------------------------
# 验证/测试：返回 loss + fault_acc + sev_acc
# --------------------------------------------------------
def evaluate(model, loader, criterion_fault, criterion_severity):
    model.eval()
    total_loss = 0.0
    correct_fault = 0
    correct_sev = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_fault, batch_severity in loader:
            batch_x = batch_x.to(device)
            batch_fault = batch_fault.to(device)
            batch_severity = batch_severity.to(device)

            fault_out, severity_out = model(batch_x)

            loss_fault = criterion_fault(fault_out, batch_fault)
            loss_severity = criterion_severity(severity_out, batch_severity)
            loss = loss_fault + SEVERITY_LOSS_WEIGHT * loss_severity

            total_loss += loss.item() * batch_x.size(0)

            _, preds_fault = fault_out.max(1)
            _, preds_sev = severity_out.max(1)

            correct_fault += preds_fault.eq(batch_fault).sum().item()
            correct_sev += preds_sev.eq(batch_severity).sum().item()
            total += batch_fault.size(0)

    return total_loss / total, correct_fault / total, correct_sev / total


# --------------------------------------------------------
# 获取预测（用于混淆矩阵）
# --------------------------------------------------------
def predict_all(model, loader):
    model.eval()
    all_fault_true, all_fault_pred = [], []
    all_sev_true, all_sev_pred = [], []

    with torch.no_grad():
        for batch_x, batch_fault, batch_severity in loader:
            batch_x = batch_x.to(device)

            fault_out, severity_out = model(batch_x)
            fault_pred = fault_out.argmax(dim=1).cpu().numpy()
            sev_pred = severity_out.argmax(dim=1).cpu().numpy()

            all_fault_true.extend(batch_fault.numpy())
            all_fault_pred.extend(fault_pred)
            all_sev_true.extend(batch_severity.numpy())
            all_sev_pred.extend(sev_pred)

    return (np.array(all_fault_true), np.array(all_fault_pred),
            np.array(all_sev_true), np.array(all_sev_pred))


# --------------------------------------------------------
# 画曲线
# --------------------------------------------------------
def plot_curves(history, save_path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # Loss curve
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_loss.png"), dpi=200)
    plt.close()

    # Fault acc curve
    plt.figure()
    plt.plot(epochs, history["train_fault_acc"], label="Train Fault Acc")
    plt.plot(epochs, history["val_fault_acc"], label="Val Fault Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_fault_acc.png"), dpi=200)
    plt.close()

    # Severity acc curve
    plt.figure()
    plt.plot(epochs, history["val_sev_acc"], label="Val Severity Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_sev_acc.png"), dpi=200)
    plt.close()


# --------------------------------------------------------
# 画混淆矩阵
# --------------------------------------------------------
def plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(6, 5))

    im = plt.imshow(
        cm,
        interpolation="nearest",
        cmap=plt.cm.Blues,   # 蓝-白色系
        vmin=0               # 0 -> 白色
    )

    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # 数值标注（深色背景用白字，其余用黑字）
    thresh = cm.max() * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



# --------------------------------------------------------
# 主程序
# --------------------------------------------------------
def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Loading raw CWRU dataset...")
    data = load_dataset(DATA_ROOT)
    print("Loaded items:", len(data))
    print("Example item:", data[0])

    print("\nPreprocessing with denoise/filter...")
    x, y_fault, y_sev, loads = preprocess_dataset_denoise(data)

    print("X shape:", x.shape)
    print("Fault unique:", np.unique(y_fault))
    print("Severity unique:", np.unique(y_sev))
    print("Load unique:", np.unique(loads))

    print("\nSplitting dataset (Train/Val: !=3HP, Test: 3HP)...")
    (x_train, x_val, x_test,
     y_fault_train, y_fault_val, y_fault_test,
     y_sev_train, y_sev_val, y_sev_test) = split_by_load(
        x, y_fault, y_sev, loads, test_load=TEST_LOAD
    )

    print("Train:", x_train.shape, y_fault_train.shape)
    print("Val:  ", x_val.shape, y_fault_val.shape)
    print("Test: ", x_test.shape, y_fault_test.shape)

    x_train, y_fault_train, y_sev_train = to_tensor(x_train, y_fault_train, y_sev_train)
    x_val, y_fault_val, y_sev_val = to_tensor(x_val, y_fault_val, y_sev_val)
    x_test, y_fault_test, y_sev_test = to_tensor(x_test, y_fault_test, y_sev_test)

    train_loader = DataLoader(TensorDataset(x_train, y_fault_train, y_sev_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_fault_val, y_sev_val),
                            batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_fault_test, y_sev_test),
                             batch_size=BATCH_SIZE, shuffle=False)

    model = MultiScale1DCNN().to(device)
    criterion_fault = nn.CrossEntropyLoss()
    criterion_severity = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_fault_acc": [],
        "val_fault_acc": [],
        "val_sev_acc": []
    }

    print("\nStart training MultiScale1DCNN (denoise)...")
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_fault_acc = train_one_epoch(
            model, train_loader, criterion_fault, criterion_severity, optimizer
        )
        val_loss, val_fault_acc, val_sev_acc = evaluate(
            model, val_loader, criterion_fault, criterion_severity
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_fault_acc"].append(train_fault_acc)
        history["val_fault_acc"].append(val_fault_acc)
        history["val_sev_acc"].append(val_sev_acc)

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f}, Train Fault Acc: {train_fault_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Fault Acc: {val_fault_acc*100:.2f}%, Val Sev Acc: {val_sev_acc*100:.2f}%")

        if val_fault_acc > best_val_acc:
            best_val_acc = val_fault_acc
            torch.save(model.state_dict(), os.path.join(RESULT_DIR, BEST_MODEL_PATH))

    print("\nTraining finished.")
    print(f"Best Validation Fault Accuracy: {best_val_acc*100:.2f}%")

    # 1) 保存曲线
    curve_base = os.path.join(RESULT_DIR, "curves.png")
    plot_curves(history, curve_base)
    print("Saved curves to:", os.path.abspath(RESULT_DIR))

    # 2) 测试集评估 + 混淆矩阵
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(RESULT_DIR, BEST_MODEL_PATH), map_location=device))

    test_loss, test_fault_acc, test_sev_acc = evaluate(
        model, test_loader, criterion_fault, criterion_severity
    )
    print(f"Test Loss: {test_loss:.4f} | Test Fault Acc: {test_fault_acc*100:.2f}% | Test Severity Acc: {test_sev_acc*100:.2f}%")

    fault_true, fault_pred, sev_true, sev_pred = predict_all(model, test_loader)

    # Fault CM（注意：你现在数据集不含 Normal，所以默认类名用 3 类）
    fault_classes_present = np.unique(np.concatenate([fault_true, fault_pred]))
    if set(fault_classes_present.tolist()) == {1, 2, 3}:
        fault_names = ["Ball", "Inner", "Outer"]
        labels_fault = [1, 2, 3]
    else:
        # 万一你后面加了 Normal，就自动扩展
        full_map = {0: "Normal", 1: "Ball", 2: "Inner", 3: "Outer"}
        labels_fault = sorted(fault_classes_present.tolist())
        fault_names = [full_map.get(i, str(i)) for i in labels_fault]

    cm_fault = confusion_matrix(fault_true, fault_pred, labels=labels_fault)
    plot_confusion_matrix(
        cm_fault, fault_names,
        title="Fault Confusion Matrix (Test)",
        save_path=os.path.join(RESULT_DIR, "cm_fault_test.png")
    )

    # Severity CM（固定 3 类 0/1/2）
    sev_names = ["Level0", "Level1", "Level2"]
    cm_sev = confusion_matrix(sev_true, sev_pred, labels=[0, 1, 2])
    plot_confusion_matrix(
        cm_sev, sev_names,
        title="Severity Confusion Matrix (Test)",
        save_path=os.path.join(RESULT_DIR, "cm_severity_test.png")
    )

    print("Saved confusion matrices to:", os.path.abspath(RESULT_DIR))


if __name__ == "__main__":
    main()
