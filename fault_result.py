# fault_result.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import TensorDataset, DataLoader

from load_dataset import load_dataset
from dataset_split import split_dataset
from model import MultiScale1DCNN


# ======================================================
# 路径配置
# ======================================================
FAULT_ROOT = r"D:\design\data\CWRU\12kDriveEndFault"
NORMAL_ROOT = r"D:\design\data\CWRU\NormalBaseline"

MODEL_PATH = r"D:\design\multiscale_best.pth"

SAVE_DIR = r"D:\design\fault_results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Normal", "Ball", "Inner", "Outer"]


# ======================================================
# 绘制混淆矩阵
# ======================================================
def plot_confusion_matrix(cm, class_names, save_path, title):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0)
    plt.title(title)
    plt.colorbar(shrink=0.8)

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ======================================================
# 将测试集转为 DataLoader
# ======================================================
def to_fault_test_loader(x_test, y_fault_test, batch_size=64):
    x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
    y_fault_test = torch.tensor(y_fault_test, dtype=torch.long)

    test_dataset = TensorDataset(x_test, y_fault_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


# ======================================================
# 加载模型权重
# ======================================================
def load_fault_model(model_path, device):
    model = MultiScale1DCNN().to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


# ======================================================
# 故障评估
# ======================================================
def evaluate_fault_results(model, test_loader, class_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_x, batch_fault in test_loader:
            batch_x = batch_x.to(device)
            batch_fault = batch_fault.to(device)

            outputs = model(batch_x)

            if isinstance(outputs, tuple):
                fault_logits = outputs[0]
            else:
                fault_logits = outputs

            preds = torch.argmax(fault_logits, dim=1)

            y_true.extend(batch_fault.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    labels = list(range(len(class_names)))

    acc = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion_matrix(
        cm,
        class_names,
        save_path=os.path.join(save_dir, "fault_confusion_matrix.png"),
        title="Fault Confusion Matrix"
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    with open(os.path.join(save_dir, "fault_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("===== Fault Classification Results =====\n\n")
        f.write(f"Model Path: {MODEL_PATH}\n\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    np.save(os.path.join(save_dir, "fault_confusion_matrix.npy"), cm)

    print("===== Fault Classification Results =====")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print(f"\n[Saved] {save_dir}")


# ======================================================
# 主程序
# ======================================================
def main():
    print("Loading dataset...")

    fault_dataset = load_dataset(FAULT_ROOT)
    normal_dataset = load_dataset(NORMAL_ROOT)
    data = fault_dataset + normal_dataset

    print("Total loaded items:", len(data))

    print("Splitting dataset...")
    x_train, x_val, x_test, \
    y_fault_train, y_fault_val, y_fault_test, \
    y_sev_train, y_sev_val, y_sev_test = split_dataset(input_dataset=data)

    print("Test set shape:", x_test.shape)
    print("Test fault labels shape:", y_fault_test.shape)
    print("Test fault unique:", np.unique(y_fault_test))

    test_loader = to_fault_test_loader(x_test, y_fault_test, batch_size=64)

    print("Loading trained model...")
    model = load_fault_model(MODEL_PATH, device)

    print("Evaluating fault prediction...")
    evaluate_fault_results(model, test_loader, CLASS_NAMES, SAVE_DIR)


if __name__ == "__main__":
    main()