# fault_result.py
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from load_dataset import load_dataset
from dataset_split import split_dataset
from model import MultiScale1DCNN


# --------------------------------------------------------
# 参数
# --------------------------------------------------------
BATCH_SIZE = 64
MODEL_PATH = "multiscale_best.pth"
RESULT_DIR = "fault_results"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULT_DIR, exist_ok=True)


# --------------------------------------------------------
# numpy → tensor
# --------------------------------------------------------
def to_tensor(x, y):
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long)
    return x, y


# --------------------------------------------------------
# 测试集评估
# --------------------------------------------------------
def evaluate_model(model, loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            fault_out, _ = model(batch_x)
            preds = torch.argmax(fault_out, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


# --------------------------------------------------------
# 绘制混淆矩阵
# --------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
    plt.close()


# --------------------------------------------------------
# 绘制识别率柱状图
# --------------------------------------------------------
def plot_accuracy_bar(y_true, y_pred, class_names):
    acc_per_class = []

    for i in range(len(class_names)):
        idx = (y_true == i)
        acc = (y_pred[idx] == i).mean() if np.sum(idx) > 0 else 0
        acc_per_class.append(acc)

    plt.figure(figsize=(6, 4))
    plt.bar(class_names, acc_per_class)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Per-Class Recognition Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "accuracy_curve.png"))
    plt.close()


# --------------------------------------------------------
# 主程序
# --------------------------------------------------------
def main():
    print("Loading dataset...")
    data = load_dataset("data/CWRU")

    _, _, x_test, \
    _, _, y_test, \
    _, _, _ = split_dataset(data)

    x_test, y_test = to_tensor(x_test, y_test)

    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Loading trained model...")
    model = MultiScale1DCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    print("Evaluating on test set...")
    y_true, y_pred = evaluate_model(model, test_loader)

    class_names = ["Normal", "Ball", "Inner", "Outer"]

    # ----------------------------------------------------
    # 指标输出
    # ----------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )

    print("Test Accuracy:", acc)
    print(report)

    with open(os.path.join(RESULT_DIR, "metrics.txt"), "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # ----------------------------------------------------
    # 绘图
    # ----------------------------------------------------
    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_accuracy_bar(y_true, y_pred, class_names)

    print("Results saved to:", RESULT_DIR)


if __name__ == "__main__":
    main()