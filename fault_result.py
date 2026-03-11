import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# ======================================================
# 通用：绘制混淆矩阵（浅蓝 + 白色）
# ======================================================
def plot_confusion_matrix(cm, class_names, save_path, title):
    plt.figure(figsize=(6, 5))
    plt.imshow(
        cm,
        interpolation="nearest",
        cmap=plt.cm.Blues,
        vmin=0
    )
    plt.title(title)
    plt.colorbar(shrink=0.8)

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ======================================================
# ⭐ 给 lolo_experiment.py 调用的接口
# ======================================================
def evaluate_fault_results(y_true, y_pred, class_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(
        cm,
        class_names,
        save_path=os.path.join(save_dir, "confusion_matrix.png"),
        title="Fault Confusion Matrix"
    )

    # ---- Accuracy ----
    acc = accuracy_score(y_true, y_pred)

    # ---- Report ----
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )

    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    print(f"[Saved] {save_dir}")


# ======================================================
# （可选）单独运行测试用
# ======================================================
if __name__ == "__main__":
    print("This file is intended to be imported by lolo_experiment.py")
