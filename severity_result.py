import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from load_dataset import load_dataset
from dataset_split import split_dataset
from model import MultiScale1DCNN


# --------------------------------------------------------
# 参数
# --------------------------------------------------------
BATCH_SIZE = 64
MODEL_PATH = "multiscale_best.pth"
RESULT_DIR = "severity_results"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULT_DIR, exist_ok=True)


# --------------------------------------------------------
# numpy → tensor
# --------------------------------------------------------
def to_tensor(x, y_fault, y_sev):
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_fault = torch.tensor(y_fault, dtype=torch.long)
    y_sev = torch.tensor(y_sev, dtype=torch.long)
    return x, y_fault, y_sev


# --------------------------------------------------------
# 评估危险等级（仅 fault ≠ Normal）
# --------------------------------------------------------
def evaluate_severity(model, loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_fault, batch_sev in loader:
            batch_x = batch_x.to(device)
            batch_fault = batch_fault.to(device)
            batch_sev = batch_sev.to(device)

            _, sev_out = model(batch_x)
            sev_pred = torch.argmax(sev_out, dim=1)

            # 仅对非 Normal 样本统计 Severity
            for i in range(len(batch_fault)):
                if batch_fault[i].item() != 0:
                    all_preds.append(sev_pred[i].item())
                    all_labels.append(batch_sev[i].item())

    return np.array(all_labels), np.array(all_preds)


# --------------------------------------------------------
# 绘制危险等级混淆矩阵（浅蓝 + 白色）
# --------------------------------------------------------
def plot_severity_confusion_matrix(y_true, y_pred):
    class_names = ["Low", "Medium", "High"]
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(
        cm,
        interpolation="nearest",
        cmap=plt.cm.Blues,  # 浅蓝色
        vmin=0              # 0 显示为白色
    )
    plt.title("Severity Confusion Matrix")
    plt.colorbar(shrink=0.8)

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names)
    plt.yticks(ticks, class_names)

    # 自动切换文字颜色
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.xlabel("Predicted Severity")
    plt.ylabel("True Severity")
    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULT_DIR, "severity_confusion_matrix.png"),
        dpi=300
    )
    plt.close()


# --------------------------------------------------------
# 主程序
# --------------------------------------------------------
def main():
    print("Loading dataset...")
    data = load_dataset("data/CWRU")

    _, _, x_test, \
    _, _, y_fault_test, \
    _, _, y_sev_test = split_dataset(data)

    x_test, y_fault_test, y_sev_test = to_tensor(
        x_test, y_fault_test, y_sev_test
    )

    test_loader = DataLoader(
        TensorDataset(x_test, y_fault_test, y_sev_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Loading trained model...")
    model = MultiScale1DCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    print("Evaluating severity prediction...")
    y_true, y_pred = evaluate_severity(model, test_loader)

    report = classification_report(
        y_true, y_pred,
        target_names=["Low", "Medium", "High"],
        digits=4
    )

    print(report)

    with open(os.path.join(RESULT_DIR, "severity_metrics.txt"), "w") as f:
        f.write(report)

    plot_severity_confusion_matrix(y_true, y_pred)

    print("Severity results saved to:", RESULT_DIR)


if __name__ == "__main__":
    main()
