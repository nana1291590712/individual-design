import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

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
MAX_DIAMETER = 0.028
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULT_DIR, exist_ok=True)


# --------------------------------------------------------
# numpy → tensor
# y_sev 保持真实直径，不做归一化
# --------------------------------------------------------
def to_tensor(x, y_fault, y_sev):
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_fault = torch.tensor(y_fault, dtype=torch.long)
    y_sev = torch.tensor(y_sev, dtype=torch.float32)
    return x, y_fault, y_sev


# --------------------------------------------------------
# fault id -> 名称
# --------------------------------------------------------
def fault_id_to_name(fault_id):
    mapping = {
        0: "Normal",
        1: "Ball",
        2: "Inner",
        3: "Outer"
    }
    return mapping.get(int(fault_id), "Unknown")


# --------------------------------------------------------
# 连续故障直径 → 严重程度类别
#   0 -> Low
#   1 -> Medium
#   2 -> High
#   3 -> Very Severe
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


def class_to_name(c):
    mapping = {
        0: "Low",
        1: "Medium",
        2: "High",
        3: "Very Severe"
    }
    return mapping[int(c)]


def class_to_range(c):
    mapping = {
        0: "[0.0000, 0.0105)",
        1: "[0.0105, 0.0175)",
        2: "[0.0175, 0.0245)",
        3: "[0.0245, 0.0280]"
    }
    return mapping[int(c)]


# --------------------------------------------------------
# 评估严重程度，并记录每个样本的预测过程
# 不打印样本过程到终端，只返回结果用于保存文件
# --------------------------------------------------------
def evaluate_severity(model, loader):
    model.eval()

    all_preds = []
    all_labels = []
    detail_records = []

    sample_idx = 0

    with torch.no_grad():
        for batch_x, batch_fault, batch_sev in loader:
            batch_x = batch_x.to(device)
            batch_fault = batch_fault.to(device)
            batch_sev = batch_sev.to(device)

            # 模型输出：归一化 severity
            _, sev_out = model(batch_x)

            # 还原到真实故障直径
            pred_diameter = sev_out * MAX_DIAMETER
            pred_diameter = torch.clamp(pred_diameter, min=0.0, max=MAX_DIAMETER)

            for i in range(len(batch_fault)):
                if batch_fault[i].item() != 0:
                    sample_idx += 1

                    fault_id = int(batch_fault[i].item())
                    fault_name = fault_id_to_name(fault_id)

                    true_d = float(batch_sev[i].item())
                    pred_norm = float(sev_out[i].item())
                    pred_d = float(pred_diameter[i].item())

                    true_class = diameter_to_class(true_d)
                    pred_class = diameter_to_class(pred_d)

                    all_labels.append(true_class)
                    all_preds.append(pred_class)

                    detail_records.append({
                        "sample_id": sample_idx,
                        "fault_id": fault_id,
                        "fault_type": fault_name,

                        "true_diameter": round(true_d, 6),
                        "pred_norm_output": round(pred_norm, 6),
                        "pred_diameter": round(pred_d, 6),

                        "true_severity_id": true_class,
                        "true_severity": class_to_name(true_class),
                        "true_range": class_to_range(true_class),

                        "pred_severity_id": pred_class,
                        "pred_severity": class_to_name(pred_class),
                        "pred_range": class_to_range(pred_class),

                        "abs_error": round(abs(pred_d - true_d), 6),
                        "correct": int(pred_class == true_class)
                    })

    return np.array(all_labels), np.array(all_preds), detail_records


# --------------------------------------------------------
# 保存详细预测过程到 CSV / HTML
# --------------------------------------------------------
def save_prediction_details(detail_records):
    df = pd.DataFrame(detail_records)

    csv_path = os.path.join(RESULT_DIR, "severity_prediction_details.csv")
    error_csv_path = os.path.join(RESULT_DIR, "severity_error_cases.csv")
    html_path = os.path.join(RESULT_DIR, "severity_prediction_details.html")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    error_df = df[df["correct"] == 0].copy()
    error_df.to_csv(error_csv_path, index=False, encoding="utf-8-sig")

    # 保存为 HTML，双击即可在浏览器中查看
    df.to_html(html_path, index=False)

    return df


# --------------------------------------------------------
# 绘制严重程度混淆矩阵
# --------------------------------------------------------
def plot_severity_confusion_matrix(y_true, y_pred):
    class_names = ["Low", "Medium", "High", "Very Severe"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0)
    plt.title("Severity Confusion Matrix")
    plt.colorbar(shrink=0.8)

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=20)
    plt.yticks(ticks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, cm[i, j],
                ha="center",
                va="center",
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
# 绘制真实直径 vs 预测直径散点图
# --------------------------------------------------------
def plot_diameter_scatter(detail_df):
    plt.figure(figsize=(6, 6))

    true_vals = detail_df["true_diameter"].values
    pred_vals = detail_df["pred_diameter"].values

    plt.scatter(true_vals, pred_vals, alpha=0.5)

    plt.plot(
        [0.0, MAX_DIAMETER],
        [0.0, MAX_DIAMETER],
        linestyle="--"
    )

    plt.xlabel("True Diameter")
    plt.ylabel("Predicted Diameter")
    plt.title("True vs Predicted Severity Diameter")
    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULT_DIR, "severity_diameter_scatter.png"),
        dpi=300
    )
    plt.close()


# --------------------------------------------------------
# 主程序
# --------------------------------------------------------
def main():
    print("Loading dataset...")

    fault_root = r"D:\design\data\CWRU\12kDriveEndFault"
    normal_root = r"D:\design\data\CWRU\NormalBaseline"

    fault_dataset = load_dataset(fault_root)
    normal_dataset = load_dataset(normal_root)
    data = fault_dataset + normal_dataset

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
    y_true, y_pred, detail_records = evaluate_severity(model, test_loader)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=["Low", "Medium", "High", "Very Severe"],
        digits=4,
        zero_division=0
    )

    with open(os.path.join(RESULT_DIR, "severity_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    detail_df = save_prediction_details(detail_records)

    plot_severity_confusion_matrix(y_true, y_pred)
    plot_diameter_scatter(detail_df)

    print("All severity files have been saved to:", RESULT_DIR)


if __name__ == "__main__":
    main()