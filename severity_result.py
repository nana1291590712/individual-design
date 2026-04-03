# severity_result.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from load_dataset import load_dataset
from dataset_split import split_dataset
from model import MultiScale1DCNN


# =========================================================
# 参数
# =========================================================
MODEL_PATH = r"D:\design\multiscale_best.pth"
SAVE_DIR = r"D:\design\severity_results"

MAX_DIAMETER = 0.028

BASE_THRESHOLDS_DIAM = [0.0105, 0.0175, 0.0245]
CLASS_CENTERS_DIAM = [0.007, 0.014, 0.021, 0.028]

BASE_THRESHOLDS_NORM = [t / MAX_DIAMETER for t in BASE_THRESHOLDS_DIAM]
CLASS_CENTERS_NORM = [c / MAX_DIAMETER for c in CLASS_CENTERS_DIAM]

CLASS_NAMES = ["Low", "Medium", "High", "Very Severe"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 工具函数
# =========================================================
def to_tensor(x, y_fault, y_severity):
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_fault = torch.tensor(y_fault, dtype=torch.long)
    y_severity = torch.tensor(y_severity / MAX_DIAMETER, dtype=torch.float32)
    return x, y_fault, y_severity


def normalized_diameter_to_class(d, thresholds=None):
    if thresholds is None:
        thresholds = BASE_THRESHOLDS_NORM

    d = float(d)

    if d < thresholds[0]:
        return 0
    elif d < thresholds[1]:
        return 1
    elif d < thresholds[2]:
        return 2
    else:
        return 3


def adaptive_soft_classify_one(d, thresholds, margins, class_centers):
    d = float(d)

    low = thresholds[0] - margins[0]
    high = thresholds[0] + margins[0]
    if low <= d <= high:
        dist0 = abs(d - class_centers[0])
        dist1 = abs(d - class_centers[1])
        return 0 if dist0 <= dist1 else 1

    low = thresholds[1] - margins[1]
    high = thresholds[1] + margins[1]
    if low <= d <= high:
        dist1 = abs(d - class_centers[1])
        dist2 = abs(d - class_centers[2])
        return 1 if dist1 <= dist2 else 2

    low = thresholds[2] - margins[2]
    high = thresholds[2] + margins[2]
    if low <= d <= high:
        dist2 = abs(d - class_centers[2])
        dist3 = abs(d - class_centers[3])
        return 2 if dist2 <= dist3 else 3

    return normalized_diameter_to_class(d, thresholds)


def adaptive_soft_classify_batch(preds, thresholds, margins, class_centers):
    return np.array(
        [adaptive_soft_classify_one(p, thresholds, margins, class_centers) for p in preds],
        dtype=np.int64
    )


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Severity Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=30)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_scatter_true_vs_pred(true_diam, pred_diam, save_path):
    pred_diam_clip = np.clip(pred_diam, 0.0, MAX_DIAMETER)
    true_diam_clip = np.clip(true_diam, 0.0, MAX_DIAMETER)

    plt.figure(figsize=(7, 6))
    plt.scatter(true_diam_clip, pred_diam_clip, alpha=0.55, s=10)

    plt.plot([0.0, MAX_DIAMETER], [0.0, MAX_DIAMETER], linestyle="--")

    plt.xlim(0.0, MAX_DIAMETER)
    plt.ylim(0.0, MAX_DIAMETER * 1.05)

    plt.xlabel("True Diameter (inch)")
    plt.ylabel("Predicted Diameter (inch)")
    plt.title("True vs Predicted Severity Scatter")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_prediction_curve_sorted(true_diam, pred_diam, thresholds, margins, save_path):
    true_diam_clip = np.clip(true_diam, 0.0, MAX_DIAMETER)
    pred_diam_clip = np.clip(pred_diam, 0.0, MAX_DIAMETER)

    sort_idx = np.argsort(true_diam_clip)
    true_sorted = true_diam_clip[sort_idx]
    pred_sorted = pred_diam_clip[sort_idx]
    x_axis = np.arange(len(true_sorted))

    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, true_sorted, linewidth=2.0, label="True Diameter")
    plt.scatter(x_axis, pred_sorted, s=8, alpha=0.55, label="Predicted Diameter")

    for i in range(len(thresholds)):
        fixed_diam = thresholds[i] * MAX_DIAMETER
        low = max(0.0, (thresholds[i] - margins[i]) * MAX_DIAMETER)
        high = min(MAX_DIAMETER, (thresholds[i] + margins[i]) * MAX_DIAMETER)

        plt.axhline(y=fixed_diam, linestyle="--", linewidth=1.0)

        if high > low:
            plt.axhspan(low, high, alpha=0.12)

    # 关键修改：给 y 轴顶部留出空白，避免 0.028 的横线和散点贴顶看不见
    y_top = MAX_DIAMETER * 1.08
    plt.ylim(0.0, y_top)

    plt.xlabel("Sample Index (sorted by true diameter)")
    plt.ylabel("Diameter (inch)")
    plt.title("Severity Prediction Visualization (Sorted by True Diameter)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_prediction_histogram(pred_diam, true_diam, save_path):
    pred_diam_clip = np.clip(pred_diam, 0.0, MAX_DIAMETER)
    true_diam_clip = np.clip(true_diam, 0.0, MAX_DIAMETER)

    bins = np.linspace(0.0, MAX_DIAMETER, 25)

    plt.figure(figsize=(8, 5))
    plt.hist(true_diam_clip, bins=bins, alpha=0.6, label="True Diameter")
    plt.hist(pred_diam_clip, bins=bins, alpha=0.6, label="Predicted Diameter")
    plt.xlabel("Diameter (inch)")
    plt.ylabel("Count")
    plt.title("Prediction Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_checkpoint(model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
        best_thresholds = checkpoint.get("best_thresholds", BASE_THRESHOLDS_NORM.copy())
        best_margins = checkpoint.get("best_margins", [0.0, 0.0, 0.0])
    else:
        model_state_dict = checkpoint
        best_thresholds = BASE_THRESHOLDS_NORM.copy()
        best_margins = [0.0, 0.0, 0.0]

    return model_state_dict, best_thresholds, best_margins


def build_soft_ranges(thresholds, margins):
    soft_ranges = []
    for i in range(len(thresholds)):
        low = thresholds[i] - margins[i]
        high = thresholds[i] + margins[i]
        soft_ranges.append((low, high))
    return soft_ranges


# =========================================================
# 主评估函数
# =========================================================
def evaluate_severity_results():
    print("Loading dataset...")

    fault_root = r"D:\design\data\CWRU\12kDriveEndFault"
    normal_root = r"D:\design\data\CWRU\NormalBaseline"

    fault_dataset = load_dataset(fault_root)
    normal_dataset = load_dataset(normal_root)
    data = fault_dataset + normal_dataset

    print("Splitting dataset...")

    x_train, x_val, x_test, \
    y_fault_train, y_fault_val, y_fault_test, \
    y_sev_train, y_sev_val, y_sev_test = split_dataset(input_dataset=data)

    x_test, y_fault_test, y_sev_test = to_tensor(x_test, y_fault_test, y_sev_test)

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_fault_test, y_sev_test),
        batch_size=64,
        shuffle=False
    )

    print("Loading trained best model...")
    model = MultiScale1DCNN().to(device)

    model_state_dict, best_thresholds, best_margins = load_checkpoint(MODEL_PATH)
    model.load_state_dict(model_state_dict)
    model.eval()

    print("Evaluating severity prediction...")

    all_pred_norm = []
    all_true_norm = []

    with torch.no_grad():
        for batch_x, _, batch_severity in test_loader:
            batch_x = batch_x.to(device)

            _, severity_out = model(batch_x)
            severity_out = torch.clamp(severity_out.view(-1), min=0.0, max=1.0)

            all_pred_norm.append(severity_out.cpu().numpy())
            all_true_norm.append(batch_severity.numpy())

    pred_norm = np.concatenate(all_pred_norm)
    true_norm = np.concatenate(all_true_norm)

    pred_cls = adaptive_soft_classify_batch(
        pred_norm,
        best_thresholds,
        best_margins,
        CLASS_CENTERS_NORM
    )

    true_cls = np.array(
        [normalized_diameter_to_class(v, BASE_THRESHOLDS_NORM) for v in true_norm],
        dtype=np.int64
    )

    pred_diam = pred_norm * MAX_DIAMETER
    true_diam = true_norm * MAX_DIAMETER

    mae = np.mean(np.abs(pred_diam - true_diam))
    rmse = np.sqrt(np.mean((pred_diam - true_diam) ** 2))

    acc = accuracy_score(true_cls, pred_cls)
    cm = confusion_matrix(true_cls, pred_cls)
    report = classification_report(
        true_cls,
        pred_cls,
        target_names=CLASS_NAMES,
        digits=4
    )

    soft_ranges_norm = build_soft_ranges(best_thresholds, best_margins)
    soft_ranges_diam = [(low * MAX_DIAMETER, high * MAX_DIAMETER) for low, high in soft_ranges_norm]

    print("===== Severity Classification Results =====")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Accuracy: {acc:.4f}")
    print(f"MAE     : {mae:.6f}")
    print(f"RMSE    : {rmse:.6f}")
    print("Calibrated Thresholds (norm):", [round(float(v), 6) for v in best_thresholds])
    print("Adaptive Margins (norm):    ", [round(float(v), 6) for v in best_margins])

    print("\nFixed Threshold + Soft Threshold Range (norm):")
    print(f"Low / Medium  : fixed = {best_thresholds[0]:.6f}, soft range = [{soft_ranges_norm[0][0]:.6f}, {soft_ranges_norm[0][1]:.6f}]")
    print(f"Medium / High : fixed = {best_thresholds[1]:.6f}, soft range = [{soft_ranges_norm[1][0]:.6f}, {soft_ranges_norm[1][1]:.6f}]")
    print(f"High / Very Severe : fixed = {best_thresholds[2]:.6f}, soft range = [{soft_ranges_norm[2][0]:.6f}, {soft_ranges_norm[2][1]:.6f}]")

    print("\nFixed Threshold + Soft Threshold Range (inch):")
    print(f"Low / Medium  : fixed = {best_thresholds[0] * MAX_DIAMETER:.6f}, soft range = [{soft_ranges_diam[0][0]:.6f}, {soft_ranges_diam[0][1]:.6f}]")
    print(f"Medium / High : fixed = {best_thresholds[1] * MAX_DIAMETER:.6f}, soft range = [{soft_ranges_diam[1][0]:.6f}, {soft_ranges_diam[1][1]:.6f}]")
    print(f"High / Very Severe : fixed = {best_thresholds[2] * MAX_DIAMETER:.6f}, soft range = [{soft_ranges_diam[2][0]:.6f}, {soft_ranges_diam[2][1]:.6f}]")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    os.makedirs(SAVE_DIR, exist_ok=True)

    cm_path = os.path.join(SAVE_DIR, "severity_confusion_matrix.png")
    txt_path = os.path.join(SAVE_DIR, "severity_report.txt")
    npy_cm_path = os.path.join(SAVE_DIR, "severity_confusion_matrix.npy")
    pred_path = os.path.join(SAVE_DIR, "severity_predictions.npz")
    scatter_path = os.path.join(SAVE_DIR, "severity_scatter_true_vs_pred.png")
    curve_path = os.path.join(SAVE_DIR, "severity_prediction_curve.png")
    hist_path = os.path.join(SAVE_DIR, "severity_prediction_histogram.png")

    plot_confusion_matrix(cm, CLASS_NAMES, cm_path)
    plot_scatter_true_vs_pred(true_diam, pred_diam, scatter_path)
    plot_prediction_curve_sorted(true_diam, pred_diam, best_thresholds, best_margins, curve_path)
    plot_prediction_histogram(pred_diam, true_diam, hist_path)

    np.save(npy_cm_path, cm)

    np.savez(
        pred_path,
        pred_norm=pred_norm,
        true_norm=true_norm,
        pred_diam=pred_diam,
        true_diam=true_diam,
        pred_cls=pred_cls,
        true_cls=true_cls,
        calibrated_thresholds=np.array(best_thresholds, dtype=np.float32),
        adaptive_margins=np.array(best_margins, dtype=np.float32),
        soft_ranges_norm=np.array(soft_ranges_norm, dtype=np.float32),
        soft_ranges_diam=np.array(soft_ranges_diam, dtype=np.float32)
    )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("===== Severity Classification Results =====\n")
        f.write(f"Model Path: {MODEL_PATH}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"MAE     : {mae:.6f}\n")
        f.write(f"RMSE    : {rmse:.6f}\n")
        f.write(f"Calibrated Thresholds (norm): {[round(float(v), 6) for v in best_thresholds]}\n")
        f.write(f"Adaptive Margins (norm): {[round(float(v), 6) for v in best_margins]}\n\n")

        f.write("Fixed Threshold + Soft Threshold Range (norm):\n")
        f.write(f"Low / Medium: fixed = {best_thresholds[0]:.6f}, soft range = [{soft_ranges_norm[0][0]:.6f}, {soft_ranges_norm[0][1]:.6f}]\n")
        f.write(f"Medium / High: fixed = {best_thresholds[1]:.6f}, soft range = [{soft_ranges_norm[1][0]:.6f}, {soft_ranges_norm[1][1]:.6f}]\n")
        f.write(f"High / Very Severe: fixed = {best_thresholds[2]:.6f}, soft range = [{soft_ranges_norm[2][0]:.6f}, {soft_ranges_norm[2][1]:.6f}]\n\n")

        f.write("Fixed Threshold + Soft Threshold Range (inch):\n")
        f.write(f"Low / Medium: fixed = {best_thresholds[0] * MAX_DIAMETER:.6f}, soft range = [{soft_ranges_diam[0][0]:.6f}, {soft_ranges_diam[0][1]:.6f}]\n")
        f.write(f"Medium / High: fixed = {best_thresholds[1] * MAX_DIAMETER:.6f}, soft range = [{soft_ranges_diam[1][0]:.6f}, {soft_ranges_diam[1][1]:.6f}]\n")
        f.write(f"High / Very Severe: fixed = {best_thresholds[2] * MAX_DIAMETER:.6f}, soft range = [{soft_ranges_diam[2][0]:.6f}, {soft_ranges_diam[2][1]:.6f}]\n\n")

        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    print(f"\n[Saved] {SAVE_DIR}")
    print("[Saved Figure] severity_confusion_matrix.png")
    print("[Saved Figure] severity_scatter_true_vs_pred.png")
    print("[Saved Figure] severity_prediction_curve.png")
    print("[Saved Figure] severity_prediction_histogram.png")


if __name__ == "__main__":
    evaluate_severity_results()