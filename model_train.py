import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from load_dataset import load_dataset
from dataset_split import split_dataset
from model import MultiScale1DCNN


# ========================================================
# 参数
# ========================================================
BATCH_SIZE = 64
LR_MODEL = 3e-4
LR_MARGIN = 8e-4
EPOCHS = 60
SEED = 42

SEVERITY_LOSS_WEIGHT = 2.0
BOUNDARY_REFINE_LOSS_WEIGHT = 1.6
MARGIN_REG_LOSS_WEIGHT = 0.15

MAX_DIAMETER = 0.028

# 三个固定阈值：绝对不改
BASE_THRESHOLDS_DIAM = [0.0105, 0.0175, 0.0245]
BASE_THRESHOLDS_NORM = [t / MAX_DIAMETER for t in BASE_THRESHOLDS_DIAM]

# 四个固定类别中心
CLASS_CENTERS_DIAM = [0.007, 0.014, 0.021, 0.028]
CLASS_CENTERS_NORM = [c / MAX_DIAMETER for c in CLASS_CENTERS_DIAM]

# 仅学习三个 ±xx
MIN_MARGIN = 0.010
MAX_MARGIN = 0.080
INIT_MARGIN = 0.028

# 类别不平衡修正
SEVERITY_CLASS_WEIGHTS = [1.00, 1.20, 1.15, 1.30]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================================================
# 固定随机种子
# ========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========================================================
# numpy -> tensor
# ========================================================
def to_tensor(x, y_fault, y_severity):
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_fault = torch.tensor(y_fault, dtype=torch.long)
    y_severity = torch.tensor(y_severity / MAX_DIAMETER, dtype=torch.float32)
    return x, y_fault, y_severity


# ========================================================
# 固定阈值下的基础类别映射
# ========================================================
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


def normalized_diameter_to_class_torch(y, thresholds=None):
    if thresholds is None:
        thresholds = BASE_THRESHOLDS_NORM

    t1, t2, t3 = thresholds
    cls = torch.zeros_like(y, dtype=torch.long)
    cls = torch.where((y >= t1) & (y < t2), torch.ones_like(cls), cls)
    cls = torch.where((y >= t2) & (y < t3), torch.full_like(cls, 2), cls)
    cls = torch.where(y >= t3, torch.full_like(cls, 3), cls)
    return cls


# ========================================================
# 只训练三个 soft margins
# ========================================================
class SoftMarginModule(nn.Module):
    def __init__(self, init_margin=INIT_MARGIN):
        super().__init__()

        init_margin = float(np.clip(init_margin, MIN_MARGIN, MAX_MARGIN))
        raw_init = np.log(np.exp(init_margin - MIN_MARGIN) - 1.0 + 1e-8)

        self.raw_m1 = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        self.raw_m2 = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        self.raw_m3 = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

    def forward(self):
        margins = torch.stack([
            F.softplus(self.raw_m1) + MIN_MARGIN,
            F.softplus(self.raw_m2) + MIN_MARGIN,
            F.softplus(self.raw_m3) + MIN_MARGIN
        ])
        margins = torch.clamp(margins, min=MIN_MARGIN, max=MAX_MARGIN)
        return margins


# ========================================================
# 严重程度回归损失
# ========================================================
def boundary_aware_severity_loss(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)

    cls = normalized_diameter_to_class_torch(target)
    class_weights = pred.new_tensor(SEVERITY_CLASS_WEIGHTS)
    sample_cls_weight = class_weights[cls]

    thresholds = pred.new_tensor(BASE_THRESHOLDS_NORM).view(1, -1)
    dist = torch.abs(target.view(-1, 1) - thresholds)
    min_dist, _ = torch.min(dist, dim=1)

    boundary_weights = 1.0 + 1.2 * torch.exp(-min_dist / 0.05)

    diff = torch.abs(pred - target)
    beta = 0.025
    smooth_l1 = torch.where(
        diff < beta,
        0.5 * (diff ** 2) / beta,
        diff - 0.5 * beta
    )

    loss = sample_cls_weight * boundary_weights * smooth_l1
    return loss.mean()


# ========================================================
# 固定阈值 + 学习到的三个 ±xx 的判定规则
# ========================================================
def refined_classify_one(pred_value, thresholds, margins, class_centers):
    pred_value = float(pred_value)
    thresholds = [float(v) for v in thresholds]
    margins = [float(v) for v in margins]
    class_centers = [float(v) for v in class_centers]

    base_cls = normalized_diameter_to_class(pred_value, thresholds)

    active_idx = -1
    active_dist = 1e9
    for i, t in enumerate(thresholds):
        d = abs(pred_value - t)
        if d <= margins[i] and d < active_dist:
            active_idx = i
            active_dist = d

    if active_idx == -1:
        return base_cls

    i = active_idx
    left_cls = i
    right_cls = i + 1

    half_step = 0.125
    s0 = half_step + margins[0]
    s1 = half_step + 0.5 * (margins[0] + margins[1])
    s2 = half_step + 0.5 * (margins[1] + margins[2])
    s3 = half_step + margins[2]
    scales = [s0, s1, s2, s3]

    score_left = abs(pred_value - class_centers[left_cls]) / (scales[left_cls] + 1e-8)
    score_right = abs(pred_value - class_centers[right_cls]) / (scales[right_cls] + 1e-8)

    if score_left <= score_right:
        return left_cls
    else:
        return right_cls


def refined_classify_batch(preds, thresholds, margins, class_centers):
    return np.array(
        [refined_classify_one(p, thresholds, margins, class_centers) for p in preds],
        dtype=np.int64
    )


# ========================================================
# 软边界细化损失：真正让三个 margin 参与训练
# ========================================================
def boundary_refine_loss(pred, target, margin_module):
    pred = pred.view(-1)
    target = target.view(-1)

    margins = margin_module()
    thresholds = pred.new_tensor(BASE_THRESHOLDS_NORM)
    centers = pred.new_tensor(CLASS_CENTERS_NORM)
    true_cls = normalized_diameter_to_class_torch(target)

    m1, m2, m3 = margins[0], margins[1], margins[2]

    half_step = pred.new_tensor(0.125)
    scales = torch.stack([
        half_step + m1,
        half_step + 0.5 * (m1 + m2),
        half_step + 0.5 * (m2 + m3),
        half_step + m3
    ])

    total_loss = pred.new_tensor(0.0)
    valid_groups = 0

    for i in range(3):
        t = thresholds[i]
        band = margins[i]

        pair_mask = (true_cls == i) | (true_cls == (i + 1))
        near_mask = torch.abs(pred - t) <= (band + 0.05)
        mask = pair_mask & near_mask

        if mask.any():
            p = pred[mask]
            y = true_cls[mask]

            left_cls = i
            right_cls = i + 1

            left_logit = -torch.abs(p - centers[left_cls]) / (scales[left_cls] + 1e-8)
            right_logit = -torch.abs(p - centers[right_cls]) / (scales[right_cls] + 1e-8)

            logits = torch.stack([left_logit, right_logit], dim=1)
            y_pair = (y == right_cls).long()

            total_loss = total_loss + F.cross_entropy(logits, y_pair)
            valid_groups += 1

    if valid_groups == 0:
        return pred.new_tensor(0.0)

    return total_loss / valid_groups


# ========================================================
# margin 正则项
# ========================================================
def margin_regularization_loss(margin_module):
    margins = margin_module()
    target = margins.new_tensor([0.028, 0.028, 0.028])
    return ((margins - target) ** 2).mean()


# ========================================================
# 训练集采样器，缓解严重程度类别不平衡
# ========================================================
def build_train_sampler(y_sev_train):
    sev_cls = np.array(
        [normalized_diameter_to_class(v / MAX_DIAMETER) for v in y_sev_train],
        dtype=np.int64
    )
    counts = np.bincount(sev_cls, minlength=4).astype(np.float64)
    counts[counts == 0] = 1.0
    class_weights = 1.0 / counts
    sample_weights = class_weights[sev_cls]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


# ========================================================
# 训练一个 epoch
# ========================================================
def train_one_epoch(model, margin_module, loader, criterion_fault, optimizer):
    model.train()
    margin_module.train()

    total_loss = 0.0
    total_fault = 0
    correct_fault = 0
    total_samples = 0
    total_sev_abs_error = 0.0
    total_interval_correct = 0

    fixed_thresholds = BASE_THRESHOLDS_NORM
    class_centers = CLASS_CENTERS_NORM

    for batch_x, batch_fault, batch_severity in loader:
        batch_x = batch_x.to(device)
        batch_fault = batch_fault.to(device)
        batch_severity = batch_severity.to(device)

        optimizer.zero_grad()

        fault_out, severity_out = model(batch_x)
        severity_out = torch.clamp(severity_out.view(-1), min=0.0, max=1.0)

        loss_fault = criterion_fault(fault_out, batch_fault)
        loss_sev = boundary_aware_severity_loss(severity_out, batch_severity)
        loss_refine = boundary_refine_loss(severity_out, batch_severity, margin_module)
        loss_margin_reg = margin_regularization_loss(margin_module)

        loss = (
            loss_fault
            + SEVERITY_LOSS_WEIGHT * loss_sev
            + BOUNDARY_REFINE_LOSS_WEIGHT * loss_refine
            + MARGIN_REG_LOSS_WEIGHT * loss_margin_reg
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(margin_module.parameters()),
            max_norm=3.0
        )
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        total_fault += batch_fault.size(0)
        total_samples += batch_x.size(0)

        _, fault_pred = fault_out.max(1)
        correct_fault += fault_pred.eq(batch_fault).sum().item()

        total_sev_abs_error += torch.abs(severity_out - batch_severity).sum().item()

        current_margins = margin_module().detach().cpu().numpy()
        pred_cls = refined_classify_batch(
            severity_out.detach().cpu().numpy(),
            fixed_thresholds,
            current_margins,
            class_centers
        )
        true_cls = np.array(
            [normalized_diameter_to_class(v) for v in batch_severity.detach().cpu().numpy()],
            dtype=np.int64
        )
        total_interval_correct += (pred_cls == true_cls).sum()

    avg_loss = total_loss / total_fault
    fault_acc = correct_fault / total_fault
    sev_mae = total_sev_abs_error / total_samples
    sev_interval_acc = total_interval_correct / total_samples

    return avg_loss, fault_acc, sev_mae, sev_interval_acc


# ========================================================
# 验证：使用当前 margin_module
# ========================================================
def evaluate(model, margin_module, loader, criterion_fault):
    model.eval()
    margin_module.eval()

    total_loss = 0.0
    total_fault = 0
    correct_fault = 0
    total_samples = 0
    total_sev_abs_error = 0.0
    total_sev_sq_error = 0.0
    total_interval_correct = 0

    fixed_thresholds = BASE_THRESHOLDS_NORM
    class_centers = CLASS_CENTERS_NORM

    with torch.no_grad():
        for batch_x, batch_fault, batch_severity in loader:
            batch_x = batch_x.to(device)
            batch_fault = batch_fault.to(device)
            batch_severity = batch_severity.to(device)

            fault_out, severity_out = model(batch_x)
            severity_out = torch.clamp(severity_out.view(-1), min=0.0, max=1.0)

            loss_fault = criterion_fault(fault_out, batch_fault)
            loss_sev = boundary_aware_severity_loss(severity_out, batch_severity)
            loss_refine = boundary_refine_loss(severity_out, batch_severity, margin_module)
            loss_margin_reg = margin_regularization_loss(margin_module)

            loss = (
                loss_fault
                + SEVERITY_LOSS_WEIGHT * loss_sev
                + BOUNDARY_REFINE_LOSS_WEIGHT * loss_refine
                + MARGIN_REG_LOSS_WEIGHT * loss_margin_reg
            )

            total_loss += loss.item() * batch_x.size(0)
            total_fault += batch_fault.size(0)
            total_samples += batch_x.size(0)

            _, fault_pred = fault_out.max(1)
            correct_fault += fault_pred.eq(batch_fault).sum().item()

            diff = severity_out - batch_severity
            total_sev_abs_error += torch.abs(diff).sum().item()
            total_sev_sq_error += (diff ** 2).sum().item()

            current_margins = margin_module().detach().cpu().numpy()
            pred_cls = refined_classify_batch(
                severity_out.detach().cpu().numpy(),
                fixed_thresholds,
                current_margins,
                class_centers
            )
            true_cls = np.array(
                [normalized_diameter_to_class(v) for v in batch_severity.detach().cpu().numpy()],
                dtype=np.int64
            )
            total_interval_correct += (pred_cls == true_cls).sum()

    avg_loss = total_loss / total_fault
    fault_acc = correct_fault / total_fault
    sev_mae = total_sev_abs_error / total_samples
    sev_rmse = np.sqrt(total_sev_sq_error / total_samples)
    sev_interval_acc = total_interval_correct / total_samples

    return avg_loss, fault_acc, sev_mae, sev_rmse, sev_interval_acc


# ========================================================
# 最终测试：显式使用 checkpoint 里的 best thresholds + best margins
# ========================================================
def evaluate_with_explicit_margins(model, loader, criterion_fault, thresholds, margins):
    model.eval()

    total_loss = 0.0
    total_fault = 0
    correct_fault = 0
    total_samples = 0
    total_sev_abs_error = 0.0
    total_sev_sq_error = 0.0
    total_interval_correct = 0

    class_centers = CLASS_CENTERS_NORM

    with torch.no_grad():
        for batch_x, batch_fault, batch_severity in loader:
            batch_x = batch_x.to(device)
            batch_fault = batch_fault.to(device)
            batch_severity = batch_severity.to(device)

            fault_out, severity_out = model(batch_x)
            severity_out = torch.clamp(severity_out.view(-1), min=0.0, max=1.0)

            loss_fault = criterion_fault(fault_out, batch_fault)
            loss_sev = boundary_aware_severity_loss(severity_out, batch_severity)

            total_loss += (loss_fault + SEVERITY_LOSS_WEIGHT * loss_sev).item() * batch_x.size(0)

            _, fault_pred = fault_out.max(1)
            correct_fault += fault_pred.eq(batch_fault).sum().item()

            diff = severity_out - batch_severity
            total_sev_abs_error += torch.abs(diff).sum().item()
            total_sev_sq_error += (diff ** 2).sum().item()

            pred_cls = refined_classify_batch(
                severity_out.detach().cpu().numpy(),
                thresholds,
                margins,
                class_centers
            )
            true_cls = np.array(
                [normalized_diameter_to_class(v) for v in batch_severity.detach().cpu().numpy()],
                dtype=np.int64
            )
            total_interval_correct += (pred_cls == true_cls).sum()

            total_fault += batch_fault.size(0)
            total_samples += batch_x.size(0)

    avg_loss = total_loss / total_fault
    fault_acc = correct_fault / total_fault
    sev_mae = total_sev_abs_error / total_samples
    sev_rmse = np.sqrt(total_sev_sq_error / total_samples)
    sev_interval_acc = total_interval_correct / total_samples

    return avg_loss, fault_acc, sev_mae, sev_rmse, sev_interval_acc


# ========================================================
# 主程序
# ========================================================
def main():
    set_seed(SEED)

    print("Loading raw CWRU dataset...")

    fault_root = r"D:\design\data\CWRU\12kDriveEndFault"
    normal_root = r"D:\design\data\CWRU\NormalBaseline"

    fault_dataset = load_dataset(fault_root)
    normal_dataset = load_dataset(normal_root)
    data = fault_dataset + normal_dataset

    print("Loaded items:", len(data))
    print("Example item:", data[0])

    print("Splitting dataset...")

    x_train, x_val, x_test, \
    y_fault_train, y_fault_val, y_fault_test, \
    y_sev_train, y_sev_val, y_sev_test = split_dataset(input_dataset=data)

    print("Train severity range:", float(np.min(y_sev_train)), "to", float(np.max(y_sev_train)))
    print("Val severity range:", float(np.min(y_sev_val)), "to", float(np.max(y_sev_val)))
    print("Test severity range:", float(np.min(y_sev_test)), "to", float(np.max(y_sev_test)))

    train_sampler = build_train_sampler(y_sev_train)

    x_train, y_fault_train, y_sev_train = to_tensor(x_train, y_fault_train, y_sev_train)
    x_val, y_fault_val, y_sev_val = to_tensor(x_val, y_fault_val, y_sev_val)
    x_test, y_fault_test, y_sev_test = to_tensor(x_test, y_fault_test, y_sev_test)

    train_loader = DataLoader(
        TensorDataset(x_train, y_fault_train, y_sev_train),
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False
    )

    val_loader = DataLoader(
        TensorDataset(x_val, y_fault_val, y_sev_val),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(x_test, y_fault_test, y_sev_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = MultiScale1DCNN().to(device)
    margin_module = SoftMarginModule().to(device)

    criterion_fault = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": LR_MODEL, "weight_decay": 5e-5},
            {"params": margin_module.parameters(), "lr": LR_MARGIN, "weight_decay": 0.0}
        ]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-5
    )

    print("Start training improved MultiScale1DCNN + learnable soft margins...")

    best_val_fault_acc = 0.0
    best_val_sev_interval_acc = 0.0
    best_val_sev_mae = 1e9
    best_val_sev_rmse = 1e9
    best_val_loss = 1e9
    best_epoch = -1

    for epoch in range(EPOCHS):
        train_loss, train_acc, train_sev_mae, train_sev_interval_acc = train_one_epoch(
            model, margin_module, train_loader, criterion_fault, optimizer
        )

        val_loss, val_acc, val_sev_mae, val_sev_rmse, val_sev_interval_acc = evaluate(
            model, margin_module, val_loader, criterion_fault
        )

        scheduler.step(val_loss)

        learned_margins = margin_module().detach().cpu().numpy()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
            f"Train Sev MAE: {train_sev_mae:.6f}, Train Sev Interval Acc: {train_sev_interval_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, "
            f"Val Sev MAE: {val_sev_mae:.6f}, Val Sev RMSE: {val_sev_rmse:.6f}, "
            f"Val Sev Interval Acc: {val_sev_interval_acc*100:.2f}% | "
            f"Fixed Thr: {[round(float(t), 4) for t in BASE_THRESHOLDS_NORM]} | "
            f"Learned Margins: {[round(float(m), 4) for m in learned_margins]}"
        )

        should_save = False

        if val_sev_interval_acc > best_val_sev_interval_acc:
            should_save = True
        elif abs(val_sev_interval_acc - best_val_sev_interval_acc) < 1e-12:
            if val_sev_rmse < best_val_sev_rmse:
                should_save = True
            elif abs(val_sev_rmse - best_val_sev_rmse) < 1e-12:
                if val_sev_mae < best_val_sev_mae:
                    should_save = True
                elif abs(val_sev_mae - best_val_sev_mae) < 1e-12:
                    if val_acc > best_val_fault_acc:
                        should_save = True
                    elif abs(val_acc - best_val_fault_acc) < 1e-12:
                        if val_loss < best_val_loss:
                            should_save = True

        if should_save:
            best_val_fault_acc = val_acc
            best_val_sev_interval_acc = val_sev_interval_acc
            best_val_sev_mae = val_sev_mae
            best_val_sev_rmse = val_sev_rmse
            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "margin_module_state_dict": margin_module.state_dict(),
                    "best_thresholds": [float(t) for t in BASE_THRESHOLDS_NORM],
                    "best_margins": [float(m) for m in learned_margins]
                },
                "multiscale_best.pth"
            )

    print("Training finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best Validation Fault Accuracy: {best_val_fault_acc*100:.2f}%")
    print(f"Best Validation Severity Interval Accuracy: {best_val_sev_interval_acc*100:.2f}%")
    print(f"Best Validation Severity MAE: {best_val_sev_mae:.6f}")
    print(f"Best Validation Severity RMSE: {best_val_sev_rmse:.6f}")
    print(f"Best Validation Loss: {best_val_loss:.6f}")

    print("Loading best model...")
    checkpoint = torch.load("multiscale_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if "margin_module_state_dict" in checkpoint:
        margin_module.load_state_dict(checkpoint["margin_module_state_dict"])

    best_thresholds = [float(v) for v in checkpoint["best_thresholds"]]
    best_margins = [float(v) for v in checkpoint["best_margins"]]

    print("Final fixed thresholds:", [round(float(t), 6) for t in best_thresholds])
    print("Final learned margins:", [round(float(m), 6) for m in best_margins])

    print("Evaluating on test set...")
    test_loss, test_acc, test_sev_mae, test_sev_rmse, test_sev_interval_acc = evaluate_with_explicit_margins(
        model,
        test_loader,
        criterion_fault,
        thresholds=best_thresholds,
        margins=best_margins
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