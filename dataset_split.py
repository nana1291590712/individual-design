# dataset_split.py
import numpy as np
from sklearn.model_selection import train_test_split
from preprocess import preprocess_dataset
from load_dataset import load_dataset


# =====================================================================
# 方案 1：固定测试负载（3HP）——分级框架版本
# =====================================================================
def split_dataset(input_dataset, window_size=1024, step=512):
    """
    分级跨负载划分（非 LOLO）：
        - 训练/验证：0HP, 1HP, 2HP
        - 测试：      3HP（unseen load）
        - Train/Val = 80/20
        - Stratify = (fault, severity)
    """

    # -------------------------------------------------
    # 1. 数据预处理
    # -------------------------------------------------
    x_all, y_fault_all, y_severity_all, loads_all = \
        preprocess_dataset(input_dataset, window_size, step)

    # -------------------------------------------------
    # 2. 基于负载划分
    # -------------------------------------------------
    train_val_idx = np.where(loads_all < 3)[0]
    test_idx = np.where(loads_all == 3)[0]

    x_train_val = x_all[train_val_idx]
    y_fault_train_val = y_fault_all[train_val_idx]
    y_sev_train_val = y_severity_all[train_val_idx]

    x_test = x_all[test_idx]
    y_fault_test = y_fault_all[test_idx]
    y_sev_test = y_severity_all[test_idx]

    # -------------------------------------------------
    # 3. 分级 stratify（fault + severity）
    # -------------------------------------------------
    stratify_labels = np.array(
        [f"{f}_{s}" for f, s in zip(y_fault_train_val, y_sev_train_val)]
    )

    x_train, x_val, \
    y_fault_train, y_fault_val, \
    y_sev_train, y_sev_val = train_test_split(
        x_train_val,
        y_fault_train_val,
        y_sev_train_val,
        test_size=0.2,
        random_state=42,
        stratify=stratify_labels
    )

    return (
        x_train, x_val, x_test,
        y_fault_train, y_fault_val, y_fault_test,
        y_sev_train, y_sev_val, y_sev_test
    )


# =====================================================================
# 方案 2：LOLO（Leave-One-Load-Out）——分级框架版本
# =====================================================================
def split_dataset_by_leave_one_load(raw_dataset, leave_out_load,
                                    window_size=1024, step=512):
    """
    LOLO + 分级约束：
        - leave_out_load 作为完全未见测试负载
        - 其余负载 → Train / Val
        - Stratify = (fault, severity)
    """

    # -------------------------------------------------
    # 1. 预处理
    # -------------------------------------------------
    x_all, y_fault_all, y_severity_all, loads_all = \
        preprocess_dataset(raw_dataset, window_size, step)

    # -------------------------------------------------
    # 2. LOLO 负载切分
    # -------------------------------------------------
    test_mask = (loads_all == leave_out_load)
    train_mask = ~test_mask

    x_train_full = x_all[train_mask]
    y_fault_train_full = y_fault_all[train_mask]
    y_sev_train_full = y_severity_all[train_mask]

    x_test = x_all[test_mask]
    y_fault_test = y_fault_all[test_mask]
    y_sev_test = y_severity_all[test_mask]

    # -------------------------------------------------
    # 3. 分级 stratify
    # -------------------------------------------------
    stratify_labels = np.array(
        [f"{f}_{s}" for f, s in zip(y_fault_train_full, y_sev_train_full)]
    )

    x_train, x_val, \
    y_fault_train, y_fault_val, \
    y_sev_train, y_sev_val = train_test_split(
        x_train_full,
        y_fault_train_full,
        y_sev_train_full,
        test_size=0.15,
        random_state=42,
        stratify=stratify_labels
    )

    train_loads = sorted(list(set(loads_all[train_mask])))

    return (
        (x_train, y_fault_train, y_sev_train),
        (x_val, y_fault_val, y_sev_val),
        (x_test, y_fault_test, y_sev_test),
        train_loads,
        leave_out_load
    )


# =====================================================================
# 自测
# =====================================================================
if __name__ == "__main__":
    root = r"D:\design\data\CWRU\12kDriveEndFault"
    dataset = load_dataset(root)

    out = split_dataset(dataset)
    print("Normal split:")
    print("Train:", out[0].shape)
    print("Val:  ", out[1].shape)
    print("Test: ", out[2].shape)

    lolo = split_dataset_by_leave_one_load(dataset, leave_out_load=3)
    print("\nLOLO leave 3HP:")
    print("Train:", lolo[0][0].shape)
    print("Val:  ", lolo[1][0].shape)
    print("Test: ", lolo[2][0].shape)
