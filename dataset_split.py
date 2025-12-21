# dataset_split.py
import numpy as np
from sklearn.model_selection import train_test_split
from preprocess import preprocess_dataset
from load_dataset import load_dataset


# =====================================================================
# 方案 1：固定测试负载（3HP）——用于普通训练（非 LOLO）
# =====================================================================
def split_dataset(input_dataset, window_size=1024, step=512):
    """
    跨负载划分（固定方案）：
        - 训练/验证：0HP, 1HP, 2HP
        - 测试：      3HP（作为 unseen load）
        - Train/Val 再按 80/20 分
    """

    # ----------------------------
    # 1. 数据预处理（归一化 + 滑窗 + 标签编码 + 负载编码）
    # ----------------------------
    x_all, y_all, loads_all = preprocess_dataset(input_dataset, window_size, step)

    # ----------------------------
    # 2. 基于负载划分
    # ----------------------------
    train_val_idx = np.where(loads_all < 3)[0]    # 0,1,2 HP
    test_idx = np.where(loads_all == 3)[0]        # 3 HP

    x_train_val = x_all[train_val_idx]
    y_train_val = y_all[train_val_idx]

    x_test = x_all[test_idx]
    y_test = y_all[test_idx]

    # ----------------------------
    # 3. Train / Val 划分（80/20）
    # ----------------------------
    validation_ratio = 0.1765     # 对应总体的 20%

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=validation_ratio,
        random_state=42,
        stratify=y_train_val
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


# =====================================================================
# 方案 2：LOLO（Leave-One-Load-Out）跨负载泛化实验
# =====================================================================
def split_dataset_by_leave_one_load(raw_dataset, leave_out_load,
                                    window_size=1024, step=512):
    """
    LOLO 负载划分：
        - leave_out_load 作为测试集负载（未见负载）
        - 其余负载全部作为训练/验证
        - Train/Val = 在训练部分内部按 85/15 随机划分

    返回：
        (x_train, y_train), (x_val, y_val), (x_test, y_test), train_loads, test_load
    """

    # ----------------------------
    # 1. 数据预处理
    # ----------------------------
    x_all, y_all, loads_all = preprocess_dataset(raw_dataset, window_size, step)

    # ----------------------------
    # 2. 基于 leave-out 划分
    # ----------------------------
    test_mask = (loads_all == leave_out_load)
    train_mask = ~test_mask

    x_train_full = x_all[train_mask]
    y_train_full = y_all[train_mask]

    x_test = x_all[test_mask]
    y_test = y_all[test_mask]

    # ----------------------------
    # 3. 训练/验证划分（85/15）
    # ----------------------------
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full,
        test_size=0.15,
        shuffle=True,
        random_state=42,
        stratify=y_train_full
    )

    train_loads = sorted(list(set(loads_all[train_mask])))
    test_load = leave_out_load

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), train_loads, test_load


# =====================================================================
# 自测代码
# =====================================================================
if __name__ == "__main__":
    root = r"D:\design\data\CWRU\12kDriveEndFault"
    dataset = load_dataset(root)

    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(dataset)

    print("Training set:   ", x_train.shape, y_train.shape)
    print("Validation set: ", x_val.shape, y_val.shape)
    print("Test set:       ", x_test.shape, y_test.shape)

    # LOLO 测试 leave 3HP
    (x_tr, y_tr), (x_v, y_v), (x_te, y_te), loads_train, l_out = \
        split_dataset_by_leave_one_load(dataset, leave_out_load=3)

    print(f"\nLOLO leave {l_out}HP:")
    print("Train loads:", loads_train)
    print("Train:", x_tr.shape)
    print("Val:  ", x_v.shape)
    print("Test: ", x_te.shape)
