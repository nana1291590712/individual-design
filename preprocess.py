# preprocess.py
import numpy as np

# -------------------------------
# 信号归一化
# -------------------------------
def normalize_signal(signal):
    """
    对信号进行标准化
    x_norm = (x - μ) / σ
    """
    return (signal - np.mean(signal)) / np.std(signal)

# -------------------------------
# 滑动窗口切片
# -------------------------------
def sliding_window(signal, window_size=1024, step=512):
    """
    将长信号切片为多个固定长度窗口
    """
    window_size = int(window_size)
    step = int(step)
    slices = []

    for start in range(0, len(signal) - window_size + 1, step):
        slice_ = signal[start:start + window_size]
        slices.append(slice_)

    return np.array(slices)

# -------------------------------
# 标签编码
# -------------------------------
label_map = {"Normal": 0, "Ball": 1, "Inner": 2, "Outer": 3}

def encode_label(label_str):
    """
    将标签字符串映射为整数；若标签不在映射表中则报错
    """
    if label_str not in label_map:
        raise ValueError(f"Unknown label: {label_str}")
    return label_map[label_str]


# -------------------------------
# 处理整个数据集
# -------------------------------
def preprocess_dataset(input_dataset, window_size=1024, step=512):
    """
    对整个数据集进行归一化 + 滑窗切片 + 标签编码
    返回：
        x_array: shape (num_slices, window_size)
        y_array: shape (num_slices,)
        loads_array: shape (num_slices,)
    """
    x_slices = []
    y_labels = []
    load_values = []

    for item in input_dataset:
        signal = normalize_signal(item['signal'])
        slices = sliding_window(signal, window_size, step)
        label_encoded = encode_label(item['label'])
        load = int(item.get('load', 0))  # load 信息，如果没有则默认 0

        x_slices.append(slices)
        y_labels.extend([label_encoded]*len(slices))
        load_values.extend([load]*len(slices))

    x_array = np.vstack(x_slices)
    y_array = np.array(y_labels, dtype=int)
    loads_array = np.array(load_values, dtype=int)

    return x_array, y_array, loads_array

# -------------------------------
# 测试 preprocess.py
# -------------------------------
if __name__ == "__main__":
    from load_dataset import load_dataset
    root = r"D:\design\data\CWRU\12kDriveEndFault"
    dataset = load_dataset(root)

    x_array, y_array, loads_array = preprocess_dataset(dataset)
    print("X shape:", x_array.shape)
    print("y shape:", y_array.shape)
    print("loads shape:", loads_array.shape)
    print("Example labels:", y_array[:10])