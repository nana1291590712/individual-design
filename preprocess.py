import numpy as np


# -------------------------------
# Signal normalization
# -------------------------------
def normalize_signal(signal):
    """
    Standardize signal:
    x_norm = (x - mean) / std
    """
    signal = np.asarray(signal, dtype=np.float32)
    std = np.std(signal)

    # 防止除以 0
    if std < 1e-12:
        return signal - np.mean(signal)

    return (signal - np.mean(signal)) / std


# -------------------------------
# Sliding window segmentation
# -------------------------------
def sliding_window(signal, window_size=1024, step=512):
    """
    Segment long signal into fixed-length windows
    """
    window_size = int(window_size)
    step = int(step)

    slices = []
    for start in range(0, len(signal) - window_size + 1, step):
        slices.append(signal[start:start + window_size])

    return np.array(slices, dtype=np.float32)


# -------------------------------
# Fault label encoding
# -------------------------------
label_map = {
    "Normal": 0,
    "Ball": 1,
    "Inner": 2,
    "Outer": 3
}


def encode_label(label_str):
    if label_str not in label_map:
        raise ValueError(f"Unknown fault label: {label_str}")
    return label_map[label_str]


# -------------------------------
# Severity index -> defect diameter
# -------------------------------
def map_severity_to_diameter(fault_label, raw_severity):
    """
    Convert raw severity index to real defect diameter.

    Mapping rule:
        Normal or missing severity -> 0.000
        1 -> 0.007
        2 -> 0.014
        3 -> 0.021
        4 -> 0.028

    Returns:
        diameter (float)
    """
    if fault_label == 0 or raw_severity == -1:
        return 0.000

    severity_to_diameter = {
        1: 0.007,
        2: 0.014,
        3: 0.021,
        4: 0.028
    }

    if raw_severity not in severity_to_diameter:
        raise ValueError(
            f"Invalid raw severity value: {raw_severity}. "
            f"Expected one of [1, 2, 3, 4]."
        )

    return severity_to_diameter[raw_severity]


# -------------------------------
# Dataset preprocessing (severity regression framework)
# -------------------------------
def preprocess_dataset(input_dataset, window_size=1024, step=512):
    """
    Preprocess dataset with fault + severity regression framework

    Returns:
        x_array        : (N, window_size)
        y_fault_array  : (N,)
        y_sev_array    : (N,)   continuous defect diameter labels
        loads_array    : (N,)
    """

    x_slices = []
    y_fault_labels = []
    y_sev_labels = []
    load_values = []

    for item in input_dataset:
        signal = normalize_signal(item["signal"])
        slices = sliding_window(signal, window_size, step)

        fault_label = encode_label(item["label"])
        raw_severity = int(item.get("severity", -1))
        load = int(item.get("load", 0))

        severity_diameter = map_severity_to_diameter(fault_label, raw_severity)

        x_slices.append(slices)
        y_fault_labels.extend([fault_label] * len(slices))
        y_sev_labels.extend([severity_diameter] * len(slices))
        load_values.extend([load] * len(slices))

    x_array = np.vstack(x_slices).astype(np.float32)
    y_fault_array = np.array(y_fault_labels, dtype=np.int64)
    y_sev_array = np.array(y_sev_labels, dtype=np.float32)
    loads_array = np.array(load_values, dtype=np.int64)

    return x_array, y_fault_array, y_sev_array, loads_array


# -------------------------------
# Self-test
# -------------------------------
if __name__ == "__main__":
    from load_dataset import load_dataset

    fault_root = r"D:\design\data\CWRU\12kDriveEndFault"
    normal_root = r"D:\design\data\CWRU\NormalBaseline"

    fault_dataset = load_dataset(fault_root)
    normal_dataset = load_dataset(normal_root)

    dataset = fault_dataset + normal_dataset

    x, y_fault, y_sev, loads = preprocess_dataset(dataset)

    print("X shape:", x.shape)
    print("Fault labels shape:", y_fault.shape)
    print("Severity labels shape:", y_sev.shape)
    print("Loads shape:", loads.shape)

    print("Fault unique:", np.unique(y_fault))
    print("Severity unique:", np.unique(y_sev))
    print("Loads unique:", np.unique(loads))