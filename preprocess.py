# preprocess.py
import numpy as np


# -------------------------------
# Signal normalization
# -------------------------------
def normalize_signal(signal):
    """
    Standardize signal:
    x_norm = (x - mean) / std
    """
    return (signal - np.mean(signal)) / np.std(signal)


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

    return np.array(slices)


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
# Dataset preprocessing (severity framework)
# -------------------------------
def preprocess_dataset(input_dataset, window_size=1024, step=512):
    """
    Preprocess dataset with fault–severity framework

    Returns:
        x_array        : (N, window_size)
        y_fault_array  : (N,)
        y_sev_array    : (N,)
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

        if fault_label == 0 or raw_severity == -1:
            severity = 0
        else:
            severity = raw_severity - 1

        if severity not in (0, 1, 2):
            raise ValueError(
                f"Invalid severity value after mapping: {severity} "
                f"(raw_severity={raw_severity})"
            )

        x_slices.append(slices)
        y_fault_labels.extend([fault_label] * len(slices))
        y_sev_labels.extend([severity] * len(slices))
        load_values.extend([load] * len(slices))

    x_array = np.vstack(x_slices)
    y_fault_array = np.array(y_fault_labels, dtype=int)
    y_sev_array = np.array(y_sev_labels, dtype=int)
    loads_array = np.array(load_values, dtype=int)

    return x_array, y_fault_array, y_sev_array, loads_array


# -------------------------------
# Self-test
# -------------------------------
if __name__ == "__main__":
    from load_dataset import load_dataset

    root = r"D:\design\data\CWRU\12kDriveEndFault"
    dataset = load_dataset(root)

    x, y_fault, y_sev, loads = preprocess_dataset(dataset)

    print("X shape:", x.shape)
    print("Fault labels shape:", y_fault.shape)
    print("Severity labels shape:", y_sev.shape)
    print("Loads shape:", loads.shape)

    print("Fault unique:", np.unique(y_fault))
    print("Severity unique:", np.unique(y_sev))
