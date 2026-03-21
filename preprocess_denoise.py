# preprocess_denoise.py
import numpy as np
from scipy.signal import butter, filtfilt, medfilt

# ==============================
# User-configurable denoise params
# ==============================
DENOISE_ENABLE = True
DENOISE_METHOD = "bandpass"  # Options: "bandpass", "median", "moving_average"
FS = 12000  # sampling frequency

# Bandpass filter params
BANDPASS_LOW = 10
BANDPASS_HIGH = 3000
BANDPASS_ORDER = 4

# Median filter params
MEDIAN_KERNEL = 5

# Moving average params
MA_WINDOW = 5

# -------------------------------
# Signal normalization
# -------------------------------
def normalize_signal(signal):
    signal = np.asarray(signal, dtype=np.float32)
    std = np.std(signal)
    if std < 1e-12:
        return signal - np.mean(signal)
    return (signal - np.mean(signal)) / std

# -------------------------------
# Denoising
# -------------------------------
def denoise_signal(signal):
    if not DENOISE_ENABLE:
        return signal

    if DENOISE_METHOD == "bandpass":
        nyq = 0.5 * FS
        low = BANDPASS_LOW / nyq
        high = BANDPASS_HIGH / nyq
        b, a = butter(BANDPASS_ORDER, [low, high], btype="band")
        return filtfilt(b, a, signal)

    elif DENOISE_METHOD == "median":
        return medfilt(signal, kernel_size=MEDIAN_KERNEL)

    elif DENOISE_METHOD == "moving_average":
        cumsum = np.cumsum(np.insert(signal, 0, 0))
        return (cumsum[MA_WINDOW:] - cumsum[:-MA_WINDOW]) / MA_WINDOW

    else:
        raise ValueError(f"Unknown DENOISE_METHOD: {DENOISE_METHOD}")

# -------------------------------
# Sliding window segmentation
# -------------------------------
def sliding_window(signal, window_size=1024, step=512):
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
    if fault_label == 0 or raw_severity == -1:
        return 0.0
    severity_to_diameter = {1: 0.007, 2: 0.014, 3: 0.021, 4: 0.028}
    if raw_severity not in severity_to_diameter:
        raise ValueError(f"Invalid raw severity value: {raw_severity}")
    return severity_to_diameter[raw_severity]

# -------------------------------
# Dataset preprocessing (with denoise)
# -------------------------------
def preprocess_dataset(input_dataset, window_size=1024, step=512):
    x_slices = []
    y_fault_labels = []
    y_sev_labels = []
    load_values = []

    for item in input_dataset:
        signal = np.asarray(item["signal"], dtype=np.float32)
        signal = denoise_signal(signal)
        signal = normalize_signal(signal)
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