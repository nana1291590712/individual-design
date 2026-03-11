# preprocess_denoise.py
import numpy as np

# =========================================================
# Config
# =========================================================
WINDOW_SIZE = 1024
STEP = 512

# ---- Denoise / filter switch ----
DENOISE_ENABLE = True

# Options: "bandpass" / "median" / "moving_average"
DENOISE_METHOD = "bandpass"

# CWRU 12k Drive End fault data commonly uses 12kHz sampling rate
FS = 12000

# Bandpass filter params (Hz)
BANDPASS_LOW = 10
BANDPASS_HIGH = 3000
BANDPASS_ORDER = 4

# Median filter params
MEDIAN_KERNEL = 5  # odd integer >= 3

# Moving average params
MA_WINDOW = 5


# =========================================================
# Fault label encoding
# =========================================================
label_map = {
    "Normal": 0,
    "Ball": 1,
    "Inner": 2,
    "Outer": 3
}


def encode_label(label_str: str) -> int:
    if label_str not in label_map:
        raise ValueError(f"Unknown fault label: {label_str}")
    return label_map[label_str]


# =========================================================
# Denoise / filtering
# =========================================================
def denoise_signal(x: np.ndarray,
                   enable: bool = DENOISE_ENABLE,
                   method: str = DENOISE_METHOD,
                   fs: int = FS) -> np.ndarray:
    """
    Apply denoise/filtering on raw signal BEFORE normalization & slicing.
    """
    if not enable:
        return x

    x = np.asarray(x, dtype=np.float64)

    if method == "bandpass":
        return butter_bandpass_filter(
            x, lowcut=BANDPASS_LOW, highcut=BANDPASS_HIGH,
            fs=fs, order=BANDPASS_ORDER
        )
    elif method == "median":
        return median_filter_1d(x, kernel_size=MEDIAN_KERNEL)
    elif method == "moving_average":
        return moving_average_filter(x, window_size=MA_WINDOW)
    else:
        raise ValueError(f"Unknown DENOISE_METHOD: {method}")


def butter_bandpass_filter(x: np.ndarray,
                           lowcut: float,
                           highcut: float,
                           fs: float,
                           order: int = 4) -> np.ndarray:
    """
    Butterworth bandpass + zero-phase filtering (filtfilt).
    Requires scipy.
    """
    try:
        from scipy.signal import butter, filtfilt
    except Exception as e:
        raise ImportError(
            "Bandpass filter requires scipy. "
            "Install scipy or switch DENOISE_METHOD to 'median'/'moving_average'."
        ) from e

    if fs <= 0:
        raise ValueError("FS must be > 0")

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    # safety
    low = max(low, 1e-6)
    high = min(high, 0.999999)
    if low >= high:
        raise ValueError(f"Invalid bandpass: lowcut={lowcut}, highcut={highcut}, fs={fs}")

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x)


def median_filter_1d(x: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Median filter (good for impulse noise).
    Uses scipy if available; otherwise uses a simple numpy fallback.
    """
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("MEDIAN_KERNEL must be an odd integer >= 3")

    try:
        from scipy.signal import medfilt
        return medfilt(x, kernel_size=kernel_size)
    except Exception:
        pad = kernel_size // 2
        xp = np.pad(x, (pad, pad), mode="edge")
        out = np.empty_like(x)
        for i in range(len(x)):
            out[i] = np.median(xp[i:i + kernel_size])
        return out


def moving_average_filter(x: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Simple moving average (fast but may smooth fault impulses).
    """
    window_size = int(window_size)
    if window_size < 2:
        return x
    kernel = np.ones(window_size, dtype=np.float64) / window_size
    return np.convolve(x, kernel, mode="same")


# =========================================================
# Normalization + slicing
# =========================================================
def normalize_signal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    std = np.std(x)
    if std < 1e-12:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def sliding_window(x: np.ndarray, window_size: int = WINDOW_SIZE, step: int = STEP) -> np.ndarray:
    window_size = int(window_size)
    step = int(step)

    slices = []
    for start in range(0, len(x) - window_size + 1, step):
        slices.append(x[start:start + window_size])

    return np.array(slices)


# =========================================================
# Main preprocessing
# =========================================================
def preprocess_dataset_denoise(input_dataset,
                               window_size: int = WINDOW_SIZE,
                               step: int = STEP,
                               denoise_enable: bool = DENOISE_ENABLE,
                               denoise_method: str = DENOISE_METHOD,
                               fs: int = FS):
    """
    New preprocessing pipeline:
        raw signal -> denoise/filter -> normalize -> sliding window -> label sync

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
        raw = item["signal"]

        # 1) denoise/filter
        sig = denoise_signal(raw, enable=denoise_enable, method=denoise_method, fs=fs)

        # 2) normalize
        sig = normalize_signal(sig)

        # 3) slice
        slices = sliding_window(sig, window_size, step)

        # 4) labels
        fault_label = encode_label(item["label"])
        raw_severity = int(item.get("severity", -1))
        load = int(item.get("load", 0))

        # same mapping rule as your original preprocess.py
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


# =========================================================
# Self-test
# =========================================================
if __name__ == "__main__":
    from load_dataset import load_dataset

    root = r"D:\design\data\CWRU\12kDriveEndFault"
    dataset = load_dataset(root)

    x, y_fault, y_sev, loads = preprocess_dataset_denoise(
        dataset,
        window_size=WINDOW_SIZE,
        step=STEP,
        denoise_enable=DENOISE_ENABLE,
        denoise_method=DENOISE_METHOD,
        fs=FS
    )

    print("=== preprocess_denoise.py self-test ===")
    print("Denoise enabled:", DENOISE_ENABLE, "| method:", DENOISE_METHOD)
    if DENOISE_METHOD == "bandpass":
        print("Bandpass:", BANDPASS_LOW, "-", BANDPASS_HIGH, "Hz | fs:", FS, "| order:", BANDPASS_ORDER)

    print("X shape:", x.shape)
    print("Fault labels shape:", y_fault.shape, "| unique:", np.unique(y_fault))
    print("Severity labels shape:", y_sev.shape, "| unique:", np.unique(y_sev))
    print("Loads shape:", loads.shape, "| unique:", np.unique(loads))
