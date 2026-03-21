import numpy as np
import matplotlib.pyplot as plt
from load_dataset import load_dataset
from preprocess_denoise import denoise_signal, normalize_signal

# =========================================================
FAULT_ROOT = r"D:\design\data\CWRU\12kDriveEndFault"
NORMAL_ROOT = r"D:\design\data\CWRU\NormalBaseline"

LOAD_LIST = [0, 1, 2, 3]
SEVERITY_ID = 1
WINDOW_SIZE = 1024
START_INDEX = 0
FS = 12000
FREQ_MAX = 6000
USE_HANN = True
USE_LOG = False
FAULT_TYPES = ["Normal", "Ball", "Inner", "Outer"]

# =========================================================
def take_window(signal, window_size, start):
    signal = np.asarray(signal)
    return signal[start:start + window_size]

def pick_first(items, cond_fn):
    for it in items:
        if cond_fn(it):
            return it
    return None

def fft_magnitude(x, fs, use_hann=True, use_log=False):
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    x = x - np.mean(x)

    if use_hann:
        w = np.hanning(n).astype(np.float32)
        xw = x * w
        amp_corr = np.mean(w)
        if amp_corr < 1e-12:
            amp_corr = 1.0
    else:
        xw = x
        amp_corr = 1.0

    X = np.fft.rfft(xw)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = (2.0 / n) * np.abs(X) / amp_corr
    if mag.size > 0:
        mag[0] = mag[0] / 2.0

    if use_log:
        eps = 1e-12
        mag = 20.0 * np.log10(mag + eps)

    return freq, mag

# =========================================================
def plot_time_domain(fault_set, normal_set):
    fig, axes = plt.subplots(
        nrows=len(FAULT_TYPES),
        ncols=len(LOAD_LIST),
        figsize=(16, 10),
        sharex=True,
        sharey=True
    )

    for col, load_id in enumerate(LOAD_LIST):
        normal_item = pick_first(normal_set, lambda it: int(it.get("load", -1)) == load_id)
        selected = {
            "Normal": normal_item,
            "Ball": pick_first(fault_set, lambda it: int(it.get("load", -1))==load_id and int(it.get("severity",-1))==SEVERITY_ID and it.get("label")=="Ball"),
            "Inner": pick_first(fault_set, lambda it: int(it.get("load", -1))==load_id and int(it.get("severity",-1))==SEVERITY_ID and it.get("label")=="Inner"),
            "Outer": pick_first(fault_set, lambda it: int(it.get("load", -1))==load_id and int(it.get("severity",-1))==SEVERITY_ID and it.get("label")=="Outer")
        }

        for row, fault in enumerate(FAULT_TYPES):
            ax = axes[row, col]
            item = selected[fault]

            if item is None:
                ax.set_title("Missing")
                ax.axis("off")
                continue

            x = take_window(item["signal"], WINDOW_SIZE, START_INDEX)
            x = denoise_signal(x)
            x = normalize_signal(x)
            ax.plot(x, linewidth=0.9)

            if col == 0:
                ax.set_ylabel(fault)
            if row == 0:
                ax.set_title(f"Load {load_id} HP")
            if row == len(FAULT_TYPES)-1:
                ax.set_xlabel("Sample index")

            ax.grid(True, alpha=0.3)

    fig.suptitle("Denoised Time-Domain Signals", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

# =========================================================
def plot_frequency_domain(fault_set, normal_set):
    fig, axes = plt.subplots(
        nrows=len(FAULT_TYPES),
        ncols=len(LOAD_LIST),
        figsize=(16, 10),
        sharex=True,
        sharey=True
    )

    for col, load_id in enumerate(LOAD_LIST):
        normal_item = pick_first(normal_set, lambda it: int(it.get("load", -1)) == load_id)
        selected = {
            "Normal": normal_item,
            "Ball": pick_first(fault_set, lambda it: int(it.get("load", -1))==load_id and int(it.get("severity",-1))==SEVERITY_ID and it.get("label")=="Ball"),
            "Inner": pick_first(fault_set, lambda it: int(it.get("load", -1))==load_id and int(it.get("severity",-1))==SEVERITY_ID and it.get("label")=="Inner"),
            "Outer": pick_first(fault_set, lambda it: int(it.get("load", -1))==load_id and int(it.get("severity",-1))==SEVERITY_ID and it.get("label")=="Outer")
        }

        for row, fault in enumerate(FAULT_TYPES):
            ax = axes[row, col]
            item = selected[fault]

            if item is None:
                ax.set_title("Missing")
                ax.axis("off")
                continue

            x = take_window(item["signal"], WINDOW_SIZE, START_INDEX)
            x = denoise_signal(x)
            x = normalize_signal(x)

            freq, mag = fft_magnitude(x, fs=FS, use_hann=USE_HANN, use_log=USE_LOG)
            idx = freq <= FREQ_MAX
            ax.plot(freq[idx], mag[idx], linewidth=0.9)

            if col == 0:
                ylab = "Magnitude" if not USE_LOG else "Magnitude (dB)"
                ax.set_ylabel(f"{fault}\n{ylab}")
            if row == 0:
                ax.set_title(f"Load {load_id} HP")
            if row == len(FAULT_TYPES)-1:
                ax.set_xlabel("Frequency (Hz)")
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"Denoised FFT Magnitude Spectrum (N={WINDOW_SIZE}, fs={FS} Hz)", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

# =========================================================
def main():
    fault_set = load_dataset(FAULT_ROOT)
    normal_set = load_dataset(NORMAL_ROOT)

    plot_time_domain(fault_set, normal_set)
    plot_frequency_domain(fault_set, normal_set)

if __name__ == "__main__":
    main()