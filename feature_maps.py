import numpy as np
import matplotlib.pyplot as plt
import pywt

from load_dataset import load_dataset
from preprocess import normalize_signal, sliding_window

# =========================
# Config
# =========================
DATA_ROOT = r"D:\design\data\CWRU"
WINDOW = 1024
STEP = 1024
FS = 12000                 # CWRU 12k Drive End
WAVELET = "morl"
SCALES = np.arange(1, 128)

FAULT_TYPES = ["Ball", "Inner", "Outer", "Normal"]

# =========================
# Load dataset
# =========================
dataset = load_dataset(DATA_ROOT)

def get_one_sample(fault):
    for item in dataset:
        if item["label"] == fault:
            sig = normalize_signal(item["signal"])
            win = sliding_window(sig, WINDOW, STEP)[0]
            return win
    raise RuntimeError(f"No data for {fault}")

# =========================
# Plot
# =========================
def plot_signal_and_cwt():
    n = len(FAULT_TYPES)

    fig, axes = plt.subplots(
        2, n,
        figsize=(3.2 * n, 6),
        constrained_layout=True
    )

    for i, fault in enumerate(FAULT_TYPES):
        signal = get_one_sample(fault)

        # -------- Time domain --------
        axes[0, i].plot(signal, linewidth=0.8)
        axes[0, i].set_title(fault)
        axes[0, i].set_ylabel("Amplitude")
        axes[0, i].set_xticks([])

        # -------- CWT --------
        coef, freqs = pywt.cwt(
            signal,
            scales=SCALES,
            wavelet=WAVELET,
            sampling_period=1 / FS
        )

        im = axes[1, i].imshow(
            np.abs(coef),
            extent=[0, len(signal), freqs[-1], freqs[0]],
            aspect="auto",
            cmap="viridis"
        )

        axes[1, i].set_xlabel("Time index")
        axes[1, i].set_ylabel("Frequency (Hz)")

    fig.suptitle(
        "Raw Signals and CWT Time-Frequency Representations",
        fontsize=16
    )

    cbar = fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.85
    )
    cbar.set_label("Magnitude")

    plt.show()

# =========================
# Run
# =========================
plot_signal_and_cwt()
