import numpy as np
import matplotlib.pyplot as plt
from load_dataset import load_dataset


# =========================================================
# 路径
# =========================================================
FAULT_ROOT = r"D:\design\data\CWRU\12kDriveEndFault"
NORMAL_ROOT = r"D:\design\data\CWRU\NormalBaseline"


# =========================================================
# 参数
# =========================================================
LOAD_LIST = [0, 1, 2, 3]      # 0/1/2/3 HP
SEVERITY_ID = 1              # 1=0.007, 2=0.014, 3=0.021

WINDOW_SIZE = 1024
START_INDEX = 0

FS = 12000                   # CWRU 12k 采样率
FREQ_MAX = 6000              # 只画到 Nyquist(=FS/2)

USE_HANN = True              # FFT 前加 Hann 窗，减少频谱泄漏
USE_LOG = False              # True: 画 20*log10(mag)；False: 线性幅值

FAULT_TYPES = ["Normal", "Ball", "Inner", "Outer"]


# =========================================================
# 工具函数
# =========================================================
def normalize_signal(x):
    x = np.asarray(x, dtype=np.float32)
    std = np.std(x)
    if std < 1e-12:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def take_window(signal, window_size, start):
    signal = np.asarray(signal)
    return signal[start:start + window_size]


def pick_first(items, cond_fn):
    for it in items:
        if cond_fn(it):
            return it
    return None


def fft_magnitude(x, fs, use_hann=True, use_log=False):
    """
    x: 1D signal (already windowed length N)
    return: freq (Hz), mag (single-sided)
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]

    # 去均值（减少 DC）
    x = x - np.mean(x)

    # 加窗
    if use_hann:
        w = np.hanning(n).astype(np.float32)
        xw = x * w
        # 粗略幅值修正：窗的平均值（避免整体被压太多）
        amp_corr = np.mean(w)
        if amp_corr < 1e-12:
            amp_corr = 1.0
    else:
        xw = x
        amp_corr = 1.0

    # 单边 FFT
    X = np.fft.rfft(xw)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)

    # 幅度谱（单边幅值，做 2/N）
    mag = (2.0 / n) * np.abs(X) / amp_corr

    # 直流分量不放大（可选）
    if mag.size > 0:
        mag[0] = mag[0] / 2.0

    if use_log:
        eps = 1e-12
        mag = 20.0 * np.log10(mag + eps)

    return freq, mag


# =========================================================
# 主流程：FFT 幅度谱 4x4 可视化
# =========================================================
def main():
    fault_set = load_dataset(FAULT_ROOT)
    normal_set = load_dataset(NORMAL_ROOT)

    fig, axes = plt.subplots(
        nrows=len(FAULT_TYPES),
        ncols=len(LOAD_LIST),
        figsize=(16, 10),
        sharex=True,
        sharey=True
    )

    for col, load_id in enumerate(LOAD_LIST):

        # -------- Normal --------
        normal_item = pick_first(
            normal_set,
            lambda it: int(it.get("load", -1)) == load_id
        )

        selected = {
            "Normal": normal_item,
            "Ball": pick_first(
                fault_set,
                lambda it: int(it.get("load", -1)) == load_id
                          and int(it.get("severity", -1)) == SEVERITY_ID
                          and it.get("label") == "Ball"
            ),
            "Inner": pick_first(
                fault_set,
                lambda it: int(it.get("load", -1)) == load_id
                          and int(it.get("severity", -1)) == SEVERITY_ID
                          and it.get("label") == "Inner"
            ),
            "Outer": pick_first(
                fault_set,
                lambda it: int(it.get("load", -1)) == load_id
                          and int(it.get("severity", -1)) == SEVERITY_ID
                          and it.get("label") == "Outer"
            ),
        }

        # -------- 画每一行 --------
        for row, fault in enumerate(FAULT_TYPES):
            ax = axes[row, col]
            item = selected[fault]

            if item is None:
                ax.set_title("Missing")
                ax.axis("off")
                continue

            raw = item["signal"]
            x = take_window(raw, WINDOW_SIZE, START_INDEX)
            x = normalize_signal(x)

            freq, mag = fft_magnitude(
                x, fs=FS, use_hann=USE_HANN, use_log=USE_LOG
            )

            # 只画到 FREQ_MAX
            idx = freq <= FREQ_MAX
            ax.plot(freq[idx], mag[idx], linewidth=0.9)

            # 只在最左侧写故障名
            if col == 0:
                ylab = "Magnitude"
                if USE_LOG:
                    ylab = "Magnitude (dB)"
                ax.set_ylabel(f"{fault}\n{ylab}")

            # 只在最上方写负载
            if row == 0:
                ax.set_title(f"Load {load_id} HP")

            # 只在最下方写横轴
            if row == len(FAULT_TYPES) - 1:
                ax.set_xlabel("Frequency (Hz)")

            ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"FFT magnitude spectrum (N={WINDOW_SIZE}, fs={FS} Hz) under different loads and fault conditions",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()