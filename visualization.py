import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from preprocess import normalize_signal, sliding_window


# ------------------------------------------------------------
# 1) 自动搜索你的 CWRU MAT 文件
# ------------------------------------------------------------
def find_mat_file():
    search_path = r"D:\design\data\CWRU\12kDriveEndFault\1730\*.mat"
    mat_files = glob.glob(search_path)

    if len(mat_files) == 0:
        raise FileNotFoundError(f"未找到 MAT 文件，请检查路径：{search_path}")

    print("Found file:", mat_files[0])
    return mat_files[0]


# ------------------------------------------------------------
# 2) 从文件中读取 DE_time 信号
# ------------------------------------------------------------
def load_DE_time(path):
    data = sio.loadmat(path)
    for k, v in data.items():
        if "DE_time" in k:
            return v.flatten()
    raise ValueError("MAT 文件中未找到 DE_time 信号！")


# ------------------------------------------------------------
# 3) 可视化信号处理步骤
# ------------------------------------------------------------
def visualize_pipeline(raw, norm, windows):
    # Step 1 — 原始信号
    plt.figure(figsize=(12, 3))
    plt.plot(raw, linewidth=0.7)
    plt.title("Step 1: Raw DE_time Signal (Original)")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    # Step 2 — 标准化信号
    plt.figure(figsize=(12, 3))
    plt.plot(norm, linewidth=0.7)
    plt.title("Step 2: Normalized Signal (x-mean)/std")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    # Step 3 — 滑窗覆盖示意（前 5 个窗口）
    plt.figure(figsize=(12, 4))
    plt.plot(norm, alpha=0.6, linewidth=0.7, label="normalized signal")

    wlen = len(windows[0])
    for i in range(5):
        start = i * 512
        plt.axvspan(start, start + wlen, color='yellow', alpha=0.2)

    plt.title("Step 3: Sliding Windows (first 5 windows)")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Step 4 — 第一个窗口（模型输入片段）
    plt.figure(figsize=(12, 3))
    plt.plot(windows[0], linewidth=0.8)
    plt.title("Step 4: Windowed Segment Used as Model Input (1024 points)")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 主程序
# ------------------------------------------------------------
if __name__ == "__main__":
    mat_path = find_mat_file()

    raw = load_DE_time(mat_path)
    norm = normalize_signal(raw)
    windows = sliding_window(norm, 1024, 512)

    visualize_pipeline(raw, norm, windows)

    print("\nSignal processing visualization completed.\n")
