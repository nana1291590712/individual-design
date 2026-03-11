# load_dataset.py
import os
import scipy.io as sio

# ---------------------------------------------------------
# 从 .mat 文件读取 DE_time 信号
# ---------------------------------------------------------
def load_mat_file(path):
    data = sio.loadmat(path)

    # 匹配 "DE_time"
    for key in data.keys():
        if key.endswith("DE_time"):
            return data[key].flatten()

    return None


# ---------------------------------------------------------
# 从文件名解析故障类型
# ---------------------------------------------------------
def parse_label(filename):
    name = filename.lower()

    if "normal" in name:
        return "Normal"
    elif "ball" in name:
        return "Ball"
    elif "inner" in name:
        return "Inner"
    elif "outer" in name:
        return "Outer"
    else:
        return "Unknown"


# ---------------------------------------------------------
# === [NEW] 从文件名解析故障等级（Severity / Level）
# ---------------------------------------------------------
def parse_severity(filename):
    """
    故障分级规则（基于 CWRU 常见命名）：
        Normal           -> level 0
        0.007 inch fault -> level 1 (Mild)
        0.014 inch fault -> level 2 (Moderate)
        0.021 inch fault -> level 3 (Severe)
    """
    name = filename.lower()

    if "normal" in name:
        return 0
    elif "0.007" in name:
        return 1
    elif "0.014" in name:
        return 2
    elif "0.021" in name:
        return 3
    else:
        # 未识别等级（可用于调试或后续过滤）
        return -1


# ---------------------------------------------------------
# 从父目录名解析负载
# ---------------------------------------------------------
def parse_load(dirpath):
    if "1730" in dirpath:
        return 3
    elif "1750" in dirpath:
        return 2
    elif "1772" in dirpath:
        return 1
    elif "1797" in dirpath:
        return 0
    else:
        return -1


# ---------------------------------------------------------
# 加载整个 CWRU 根目录
# ---------------------------------------------------------
def load_dataset(root_dir):
    dataset = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith(".mat"):
                continue

            full_path = os.path.join(dirpath, filename)

            signal = load_mat_file(full_path)
            if signal is None:
                continue

            label = parse_label(filename)
            load_value = parse_load(dirpath)

            # === [NEW] 故障等级
            severity = parse_severity(filename)

            dataset.append({
                "signal": signal,
                "label": label,
                "severity": severity,   # === [NEW]
                "filename": filename,
                "load": load_value
            })

    print(f"Loaded files: {len(dataset)}")
    return dataset


# ---------------------------------------------------------
# 测试 load_dataset.py
# ---------------------------------------------------------
if __name__ == "__main__":
    data = load_dataset("data/CWRU")

    print("Example item:")
    if len(data) > 0:
        print(data[0])
