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
# 说明：
# 当前仍优先按文件名关键词判断；
# 若文件名不包含关键词，则返回 "Unknown"，后续通过调试输出检查。
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
# 从文件名解析故障等级（Severity / defect diameter index）
# ---------------------------------------------------------
def parse_severity(filename):
    """
    Severity index mapping:
        Normal           -> 0
        0.007 inch fault -> 1
        0.014 inch fault -> 2
        0.021 inch fault -> 3
        0.028 inch fault -> 4
        Unknown          -> -1
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
    elif "0.028" in name:
        return 4
    else:
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

    unknown_label_files = []
    unknown_severity_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith(".mat"):
                continue

            full_path = os.path.join(dirpath, filename)

            signal = load_mat_file(full_path)
            if signal is None:
                continue

            label = parse_label(filename)
            severity = parse_severity(filename)
            load_value = parse_load(dirpath)

            if label == "Unknown":
                unknown_label_files.append(full_path)

            if severity == -1:
                unknown_severity_files.append(full_path)

            dataset.append({
                "signal": signal,
                "label": label,
                "severity": severity,
                "filename": filename,
                "load": load_value
            })

    print(f"Loaded files: {len(dataset)}")

    # 调试输出
    label_count = {}
    severity_count = {}
    load_count = {}

    for item in dataset:
        label_count[item["label"]] = label_count.get(item["label"], 0) + 1
        severity_count[item["severity"]] = severity_count.get(item["severity"], 0) + 1
        load_count[item["load"]] = load_count.get(item["load"], 0) + 1

    print("Label count:", label_count)
    print("Severity count:", severity_count)
    print("Load count:", load_count)

    if len(unknown_label_files) > 0:
        print("\n[Warning] Files with Unknown label:")
        for p in unknown_label_files[:20]:
            print(p)
        if len(unknown_label_files) > 20:
            print(f"... and {len(unknown_label_files) - 20} more")

    if len(unknown_severity_files) > 0:
        print("\n[Warning] Files with Unknown severity:")
        for p in unknown_severity_files[:20]:
            print(p)
        if len(unknown_severity_files) > 20:
            print(f"... and {len(unknown_severity_files) - 20} more")

    return dataset


# ---------------------------------------------------------
# 测试 load_dataset.py
# ---------------------------------------------------------
if __name__ == "__main__":
    data = load_dataset("data/CWRU")

    print("\nExample item:")
    if len(data) > 0:
        print(data[0])