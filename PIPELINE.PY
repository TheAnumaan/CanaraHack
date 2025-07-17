import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Directory setup
base_dir = "/Users/riyamehdiratta/Desktop/hackathon/HUMI_final"
gesture_folders = ["touch", "scrollup", "scrolldown"]

# Helper to get expected columns

def get_expected_cols(file_lower, parent_folder, grandparent_folder):
    if parent_folder == "sensors" and grandparent_folder in gesture_folders:
        if any(sensor in file_lower for sensor in ["gyro", "lacc", "magn", "nacc"]):
            return ["timestamp(ms)", "orientation", "x", "y", "z"]
        elif "grav" in file_lower:
            return ["timestamp(ms)", "orientation", "gravity_data"]
        elif "ligh" in file_lower:
            return ["timestamp(ms)", "orientation", "light_data"]
        elif "prox" in file_lower:
            return ["timestamp(ms)", "orientation", "proximity"]
        elif "temp" in file_lower:
            return ["timestamp(ms)", "orientation", "temperature"]
        elif "swipe" in file_lower or "touch" in file_lower:
            return ["timestamp(ms)", "orientation", "x", "y", "p", "action"]
    elif "swipe" in file_lower or "touch" in file_lower:
        return ["timestamp(ms)", "orientation", "x", "y", "p", "action"]
    elif any(sensor in file_lower for sensor in ["gyro", "lacc", "magn", "nacc"]):
        return ["timestamp(ms)", "orientation", "x", "y", "z"]
    elif "grav" in file_lower:
        return ["timestamp(ms)", "orientation", "gravity_data"]
    elif "ligh" in file_lower:
        return ["timestamp(ms)", "orientation", "light_data"]
    elif "prox" in file_lower:
        return ["timestamp(ms)", "orientation", "proximity"]
    elif "temp" in file_lower:
        return ["timestamp(ms)", "orientation", "temperature"]
    elif "wifi" in file_lower:
        return ["timestamp(ms)", "SSID", "level", "info", "channel", "frequency"]
    return None

# 2. Data aggregation
raw_data = []
raw_labels = []
shapes = []

print(f"Scanning base directory: {base_dir}")

for session_num in tqdm(range(0, 599)):
    session_folder = os.path.join(base_dir, f"{session_num:03d}")
    if not os.path.isdir(session_folder):
        continue
    for session_sub in os.listdir(session_folder):
        session_sub_folder = os.path.join(session_folder, session_sub)
        if not os.path.isdir(session_sub_folder):
            continue
        for finger_num in range(10):
            finger_folder = os.path.join(session_sub_folder, f"finger_{finger_num}")
            if not os.path.isdir(finger_folder):
                continue
            for root, dirs, files in os.walk(finger_folder):
                for file in files:
                    if file.endswith(".csv"):
                        file_path = os.path.join(root, file)
                        parent_folder = os.path.basename(os.path.dirname(file_path)).lower()
                        grandparent_folder = os.path.basename(os.path.dirname(os.path.dirname(file_path))).lower()
                        file_lower = file.lower()
                        expected_cols = get_expected_cols(file_lower, parent_folder, grandparent_folder)
                        if expected_cols is None:
                            continue
                        try:
                            df = pd.read_csv(file_path, header=0)
                            if list(df.columns) != expected_cols:
                                df = pd.read_csv(file_path, header=None)
                                if df.shape[1] != len(expected_cols):
                                    continue
                                df.columns = expected_cols
                            numeric_cols = [col for col in expected_cols if col not in ["SSID", "MAC", "info", "channel", "frequency", "name", "action", "field"]]
                            arr = df[numeric_cols].values
                            fixed_len = 100
                            if arr.shape[0] < fixed_len:
                                arr = np.pad(arr, ((0, fixed_len - arr.shape[0]), (0, 0)), 'constant')
                            else:
                                arr = arr[:fixed_len]
                            shapes.append(arr.shape)
                            raw_data.append(arr)
                            raw_labels.append(finger_num)
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")

print(f"Total loaded samples: {len(raw_data)}")

# Find the most common shape
shape_counts = Counter(shapes)
if not shape_counts:
    print("No data loaded. Please check your data path and structure.")
    exit(1)
most_common_shape, _ = shape_counts.most_common(1)[0]
print(f"Most common shape: {most_common_shape}")

# Filter data to only keep samples with the most common shape
data = [d for d, s in zip(raw_data, shapes) if s == most_common_shape]
labels = [l for l, s in zip(raw_labels, shapes) if s == most_common_shape]

data = np.array(data)
labels = np.array(labels)
print("Filtered data shape:", data.shape)
print("Filtered labels shape:", labels.shape)

# Normalize
scaler = StandardScaler()
data_reshaped = data.reshape(-1, data.shape[-1])
data_reshaped = scaler.fit_transform(data_reshaped)
data = data_reshaped.reshape(data.shape)
print("Normalized data shape:", data.shape)

# Encode labels
le = LabelEncoder()
labels_enc = le.fit_transform(labels)
print("Encoded labels shape:", labels_enc.shape)
