import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tcn import TCN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# 1. Directory setup
base_dir = "/Users/riyamehdiratta/Desktop/hackathon/HUMI_final"
gesture_folders = ["touch", "scrollup", "scrolldown"]

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
    

# 2. Data aggregation
data = []
labels = []

for session_num in tqdm(range(0, 599)):
    session_folder = os.path.join(base_dir, f"{session_num:03d}")
    if not os.path.isdir(session_folder):
        continue
    for finger_num in range(10):
        finger_folder = os.path.join(session_folder, f"Finger_{finger_num}")
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
                        # Use only numeric columns for TCN
                        numeric_cols = [col for col in expected_cols if col not in ["SSID", "MAC", "info", "channel", "frequency", "name", "action", "field"]]
                        arr = df[numeric_cols].values
                        # Pad or truncate to fixed length (e.g., 100)
                        fixed_len = 100
                        if arr.shape[0] < fixed_len:
                            arr = np.pad(arr, ((0, fixed_len - arr.shape[0]), (0, 0)), 'constant')
                        else:
                            arr = arr[:fixed_len]
                        data.append(arr)
                        labels.append(finger_num)  # or session_num, or a tuple
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

data = np.array(data)
labels = np.array(labels)
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# 3. Normalize
scaler = StandardScaler()
data_reshaped = data.reshape(-1, data.shape[-1])
data_reshaped = scaler.fit_transform(data_reshaped)
data = data_reshaped.reshape(data.shape)

# 4. Encode labels
le = LabelEncoder()
labels_enc = le.fit_transform(labels)
num_classes = len(np.unique(labels_enc))
labels_cat = to_categorical(labels_enc, num_classes=num_classes)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(data, labels_cat, test_size=0.2, random_state=42, stratify=labels_enc)

# 6. Build TCN model
model = Sequential([
    TCN(input_shape=(data.shape[1], data.shape[2])),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Train
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# 8. Evaluate
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
acc = accuracy_score(y_true, y_pred_classes)
print(f"Test Accuracy: {acc:.4f}") 