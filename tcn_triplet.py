import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tcn import TCN
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras.models import Model

# 1. Directory setup
base_dir = "/Users/riyamehdiratta/Desktop/hackathon/HUMI_final"
gesture_folders = ["touch", "scrollup", "scrolldown"]

batch_size = 20  # Set batch size for training

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
data = []
labels = []

print(f"Scanning base directory: {base_dir}")

for session_num in tqdm(range(0, 599)):
    session_folder = os.path.join(base_dir, f"{session_num:03d}")
    if not os.path.isdir(session_folder):
        print(f"Session folder not found: {session_folder}")
        continue
    # NEW: Look for session_X folders inside each session_folder
    for session_sub in os.listdir(session_folder):
        session_sub_folder = os.path.join(session_folder, session_sub)
        if not os.path.isdir(session_sub_folder):
            continue
        for finger_num in range(10):
            finger_folder = os.path.join(session_sub_folder, f"finger_{finger_num}")
            if not os.path.isdir(finger_folder):
                print(f"  Finger folder not found: {finger_folder}")
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
                            print(f"    Skipping file (unknown type): {file_path}")
                            continue
                        try:
                            df = pd.read_csv(file_path, header=0)
                            if list(df.columns) != expected_cols:
                                df = pd.read_csv(file_path, header=None)
                                if df.shape[1] != len(expected_cols):
                                    print(f"    Skipping file (col mismatch): {file_path}")
                                    continue
                                df.columns = expected_cols
                            # Use only numeric columns for TCN
                            numeric_cols = [col for col in expected_cols if col not in ["SSID", "MAC", "info", "channel", "frequency", "name", "action", "field"]]
                            arr = df[numeric_cols].values
                            # Pad or truncate to fixed length (e.g., 100)
                            fixed_len = 100
                            if arr.shape[0] < fixed_len:
                                arr = np.pad(arr, ((0, fixed_len - arr.shape[0]), (0, 0)), constant_values=np.nan)
                            else:
                                arr = arr[:fixed_len]
                            data.append(arr)
                            labels.append(finger_num)
                            print(f"    Loaded file: {file_path}")
                        except Exception as e:
                            print(f"    Error reading {file_path}: {e}")

print(f"Total loaded samples: {len(data)}")

if len(data) == 0:
    print("No data was loaded. Please check your base_dir, folder structure, and CSV files.")
    exit(1)

# Find the maximum number of features
max_features = max(arr.shape[1] for arr in data)

# Pad all arrays to have the same number of features using np.nan
data_padded = []
for arr in data:
    if arr.shape[1] < max_features:
        pad_width = max_features - arr.shape[1]
        arr = np.pad(arr, ((0, 0), (0, pad_width)), constant_values=np.nan)
    data_padded.append(arr)

data = np.array(data_padded)

# 3. Normalize (ignore NaNs)
scaler = StandardScaler()
data_reshaped = data.reshape(-1, data.shape[-1])
mask = ~np.isnan(data_reshaped).any(axis=1)
scaler.fit(data_reshaped[mask])
data_reshaped[mask] = scaler.transform(data_reshaped[mask])
data = data_reshaped.reshape(data.shape)

# 4. Encode labels
le = LabelEncoder()
labels_enc = le.fit_transform(labels)

# 5. Triplet Loss

def triplet_loss(margin=1.0):
    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:,0,:], y_pred[:,1,:], y_pred[:,2,:]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss

# 6. Build TCN embedding model
input_shape = (data.shape[1], data.shape[2])
embedding_dim = 64

def create_embedding_model():
    inp = Input(shape=input_shape)
    x = TCN()(inp)
    x = Dense(embedding_dim)(x)
    return Model(inp, x)

embedding_model = create_embedding_model()

# 7. Triplet model
anchor_in = Input(shape=input_shape, name='anchor_in')
positive_in = Input(shape=input_shape, name='positive_in')
negative_in = Input(shape=input_shape, name='negative_in')

anchor_emb = embedding_model(anchor_in)
positive_emb = embedding_model(positive_in)
negative_emb = embedding_model(negative_in)

merged = Lambda(lambda x: tf.stack(x, axis=1))([anchor_emb, positive_emb, negative_emb])
triplet_model = Model(inputs=[anchor_in, positive_in, negative_in], outputs=merged)
triplet_model.compile(optimizer='adam', loss=triplet_loss(margin=1.0))

# 8. Prepare triplets

def create_triplets(data, labels, num_triplets=1000):
    triplets = []
    for _ in range(num_triplets):
        idx_anchor = np.random.randint(0, len(data))
        label = labels[idx_anchor]
        idx_positive = np.random.choice(np.where(labels == label)[0])
        idx_negative = np.random.choice(np.where(labels != label)[0])
        triplets.append((data[idx_anchor], data[idx_positive], data[idx_negative]))
    return np.array(triplets)

triplets = create_triplets(data, labels_enc, num_triplets=1000)
anchor, positive, negative = triplets[:,0], triplets[:,1], triplets[:,2]
dummy_y = np.zeros((anchor.shape[0], 1))  # Not used

# 9. Train and print triplet loss
history = triplet_model.fit([anchor, positive, negative], dummy_y, epochs=10, batch_size=batch_size)
print("Final triplet loss:", history.history['loss'][-1]) 