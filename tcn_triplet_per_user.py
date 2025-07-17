import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tcn import TCN
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras.models import Model

base_dir = "/Users/riyamehdiratta/Desktop/hackathon/HUMI_final"
gesture_folders = ["touch", "scrollup", "scrolldown"]
batch_size = 20

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
    return None

def robust_read_csv(file_path, expected_cols):
    try:
        df = pd.read_csv(file_path, header=0)
        if list(df.columns) != expected_cols:
            df = pd.read_csv(file_path, header=None)
            if df.shape[1] != len(expected_cols):
                print(f"    Skipping file (col mismatch): {file_path}")
                return None
            df.columns = expected_cols
    except Exception as e:
        print(f"    Error reading {file_path}: {e}")
        return None
    return df

def extract_numeric_array(df, expected_cols):
    numeric_cols = [col for col in expected_cols if col not in ["SSID", "MAC", "info", "name", "action", "field"]]
    arr = df[numeric_cols].values
    return arr

def triplet_loss(margin=1.0):
    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:,0,:], y_pred[:,1,:], y_pred[:,2,:]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss

def create_embedding_model(input_shape, embedding_dim=64):
    inp = Input(shape=input_shape)
    x = TCN()(inp)
    x = Dense(embedding_dim)(x)
    return Model(inp, x)

def create_triplets(data, labels, num_triplets=1000):
    triplets = []
    for _ in range(num_triplets):
        idx_anchor = np.random.randint(0, len(data))
        label = labels[idx_anchor]
        idx_positive = np.random.choice(np.where(labels == label)[0])
        idx_negative = np.random.choice(np.where(labels != label)[0])
        triplets.append((data[idx_anchor], data[idx_positive], data[idx_negative]))
    return np.array(triplets)

user_indices = list(range(0, 599))
batch_user_size = 20

for batch_start in range(0, len(user_indices), batch_user_size):
    batch_users = user_indices[batch_start:batch_start+batch_user_size]
    print(f"\nProcessing user batch: {batch_users[0]:03d} to {batch_users[-1]:03d}")
    batch_data = []
    batch_labels = []
    user_data_dict = {}
    user_label_dict = {}
    for user_num in batch_users:
        user_folder = os.path.join(base_dir, f"{user_num:03d}")
        if not os.path.isdir(user_folder):
            print(f"User folder not found: {user_folder}")
            continue
        data = []
        labels = []
        for session_sub in os.listdir(user_folder):
            session_sub_folder = os.path.join(user_folder, session_sub)
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
                                print(f"    Skipping file (unknown type): {file_path}")
                                continue
                            df = robust_read_csv(file_path, expected_cols)
                            if df is None:
                                continue
                            arr = extract_numeric_array(df, expected_cols)
                            data.extend(arr)
                            labels.append(finger_num)
        if len(data) < 10:
            print(f"  Not enough samples for user {user_num:03d}, skipping.")
            continue
        batch_data.extend(data)
        batch_labels.extend(labels)
        user_data_dict[user_num] = data
        user_label_dict[user_num] = labels
    if not batch_data:
        print("  No data in this batch, skipping.")
        continue
    # Only consider arrays with at least 2 dimensions and nonzero shape
    valid_arrays = [arr for arr in batch_data if len(arr.shape) == 2 and arr.shape[1] > 0]
    if not valid_arrays:
        print("  No valid data in this batch, skipping.")
        continue
    max_features = max(arr.shape[1] for arr in valid_arrays)
    for user_num in user_data_dict:
        data = user_data_dict[user_num]
        data_padded = []
        for arr in data:
            if len(arr.shape) != 2 or arr.shape[1] == 0:
                continue  # skip invalid arrays
            if arr.shape[1] < max_features:
                pad_width = max_features - arr.shape[1]
                arr = np.pad(arr, ((0, 0), (0, pad_width)), constant_values=np.nan)
            data_padded.append(arr)
        user_data_dict[user_num] = np.array(data_padded)
        user_label_dict[user_num] = np.array(user_label_dict[user_num])
    for user_num in user_data_dict:
        data = user_data_dict[user_num]
        labels = user_label_dict[user_num]
        scaler = StandardScaler()
        data_reshaped = data.reshape(-1, data.shape[-1])
        mask = ~np.isnan(data_reshaped).any(axis=1)
        scaler.fit(data_reshaped[mask])
        data_reshaped[mask] = scaler.transform(data_reshaped[mask])
        data = data_reshaped.reshape(data.shape)
        le = LabelEncoder()
        labels_enc = le.fit_transform(labels)
        input_shape = (data.shape[1], data.shape[2])
        embedding_model = create_embedding_model(input_shape)
        anchor_in = Input(shape=input_shape, name='anchor_in')
        positive_in = Input(shape=input_shape, name='positive_in')
        negative_in = Input(shape=input_shape, name='negative_in')
        anchor_emb = embedding_model(anchor_in)
        positive_emb = embedding_model(positive_in)
        negative_emb = embedding_model(negative_in)
        merged = Lambda(lambda x: tf.stack(x, axis=1))([anchor_emb, positive_emb, negative_emb])
        triplet_model = Model(inputs=[anchor_in, positive_in, negative_in], outputs=merged)
        triplet_model.compile(optimizer='adam', loss=triplet_loss(margin=1.0))
        num_triplets = min(500, len(data))
        triplets = create_triplets(data, labels_enc, num_triplets=num_triplets)
        anchor, positive, negative = triplets[:,0], triplets[:,1], triplets[:,2]
        dummy_y = np.zeros((anchor.shape[0], 1))
        history = triplet_model.fit([anchor, positive, negative], dummy_y, epochs=3, batch_size=batch_size, verbose=0)
        print(f"  Final triplet loss for user {user_num:03d}: {history.history['loss'][-1]:.4f}") 