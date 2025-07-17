import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import argparse
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import pickle
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# --- Robust CSV loading utilities ---
gesture_folders = set(["gestures"])  # Placeholder, update as needed

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

def iterate_user_sessions(base_dir, user_indices, batch_user_size=20):
    """
    Iterate over users, sessions, and finger folders, loading CSVs robustly.
    Prints progress and skips files/folders as needed.
    """
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
                                # You may want to add your own numeric extraction logic here
                                # arr = extract_numeric_array(df, expected_cols)
                                # data.extend(arr)
                                # labels.append(finger_num)
                                data.append(df)
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
    # Return collected data if needed
    return batch_data, batch_labels, user_data_dict, user_label_dict

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class TemporalConvBlock(nn.Module):
    """Single temporal convolutional block with residual connection"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              dilation=dilation, padding=(kernel_size-1)*dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                              dilation=dilation, padding=(kernel_size-1)*dilation)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)
        residual = x
        
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # Remove padding
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]  # Remove padding
        out = self.norm2(out)
        
        if self.residual is not None:
            residual = self.residual(residual)
            
        out += residual
        return self.relu(out)


class TCNEncoder(nn.Module):
    """Lightweight TCN encoder for variable-length time series"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=4, kernel_size=3, 
                 embedding_dim=128, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        
        # TCN blocks with increasing dilation
        self.tcn_blocks = nn.ModuleList([
            TemporalConvBlock(hidden_dim, hidden_dim, kernel_size, 
                            dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])
        
        # Global pooling and final projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, embedding_dim)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Handle single sequence case
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Convert to (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # Apply TCN blocks
        for block in self.tcn_blocks:
            x = block(x)
        
        # Global pooling
        x = self.global_pool(x)  # (batch_size, hidden_dim, 1)
        x = x.squeeze(-1)  # (batch_size, hidden_dim)
        
        # Final projection to embedding
        embedding = self.final_proj(x)
        
        # L2 normalize embedding
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding

def extract_numeric_array(df, expected_cols):
    numeric_cols = [col for col in expected_cols if col not in ["SSID", "MAC", "info", "name", "action", "field"]]
    arr = df[numeric_cols].values
    return arr

# Utility: Iterate over all user/session/finger/csv and apply TCNEncoder to each sample

def apply_tcn_to_all_samples(base_dir, tcn_model, device, user_indices=None, batch_user_size=20):
    """
    Iterate over all user/session/finger/csv, load and preprocess, and apply TCNEncoder to each sample.
    Prints embedding shape for each file.
    """
    if user_indices is None:
        user_indices = [int(u) for u in os.listdir(base_dir) if u.isdigit() and len(u) == 3]
        user_indices.sort()
    for batch_start in range(0, len(user_indices), batch_user_size):
        batch_users = user_indices[batch_start:batch_start+batch_user_size]
        print(f"\nProcessing user batch: {batch_users[0]:03d} to {batch_users[-1]:03d}")
        for user_num in batch_users:
            user_folder = os.path.join(base_dir, f"{user_num:03d}")
            if not os.path.isdir(user_folder):
                print(f"User folder not found: {user_folder}")
                continue
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
                                if arr.shape[0] == 0 or arr.shape[1] == 0:
                                    print(f"    Skipping file (no valid numeric data): {file_path}")
                                    continue
                                # Preprocess: z-score normalization
                                mean = np.mean(arr, axis=0, keepdims=True)
                                std = np.std(arr, axis=0, keepdims=True)
                                std = np.where(std == 0, 1, std)
                                arr = (arr - mean) / std
                                # Convert to torch tensor
                                tensor_data = torch.FloatTensor(arr).unsqueeze(0).to(device)  # (1, seq_len, input_dim)
                                with torch.no_grad():
                                    embedding = tcn_model(tensor_data)
                                print(f"    {file_path}: embedding shape {embedding.shape}")

def discover_data_structure_nested(data_dir):
    """
    Discover users, sessions, fingers, and CSVs in nested folder structure:
    data_dir/user/session/finger_x/*.csv
    Returns:
        users: list of user folder names (e.g., ['000', '001', ...])
        sessions_per_user: dict user -> list of session names
        fingers_per_user_session: dict (user, session) -> list of finger names
        csvs_per_user_session_finger: dict (user, session, finger) -> list of csv file paths
    """
    users = []
    sessions_per_user = {}
    fingers_per_user_session = {}
    csvs_per_user_session_finger = {}
    for user in sorted(os.listdir(data_dir)):
        user_path = os.path.join(data_dir, user)
        if not os.path.isdir(user_path) or not user.isdigit() or len(user) != 3:
            continue
        users.append(user)
        sessions = []
        for session in sorted(os.listdir(user_path)):
            session_path = os.path.join(user_path, session)
            if not os.path.isdir(session_path):
                continue
            sessions.append(session)
            fingers = []
            for finger in sorted(os.listdir(session_path)):
                if not finger.startswith("finger_"):
                    continue
                finger_path = os.path.join(session_path, finger)
                if not os.path.isdir(finger_path):
                    continue
                fingers.append(finger)
                csvs = []
                for root, dirs, files in os.walk(finger_path):
                    for file in files:
                        if file.endswith(".csv"):
                            csvs.append(os.path.join(root, file))
                csvs_per_user_session_finger[(user, session, finger)] = csvs
            fingers_per_user_session[(user, session)] = fingers
        sessions_per_user[user] = sessions
    return users, sessions_per_user, fingers_per_user_session, csvs_per_user_session_finger

class TripletDataset(Dataset):
    """Dataset for triplet loss training with nested folder structure"""
    def __init__(self, data_dir: str, users: list, sessions_per_user: dict, fingers_per_user_session: dict, csvs_per_user_session_finger: dict, samples_per_epoch: int = 1000, normalize: bool = True):
        self.data_dir = data_dir
        self.users = users
        self.sessions_per_user = sessions_per_user
        self.fingers_per_user_session = fingers_per_user_session
        self.csvs_per_user_session_finger = csvs_per_user_session_finger
        self.samples_per_epoch = samples_per_epoch
        self.normalize = normalize
        self.data_cache = {}
        
    def __len__(self):
        return self.samples_per_epoch
    
    def load_csv_data(self, filepath: str) -> torch.Tensor:
        if filepath in self.data_cache:
            return self.data_cache[filepath]
        try:
            file_lower = os.path.basename(filepath).lower()
            parent_folder = os.path.basename(os.path.dirname(filepath)).lower()
            grandparent_folder = os.path.basename(os.path.dirname(os.path.dirname(filepath))).lower()
            expected_cols = get_expected_cols(file_lower, parent_folder, grandparent_folder)
            if expected_cols is None:
                expected_cols = [col for col in pd.read_csv(filepath, nrows=1).columns]
            df = robust_read_csv(filepath, expected_cols)
            if df is None:
                return torch.zeros(10, 6)
            data = df.select_dtypes(include=[np.number]).values
            if self.normalize:
                mean = np.mean(data, axis=0, keepdims=True)
                std = np.std(data, axis=0, keepdims=True)
                std = np.where(std == 0, 1, std)
                data = (data - mean) / std
            tensor_data = torch.FloatTensor(data)
            if len(self.data_cache) < 100:
                self.data_cache[filepath] = tensor_data
            return tensor_data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return torch.zeros(10, 6)
    
    def __getitem__(self, idx):
        # Sample anchor user, session, finger, csv
        anchor_user = random.choice(self.users)
        anchor_sessions = self.sessions_per_user[anchor_user]
        anchor_session = random.choice(anchor_sessions)
        anchor_fingers = self.fingers_per_user_session[(anchor_user, anchor_session)]
        anchor_finger = random.choice(anchor_fingers)
        anchor_csvs = self.csvs_per_user_session_finger[(anchor_user, anchor_session, anchor_finger)]
        anchor_csv = random.choice(anchor_csvs)
        anchor_data = self.load_csv_data(anchor_csv)

        # Positive: same user, different session or finger or csv
        positive_session = anchor_session
        positive_finger = anchor_finger
        positive_csv = anchor_csv
        # Try to pick a different csv, finger, or session
        positive_options = []
        for s in self.sessions_per_user[anchor_user]:
            for f in self.fingers_per_user_session[(anchor_user, s)]:
                for c in self.csvs_per_user_session_finger[(anchor_user, s, f)]:
                    if not (s == anchor_session and f == anchor_finger and c == anchor_csv):
                        positive_options.append((s, f, c))
        if positive_options:
            positive_session, positive_finger, positive_csv = random.choice(positive_options)
        positive_data = self.load_csv_data(positive_csv)

        # Negative: different user
        negative_users = [u for u in self.users if u != anchor_user]
        negative_user = random.choice(negative_users)
        negative_sessions = self.sessions_per_user[negative_user]
        negative_session = random.choice(negative_sessions)
        negative_fingers = self.fingers_per_user_session[(negative_user, negative_session)]
        negative_finger = random.choice(negative_fingers)
        negative_csvs = self.csvs_per_user_session_finger[(negative_user, negative_session, negative_finger)]
        negative_csv = random.choice(negative_csvs)
        negative_data = self.load_csv_data(negative_csv)
        
        return anchor_data, positive_data, negative_data

class TestDataset(Dataset):
    """Dataset for evaluation with nested folder structure"""
    def __init__(self, data_dir: str, test_pairs: list, normalize: bool = True):
        self.data_dir = data_dir
        self.test_pairs = test_pairs  # list of (csv1, csv2, label)
        self.normalize = normalize
        self.data_cache = {}
    
    def load_csv_data(self, filepath: str) -> torch.Tensor:
        if filepath in self.data_cache:
            return self.data_cache[filepath]
        try:
            file_lower = os.path.basename(filepath).lower()
            parent_folder = os.path.basename(os.path.dirname(filepath)).lower()
            grandparent_folder = os.path.basename(os.path.dirname(os.path.dirname(filepath))).lower()
            expected_cols = get_expected_cols(file_lower, parent_folder, grandparent_folder)
            if expected_cols is None:
                expected_cols = [col for col in pd.read_csv(filepath, nrows=1).columns]
            df = robust_read_csv(filepath, expected_cols)
            if df is None:
                return torch.zeros(10, 6)
            data = df.select_dtypes(include=[np.number]).values
            if self.normalize:
                mean = np.mean(data, axis=0, keepdims=True)
                std = np.std(data, axis=0, keepdims=True)
                std = np.where(std == 0, 1, std)
                data = (data - mean) / std
            tensor_data = torch.FloatTensor(data)
            if len(self.data_cache) < 200:
                self.data_cache[filepath] = tensor_data
            return tensor_data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return torch.zeros(10, 6)
    
    def __len__(self):
        return len(self.test_pairs)
    
    def __getitem__(self, idx):
        file1, file2, label = self.test_pairs[idx]
        data1 = self.load_csv_data(file1)
        data2 = self.load_csv_data(file2)
        return data1, data2, label

def collate_fn(batch):
    """Custom collate function for variable-length sequences"""
    if len(batch[0]) == 3 and torch.is_tensor(batch[0][2]):  # Test data
        data1_list, data2_list, labels = zip(*batch)
        return list(data1_list), list(data2_list), torch.LongTensor(labels)
    else:  # Training triplets
        anchors, positives, negatives = zip(*batch)
        return list(anchors), list(positives), list(negatives)

class BehavioralAuthenticator:
    """Main class for behavioral authentication"""
    def __init__(self, data_dir: str, input_size: int = 6, embedding_dim: int = 128, 
                 kernel_size: int = 3,
                 dropout: float = 0.2, lr: float = 0.001, device: str = 'auto'):
        
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        
        # Initialize model
        self.model = TCNEncoder(input_size, hidden_dim=64, num_layers=4, kernel_size=kernel_size, 
                               embedding_dim=embedding_dim, dropout=dropout)
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.TripletMarginLoss(margin=1.0, p=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.8)
        
        # Training history
        self.train_losses = []
    
    def prepare_datasets(self, train_ratio: float = 0.8, samples_per_epoch: int = 1000):
        """Prepare training and test datasets"""
        users, sessions_per_user, fingers_per_user_session, csvs_per_user_session_finger = discover_data_structure_nested(self.data_dir)
        
        if not users:
            raise ValueError(f"No users found in data directory '{self.data_dir}'. Please ensure your data is present and named correctly.")
        if not sessions_per_user:
            raise ValueError(f"No sessions found in data directory '{self.data_dir}'. Please ensure your data is present and named correctly.")
        
        # Split users into train and test
        random.shuffle(users)
        split_idx = int(len(users) * train_ratio)
        train_users = users[:split_idx]
        test_users = users[split_idx:]
        
        if not train_users:
            raise ValueError("No training users found after split. Please check your data and train_ratio.")
        if not test_users:
            print("[WARNING] No test users found after split. All users are in training set.")
        
        # Create training dataset
        self.train_dataset = TripletDataset(self.data_dir, train_users, sessions_per_user, fingers_per_user_session, csvs_per_user_session_finger, samples_per_epoch)
        
        # Create test pairs
        test_pairs = []
        for user in test_users:
            for session in sessions_per_user[user]:
                for finger in fingers_per_user_session[(user, session)]:
                    csvs = csvs_per_user_session_finger[(user, session, finger)]
                    # Positive pairs
                    for i in range(len(csvs)):
                        for j in range(i+1, len(csvs)):
                            test_pairs.append((csvs[i], csvs[j], 1))
        # Negative pairs
        for i, user1 in enumerate(test_users):
            for user2 in test_users[i+1:]:
                # Pick one csv from each user, but only if all levels exist and are non-empty
                sessions1 = sessions_per_user[user1]
                sessions2 = sessions_per_user[user2]
                if not sessions1 or not sessions2:
                    continue
                session1 = sessions1[0]
                session2 = sessions2[0]
                fingers1 = fingers_per_user_session.get((user1, session1), [])
                fingers2 = fingers_per_user_session.get((user2, session2), [])
                if not fingers1 or not fingers2:
                    continue
                finger1 = fingers1[0]
                finger2 = fingers2[0]
                csvs1 = csvs_per_user_session_finger.get((user1, session1, finger1), [])
                csvs2 = csvs_per_user_session_finger.get((user2, session2, finger2), [])
                if not csvs1 or not csvs2:
                    continue
                csv1 = csvs1[0]
                csv2 = csvs2[0]
                test_pairs.append((csv1, csv2, 0))
        self.test_dataset = TestDataset(self.data_dir, test_pairs[:min(len(test_pairs), 500)])
        
        print(f"Training users: {len(train_users)}, Test users: {len(test_users)}")
        print(f"Test pairs: {len(test_pairs)}")
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            anchors, positives, negatives = batch
            
            # Process variable-length sequences
            anchor_embeddings = []
            positive_embeddings = []
            negative_embeddings = []
            
            for anchor, positive, negative in zip(anchors, positives, negatives):
                anchor = anchor.unsqueeze(0).to(self.device)
                positive = positive.unsqueeze(0).to(self.device)
                negative = negative.unsqueeze(0).to(self.device)
                
                anchor_emb = self.model(anchor)
                positive_emb = self.model(positive)
                negative_emb = self.model(negative)
                
                anchor_embeddings.append(anchor_emb)
                positive_embeddings.append(positive_emb)
                negative_embeddings.append(negative_emb)
            
            # Stack embeddings
            anchor_embeddings = torch.cat(anchor_embeddings, dim=0)
            positive_embeddings = torch.cat(positive_embeddings, dim=0)
            negative_embeddings = torch.cat(negative_embeddings, dim=0)
            
            # Compute loss
            loss = self.criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, dataloader, threshold: float = 0.5):
        """Evaluate the model"""
        self.model.eval()
        distances = []
        labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                data1_list, data2_list, batch_labels = batch
                
                for data1, data2, label in zip(data1_list, data2_list, batch_labels):
                    data1 = data1.unsqueeze(0).to(self.device)
                    data2 = data2.unsqueeze(0).to(self.device)
                    
                    emb1 = self.model(data1)
                    emb2 = self.model(data2)
                    
                    # Compute cosine distance
                    distance = 1 - F.cosine_similarity(emb1, emb2, dim=1)
                    distances.append(distance.item())
                    labels.append(label.item())
        
        # Convert to numpy arrays
        distances = np.array(distances)
        labels = np.array(labels)
        
        # Compute metrics
        predictions = (distances < threshold).astype(int)
        accuracy = np.mean(predictions == labels)
        
        # ROC-AUC (invert distances for ROC since we want higher similarity for positive pairs)
        similarities = 1 - distances
        auc = roc_auc_score(labels, similarities)
        
        return accuracy, auc, distances, labels
    
    def train(self, epochs: int = 20, batch_size: int = 32, samples_per_epoch: int = 1000):
        """Train the model"""
        print("Preparing datasets...")
        self.prepare_datasets(samples_per_epoch=samples_per_epoch)
        
        # Create data loaders
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, 
                                 shuffle=True, collate_fn=collate_fn, num_workers=0)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, 
                                shuffle=False, collate_fn=collate_fn, num_workers=0)
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_auc = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Evaluate
            accuracy, auc, _, _ = self.evaluate(test_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Test Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            
            # Save best model
            if auc > best_auc:
                best_auc = auc
                self.save_model('best_model.pth')
        
        print(f"\nBest AUC: {best_auc:.4f}")
        return self.train_losses
    
    def save_model(self, filepath: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
    
    def export_to_onnx(self, filepath: str, input_size: int = 6, seq_len: int = 100):
        """Export model to ONNX format"""
        self.model.eval()
        dummy_input = torch.randn(1, seq_len, input_size).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            filepath,
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['embedding'],
            dynamic_axes={'input': {1: 'seq_len'}}
        )
        print(f"Model exported to {filepath}")
    
    def export_to_torchscript(self, filepath: str):
        """Export model to TorchScript format"""
        self.model.eval()
        scripted_model = torch.jit.script(self.model)
        scripted_model.save(filepath)
        print(f"Model exported to {filepath}")

def create_sample_data(data_dir: str, num_users: int = 5, sessions_per_user: int = 3, 
                      seq_lengths: List[int] = [50, 100, 150], num_channels: int = 6):
    """Create sample CSV data for testing"""
    os.makedirs(data_dir, exist_ok=True)
    
    for user_id in range(num_users):
        for session_id in range(sessions_per_user):
            # Random sequence length
            seq_len = random.choice(seq_lengths)
            
            # Generate synthetic sensor data
            data = np.random.randn(seq_len, num_channels)
            
            # Add some user-specific patterns
            user_pattern = np.sin(np.linspace(0, 2*np.pi*user_id, seq_len))
            data[:, 0] += user_pattern.reshape(-1, 1)
            
            # Save as CSV
            filename = f"user{user_id}_session{session_id}.csv"
            filepath = os.path.join(data_dir, filename)
            
            # Create DataFrame with column names
            columns = [f'sensor_{i}' for i in range(num_channels)]
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(filepath, index=False)
    
    print(f"Sample data created in {data_dir}")

def main():
    parser = argparse.ArgumentParser(description='TCN Behavioral Authentication')
    parser.add_argument('--mode', choices=['train', 'eval', 'export'], required=True,
                       help='Mode: train, eval, or export')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing CSV files')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                       help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--input_size', type=int, default=6,
                       help='Number of input channels')
    parser.add_argument('--samples_per_epoch', type=int, default=1000,
                       help='Number of triplet samples per epoch')
    parser.add_argument('--create_sample_data', action='store_true',
                       help='Create sample data for testing')
    parser.add_argument('--export_format', choices=['onnx', 'torchscript'], default='onnx',
                       help='Export format')
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample_data:
        create_sample_data(args.data_dir)
        return
    
    # Initialize authenticator
    authenticator = BehavioralAuthenticator(
        data_dir=args.data_dir,
        input_size=args.input_size,
        embedding_dim=args.embedding_dim,
        lr=args.lr
    )
    
    if args.mode == 'train':
        print("Starting training...")
        losses = authenticator.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            samples_per_epoch=args.samples_per_epoch
        )
        
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('training_loss.png')
        plt.show()
        
    elif args.mode == 'eval':
        print("Loading model and evaluating...")
        authenticator.load_model(args.model_path)
        authenticator.prepare_datasets()
        
        test_loader = DataLoader(authenticator.test_dataset, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate_fn, num_workers=0)
        
        accuracy, auc, distances, labels = authenticator.evaluate(test_loader)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC: {auc:.4f}")
        
        # Plot ROC curve
        similarities = 1 - distances
        fpr, tpr, _ = roc_curve(labels, similarities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig('roc_curve.png')
        plt.show()
        
    elif args.mode == 'export':
        print("Loading model and exporting...")
        authenticator.load_model(args.model_path)
        
        if args.export_format == 'onnx':
            authenticator.export_to_onnx('model.onnx', args.input_size)
        elif args.export_format == 'torchscript':
            authenticator.export_to_torchscript('model.pt')

if __name__ == "__main__":
    main()