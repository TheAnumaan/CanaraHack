import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import random
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
import json
from pathlib import Path
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
warnings.filterwarnings('ignore')

# Level 2: Enhanced Data Processing - Parallel CSV loading

def process_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_csv_parallel(file_paths, num_workers=4):
    """Parallel CSV processing using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_csv_file, file_paths))
    return results

# Level 3: Advanced Multiprocessing - Parallel triplet generation

def generate_triplet_batch(args):
    file_groups, user = args
    user_files = file_groups[user]
    triplets = []
    if len(user_files) < 2:
        return triplets
    for i in range(len(user_files)):
        for j in range(i+1, len(user_files)):
            anchor_file = user_files[i]
            positive_file = user_files[j]
            # Select negative from different user
            other_users = [u for u in file_groups if u != user]
            if other_users:
                import random
                negative_user = random.choice(other_users)
                negative_file = random.choice(file_groups[negative_user])
                triplets.append((anchor_file, positive_file, negative_file))
    return triplets

def generate_triplets_parallel(file_groups, num_workers=4):
    """Parallel triplet generation using ProcessPoolExecutor."""
    users = list(file_groups.keys())
    args_list = [(file_groups, user) for user in users]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(generate_triplet_batch, args_list))
    # Flatten the list of lists
    triplets = [triplet for sublist in results for triplet in sublist]
    return triplets

# Level 1: Basic Multiprocessing - DataLoader with num_workers
class TripletDataset(Dataset):
    def __init__(self, file_groups, triplets):
        self.file_groups = file_groups
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_file, positive_file, negative_file = self.triplets[idx]
        anchor = pd.read_csv(anchor_file).values
        positive = pd.read_csv(positive_file).values
        negative = pd.read_csv(negative_file).values
        # Convert to torch tensors (no normalization for brevity)
        anchor = torch.tensor(anchor, dtype=torch.float32)
        positive = torch.tensor(positive, dtype=torch.float32)
        negative = torch.tensor(negative, dtype=torch.float32)
        return anchor, positive, negative

def organize_files(data_dir):
    """Recursively organize CSV files by user/session/gesture, but only for users 000 to 049."""
    file_groups = {}
    for user in sorted(os.listdir(data_dir)):
        if not (user.isdigit() and len(user) == 3 and 0 <= int(user) <= 49):
            continue
        user_path = os.path.join(data_dir, user)
        if not os.path.isdir(user_path):
            continue
        for session in sorted(os.listdir(user_path)):
            session_path = os.path.join(user_path, session)
            if not os.path.isdir(session_path):
                continue
            for gesture in sorted(os.listdir(session_path)):
                gesture_path = os.path.join(session_path, gesture)
                if not os.path.isdir(gesture_path):
                    continue
                for root, dirs, files in os.walk(gesture_path):
                    for file in files:
                        if file.endswith(".csv"):
                            file_path = os.path.join(root, file)
                            if user not in file_groups:
                                file_groups[user] = []
                            file_groups[user].append(file_path)
    return file_groups

def main():
    data_dir = "/Users/riyamehdiratta/Desktop/hackathon/HUMI_final"  # Update as needed
    num_workers = 4  # Adjust based on your CPU

    print("Organizing files...")
    file_groups = organize_files(data_dir)

    print("Generating triplets in parallel...")
    triplets = generate_triplets_parallel(file_groups, num_workers=num_workers)
    print(f"Generated {len(triplets)} triplets.")

    print("Creating dataset and dataloader with multiprocessing...")
    dataset = TripletDataset(file_groups, triplets)
    train_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=num_workers,  # Enable multiprocessing
        pin_memory=True,         # Faster GPU transfer
        persistent_workers=True  # Keep workers alive between epochs
    )

    # Example: Iterate through one epoch with tqdm
    for batch in tqdm(train_loader, desc="Training epoch (multiprocess)"):
        anchor, positive, negative = batch
        # Here you would pass to your TCN model, e.g. model(anchor)
        pass

    print("Multiprocessing TCN pipeline complete.")

if __name__ == "__main__":
    main()