import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from multiprocessing import cpu_count
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import glob

# Device selection: prefer MPS (Mac GPU), then CUDA, then CPU
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"[INFO] Using device: {DEVICE}")

# Directory path where all user folders (e.g., 001, 002, ...) are located
DATA_ROOT = "/Users/riyamehdiratta/Desktop/hackathon/HUMI_final"

class ResidualTCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        # First layer: input_dim -> output_dim
        dilation = 1
        padding = (kernel_size - 1) * dilation // 2
        self.layers.append(
            nn.Conv1d(input_dim, output_dim, kernel_size, padding=padding, dilation=dilation)
        )
        # Remaining layers: output_dim -> output_dim
        for i in range(1, num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            self.layers.append(
                nn.Conv1d(output_dim, output_dim, kernel_size, padding=padding, dilation=dilation)
            )
        self.proj = nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, T)
        residual = self.proj(x)
        out = x
        for layer in self.layers:
            out = layer(out)
            out = torch.relu(out)
        out = out + residual
        return torch.mean(out, dim=2)  # (B, D_out)

class ModalityEncoder(nn.Module):
    def __init__(self, sensor_type, input_dim, hidden_dim=32):
        super().__init__()
        self.sensor_type = sensor_type
        print(f"[ModalityEncoder] Initializing encoder for: {sensor_type}, input_dim={input_dim}")

        if sensor_type in ['gps', 'sensor_grav', 'sensor_gyro', 'sensor_lacc', 'sensor_magn',
                           'sensor_nacc', 'sensor_prox', 'sensor_temp', 'sensor_ligh', 'sensor_humd',
                           'swipe', 'scroll', 'touch', 'f_x_touch', 'key_data', 'wifi']:
            self.encoder = ResidualTCNBlock(input_dim, hidden_dim)
        elif sensor_type == 'bluetooth':
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")

    def forward(self, x):
        if isinstance(self.encoder, nn.Sequential):
            # Apply mean pooling over time dimension if using Linear encoder
            x = torch.mean(x, dim=1)  # from (B, T, D) → (B, D)
        return self.encoder(x)

class MultimodalFusion(nn.Module):
    def __init__(self, modality_dims, fusion_dim=128):
        super().__init__()
        print(f"[MultimodalFusion] Initializing with modality dims: {modality_dims}")
        self.fusion = nn.Sequential(
            nn.Linear(sum(modality_dims), fusion_dim),
            nn.ReLU()
        )

    def forward(self, embeddings):
        for i, e in enumerate(embeddings):
            print(f"[Fusion] Embedding {i} shape: {e.shape}")
        x = torch.cat(embeddings, dim=1)
        return self.fusion(x)

class SigLipLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
        label_matrix = label_matrix.float()
        mask = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        sim_matrix = sim_matrix[mask].view(len(labels), -1)
        label_matrix = label_matrix[mask].view(len(labels), -1)
        sim_scores = torch.sigmoid(sim_matrix)
        # Fix for MPS: skip empty tensors
        if sim_scores.numel() == 0 or label_matrix.numel() == 0:
            print("[WARN] Empty sim_scores or label_matrix in loss, skipping batch.")
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        loss = F.binary_cross_entropy(sim_scores, label_matrix)
        return loss

class MultimodalEmbeddingModel(nn.Module):
    def __init__(self, sensors, input_dim: dict, hidden_dim=32):
        super().__init__()
        self.encoders = nn.ModuleDict({
            sensor: ModalityEncoder(sensor, input_dim[sensor], hidden_dim)
            for sensor in sensors
        })
        self.fusion = MultimodalFusion([hidden_dim] * len(sensors))
        self.sensor_dims = input_dim  # Save for slicing in forward

    def forward(self, inputs):
        embeddings = []
        for i, (sensor, encoder) in enumerate(self.encoders.items()):
            print(f"[Forward] Encoding sensor: {sensor}")
            num_dim = self.sensor_dims[sensor]
            x = inputs[:, i, :, :num_dim]  # shape: (B, T, num_dim)
            embeddings.append(encoder(x))
        return self.fusion(embeddings)


import glob

class MultimodalSessionDataset(Dataset):
    def __init__(self, root_dir, sensor_list, sensor_dims, max_len=1000):
        self.samples = []
        self.max_len = max_len
        self.sensor_list = sensor_list
        self.sensor_dims = sensor_dims

        print(f"[Dataset Init] Scanning dataset directory: {root_dir}")

        for user_folder in os.listdir(root_dir):
            user_path = os.path.join(root_dir, user_folder)
            if not os.path.isdir(user_path):
                continue
            try:
                user_id = int(user_folder)
            except ValueError:
                continue
            for session_folder in os.listdir(user_path):
                session_path = os.path.join(user_path, session_folder)
                if not os.path.isdir(session_path):
                    continue
                for finger_folder in os.listdir(session_path):
                    finger_path = os.path.join(session_path, finger_folder)
                    if not os.path.isdir(finger_path):
                        continue
                    sensors_path = os.path.join(finger_path, 'sensors')
                    if not os.path.isdir(sensors_path):
                        continue
                    data_paths = {}
                    for sensor in sensor_list:
                        if sensor == 'f_x_touch':
                            # Extract finger number from finger_folder (e.g., 'finger_0' -> 0)
                            try:
                                finger_number = int(finger_folder.split('_')[-1])
                            except Exception:
                                finger_number = 0
                            data_paths[sensor] = os.path.join(session_path, f"f_{finger_number}_touch.csv")
                        elif sensor == 'swipe':
                            data_paths[sensor] = os.path.join(session_path, "swipe.csv")
                        else:
                            data_paths[sensor] = os.path.join(sensors_path, f"{sensor}.csv")
                    self.samples.append({
                        'user_id': user_id,
                        'session_id': session_folder,
                        'finger_id': finger_folder,
                        'paths': data_paths
                    })
        print(f"[Dataset Init Complete] Loaded {len(self.samples)} samples across users/sessions/fingers")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._load_sample(self.samples[idx])

    def _load_sample(self, sample):
        print(f"[Sample Load] Loading user {sample['user_id']} session {sample['session_id']} finger {sample['finger_id']}")
        tensors = []
        for sensor in self.sensor_list:
            path = sample['paths'].get(sensor)
            print(f"  [Sensor] {sensor} -> {path}")
            num_dim = self.sensor_dims[sensor]
            if path and os.path.exists(path):
                try:
                    df = pd.read_csv(path, header=None)
                    # Ensure all data is numeric
                    df_numeric = df.apply(pd.to_numeric, errors='coerce').fillna(0)
                    data = torch.tensor(df_numeric.values, dtype=torch.float)
                    T, D = data.shape
                    if D > num_dim:
                        data = data[:, :num_dim]
                    elif D < num_dim:
                        pad = torch.zeros(T, num_dim - D)
                        data = torch.cat([data, pad], dim=1)
                    if T > self.max_len:
                        data = data[:self.max_len]
                    elif T < self.max_len:
                        pad = torch.zeros(self.max_len - T, num_dim)
                        data = torch.cat([data, pad], dim=0)
                except pd.errors.EmptyDataError:
                    print(f"    [Warning] Empty file for sensor {sensor}, using zero tensor")
                    data = torch.zeros(self.max_len, num_dim)
                tensors.append(data)
            else:
                print(f"    [Warning] Missing file for sensor {sensor}, using zero tensor")
                tensors.append(torch.zeros(self.max_len, num_dim))
        # Pad all tensors to the same number of columns (max_dim)
        max_dim = max(t.shape[1] for t in tensors)
        padded = []
        for t in tensors:
            if t.shape[1] < max_dim:
                pad = torch.zeros(t.shape[0], max_dim - t.shape[1])
                t = torch.cat([t, pad], dim=1)
            padded.append(t)
        tensor_stack = torch.stack(padded)  # [num_sensors, max_len, max_dim]
        return tensor_stack, sample['user_id']


# Training + Evaluation

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for x, labels in dataloader:
        x, labels = x.to(device), labels.to(device)
        embeddings = model(x)
        loss = loss_fn(embeddings, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for x, labels in dataloader:
            x, labels = x.to(device), labels.to(device)
            embeddings = model(x)
            all_embeddings.append(embeddings)
            all_labels.append(labels)

    embeddings = torch.cat(all_embeddings)
    labels = torch.cat(all_labels)
    sims = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

    mask = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    sims = sims[mask]
    targets = label_matrix[mask]

    fpr, tpr, thresholds = roc_curve(targets.cpu().numpy(), sims.cpu().numpy())
    eer = fpr[np.nanargmin(np.absolute((1 - tpr - fpr)))]
    auc = roc_auc_score(targets.cpu(), sims.cpu())

    plt.hist(sims[targets == 1].cpu().numpy(), bins=50, alpha=0.6, label='Same User')
    plt.hist(sims[targets == 0].cpu().numpy(), bins=50, alpha=0.6, label='Different User')
    plt.legend()
    plt.title("Cosine Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.show()

    return eer, auc

# Split users into train/test

def split_dataset_by_user(dataset, test_size=0.2, random_state=42):
    user_ids = list(set(sample['user_id'] for sample in dataset.samples))
    train_users, test_users = train_test_split(user_ids, test_size=test_size, random_state=random_state)
    train_indices = [i for i, s in enumerate(dataset.samples) if s['user_id'] in train_users]
    test_indices = [i for i, s in enumerate(dataset.samples) if s['user_id'] in test_users]
    return Subset(dataset, train_indices), Subset(dataset, test_indices)

# Configure DataLoader for multiprocessing
DataLoaderMP = lambda dataset, batch_size, shuffle=True: DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=min(cpu_count(), 8),
    pin_memory=True
)

# (Previous code remains unchanged)

from torch.utils.data import random_split
from tqdm import tqdm

def split_dataset_by_user(dataset, test_ratio=0.2, min_test_users=2):
    user_ids = list({s['user_id'] for s in dataset.samples})
    user_ids.sort()

    if len(user_ids) < min_test_users + 1:
        raise ValueError(f"Need at least {min_test_users + 1} users to split. Found only {len(user_ids)}.")

    # Use train_test_split to randomly select test users
    test_size = max(min_test_users, int(len(user_ids) * test_ratio))
    train_users, test_users = train_test_split(user_ids, test_size=test_size, random_state=42)

    train_idx = [i for i, s in enumerate(dataset.samples) if s['user_id'] in train_users]
    test_idx = [i for i, s in enumerate(dataset.samples) if s['user_id'] in test_users]

    return Subset(dataset, train_idx), Subset(dataset, test_idx)


def evaluate_model(model, dataloader):
    print("[Eval] Computing embeddings and similarities...")
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.cuda(), y.cuda()
            emb = model(x)
            embeddings.append(emb)
            labels.append(y)

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    if len(labels.unique()) < 2:
        print("[ERROR] Only one user present in evaluation set. Skipping evaluation.")
        return None, None

    # Compute pairwise cosine similarities
    sims = torch.matmul(F.normalize(embeddings, dim=1), F.normalize(embeddings, dim=1).T)
    same_user = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)

    y_true = same_user[mask].cpu().numpy().astype(int)
    y_score = sims[mask].cpu().numpy()

    print(f"[Eval] y_true distribution: {np.bincount(y_true)}")

    if np.all(y_true == 0) or np.all(y_true == 1):
        print("[WARN] Only one class in similarity labels. Skipping AUC/EER.")
        return None, None

    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_score)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    auc = roc_auc_score(y_true, y_score)

    print(f"[Eval] AUC: {auc:.4f}, EER: {eer:.4f}")
    return auc, eer

def balanced_similarity_scores(embeddings, labels):
    from sklearn.metrics import roc_auc_score, roc_curve
    labels = labels.cpu()
    embeddings = embeddings.cpu()

    same_user_mask = (labels.unsqueeze(1) == labels.unsqueeze(0))
    sims = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

    # Only consider off-diagonal pairs
    mask = ~torch.eye(len(labels), dtype=torch.bool)
    pos_pairs = sims[same_user_mask & mask]
    neg_pairs = sims[~same_user_mask & mask]

    n = min(len(pos_pairs), len(neg_pairs))
    if n == 0:
        print("[WARN] Not enough positive/negative pairs for ROC/AUC calculation. Skipping.")
        return None, None

    pos_sample = pos_pairs[:n]
    neg_sample = neg_pairs[:n]

    y_true = np.array([1] * n + [0] * n)
    y_score = np.concatenate([pos_sample.numpy(), neg_sample.numpy()])

    # Check if y_true contains both classes
    if len(np.unique(y_true)) < 2:
        print("[WARN] Only one class present in y_true. Skipping ROC/AUC calculation.")
        return None, None

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]

    return auc, eer

# === Hyperparameters ===
hidden_dim = 32
kernel_size = 3
num_layers = 3
fusion_dim = 128
learning_rate = 1e-3
epochs = 5
batch_size = 2
# =====================

if __name__ == "__main__":
    DATA_ROOT = "/Users/riyamehdiratta/Desktop/hackathon/HUMI_final"

    # Updated sensor list and dimensions
    sensor_list = [
        'sensor_grav', 'sensor_gyro', 'sensor_humd', 'sensor_lacc',
        'sensor_ligh', 'sensor_magn', 'sensor_nacc', 'sensor_prox',
        'sensor_temp', 'f_x_touch', 'swipe'
    ]

    sensor_dims = {
        'sensor_grav': 3,
        'sensor_gyro': 3,
        'sensor_humd': 3,
        'sensor_lacc': 3,
        'sensor_ligh': 1,
        'sensor_magn': 3,
        'sensor_nacc': 3,
        'sensor_prox': 1,
        'sensor_temp': 1,
        'f_x_touch': 4,   # Adjust as needed
        'swipe': 4        # Adjust as needed
    }

    print("[Main] Creating dataset...")
    dataset = MultimodalSessionDataset(DATA_ROOT, sensor_list, sensor_dims)
    print(f"[Main] Total samples in dataset: {len(dataset)}")

    print("[Main] Splitting train/test users...")
    # Select up to 5 unique users for the test set
    user_ids = list({s['user_id'] for s in dataset.samples})
    user_ids.sort()
    test_user_count = min(5, len(user_ids))
    test_users = user_ids[:test_user_count]
    train_users = user_ids[test_user_count:]
    train_indices = [i for i, s in enumerate(dataset.samples) if s['user_id'] in train_users]
    test_indices = [i for i, s in enumerate(dataset.samples) if s['user_id'] in test_users]
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("[Main] Building model...")
    model = MultimodalEmbeddingModel(sensor_list, input_dim=sensor_dims, hidden_dim=hidden_dim).to(DEVICE)
    loss_fn = SigLipLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("[Main] Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            emb = model(x)
            loss = loss_fn(emb, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        embeddings = []
        labels = []
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc=f"Epoch {epoch} [Val]"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                emb = model(x)
                embeddings.append(emb)
                labels.append(y)
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        auc, eer = balanced_similarity_scores(embeddings, labels)
        if auc is not None and eer is not None:
            print(f"[Epoch {epoch}] [Val] AUC: {auc:.4f}, EER: {eer:.4f}")
        else:
            print(f"[Epoch {epoch}] [Val] Not enough data to compute AUC/EER.")