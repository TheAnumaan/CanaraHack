import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from multiprocessing import cpu_count
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

class TCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(nn.Conv1d(
                input_dim if i == 0 else output_dim,
                output_dim,
                kernel_size,
                padding=(kernel_size - 1) * dilation,
                dilation=dilation
            ))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, D, T] for Conv1d
        out = self.network(x)
        out = torch.mean(out, dim=2)  # Global average pooling
        return out

class ModalityEncoder(nn.Module):
    def __init__(self, sensor_type, input_dim, hidden_dim=32):
        super().__init__()
        self.sensor_type = sensor_type

        if sensor_type in ['gps', 'sensor_grav', 'sensor_gyro', 'sensor_lacc', 'sensor_magn', 'sensor_nacc', 'sensor_prox', 'sensor_temp', 'sensor_ligh', 'sensor_humd']:
            self.encoder = TCNBlock(input_dim, hidden_dim)
        elif sensor_type in ['swipe', 'scroll_X_touch', 'touch_touch', 'f_X_touch']:
            self.encoder = TCNBlock(input_dim, hidden_dim)
        elif sensor_type == 'wifi':
            self.encoder = TCNBlock(input_dim, hidden_dim)
        elif sensor_type == 'bluetooth':
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            )
        elif sensor_type == 'key_data':
            self.encoder = TCNBlock(input_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")

    def forward(self, x):
        return self.encoder(x)

class MultimodalFusion(nn.Module):
    def __init__(self, modality_dims, fusion_dim=128):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(sum(modality_dims), fusion_dim),
            nn.ReLU()
        )

    def forward(self, embeddings):
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
        loss = F.binary_cross_entropy(sim_scores, label_matrix)
        return loss

class MultimodalSessionDataset(Dataset):
    def __init__(self, root_dir, sensor_list, max_len=1000):
        self.samples = []
        self.max_len = max_len
        self.sensor_list = sensor_list

        for user_folder in os.listdir(root_dir):
            if not user_folder.startswith("user_"):
                continue
            user_id = int(user_folder.split('_')[-1])
            user_path = os.path.join(root_dir, user_folder)
            sessions = os.listdir(user_path)
            for session_file in sessions:
                if session_file.endswith('.csv'):
                    session_name = '_'.join(session_file.split('_')[:-1])
                    for sensor in sensor_list:
                        if session_file.startswith(sensor):
                            self.samples.append({
                                'user_id': user_id,
                                'session_id': session_file.split('.')[0],
                                'paths': {
                                    sensor: os.path.join(user_path, session_file)
                                }
                            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._load_sample(self.samples[idx])

    def _load_sample(self, sample):
        tensors = []
        for sensor in self.sensor_list:
            path = sample['paths'].get(sensor)
            if path and os.path.exists(path):
                df = pd.read_csv(path, header=None)
                if sensor == 'gps':
                    data = torch.tensor(df.iloc[:, [2, 3, 4, 5, 6]].values, dtype=torch.float)
                elif sensor == 'bluetooth':
                    data = df.iloc[:, [1, 2]].astype(str)
                    data = torch.tensor([
                        [len(name), float(int("0x" + mac.replace(":", ""), 16) % 1e9)]
                        for name, mac in data.values
                    ], dtype=torch.float)
                elif sensor == 'wifi':
                    data = torch.tensor(df.iloc[:, [2, 4, 5]].values, dtype=torch.float)
                elif sensor.startswith('sensor_'):
                    data = torch.tensor(df.iloc[:, 2:].values, dtype=torch.float)
                elif sensor in ['swipe', 'scroll_X_touch', 'touch_touch', 'f_X_touch']:
                    data = torch.tensor(df.iloc[:, 2:6].values, dtype=torch.float)
                elif sensor == 'key_data':
                    ascii_vals = pd.to_numeric(df.iloc[:, 2], errors='coerce').fillna(0)
                    data = torch.tensor(ascii_vals.values.reshape(-1, 1), dtype=torch.float)
                else:
                    data = torch.zeros(self.max_len, 1)

                T, D = data.shape
                if T > self.max_len:
                    data = data[:self.max_len]
                elif T < self.max_len:
                    pad = torch.zeros(self.max_len - T, D)
                    data = torch.cat([data, pad], dim=0)
                tensors.append(data)
            else:
                tensors.append(torch.zeros(self.max_len, 1))

        tensor_stack = torch.stack(tensors)
        return tensor_stack, sample['user_id']

class MultimodalEmbeddingModel(nn.Module):
    def __init__(self, sensor_dims, hidden_dim=32, fusion_dim=128):
        super().__init__()
        self.encoders = nn.ModuleDict({
            sensor: ModalityEncoder(sensor, dim, hidden_dim)
            for sensor, dim in sensor_dims.items()
        })
        self.fusion = MultimodalFusion([hidden_dim for _ in sensor_dims])

    def forward(self, inputs):
        embeddings = []
        for i, (sensor, encoder) in enumerate(self.encoders.items()):
            x = inputs[:, i, :, :]
            embeddings.append(encoder(x))
        return self.fusion(embeddings)

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

# Configure DataLoader for multiprocessing
DataLoaderMP = lambda dataset, batch_size, shuffle=True: DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=min(cpu_count(), 8),
    pin_memory=True
)

# this is a fusion model which chatgpt gave to me i'll see if i can use it

# class MultimodalUserEncoder(nn.Module):
#     def __init__(self, dims):
#         super().__init__()
#         self.gps_tcn       = TCNBlock(input_dim=dims['gps'], output_dim=32)
#         self.wifi_tcn      = TCNBlock(input_dim=dims['wifi'], output_dim=32)
#         self.keystroke_tcn = TCNBlock(input_dim=dims['keystroke'], output_dim=64)
#         # Add more modalities here

#     def forward(self, gps_seq, wifi_seq, keystroke_seq):
#         gps_embed       = self.gps_tcn(gps_seq)        # shape: (B, 32)
#         wifi_embed      = self.wifi_tcn(wifi_seq)       # shape: (B, 32)
#         keystroke_embed = self.keystroke_tcn(keystroke_seq)  # shape: (B, 64)

#         # Concatenate all modality embeddings
#         user_vector = torch.cat([gps_embed, wifi_embed, keystroke_embed], dim=1)  # shape: (B, 128)
#         return user_vector

# this is a profile viewer chatgpt gave to me
# def create_user_profile(model, user_sessions):
#     # user_sessions: list of dicts {modality_name: Tensor} of shape (T, D)
#     vectors = []
#     for session in user_sessions:
#         # Add batch dim
#         gps = session['gps'].unsqueeze(0)
#         wifi = session['wifi'].unsqueeze(0)
#         keys = session['keystroke'].unsqueeze(0)
#         with torch.no_grad():
#             vec = model(gps, wifi, keys)  # shape: (1, D)
#         vectors.append(vec.squeeze(0))    # shape: (D,)
#     return torch.stack(vectors).mean(dim=0)  # Averaged profile vector (D,)

# for comparing to a new session
# def verify_user(model, session, stored_vector, threshold=0.8):
#     gps = session['gps'].unsqueeze(0)
#     wifi = session['wifi'].unsqueeze(0)
#     keys = session['keystroke'].unsqueeze(0)
    
#     with torch.no_grad():
#         vec = model(gps, wifi, keys)  # shape: (1, D)
#         score = F.cosine_similarity(vec, stored_vector.unsqueeze(0))  # shape: (1,)
#     return score.item() > threshold, score.item()