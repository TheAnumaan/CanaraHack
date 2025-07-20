import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# =================================================================================================
# --- 1. Global Configuration ---
# =================================================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 1000      # Sequence length for sensor data
DATA_ROOT = "/home/krish/Downloads/HuMI"  # IMPORTANT: Update this to your dataset path

# =================================================================================================
# --- 2. Model Architecture ---
# =================================================================================================

class TCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, num_layers, dropout_rate, sequence_length):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else output_dim
            layers.append(nn.Conv1d(
                in_channels=in_ch, out_channels=output_dim, kernel_size=kernel_size,
                padding=(kernel_size - 1) * (2 ** i), dilation=2 ** i
            ))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.Dropout(dropout_rate))

        self.network = nn.Sequential(*layers)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim, sequence_length)
            dummy_output = self.network(dummy_input)
            flattened_size = dummy_output.flatten(1).shape[1]
            
        self.project = nn.Linear(flattened_size, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1) # (B, D, T)
        conv_out = self.network(x)
        flattened = conv_out.flatten(start_dim=1)
        return self.project(flattened)

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, tcn_layers, dropout_rate, sequence_length):
        super().__init__()
        self.encoder = TCNBlock(
            input_dim=input_dim, output_dim=hidden_dim, kernel_size=3,
            num_layers=tcn_layers, dropout_rate=dropout_rate, sequence_length=sequence_length
        )

    def forward(self, x):
        return self.encoder(x)

class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim, attention_dim=128):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        self.fusion_layer = nn.Linear(hidden_dim, 128)

    def forward(self, embeddings_list):
        stacked_embeddings = torch.stack(embeddings_list, dim=1) # (B, Num_Modalities, D)
        attn_weights = self.attention_net(stacked_embeddings)    # (B, Num_Modalities, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted_embeddings = stacked_embeddings * attn_weights
        context_vector = torch.sum(weighted_embeddings, dim=1) # (B, D)
        return self.fusion_layer(context_vector)

class MultimodalEmbeddingModel(nn.Module):
    def __init__(self, sensors, sensor_dims, params):
        super().__init__()
        max_feature_dim = max(sensor_dims.values())
        
        self.encoders = nn.ModuleDict({
            sensor: ModalityEncoder(
                input_dim=max_feature_dim,
                hidden_dim=params['hidden_dim'],
                tcn_layers=params['tcn_layers'],
                dropout_rate=params['dropout_rate'],
                sequence_length=params['sequence_length']
            ) for sensor in sensors
        })

        fusion_input_dim = params['hidden_dim'] * len(sensors)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(params['dropout_rate'])
        )

        self.projection = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, params['proj_dim'])
        )

    def forward(self, inputs):
        embeddings = [self.encoders[sensor](inputs[:, i, :, :]) for i, sensor in enumerate(self.encoders.keys())]
        fused_input = torch.cat(embeddings, dim=1)
        fused = self.fusion(fused_input)
        return self.projection(fused)

# =================================================================================================
# --- 3. Data Handling ---
# =================================================================================================

class MultimodalSessionDataset(Dataset):
    def __init__(self, root_dir, sensor_list, max_len=1000):
        self.samples = []
        self.max_len = max_len
        self.sensor_list = sensor_list
        self.user_to_idx = {}
        self.idx_to_user = {}

        print(f"[Dataset] Scanning {root_dir} and building user map...")
        all_user_ids = sorted([int(d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()])
        
        for idx, user_id in enumerate(all_user_ids):
            self.user_to_idx[user_id] = idx
            self.idx_to_user[idx] = user_id

        for user_folder in os.listdir(root_dir):
            user_path = os.path.join(root_dir, user_folder)
            if not os.path.isdir(user_path): continue
            try: user_id = int(user_folder)
            except ValueError: continue

            for session_folder in os.listdir(user_path):
                session_path = os.path.join(user_path, session_folder)
                if not os.path.isdir(session_path): continue
                
                data_paths = { s: os.path.join(session_path, 'SENSORS', f'{s}.csv') for s in self.sensor_list if s.startswith('sensor_') }
                data_paths.update({ s: os.path.join(session_path, 'TOUCH', f'{s}.csv') for s in self.sensor_list if s.startswith('touch') or s in ['swipe', 'f_X_touch']})
                data_paths.update({ s: os.path.join(session_path, 'KEYSTROKE', f'{s}.csv') for s in self.sensor_list if s.startswith('key')})
                self.samples.append({'user_id': user_id, 'session_id': session_folder, 'paths': data_paths})
        print(f"[Dataset] Loaded {len(self.samples)} sessions across {len(self.user_to_idx)} unique users.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        original_user_id = sample['user_id']
        mapped_label = self.user_to_idx[original_user_id]
        tensors = []
        
        for sensor in self.sensor_list:
            path = sample['paths'].get(sensor)
            data_loaded = False
            if path and os.path.exists(path) and os.path.getsize(path) > 5:
                try:
                    with open(path, 'r', errors='ignore') as f: first_line = f.readline()
                    separator = ';' if ';' in first_line else ','
                    header_row = 0 if any(c.isalpha() for c in first_line) else None
                    df = pd.read_csv(path, sep=separator, header=header_row, engine='python', on_bad_lines='skip')
                    if not df.empty:
                        numeric_df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
                        if not numeric_df.empty:
                            numeric_df.fillna(0, inplace=True)
                            if numeric_df.std().sum() > 0:
                                numeric_df = (numeric_df - numeric_df.mean()) / (numeric_df.std().replace(0, 1))
                            data = torch.tensor(numeric_df.values, dtype=torch.float)
                            data_loaded = True
                except Exception as e:
                    pass
            if not data_loaded:
                data = torch.zeros(self.max_len, 1)

            T, D = data.shape
            if T > self.max_len: data = data[:self.max_len]
            elif T < self.max_len: data = torch.cat([data, torch.zeros(self.max_len - T, D)], dim=0)
            tensors.append(data)

        max_dim = max(t.shape[1] for t in tensors) if tensors else 1
        padded = [torch.cat([t, torch.zeros(t.shape[0], max_dim - t.shape[1])], dim=1) if t.shape[1] < max_dim else t for t in tensors]
        tensor_stack = torch.stack(padded)
        return tensor_stack, mapped_label

def split_dataset_by_user(dataset, test_ratio=0.2, min_sessions_for_test=2, random_state=42):
    user_sessions = {}
    for i, sample in enumerate(dataset.samples):
        user_id = sample['user_id']
        user_sessions.setdefault(user_id, []).append(i)
    eligible_test_users = [uid for uid, idxs in user_sessions.items() if len(idxs) >= min_sessions_for_test]
    other_users = [uid for uid, idxs in user_sessions.items() if len(idxs) < min_sessions_for_test]
    if len(eligible_test_users) < 2:
        raise ValueError(f"Not enough users with >= {min_sessions_for_test} sessions for a test set.")
    test_user_count = max(2, int(len(eligible_test_users) * test_ratio))
    train_eligible_users, test_users = train_test_split(eligible_test_users, test_size=test_user_count, random_state=random_state)
    train_users = set(train_eligible_users) | set(other_users)
    train_idx = [i for uid in train_users for i in user_sessions[uid]]
    test_idx = [i for uid in test_users for i in user_sessions[uid]]
    print(f"[Split] Train users: {len(train_users)}, Test users: {len(test_users)}")
    return Subset(dataset, train_idx), Subset(dataset, test_idx)

# =================================================================================================
# --- 4. Training and Evaluation Functions ---
# =================================================================================================

class DataAugmentation:
    def __init__(self, sigma=0.02, shift_fraction=0.05):
        self.sigma = sigma
        self.shift_fraction = shift_fraction

    def __call__(self, sample_batch):
        noise = torch.randn_like(sample_batch) * self.sigma
        augmented_batch = sample_batch + noise
        max_shift = int(sample_batch.shape[2] * self.shift_fraction)
        shifts = torch.randint(-max_shift, max_shift, (sample_batch.shape[0],))
        for i in range(sample_batch.shape[0]):
            augmented_batch[i] = torch.roll(augmented_batch[i], shifts=shifts[i].item(), dims=1)
        return augmented_batch

def train_model(model, dataloader, loss_fn, optimizer, scheduler, num_epochs, device):
    augment = DataAugmentation(sigma=0.01, shift_fraction=0.1)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for i, (inputs, labels) in enumerate(progress_bar):
            inputs = augment(inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            embeddings = model(inputs)
            
            dist_matrix = torch.cdist(embeddings, embeddings, p=2)
            is_pos = labels.unsqueeze(1) == labels.unsqueeze(0)
            is_neg = ~is_pos
            
            dist_ap = dist_matrix.clone()
            dist_ap[~is_pos] = -float('inf')
            hardest_positive_dist, _ = torch.max(dist_ap, dim=1)

            dist_an = dist_matrix.clone()
            dist_an[is_pos] = float('inf')
            hardest_negative_dist, _ = torch.min(dist_an, dim=1)
            
            loss = torch.mean(F.relu(hardest_positive_dist - hardest_negative_dist + loss_fn.margin))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"** End of Epoch {epoch+1} | Average Triplet Loss: {avg_loss:.4f} **")
        scheduler.step()

def evaluate_model(model, dataloader, device):
    model.eval()
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating", leave=False):
            all_embeddings.append(model(x.to(device)).cpu())
            all_labels.append(y.cpu())
    embeddings = F.normalize(torch.cat(all_embeddings), dim=1)
    labels = torch.cat(all_labels)
    user_to_embeddings = {}
    for user_id in torch.unique(labels):
        indices = (labels == user_id).nonzero(as_tuple=True)[0]
        if len(indices) > 1: user_to_embeddings[user_id.item()] = embeddings[indices]
    if len(user_to_embeddings) < 2:
        print("[Error] Evaluation failed: Need at least 2 users with multiple sessions.")
        return None, None
    genuine_scores = [F.cosine_similarity(embs[i].unsqueeze(0), embs[j].unsqueeze(0)).item() for embs in user_to_embeddings.values() for i in range(len(embs)) for j in range(i + 1, len(embs))]
    if not genuine_scores:
        print("[Error] No genuine pairs found for evaluation.")
        return None, None
    impostor_scores = []
    user_ids = list(user_to_embeddings.keys())
    num_impostors = min(len(genuine_scores) * 2, 20000)
    for _ in range(num_impostors):
        u1_id, u2_id = random.sample(user_ids, 2)
        emb1 = random.choice(user_to_embeddings[u1_id])
        emb2 = random.choice(user_to_embeddings[u2_id])
        impostor_scores.append(F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item())
    if not impostor_scores:
        print("[Error] No impostor pairs found for evaluation.")
        return None, None
        
    scores = np.array(genuine_scores + impostor_scores)
    y_true = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))
    if len(np.unique(y_true)) < 2:
        print("[Error] Cannot calculate AUC/EER with only one class.")
        return None, None
    fpr, tpr, _ = roc_curve(y_true, scores)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    auc = roc_auc_score(y_true, scores)
    plt.figure(figsize=(10, 6))
    plt.hist(genuine_scores, bins=50, alpha=0.7, density=True, label=f'Genuine')
    plt.hist(impostor_scores, bins=50, alpha=0.7, density=True, label=f'Impostor')
    plt.title(f"Final Model Performance (AUC: {auc:.4f}, EER: {eer:.4f})")
    plt.xlabel('Cosine Similarity'); plt.ylabel('Density'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig("final_evaluation_scores.png"); plt.close()
    return auc, eer

# =================================================================================================
# --- 5. Main Execution Block ---
# =================================================================================================
if __name__ == "__main__":
    print(f"[INFO] Using device: {DEVICE}")

    # --- Configuration ---
    model_params = {
        'hidden_dim': 128,
        'proj_dim': 128,
        'tcn_layers': 5,
        'dropout_rate': 0.4,
        'sequence_length': MAX_LEN
    }
    training_params = {
        'num_epochs': 40,
        'batch_size': 32,
        'learning_rate': 0.0002,
        'triplet_margin': 0.5
    }

    sensor_list = [
        'key_data', 'swipe', 'touch_touch', 'sensor_grav', 'sensor_gyro',
        'sensor_lacc', 'sensor_magn', 'sensor_nacc'
    ]
    sensor_dims = {
        'key_data': 1, 'swipe': 6, 'touch_touch': 6, 'sensor_grav': 3,
        'sensor_gyro': 3, 'sensor_lacc': 3, 'sensor_magn': 3, 'sensor_nacc': 3
    }

    # --- Data Loading & Splitting ---
    print("[Main] Loading and splitting data...")
    full_dataset = MultimodalSessionDataset(DATA_ROOT, sensor_list, max_len=MAX_LEN)
    train_subset, test_subset = split_dataset_by_user(full_dataset)
    
    train_loader = DataLoader(train_subset, batch_size=training_params['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=training_params['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # --- Model, Loss, and Optimizer Setup ---
    print("[Main] Initializing model...")
    model = MultimodalEmbeddingModel(sensor_list, sensor_dims, model_params).to(DEVICE)
    loss_fn = nn.TripletMarginLoss(margin=training_params['triplet_margin'])
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_params['num_epochs'])

    # --- Start Training ---
    print(f"[Main] Starting training for {training_params['num_epochs']} epochs...")
    train_model(model, train_loader, loss_fn, optimizer, scheduler, training_params['num_epochs'], DEVICE)

    # =========================================================================
    # --- ✅ SAVE THE MODEL STATE DICTIONARY ---
    # =========================================================================
    MODEL_SAVE_PATH = "multimodal_authentication_model.pkl"
    print(f"\n[Main] Saving model state dictionary to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("[Main] Model saved successfully.")
    # =========================================================================

    # --- Final Evaluation ---
    print("\n[Main] Evaluating final model on the test set...")
    auc, eer = evaluate_model(model, test_loader, device=DEVICE)
    
    if auc is not None and eer is not None:
        print("\n--- FINAL PERFORMANCE ---")
        print(f"AUC: {auc:.4f}")
        print(f"EER: {eer:.4f}")
        print("Evaluation plot saved to 'final_evaluation_scores.png'")
    else:
        print("Evaluation could not be completed.")