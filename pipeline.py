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
import random
from sklearn.metrics import roc_curve, roc_auc_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# Directory path where all user folders (e.g., 000, 001, ...) are located
DATA_ROOT = "/home/krish/Downloads/HuMI/"

class TCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        layers = []
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else output_dim
            layers.append(nn.Conv1d(
                in_channels=in_ch,
                out_channels=output_dim,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) * (2 ** i),
                dilation=2 ** i
            ))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))  # Add after ReLU or before BatchNorm

            layers.append(nn.BatchNorm1d(output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, D, T)
        if x.shape[1] != self.input_dim:
            print(f"[TCNBlock Warning] Reshaping from {x.shape[1]} to {self.input_dim}")
            if x.shape[1] > self.input_dim:
                x = x[:, :self.input_dim, :]  # Truncate channels
            else:
                pad = torch.zeros(x.size(0), self.input_dim - x.shape[1], x.size(2)).to(x.device)
                x = torch.cat([x, pad], dim=1)
        out = self.network(x)
        return torch.mean(out, dim=2)


class ModalityEncoder(nn.Module):
    def __init__(self, sensor_type, input_dim, hidden_dim=32):
        super().__init__()
        self.sensor_type = sensor_type

        if sensor_type in ['gps', 'sensor_grav', 'sensor_gyro', 'sensor_lacc', 'sensor_magn',
                           'sensor_nacc', 'sensor_prox', 'sensor_temp', 'sensor_ligh', 'sensor_humd',
                           'swipe', 'f_X_touch', 'key_data','touch_touch']:
            self.encoder = TCNBlock(input_dim, output_dim=64, num_layers=4)


        elif sensor_type == 'bluetooth':
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            )
        elif sensor_type == 'wifi':
            self.encoder = TCNBlock(input_dim, output_dim=64, num_layers=4)


        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")

    def forward(self, x):
        if isinstance(self.encoder, nn.Sequential):
            x = torch.mean(x, dim=1)
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
        loss = F.binary_cross_entropy(sim_scores, label_matrix)
        return loss
class MultimodalEmbeddingModel(nn.Module):
    def __init__(self, sensors, sensor_dims, hidden_dim=64, proj_dim=64): # Note: hidden_dim is now the TCN output
        super().__init__()
        self.encoders = nn.ModuleDict({
            sensor: ModalityEncoder(sensor, input_dim=sensor_dims.get(sensor, 1), hidden_dim=hidden_dim)
            for sensor in sensors
        })

        # --- THIS IS THE CRITICAL FIX ---
        # The TCN encoders output `hidden_dim`. The total dimension for the fusion
        # layer is the sum of all these output dimensions.
        fusion_input_dim = hidden_dim * len(sensors)
        self.fusion = MultimodalFusion(modality_dims=[fusion_input_dim], fusion_dim=128) # Pass a single value

        # 🔥 Projection Head
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, proj_dim)
        )

    def forward(self, inputs):
        embeddings = []
        for i, (sensor, encoder) in enumerate(self.encoders.items()):
            x = inputs[:, i, :, :]  # (B, T, D)
            embeddings.append(encoder(x))
        
        # Concatenate embeddings before passing to the fusion layer
        fused_input = torch.cat(embeddings, dim=1)
        fused = self.fusion(fused_input)
        return self.projection(fused)


class MultimodalFusion(nn.Module):
    def __init__(self, modality_dims, fusion_dim=128):
        super().__init__()
        # modality_dims is now a list with a single element: the total concatenated dimension
        total_input_dim = modality_dims[0]
        print(f"[MultimodalFusion] Initializing with total input dim: {total_input_dim}")
        self.fusion = nn.Sequential(
            nn.Linear(total_input_dim, fusion_dim),
            nn.ReLU()
        )

    def forward(self, x): # Now takes the already concatenated tensor
        # No need to loop and print, we pass the full tensor now
        # x = torch.cat(embeddings, dim=1)
        return self.fusion(x)


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os

class MultimodalSessionDataset(Dataset):
    def __init__(self, root_dir, sensor_list, max_len=1000):
        self.samples = []
        self.max_len = max_len
        self.sensor_list = sensor_list

        print(f"[Dataset Init] Scanning dataset directory: {root_dir}")

        # ... (rest of the __init__ code is correct)
        for user_folder in os.listdir(root_dir):
            user_path = os.path.join(root_dir, user_folder)
            if not os.path.isdir(user_path): continue
            try: user_id = int(user_folder)
            except ValueError: continue

            for session_folder in os.listdir(user_path):
                session_path = os.path.join(user_path, session_folder)
                if not os.path.isdir(session_path): continue
                data_paths = {
                    'key_data': os.path.join(session_path, 'KEYSTROKE', 'key_data.csv'),
                    'swipe': os.path.join(session_path, 'TOUCH', 'swipe.csv'),
                    'f_0_touch': os.path.join(session_path, 'TOUCH', 'f_0_touch.csv'),
                    'touch_touch': os.path.join(session_path, 'TOUCH', 'touch_touch.csv'),
                    'sensor_grav': os.path.join(session_path, 'SENSORS', 'sensor_grav.csv'),
                    'sensor_gyro': os.path.join(session_path, 'SENSORS', 'sensor_gyro.csv'),
                    'sensor_humd': os.path.join(session_path, 'SENSORS', 'sensor_humd.csv'),
                    'sensor_lacc': os.path.join(session_path, 'SENSORS', 'sensor_lacc.csv'),
                    'sensor_ligh': os.path.join(session_path, 'SENSORS', 'sensor_ligh.csv'),
                    'sensor_magn': os.path.join(session_path, 'SENSORS', 'sensor_magn.csv'),
                    'sensor_nacc': os.path.join(session_path, 'SENSORS', 'sensor_nacc.csv'),
                    'sensor_prox': os.path.join(session_path, 'SENSORS', 'sensor_prox.csv'),
                    'sensor_temp': os.path.join(session_path, 'SENSORS', 'sensor_temp.csv')
                }
                self.samples.append({'user_id': user_id, 'session_id': session_folder, 'paths': data_paths})
        print(f"[Dataset Init Complete] Loaded {len(self.samples)} sessions across users.")

    def __len__(self):
        return len(self.samples)

    # In your MultimodalSessionDataset class
    def __getitem__(self, idx):
        sample = self.samples[idx]
        tensors = []
        
        for sensor in self.sensor_list:
            path = sample['paths'].get(sensor)
            data_loaded = False
            
            if path and os.path.exists(path):
                try:
                    # --- NEW ROBUST LOADING LOGIC ---
                    # 1. Open the file to "sniff" the format from the first line.
                    with open(path, 'r') as f:
                        first_line = f.readline()

                    # 2. Auto-detect the separator.
                    if ';' in first_line:
                        separator = ';'
                    else:
                        separator = ',' # Default to comma if no semicolon is found

                    # 3. Auto-detect if there is a header.
                    # If the first line contains letters, it's a header. Otherwise, no header.
                    if any(c.isalpha() for c in first_line):
                        header_row = 0
                    else:
                        header_row = None
                    
                    # 4. Read the CSV using the detected format.
                    df = pd.read_csv(path, sep=separator, header=header_row, engine='python')
                    # --- End of new logic ---

                    if not df.empty:
                        # Select all columns that can be converted to numeric data
                        numeric_df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1)

                        # Normalize the valid numeric columns
                        if not numeric_df.empty:
                            numeric_df = (numeric_df - numeric_df.mean()) / (numeric_df.std() + 1e-6)
                            data = torch.tensor(numeric_df.values, dtype=torch.float)
                            data_loaded = True

                except Exception as e:
                    print(f"  [FAILURE] Could not process {sensor} from {path}. Reason: {e}")
                    pass
            
            if not data_loaded:
                data = torch.zeros(self.max_len, 1)

            # Padding/truncation to max_len
            T, D = data.shape
            if T > self.max_len:
                data = data[:self.max_len]
            elif T < self.max_len:
                pad = torch.zeros(self.max_len - T, D)
                data = torch.cat([data, pad], dim=0)
            tensors.append(data)

        # Padding features and stacking remains the same
        max_dim = max(t.shape[1] for t in tensors) if tensors else 1
        padded = []
        for t in tensors:
            if t.shape[1] < max_dim:
                pad = torch.zeros(t.shape[0], max_dim - t.shape[1])
                t = torch.cat([t, pad], dim=1)
            padded.append(t)
        tensor_stack = torch.stack(padded)

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

def split_dataset_by_user(dataset, test_ratio=0.2, min_sessions_for_test=2, random_state=42):
    """
    Splits the dataset by user, ensuring that all users in the test set have a minimum
    number of sessions for robust evaluation.
    """
    # Group sample indices by user ID
    user_sessions = {}
    for i, sample in enumerate(dataset.samples):
        user_id = sample['user_id']
        user_sessions.setdefault(user_id, []).append(i)

    # Identify users eligible for the test set (i.e., have enough sessions)
    eligible_test_users = [uid for uid, idxs in user_sessions.items() if len(idxs) >= min_sessions_for_test]
    other_users = [uid for uid, idxs in user_sessions.items() if len(idxs) < min_sessions_for_test]

    if len(eligible_test_users) < 2:
        raise ValueError(f"Cannot create a test set. Need at least 2 users with >= {min_sessions_for_test} sessions.")

    # Split the eligible users into training and testing
    test_user_count = max(2, int(len(eligible_test_users) * test_ratio))
    train_eligible_users, test_users = train_test_split(eligible_test_users, test_size=test_user_count, random_state=random_state)

    # All users who weren't eligible for the test set automatically go into training
    train_users = set(train_eligible_users) | set(other_users)

    print(f"[Split] Train users: {len(train_users)}, Test users: {len(test_users)}")
    print(f"[Split] Test users are: {sorted(test_users)}")


    # Collect the final indices for the subsets
    train_idx = [i for uid in train_users for i in user_sessions[uid]]
    test_idx = [i for uid in test_users for i in user_sessions[uid]]

    return Subset(dataset, train_idx), Subset(dataset, test_idx)

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




def balanced_pairs(embeddings, labels, max_pos_pairs=5000, max_neg_pairs=5000):
    labels = labels.cpu().numpy()
    embeddings = embeddings.cpu()

    label_to_indices = {}
    for i, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(i)

    pos_pairs = []
    for indices in label_to_indices.values():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                pos_pairs.append((indices[i], indices[j]))
    random.shuffle(pos_pairs)
    pos_pairs = pos_pairs[:max_pos_pairs]

    all_indices = list(range(len(labels)))
    neg_pairs = []
    while len(neg_pairs) < max_neg_pairs:
        i, j = random.sample(all_indices, 2)
        if labels[i] != labels[j]:
            neg_pairs.append((i, j))

    return pos_pairs, neg_pairs

def evaluate_model(model, dataloader, device):
    """
    Evaluates the model by calculating genuine and impostor scores.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    print("\n[Eval] Generating embeddings for the test set...")
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x = x.to(device)
            emb = model(x)
            all_embeddings.append(emb.cpu())
            all_labels.append(y.cpu())

    embeddings = F.normalize(torch.cat(all_embeddings), dim=1)
    labels = torch.cat(all_labels)

    # Group embeddings by user ID
    user_to_embeddings = {}
    for user_id in torch.unique(labels):
        indices = (labels == user_id).nonzero(as_tuple=True)[0]
        user_to_embeddings[user_id.item()] = embeddings[indices]

    print(f"[Eval] Found {len(user_to_embeddings)} unique users in the test set.")
    
    genuine_scores = []
    impostor_scores = []

    # --- Calculate Genuine Scores (Same User, Different Sessions) ---
    print("[Eval] Calculating genuine scores...")
    for user_id, embs in user_to_embeddings.items():
        if len(embs) > 1:
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    sim = F.cosine_similarity(embs[i].unsqueeze(0), embs[j].unsqueeze(0)).item()
                    genuine_scores.append(sim)

    # --- Calculate Impostor Scores (Different Users) ---
    print("[Eval] Calculating impostor scores...")
    user_ids = list(user_to_embeddings.keys())
    # To keep computation manageable, we sample a number of impostor pairs
    num_impostor_pairs = len(genuine_scores) * 2 # Aim for a 2:1 ratio of impostors to genuines
    
    for _ in range(num_impostor_pairs):
        u1_id, u2_id = random.sample(user_ids, 2)
        emb1 = random.choice(user_to_embeddings[u1_id])
        emb2 = random.choice(user_to_embeddings[u2_id])
        sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        impostor_scores.append(sim)

    if not genuine_scores or not impostor_scores:
        print("[Error] Could not form both genuine and impostor pairs. Cannot evaluate.")
        return

    # --- Calculate Metrics ---
    scores = np.array(genuine_scores + impostor_scores)
    y_true = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))

    fpr, tpr, _ = roc_curve(y_true, scores)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    auc = roc_auc_score(y_true, scores)

    print("\n--- Evaluation Results ---")
    print(f"Genuine Pairs Found: {len(genuine_scores)}")
    print(f"Impostor Pairs Sampled: {len(impostor_scores)}")
    print(f"AUC: {auc:.4f}")
    print(f"EER (Equal Error Rate): {eer:.4f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(genuine_scores, bins=50, alpha=0.7, density=True, label='Genuine Scores (Same User)')
    plt.hist(impostor_scores, bins=50, alpha=0.7, density=True, label='Impostor Scores (Different User)')
    plt.title('Similarity Score Distribution on Test Set')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def split_dataset_by_user(dataset, test_ratio=0.2, min_sessions_for_test=2, random_state=42):
    """
    Splits the dataset by user, ensuring that all users in the test set have a minimum
    number of sessions for robust evaluation.
    """
    # Group sample indices by user ID
    user_sessions = {}
    for i, sample in enumerate(dataset.samples):
        user_id = sample['user_id']
        user_sessions.setdefault(user_id, []).append(i)

    # Identify users eligible for the test set (i.e., have enough sessions)
    eligible_test_users = [uid for uid, idxs in user_sessions.items() if len(idxs) >= min_sessions_for_test]
    other_users = [uid for uid, idxs in user_sessions.items() if len(idxs) < min_sessions_for_test]

    if len(eligible_test_users) < 2:
        raise ValueError(f"Cannot create a test set. Need at least 2 users with >= {min_sessions_for_test} sessions.")

    # Split the eligible users into training and testing
    test_user_count = max(2, int(len(eligible_test_users) * test_ratio))
    train_eligible_users, test_users = train_test_split(eligible_test_users, test_size=test_user_count, random_state=random_state)

    # All users who weren't eligible for the test set automatically go into training
    train_users = set(train_eligible_users) | set(other_users)

    print(f"[Split] Train users: {len(train_users)}, Test users: {len(test_users)}")
    print(f"[Split] Test users are: {sorted(test_users)}")


    # Collect the final indices for the subsets
    train_idx = [i for uid in train_users for i in user_sessions[uid]]
    test_idx = [i for uid in test_users for i in user_sessions[uid]]

    return Subset(dataset, train_idx), Subset(dataset, test_idx)

class TripletWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset  # This is a Subset
        self.underlying_dataset = self.dataset.dataset
        self.user_to_indices = {}

        # Map user_id to their indices *within the original dataset*
        for idx in self.dataset.indices:
            user_id = self.underlying_dataset.samples[idx]['user_id']
            self.user_to_indices.setdefault(user_id, []).append(idx)

        # We can only create valid triplets for users with >1 session in the training set
        self.trainable_indices = []
        for user_id, indices in self.user_to_indices.items():
            if len(indices) > 1:
                self.trainable_indices.extend(indices)
        
    def __len__(self):
        return len(self.trainable_indices)

    def __getitem__(self, idx):
        # The index for our anchor sample within self.trainable_indices
        anchor_original_idx = self.trainable_indices[idx]
        anchor_data, anchor_label = self.underlying_dataset[anchor_original_idx]

        # 1. Select a POSITIVE sample (same user, different session)
        positive_options = [i for i in self.user_to_indices[anchor_label] if i != anchor_original_idx]
        positive_original_idx = random.choice(positive_options)
        pos_data, _ = self.underlying_dataset[positive_original_idx]

        # 2. Select a NEGATIVE sample (different user)
        neg_label = anchor_label
        while neg_label == anchor_label:
            neg_label = random.choice(list(self.user_to_indices.keys()))
        
        negative_original_idx = random.choice(self.user_to_indices[neg_label])
        neg_data, _ = self.underlying_dataset[negative_original_idx]

        return anchor_data, pos_data, neg_data

if __name__ == "__main__":
    DATA_ROOT = "/home/krish/Downloads/HuMI"

    sensor_list = [
        'key_data', 'swipe', 'f_X_touch', 'touch_touch',
        'sensor_grav', 'sensor_gyro', 'sensor_humd', 'sensor_lacc',
        'sensor_ligh', 'sensor_magn', 'sensor_nacc', 'sensor_prox', 'sensor_temp'
    ]

    print("[Main] Creating dataset...")
    dataset = MultimodalSessionDataset(DATA_ROOT, sensor_list)
    print(f"[Main] Total users in dataset: {len(dataset)}")

    print("[Main] Splitting train/test users...")
    train_dataset, test_dataset = split_dataset_by_user(dataset)

    from torch.utils.data import DataLoader
    import random

    

    triplet_train_dataset = TripletWrapper(train_dataset)
    triplet_loader = DataLoader(triplet_train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    print("[Main] Building model...")

    sensor_dims = {
        'key_data': 1, 'swipe': 4, 'f_X_touch': 4, 'touch_touch': 4,
        'sensor_grav': 3, 'sensor_gyro': 3, 'sensor_humd': 3, 'sensor_lacc': 3,
        'sensor_ligh': 1, 'sensor_magn': 3, 'sensor_nacc': 3, 'sensor_prox': 1, 'sensor_temp': 1
    }

    model = MultimodalEmbeddingModel(sensors=sensor_list, sensor_dims=sensor_dims, hidden_dim=64).to(DEVICE)

    loss_fn = nn.TripletMarginLoss(margin=0.5, p=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    print("[Main] Training model with Triplet Loss...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for anchor, positive, negative in tqdm(triplet_loader):
            anchor = anchor.to(DEVICE)
            positive = positive.to(DEVICE)
            negative = negative.to(DEVICE)

            optimizer.zero_grad()
            try:
                anchor_emb = F.normalize(model(anchor), dim=1)
                pos_emb = F.normalize(model(positive), dim=1)
                neg_emb = F.normalize(model(negative), dim=1)

                loss = loss_fn(anchor_emb, pos_emb, neg_emb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except Exception as e:
                print(f"[Error] Skipping triplet due to error: {e}")

        avg_loss = total_loss / max(1, len(triplet_loader))
        print(f"[Epoch {epoch+1}] Triplet Loss: {avg_loss:.4f}")
        scheduler.step()

    print("[Main] Evaluating model...")
    evaluate_model(model, test_loader, device=DEVICE)
