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

import torch
import torch.nn as nn
import torch.nn.functional as F

# TCNBlock now accepts layers and dropout as parameters
# Correct TCNBlock that requires all parameters
# Correct TCNBlock that requires all parameters
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
        x = x.permute(0, 2, 1)
        conv_out = self.network(x)
        flattened = conv_out.flatten(start_dim=1)
        return self.project(flattened)


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

        # Use the new AttentionFusion module
        self.fusion = AttentionFusion(hidden_dim=params['hidden_dim'])

        self.projection = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, params['proj_dim'])
        )

    def forward(self, inputs):
        # The dataloader produces tensors of shape (B, num_modalities, T, D_max)
        # Create a list of embeddings, one for each modality
        embeddings = [
            self.encoders[sensor](inputs[:, i, :, :]) for i, sensor in enumerate(self.encoders.keys())
        ]
        
        # Pass the list of embeddings to the attention fusion layer
        fused = self.fusion(embeddings)
        
        return self.projection(fused)

# Additive Angular Margin Loss (ArcFace)
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
        self.th = torch.cos(torch.tensor(torch.pi) - m)
        self.mm = torch.sin(torch.tensor(torch.pi) - m) * m

    def forward(self, embeddings, labels):
        # embeddings are (B, D), labels are (B)
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return F.cross_entropy(output, labels)
    
import optuna

# Keep your Dataset, TripletWrapper, and evaluate_model functions as they are.

import optuna
from torch.utils.data import DataLoader

from tqdm import tqdm

def objective(trial, sensors, sensor_dims, train_dataset, test_dataset, num_train_users):
    """
    An Optuna objective function to find the best hyperparameters.
    Includes hard negative mining for TripletLoss.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Define the search space for hyperparameters
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128]),
        'proj_dim': trial.suggest_categorical('proj_dim', [64, 128]),
        'tcn_layers': trial.suggest_int('tcn_layers', 3, 6),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
        'loss_function': trial.suggest_categorical('loss_function', ['TripletLoss', 'ArcFaceLoss']),
        'sequence_length': 1000
    }

    # 2. Create model
    model = MultimodalEmbeddingModel(sensors, sensor_dims, params).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=1e-5)
    
    # 3. Select loss function and create the appropriate DataLoader
    if params['loss_function'] == 'TripletLoss':
        margin = trial.suggest_float('triplet_margin', 0.2, 1.0)
        loss_fn = nn.TripletMarginLoss(margin=margin, reduction='mean')
        # Hard negative mining requires batches with multiple samples from the same class
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    else: # ArcFaceLoss
        s = trial.suggest_float('arcface_s', 20.0, 40.0)
        m = trial.suggest_float('arcface_m', 0.3, 0.7)
        loss_fn = ArcFaceLoss(in_features=params['proj_dim'], out_features=num_train_users, s=s, m=m).to(DEVICE)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # 4. Run the training loop
    NUM_TUNING_EPOCHS = 20 # Increased epochs for better convergence
    for epoch in range(NUM_TUNING_EPOCHS):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch+1}/{NUM_TUNING_EPOCHS}", leave=False)
        for data in progress_bar:
            optimizer.zero_grad()
            
            if params['loss_function'] == 'TripletLoss':
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                embeddings = model(inputs)
                
                # --- Hard Negative Mining Logic ---
                dist_matrix = torch.cdist(embeddings, embeddings, p=2)
                is_pos = labels.unsqueeze(1) == labels.unsqueeze(0)
                is_neg = ~is_pos
                
                # For each anchor, find the hardest positive (furthest sample of same class)
                dist_ap = dist_matrix.clone()
                dist_ap[~is_pos] = -float('inf')
                hardest_positive_dist, _ = torch.max(dist_ap, dim=1)

                # For each anchor, find the hardest negative (closest sample of different class)
                dist_an = dist_matrix.clone()
                dist_an[is_pos] = float('inf')
                hardest_negative_dist, _ = torch.min(dist_an, dim=1)
                
                loss = torch.mean(F.relu(hardest_positive_dist - hardest_negative_dist + loss_fn.margin))
            
            else: # ArcFaceLoss
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                embeddings = model(inputs)
                loss = loss_fn(embeddings, labels)

            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    # 5. Evaluate the model and return the score to minimize (EER)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    auc, eer = evaluate_model(model, test_loader, device=DEVICE, trial=trial)
    
    if eer is None:
        return 1.0  # Return the worst possible EER for a failed trial

    return eer


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, tcn_layers, dropout_rate, sequence_length):
        super().__init__()
        self.encoder = TCNBlock(
            input_dim=input_dim,
            output_dim=hidden_dim,
            kernel_size=3,  # Using a fixed kernel size
            num_layers=tcn_layers, # Using the parameter from Optuna
            dropout_rate=dropout_rate, # Using the parameter from Optuna
            sequence_length=sequence_length
        )

    def forward(self, x):
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
# Correct main model that initializes the other two correctly
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
        fused_input = torch.cat(embeddings, dim=1) # Corrected fusion input
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

class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim, attention_dim=128):
        """
        Initializes the Attention Fusion mechanism.
        Args:
            hidden_dim (int): The feature dimension of each incoming modality embedding.
            attention_dim (int): The size of the hidden layer in the attention network.
        """
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, 128), # Input is the weighted-sum embedding, so its size is hidden_dim
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, embeddings_list):
        """
        Processes a list of embeddings from different modalities.
        Args:
            embeddings_list (list of Tensors): A list where each element is a tensor
                                               of shape (batch_size, hidden_dim).
        Returns:
            Tensor: A fused embedding of shape (batch_size, 128).
        """
        # Stack embeddings to create a (batch_size, num_modalities, hidden_dim) tensor
        stacked_embeddings = torch.stack(embeddings_list, dim=1)
        
        # Compute attention weights for each modality
        # attn_weights will have shape (batch_size, num_modalities, 1)
        attn_weights = self.attention_net(stacked_embeddings)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Apply attention weights to the embeddings
        # (B, N, D) * (B, N, 1) -> (B, N, D) where N is num_modalities
        weighted_embeddings = stacked_embeddings * attn_weights
        
        # Sum the weighted embeddings to get a single context vector
        # Shape: (B, D)
        context_vector = torch.sum(weighted_embeddings, dim=1)
        
        # Pass the final context vector through the fusion layers
        return self.fusion(context_vector)


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

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn.functional as F

def evaluate_model(model, dataloader, device, trial=None):
    """
    Evaluates the model by calculating genuine and impostor scores.
    Saves plots to a file and handles evaluation failures gracefully.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            emb = model(x)
            all_embeddings.append(emb.cpu())
            all_labels.append(y.cpu())

    embeddings = F.normalize(torch.cat(all_embeddings), dim=1)
    labels = torch.cat(all_labels)

    # Group embeddings by user ID, only including users with multiple sessions
    user_to_embeddings = {}
    for user_id in torch.unique(labels):
        indices = (labels == user_id).nonzero(as_tuple=True)[0]
        if len(indices) > 1:
            user_to_embeddings[user_id.item()] = embeddings[indices]

    # Check if evaluation is possible
    if len(user_to_embeddings) < 2:
        print("[Error] Evaluation failed. Need at least 2 users with multiple sessions.")
        return None, None

    # Calculate genuine scores
    genuine_scores = []
    for user_id, embs in user_to_embeddings.items():
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                sim = F.cosine_similarity(embs[i].unsqueeze(0), embs[j].unsqueeze(0)).item()
                genuine_scores.append(sim)

    if not genuine_scores:
        print("[Error] No genuine pairs could be formed.")
        return None, None

    # Calculate impostor scores
    impostor_scores = []
    user_ids = list(user_to_embeddings.keys())
    for _ in range(len(genuine_scores) * 2): # Sample a reasonable number of impostor pairs
        u1_id, u2_id = random.sample(user_ids, 2)
        emb1 = random.choice(user_to_embeddings[u1_id])
        emb2 = random.choice(user_to_embeddings[u2_id])
        impostor_scores.append(F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item())

    scores = np.array(genuine_scores + impostor_scores)
    y_true = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))

    fpr, tpr, _ = roc_curve(y_true, scores)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    auc = roc_auc_score(y_true, scores)

    # Save plot to file instead of showing it
    plt.figure(figsize=(10, 6))
    plt.hist(genuine_scores, bins=50, alpha=0.7, density=True, label=f'Genuine (Same User)')
    plt.hist(impostor_scores, bins=50, alpha=0.7, density=True, label=f'Impostor (Different User)')
    plt.title(f"Trial {trial.number if trial else 'N/A'} - Scores (EER: {eer:.4f})")
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"trial_{trial.number}_scores.png" if trial else "final_evaluation_scores.png"
    plt.savefig(filename)
    plt.close() # Close the plot to free up memory

    return auc, eer

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
    # --- Basic Config ---
    DATA_ROOT = "/home/krish/Downloads/HuMI/"
    sensor_list = [
        'key_data', 'swipe', 'touch_touch', 'sensor_grav', 'sensor_gyro',
        'sensor_lacc', 'sensor_magn', 'sensor_nacc'
    ]
    sensor_dims = {
        'key_data': 1, 'swipe': 6, 'touch_touch': 6, 'sensor_grav': 3,
        'sensor_gyro': 3, 'sensor_lacc': 3, 'sensor_magn': 3, 'sensor_nacc': 3
    }

    # --- Load and Split Data ONCE ---
    print("[Main] Loading and splitting data...")
    dataset = MultimodalSessionDataset(DATA_ROOT, sensor_list, max_len=1000)
    train_dataset, test_dataset = split_dataset_by_user(dataset)
    
    # We need the number of unique users for the ArcFace loss function's output layer
    train_user_ids = {s['user_id'] for i, s in enumerate(dataset.samples) if i in train_dataset.indices}
    num_train_users = len(train_user_ids)
    print(f"[Main] Found {num_train_users} unique users in the training set.")


    # --- Optuna Study ---
    print("[Main] Starting hyperparameter search with Optuna...")
    # We pass the datasets as extra arguments to the objective function using a lambda
    study = optuna.create_study(direction='minimize') # We want to MINIMIZE the EER
# NEW
    study.optimize(
        lambda trial: objective(trial, sensor_list, sensor_dims, train_dataset, test_dataset, num_train_users),
        n_trials=50 
    )

    # --- Print Best Results ---
    print("\n\n--- Optuna Search Complete ---")
    print(f"Best trial EER: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
