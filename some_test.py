import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import optuna # The star of the show

# =================================================================================================
# --- 1. Global Configuration & Model Definitions (Same as before) ---
# =================================================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 1000
DATA_ROOT = "/home/krish/Downloads/HuMI" # IMPORTANT: Update this path

# --- Model classes are included here for a self-contained script ---
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

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, tcn_layers, dropout_rate, sequence_length):
        super().__init__()
        self.encoder = TCNBlock(input_dim=input_dim, output_dim=hidden_dim, kernel_size=3, num_layers=tcn_layers, dropout_rate=dropout_rate, sequence_length=sequence_length)
    def forward(self, x): return self.encoder(x)

class MultimodalEmbeddingModel(nn.Module):
    def __init__(self, sensors, sensor_dims, params):
        super().__init__()
        max_feature_dim = max(sensor_dims.values())
        self.encoders = nn.ModuleDict({
            sensor: ModalityEncoder(input_dim=max_feature_dim, hidden_dim=params['hidden_dim'], tcn_layers=params['tcn_layers'], dropout_rate=params['dropout_rate'], sequence_length=params['sequence_length'])
            for sensor in sensors
        })
        fusion_input_dim = params['hidden_dim'] * len(sensors)
        self.fusion = nn.Sequential(nn.Linear(fusion_input_dim, 128), nn.ReLU(), nn.Dropout(params['dropout_rate']))
        self.projection = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, params['proj_dim']))
    def forward(self, inputs):
        embeddings = [self.encoders[sensor](inputs[:, i, :, :]) for i, sensor in enumerate(self.encoders.keys())]
        fused_input = torch.cat(embeddings, dim=1)
        fused = self.fusion(fused_input)
        return self.projection(fused)

# --- Data Handling and Training/Evaluation functions are also included ---
# (These are the same as your original script, no changes needed)
class MultimodalSessionDataset(Dataset):
    def __init__(self, root_dir, sensor_list, max_len=1000):
        self.samples, self.max_len, self.sensor_list = [], max_len, sensor_list
        self.user_to_idx, self.idx_to_user = {}, {}
        all_user_ids = sorted([int(d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()])
        for idx, user_id in enumerate(all_user_ids): self.user_to_idx[user_id], self.idx_to_user[idx] = idx, user_id
        for user_folder in os.listdir(root_dir):
            user_path = os.path.join(root_dir, user_folder)
            if not os.path.isdir(user_path): continue
            try: user_id = int(user_folder)
            except ValueError: continue
            for session_folder in os.listdir(user_path):
                session_path = os.path.join(user_path, session_folder)
                if not os.path.isdir(session_path): continue
                data_paths = {s: os.path.join(session_path, 'SENSORS', f'{s}.csv') for s in self.sensor_list if s.startswith('sensor_')}
                data_paths.update({s: os.path.join(session_path, 'TOUCH', f'{s}.csv') for s in self.sensor_list if s.startswith('touch') or s in ['swipe', 'f_X_touch']})
                data_paths.update({s: os.path.join(session_path, 'KEYSTROKE', f'{s}.csv') for s in self.sensor_list if s.startswith('key')})
                self.samples.append({'user_id': user_id, 'session_id': session_folder, 'paths': data_paths})
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        sample, original_user_id = self.samples[idx], self.samples[idx]['user_id']
        mapped_label = self.user_to_idx[original_user_id]
        tensors = []
        for sensor in self.sensor_list:
            path, data_loaded = sample['paths'].get(sensor), False
            if path and os.path.exists(path) and os.path.getsize(path) > 5:
                try:
                    with open(path, 'r', errors='ignore') as f: first_line = f.readline()
                    separator, header_row = (';', 0) if ';' in first_line else (',', 0 if any(c.isalpha() for c in first_line) else None)
                    df = pd.read_csv(path, sep=separator, header=header_row, engine='python', on_bad_lines='skip').apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
                    if not df.empty:
                        df.fillna(0, inplace=True)
                        if df.std().sum() > 0: df = (df - df.mean()) / (df.std().replace(0, 1))
                        data, data_loaded = torch.tensor(df.values, dtype=torch.float), True
                except Exception: pass
            if not data_loaded: data = torch.zeros(self.max_len, 1)
            T, D = data.shape
            if T > self.max_len: data = data[:self.max_len]
            elif T < self.max_len: data = torch.cat([data, torch.zeros(self.max_len - T, D)], dim=0)
            tensors.append(data)
        max_dim = max(t.shape[1] for t in tensors) if tensors else 1
        padded = [torch.cat([t, torch.zeros(t.shape[0], max_dim - t.shape[1])], dim=1) if t.shape[1] < max_dim else t for t in tensors]
        return torch.stack(padded), mapped_label

def split_dataset_by_user(dataset, test_ratio=0.2, min_sessions_for_test=2, random_state=42):
    user_sessions = {}
    for i, sample in enumerate(dataset.samples): user_sessions.setdefault(sample['user_id'], []).append(i)
    eligible_test_users = [uid for uid, idxs in user_sessions.items() if len(idxs) >= min_sessions_for_test]
    other_users = [uid for uid, idxs in user_sessions.items() if len(idxs) < min_sessions_for_test]
    if len(eligible_test_users) < 2: raise ValueError("Not enough users for a test set.")
    test_user_count = max(2, int(len(eligible_test_users) * test_ratio))
    train_eligible_users, test_users = train_test_split(eligible_test_users, test_size=test_user_count, random_state=random_state)
    train_users = set(train_eligible_users) | set(other_users)
    train_idx = [i for uid in train_users for i in user_sessions[uid]]
    test_idx = [i for uid in test_users for i in user_sessions[uid]]
    return Subset(dataset, train_idx), Subset(dataset, test_idx)

class DataAugmentation:
    def __init__(self, sigma=0.02, shift_fraction=0.05): self.sigma, self.shift_fraction = sigma, shift_fraction
    def __call__(self, batch):
        noise = torch.randn_like(batch) * self.sigma
        aug_batch = batch + noise
        max_shift = int(batch.shape[2] * self.shift_fraction)
        shifts = torch.randint(-max_shift, max_shift, (batch.shape[0],))
        for i in range(batch.shape[0]): aug_batch[i] = torch.roll(aug_batch[i], shifts=shifts[i].item(), dims=1)
        return aug_batch

def train_model(model, dataloader, loss_fn, optimizer, scheduler, num_epochs, device):
    augment = DataAugmentation(sigma=0.01, shift_fraction=0.1)
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, ncols=100)
        for inputs, labels in progress_bar:
            inputs, labels = augment(inputs).to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings = model(inputs)
            dist_matrix = torch.cdist(embeddings, embeddings, p=2)
            is_pos, is_neg = labels.unsqueeze(1) == labels.unsqueeze(0), labels.unsqueeze(1) != labels.unsqueeze(0)
            dist_ap, dist_an = dist_matrix.clone(), dist_matrix.clone()
            dist_ap[is_neg], dist_an[is_pos] = -float('inf'), float('inf')
            hardest_pos, _ = torch.max(dist_ap, dim=1)
            hardest_neg, _ = torch.min(dist_an, dim=1)
            loss = torch.mean(F.relu(hardest_pos - hardest_neg + loss_fn.margin))
            loss.backward()
            optimizer.step()
        scheduler.step()

def evaluate_model(model, dataloader, device):
    model.eval()
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            all_embeddings.append(model(x.to(device)).cpu())
            all_labels.append(y.cpu())
    embeddings, labels = F.normalize(torch.cat(all_embeddings), dim=1), torch.cat(all_labels)
    user_to_embeddings = {uid.item(): embeddings[(labels == uid).nonzero(as_tuple=True)[0]] for uid in torch.unique(labels)}
    user_to_embeddings = {k: v for k, v in user_to_embeddings.items() if len(v) > 1}
    if len(user_to_embeddings) < 2: return None, None # Cannot evaluate
    genuine_scores = [F.cosine_similarity(embs[i].unsqueeze(0), embs[j].unsqueeze(0)).item() for embs in user_to_embeddings.values() for i in range(len(embs)) for j in range(i + 1, len(embs))]
    if not genuine_scores: return None, None
    user_ids = list(user_to_embeddings.keys())
    impostor_scores = [F.cosine_similarity(random.choice(user_to_embeddings[u1]).unsqueeze(0), random.choice(user_to_embeddings[u2]).unsqueeze(0)).item() for _ in range(len(genuine_scores) * 2) for u1, u2 in [random.sample(user_ids, 2)]]
    if not impostor_scores: return None, None
    scores, y_true = np.array(genuine_scores + impostor_scores), np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))
    if len(np.unique(y_true)) < 2: return None, None
    fpr, tpr, _ = roc_curve(y_true, scores)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    auc = roc_auc_score(y_true, scores)
    return auc, eer

# =================================================================================================
# --- 2. OPTUNA Objective Function ---
# This is the core of the optimization task.
# =================================================================================================

def objective(trial, train_loader, val_loader, sensor_list, sensor_dims):
    """
    This function is called by Optuna for each trial.
    It builds a model with suggested hyperparameters, trains it, evaluates it,
    and returns a performance score for Optuna to minimize.
    """
    # --- Step 1: Let Optuna suggest hyperparameters ---
    # We define a search space for Optuna to explore.
    model_params = {
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
        'proj_dim': 128, # Keep projection dim fixed for simplicity
        'tcn_layers': trial.suggest_int('tcn_layers', 3, 6),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.6),
        'sequence_length': MAX_LEN
    }
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    triplet_margin = trial.suggest_float('triplet_margin', 0.3, 1.0)
    
    # --- Step 2: Build and train the model with these parameters ---
    model = MultimodalEmbeddingModel(sensor_list, sensor_dims, model_params).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # For speed, we train for fewer epochs during the search
    num_search_epochs = 15 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_search_epochs)
    loss_fn = nn.TripletMarginLoss(margin=triplet_margin)

    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Params: {trial.params}")
    
    train_model(model, train_loader, loss_fn, optimizer, scheduler, num_search_epochs, DEVICE)
    
    # --- Step 3: Evaluate on the validation set ---
    auc, eer = evaluate_model(model, val_loader, device=DEVICE)
    
    # Handle cases where evaluation might fail (e.g., not enough pairs)
    if eer is None:
        print(f"Trial {trial.number} failed evaluation. Returning high EER.")
        return 1.0 # Return a high value to penalize this trial

    print(f"--- Trial {trial.number} Finished | EER: {eer:.4f} | AUC: {auc:.4f} ---")

    # --- Step 4: Return the metric for Optuna to minimize ---
    return eer


# =================================================================================================
# --- 3. Main Execution Block ---
# =================================================================================================

if __name__ == "__main__":
    print(f"[INFO] Using device: {DEVICE}")
    
    # --- Sensor Config ---
    sensor_list = ['key_data', 'swipe', 'touch_touch', 'sensor_grav', 'sensor_gyro', 'sensor_lacc', 'sensor_magn', 'sensor_nacc']
    sensor_dims = {'key_data': 1, 'swipe': 6, 'touch_touch': 6, 'sensor_grav': 3, 'sensor_gyro': 3, 'sensor_lacc': 3, 'sensor_magn': 3, 'sensor_nacc': 3}

    # --- Data Loading ---
    print("[Main] Loading and splitting data...")
    full_dataset = MultimodalSessionDataset(DATA_ROOT, sensor_list, max_len=MAX_LEN)
    # This is the original training set, containing only training users
    train_val_subset, test_subset = split_dataset_by_user(full_dataset)

    # --- Create a Validation Set from the training data ---
    # We need to split train_val_subset further for tuning.
    # We'll do a simple random split here. A user-based split would be even better if time permits.
    train_indices, val_indices = train_test_split(
        list(range(len(train_val_subset))),
        test_size=0.2, # Use 20% of the training data for validation
        random_state=42
    )
    train_subset = Subset(train_val_subset, train_indices)
    val_subset = Subset(train_val_subset, val_indices)
    
    print(f"[Split] Original training users: {len(train_val_subset)} sessions")
    print(f"[Split] New training set for tuning: {len(train_subset)} sessions")
    print(f"[Split] New validation set for tuning: {len(val_subset)} sessions")
    print(f"[Split] Final test set (untouched): {len(test_subset)} sessions")

    # --- Create DataLoaders ---
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # --- Run the Optuna Study ---
    # We want to minimize the Equal Error Rate (EER)
    study = optuna.create_study(direction='minimize')
    
    # We pass the necessary data via a lambda function to the objective
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, sensor_list, sensor_dims),
        n_trials=50, # Run 50 trials. Increase this if you have more time.
        timeout=60 * 60 * 10 # Set a 10-hour timeout
    )

    # --- Print the results ---
    print("\n\n--- OPTIMIZATION FINISHED ---")
    print(f"Number of finished trials: {len(study.trials)}")
    
    print("\n--- BEST TRIAL ---")
    best_trial = study.best_trial
    print(f"  Value (EER): {best_trial.value:.4f}")
    
    print("  Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # You can also visualize the results
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image("optuna_history.png")
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image("optuna_importances.png")
    except Exception as e:
        print(f"\nCould not generate plots. Install plotly and kaleido: {e}")

