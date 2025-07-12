import torch
import torch.nn as nn
import os
import pandas as pd
from torch.utils.data import Dataset

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

class CSVSessionDataset(Dataset):
    def __init__(self, root_dir, sensor='gps', max_len=1000):
        self.samples = []
        self.max_len = max_len
        self.sensor = sensor

        for user_folder in os.listdir(root_dir):
            if not user_folder.startswith("user_"):
                continue
            user_id = int(user_folder.split('_')[-1])
            user_path = os.path.join(root_dir, user_folder)

            for file in os.listdir(user_path):
                if file.startswith(sensor) and file.endswith('.csv'):
                    self.samples.append({
                        'csv_path': os.path.join(user_path, file),
                        'user_id': user_id
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        df = pd.read_csv(sample['csv_path'], header=None)

        if self.sensor == 'gps':
            # Columns: timestamp, orientation, lat, lon, alt, bearing, accuracy
            data = torch.tensor(df.iloc[:, [2, 3, 4, 5, 6]].values, dtype=torch.float)
        elif self.sensor == 'bluetooth':
            # Columns: timestamp, name, MAC
            # Use name length and MAC encoded as float (hash-based simple encoding)
            data = df.iloc[:, [1, 2]].astype(str)
            data = torch.tensor([
                [len(name), float(int("0x" + mac.replace(":", ""), 16) % 1e9)]
                for name, mac in data.values
            ], dtype=torch.float)
        elif self.sensor == 'wifi':
            # Columns: timestamp, name, level, info, channel, frequency
            data = torch.tensor(df.iloc[:, [2, 4, 5]].values, dtype=torch.float)
        elif self.sensor.startswith('sensor_'):
            # Handle all sensor files with common logic
            data = torch.tensor(df.iloc[:, 2:].values, dtype=torch.float)
        elif self.sensor in ['swipe', 'scroll_X_touch', 'touch_touch', 'f_X_touch']:
            data = torch.tensor(df.iloc[:, 2:6].values, dtype=torch.float)
        elif self.sensor == 'key_data':
            # Columns: timestamp, field, ASCII code (may be NaN)
            ascii_vals = pd.to_numeric(df.iloc[:, 2], errors='coerce').fillna(0)
            data = torch.tensor(ascii_vals.values.reshape(-1, 1), dtype=torch.float)
        else:
            raise ValueError(f"Unknown sensor type: {self.sensor}")

        T, D = data.shape
        if T > self.max_len:
            data = data[:self.max_len]
        elif T < self.max_len:
            pad = torch.zeros(self.max_len - T, D)
            data = torch.cat([data, pad], dim=0)

        return data, sample['user_id']
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
