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
warnings.filterwarnings('ignore')

def has_header(row):
    header_keywords = ["timestamp", "orientation", "x", "y", "z", "SSID", "MAC", "altitude", "accuracy", "action"]
    return any(kw.lower() in str(row).lower() for kw in header_keywords)

def get_expected_cols(file_lower, parent_folder, grandparent_folder):
    """Determine expected columns based on file name and parent/grandparent folders."""
    if "accel" in file_lower or "gyro" in file_lower:
        return ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    elif "orientation" in file_lower:
        return ["orientation_x", "orientation_y", "orientation_z"]
    elif "wifi" in file_lower:
        return ["SSID", "MAC", "altitude", "accuracy"]
    elif "action" in file_lower:
        return ["action"]
    else:
        return None

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

class TripletDataset(Dataset):
    """Dataset for loading triplet samples from CSV files"""
    def __init__(self, data_dir, user_filter=None, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.user_filter = user_filter
        self.file_groups = self._organize_files()
        self.triplets = self._generate_triplets()
        
    def _organize_files(self):
        """Recursively organize CSV files by user/session/gesture, robust to missing headers."""
        file_groups = {}
        for user in sorted(os.listdir(self.data_dir)):
            user_path = self.data_dir / user
            if not user_path.is_dir() or not user.isdigit() or len(user) != 3:
                continue
            if self.user_filter and user not in self.user_filter:
                continue
            for session in sorted(os.listdir(user_path)):
                session_path = user_path / session
                if not session_path.is_dir():
                    continue
                for gesture in sorted(os.listdir(session_path)):
                    gesture_path = session_path / gesture
                    if not gesture_path.is_dir():
                        continue
                    for root, dirs, files in os.walk(gesture_path):
                        for file in files:
                            if file.endswith(".csv"):
                                file_path = Path(root) / file
                                if user not in file_groups:
                                    file_groups[user] = []
                                file_groups[user].append(file_path)
        return file_groups
    
    def _generate_triplets(self):
        """Generate triplet combinations"""
        triplets = []
        users = list(self.file_groups.keys())
        
        for user in users:
            user_files = self.file_groups[user]
            if len(user_files) < 2:
                continue
                
            # Generate positive pairs within same user
            for i in range(len(user_files)):
                for j in range(i+1, len(user_files)):
                    anchor_file = user_files[i]
                    positive_file = user_files[j]
                    
                    # Select negative from different user
                    other_users = [u for u in users if u != user]
                    if other_users:
                        negative_user = random.choice(other_users)
                        negative_file = random.choice(self.file_groups[negative_user])
                        
                        triplets.append((anchor_file, positive_file, negative_file))
        
        return triplets
    
    REQUIRED_COLUMNS = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    
    def _load_and_normalize_csv(self, file_path):
        """Load CSV file, robustly handle headers, and return the data tensor (no normalization)."""
        try:
            parent_folder = file_path.parent.name.lower()
            grandparent_folder = file_path.parent.parent.name.lower()
            file_lower = file_path.name.lower()
            with open(file_path, 'r') as f:
                first_line = f.readline()
            if has_header(first_line):
                df = pd.read_csv(file_path)
            else:
                expected_cols = get_expected_cols(file_lower, parent_folder, grandparent_folder)
                df = pd.read_csv(file_path, header=None)
                if expected_cols and df.shape[1] == len(expected_cols):
                    df.columns = expected_cols

            # Ensure all required columns are present
            for col in self.REQUIRED_COLUMNS:
                if col not in df.columns:
                    df[col] = 0.0  # or np.nan

            # Select and order the required columns
            data_arr = df[self.REQUIRED_COLUMNS].values
            data = torch.tensor(data_arr, dtype=torch.float32)

            if data.shape[0] < 2:
                print(f"[WARNING] Sequence too short (len={data.shape[0]}) in file: {file_path}. Skipping this file.")
                return torch.randn(2, len(self.REQUIRED_COLUMNS))
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return torch.randn(10, len(self.REQUIRED_COLUMNS))  # Fallback
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_file, positive_file, negative_file = self.triplets[idx]
        
        anchor = self._load_and_normalize_csv(anchor_file)
        positive = self._load_and_normalize_csv(positive_file)
        negative = self._load_and_normalize_csv(negative_file)
        
        return anchor, positive, negative

def collate_fn(batch):
    """Custom collate function for variable-length sequences"""
    anchors, positives, negatives = zip(*batch)
    return list(anchors), list(positives), list(negatives)

class TCNTrainer:
    def __init__(self, model, device, lr=0.001, margin=1.0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.criterion = nn.TripletMarginLoss(margin=margin)
        self.train_losses = []
        
    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0.0
        
        for batch_idx, (anchors, positives, negatives) in enumerate(tqdm(dataloader, desc="Training")):
            self.optimizer.zero_grad()
            
            # Process each triplet in the batch individually
            anchor_embeddings = []
            positive_embeddings = []
            negative_embeddings = []
            
            for anchor, positive, negative in zip(anchors, positives, negatives):
                # Move to device and ensure proper shape
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                # Add batch dimension if needed
                if anchor.dim() == 2:
                    anchor = anchor.unsqueeze(0)
                if positive.dim() == 2:
                    positive = positive.unsqueeze(0)
                if negative.dim() == 2:
                    negative = negative.unsqueeze(0)
                
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
            
            # Compute triplet loss
            loss = self.criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        self.train_losses.append(avg_loss)
        self.scheduler.step(avg_loss)
        
        return avg_loss
    
    def train(self, train_loader, num_epochs=50):
        print(f"Training TCN for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
    
    def save_checkpoint(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
        }, filepath)
        print(f"Checkpoint saved: {filepath}")

class TCNEvaluator:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
    def compute_embeddings(self, dataloader):
        """Compute embeddings for all samples"""
        self.model.eval()
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for batch_idx, (anchors, positives, negatives) in enumerate(tqdm(dataloader, desc="Computing embeddings")):
                # Process anchors and positives (same user)
                for anchor, positive in zip(anchors, positives):
                    anchor = anchor.to(self.device)
                    positive = positive.to(self.device)
                    
                    # Add batch dimension if needed
                    if anchor.dim() == 2:
                        anchor = anchor.unsqueeze(0)
                    if positive.dim() == 2:
                        positive = positive.unsqueeze(0)
                    
                    anchor_emb = self.model(anchor)
                    positive_emb = self.model(positive)
                    
                    embeddings.extend([anchor_emb.cpu(), positive_emb.cpu()])
                    labels.extend([batch_idx, batch_idx])  # Same user label
                
                # Process negatives (different user)
                for negative in negatives:
                    negative = negative.to(self.device)
                    
                    # Add batch dimension if needed
                    if negative.dim() == 2:
                        negative = negative.unsqueeze(0)
                    
                    negative_emb = self.model(negative)
                    
                    embeddings.append(negative_emb.cpu())
                    labels.append(batch_idx + 1000)  # Different user label
        
        return torch.cat(embeddings, dim=0), labels
    
    def evaluate_verification(self, test_loader, threshold=0.5):
        """Evaluate verification accuracy"""
        embeddings, labels = self.compute_embeddings(test_loader)
        
        # Compute pairwise distances
        distances = []
        true_labels = []
        
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                # Cosine distance
                dist = 1 - F.cosine_similarity(embeddings[i:i+1], embeddings[j:j+1]).item()
                distances.append(dist)
                
                # Same user if labels match
                same_user = (labels[i] == labels[j])
                true_labels.append(1 if same_user else 0)
        
        # Convert to tensors
        distances = torch.tensor(distances)
        true_labels = torch.tensor(true_labels)
        
        # Compute predictions using threshold
        predictions = (distances < threshold).float()
        
        # Compute metrics
        accuracy = accuracy_score(true_labels, predictions)
        auc = roc_auc_score(true_labels, 1 - distances)  # 1 - distance for similarity
        
        return accuracy, auc, distances, true_labels

def export_model(model, input_dim, export_path, format='onnx'):
    """Export model to ONNX or TorchScript"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 100, input_dim)  # Batch=1, seq_len=100
    
    if format == 'onnx':
        torch.onnx.export(
            model,
            dummy_input,
            export_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['embedding'],
            dynamic_axes={
                'input': {1: 'sequence_length'},
                'embedding': {0: 'batch_size'}
            }
        )
        print(f"Model exported to ONNX: {export_path}")
    
    elif format == 'torchscript':
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(export_path)
        print(f"Model exported to TorchScript: {export_path}")

def create_sample_data(data_dir, num_users=10, sessions_per_user=5):
    """Create sample CSV data for testing"""
    data_dir = Path(data_dir)
    
    for split in ['train', 'test']:
        for user_id in range(num_users):
            user_dir = data_dir / split / f"user_{user_id:03d}"
            user_dir.mkdir(parents=True, exist_ok=True)
            
            for session in range(sessions_per_user):
                # Generate random sensor data (6 channels: 3 accel + 3 gyro)
                seq_len = random.randint(50, 200)
                data = torch.randn(seq_len, 6)
                
                # Add some user-specific patterns
                data[:, :3] += torch.sin(torch.linspace(0, 10 * user_id, seq_len)).unsqueeze(1)
                data[:, 3:] += torch.cos(torch.linspace(0, 5 * user_id, seq_len)).unsqueeze(1)
                
                # Save to CSV
                df = pd.DataFrame(data.numpy(), columns=[
                    'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'
                ])
                df.to_csv(user_dir / f"session_{session:03d}.csv", index=False)
    
    print(f"Sample data created in {data_dir}")

def main():
    parser = argparse.ArgumentParser(description='TCN Behavioral Authentication')
    parser.add_argument('--mode', choices=['train', 'eval', 'export', 'create_data'], 
                       default='train', help='Mode to run')
    parser.add_argument('--data_dir', type=str, default='./data', 
                       help='Directory containing CSV files')
    parser.add_argument('--model_path', type=str, default='tcn_model.pth',
                       help='Path to save/load model')
    parser.add_argument('--export_path', type=str, default='tcn_model.onnx',
                       help='Path to export model')
    parser.add_argument('--export_format', choices=['onnx', 'torchscript'], 
                       default='onnx', help='Export format')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--input_dim', type=int, default=6, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.mode == 'create_data':
        create_sample_data(args.data_dir)
        return
    
    # Initialize model
    model = TCNEncoder(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim
    )
    
    if args.mode == 'train':
        # List all users in the root data_dir, but only users 000 to 049
        all_users = [u for u in sorted(os.listdir(args.data_dir)) if u.isdigit() and len(u) == 3 and 0 <= int(u) <= 49]
        random.shuffle(all_users)
        split_idx = int(0.8 * len(all_users))
        train_users = set(all_users[:split_idx])
        test_users = set(all_users[split_idx:])

        # Pass the split to the dataset:
        train_dataset = TripletDataset(args.data_dir, user_filter=train_users)
        test_dataset = TripletDataset(args.data_dir, user_filter=test_users)
        
        # Create datasets
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        # Train model
        trainer = TCNTrainer(model, device, lr=args.lr)
        trainer.train(train_loader, args.epochs)
        
        # Save model
        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved: {args.model_path}")
        
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(trainer.train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('training_loss.png')
        plt.show()
    
    elif args.mode == 'eval':
        # Load model
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        # Create test dataset
        test_dataset = TripletDataset(args.data_dir, split='test')
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        # Evaluate model
        evaluator = TCNEvaluator(model, device)
        accuracy, auc, distances, true_labels = evaluator.evaluate_verification(test_loader)
        
        print(f"Verification Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        
        # Plot ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(true_labels, 1 - distances)
        
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
        # Load model
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        # Export model
        export_model(model, args.input_dim, args.export_path, args.export_format)

if __name__ == "__main__":
    main()