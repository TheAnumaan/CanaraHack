import torch
import torch.nn as nn
import torch.nn.functional as F

# =================================================================================================
# --- Model Architecture Definitions ---
# This file contains the PyTorch nn.Module classes required to instantiate the model.
# It should be in the same directory as your api.py file.
# =================================================================================================

class TCNBlock(nn.Module):
    """
    A Temporal Convolutional Network block, which is the core of the modality encoders.
    """
    def __init__(self, input_dim, output_dim, kernel_size, num_layers, dropout_rate, sequence_length):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else output_dim
            # Convolutional layer with increasing dilation
            layers.append(nn.Conv1d(
                in_channels=in_ch, out_channels=output_dim, kernel_size=kernel_size,
                padding=(kernel_size - 1) * (2 ** i), dilation=2 ** i
            ))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.Dropout(dropout_rate))

        self.network = nn.Sequential(*layers)
        
        # This part dynamically calculates the flattened size after convolution
        # to correctly size the final linear layer.
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim, sequence_length)
            dummy_output = self.network(dummy_input)
            flattened_size = dummy_output.flatten(1).shape[1]
            
        self.project = nn.Linear(flattened_size, output_dim)

    def forward(self, x):
        # Input shape: (Batch, Time, Features) -> (B, T, D)
        # Permute to: (Batch, Features, Time) -> (B, D, T) for Conv1d
        x = x.permute(0, 2, 1)
        conv_out = self.network(x)
        # Flatten the output of the convolutional layers
        flattened = conv_out.flatten(start_dim=1)
        return self.project(flattened)

class ModalityEncoder(nn.Module):
    """
    An encoder for a single sensor modality, using a TCNBlock.
    """
    def __init__(self, input_dim, hidden_dim, tcn_layers, dropout_rate, sequence_length):
        super().__init__()
        self.encoder = TCNBlock(
            input_dim=input_dim, output_dim=hidden_dim, kernel_size=3,
            num_layers=tcn_layers, dropout_rate=dropout_rate, sequence_length=sequence_length
        )

    def forward(self, x):
        return self.encoder(x)

class MultimodalEmbeddingModel(nn.Module):
    """
    The main model that takes data from all sensors, encodes them individually,
    fuses them, and projects them into a final embedding space.
    """
    def __init__(self, sensors, sensor_dims, params):
        super().__init__()
        # The model expects all inputs to be padded to the maximum feature dimension
        max_feature_dim = max(sensor_dims.values())
        
        # A dictionary of encoders, one for each sensor modality
        self.encoders = nn.ModuleDict({
            sensor: ModalityEncoder(
                input_dim=max_feature_dim,
                hidden_dim=params['hidden_dim'],
                tcn_layers=params['tcn_layers'],
                dropout_rate=params['dropout_rate'],
                sequence_length=params['sequence_length']
            ) for sensor in sensors
        })

        # Simple fusion by concatenating the outputs of all encoders
        fusion_input_dim = params['hidden_dim'] * len(sensors)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(params['dropout_rate'])
        )

        # Final projection head to get the desired output embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, params['proj_dim'])
        )

    def forward(self, inputs):
        # inputs shape: (Batch, Num_Modalities, Time, Features)
        # Process each modality through its corresponding encoder
        embeddings = [self.encoders[sensor](inputs[:, i, :, :]) for i, sensor in enumerate(self.encoders.keys())]
        
        # Concatenate the resulting embeddings
        fused_input = torch.cat(embeddings, dim=1)
        
        # Pass through the fusion and projection layers
        fused = self.fusion(fused_input)
        return self.projection(fused)
