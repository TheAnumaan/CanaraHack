import torch
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict

# Import the model classes from your model definition file
from model_definition import MultimodalEmbeddingModel

# =================================================================================================
# --- 1. API Setup and Global Configuration ---
# =================================================================================================

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Biometric Authentication API",
    description="An API to generate user embeddings from multimodal sensor data for authentication.",
    version="1.0.0"
)

# --- Configuration (MUST MATCH TRAINING SCRIPT) ---
DEVICE = torch.device("cpu") # Run inference on CPU
MAX_LEN = 1000

# Model parameters used during training
MODEL_PARAMS = {
    'hidden_dim': 128,
    'proj_dim': 128,
    'tcn_layers': 5,
    'dropout_rate': 0.4,
    'sequence_length': MAX_LEN
}

# Sensor list and their original dimensions
SENSOR_LIST = [
    'key_data', 'swipe', 'touch_touch', 'sensor_grav', 'sensor_gyro',
    'sensor_lacc', 'sensor_magn', 'sensor_nacc'
]
SENSOR_DIMS = {
    'key_data': 1, 'swipe': 6, 'touch_touch': 6, 'sensor_grav': 3,
    'sensor_gyro': 3, 'sensor_lacc': 3, 'sensor_magn': 3, 'sensor_nacc': 3
}

# The maximum feature dimension across all sensors, used for padding
MAX_FEATURE_DIM = max(SENSOR_DIMS.values())

# =================================================================================================
# --- 2. Load The Trained Model ---
# =================================================================================================

# Instantiate the model with the same architecture and parameters
model = MultimodalEmbeddingModel(SENSOR_LIST, SENSOR_DIMS, MODEL_PARAMS)

# Load the saved state dictionary from the .pkl file
try:
    model.load_state_dict(torch.load("multimodal_authentication_model.pkl", map_location=DEVICE))
except FileNotFoundError:
    raise RuntimeError("Model file 'multimodal_authentication_model.pkl' not found. Make sure it's in the same directory.")
except Exception as e:
    raise RuntimeError(f"Error loading model state_dict: {e}")


# Set the model to evaluation mode (this disables dropout and batch norm running averages)
model.eval()
# Move model to the specified device
model.to(DEVICE)

print("[INFO] Model loaded successfully and is ready for inference.")


# =================================================================================================
# --- 3. Define Request and Response Models (Pydantic) ---
# =================================================================================================

# Pydantic model to define the structure of the incoming JSON data.
# This provides automatic validation for the request body.
class SensorData(BaseModel):
    # A dictionary where keys are sensor names and values are 2D lists of numbers.
    # Example: {"sensor_gyro": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], ...]}
    # The `Field` with a default_factory of dict ensures that if a sensor is missing,
    # it defaults to an empty dictionary, which we can handle gracefully.
    data: Dict[str, List[List[float]]] = Field(default_factory=dict)

class EmbeddingResponse(BaseModel):
    # The response will contain the generated embedding as a list of floats.
    embedding: List[float]

# =================================================================================================
# --- 4. Preprocessing Logic ---
# =================================================================================================

def preprocess_input(sensor_data: SensorData) -> torch.Tensor:
    """
    Transforms raw sensor data from the request into a tensor suitable for the model.
    This function mimics the preprocessing from the original training script's Dataset class.
    """
    tensors = []
    
    for sensor in SENSOR_LIST:
        sensor_readings = sensor_data.data.get(sensor, [])
        
        if not sensor_readings:
            # If sensor data is missing, create a zero tensor with the expected feature dimension
            data = torch.zeros(MAX_LEN, SENSOR_DIMS[sensor])
        else:
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(sensor_readings)
            
            # --- Normalization ---
            # NOTE: In a production system, it's better to normalize using pre-calculated
            # mean/std from the entire training set. Here, we mimic the per-sample
            # normalization from the training script for consistency.
            if not df.empty and df.std().sum() > 0:
                df = (df - df.mean()) / (df.std().replace(0, 1))
            df.fillna(0, inplace=True)
            
            data = torch.tensor(df.values, dtype=torch.float32)

        # --- Padding / Truncating Time Steps ---
        T, D = data.shape
        if T > MAX_LEN:
            data = data[:MAX_LEN]
        elif T < MAX_LEN:
            padding = torch.zeros(MAX_LEN - T, D)
            data = torch.cat([data, padding], dim=0)
            
        tensors.append(data)

    # --- Padding Feature Dimension ---
    # Pad each sensor's feature dimension to be the same (MAX_FEATURE_DIM)
    padded_tensors = []
    for t in tensors:
        current_dim = t.shape[1]
        if current_dim < MAX_FEATURE_DIM:
            padding = torch.zeros(t.shape[0], MAX_FEATURE_DIM - current_dim)
            t = torch.cat([t, padding], dim=1)
        padded_tensors.append(t)
        
    # Stack all tensors into a single tensor for the model
    # Shape: (Num_Modalities, Time, Features)
    tensor_stack = torch.stack(padded_tensors)
    
    # Add a batch dimension at the beginning
    # Shape: (1, Num_Modalities, Time, Features)
    return tensor_stack.unsqueeze(0)


# =================================================================================================
# --- 5. API Endpoint Definition ---
# =================================================================================================

@app.post("/predict", response_model=EmbeddingResponse)
async def predict(sensor_data: SensorData):
    """
    Receives sensor data, preprocesses it, and returns the generated biometric embedding.

    **How to use this endpoint:**
    1.  **Enrollment:** An Android app collects a user's sensor data, sends it to this
        endpoint to get an embedding, and stores this "enrollment embedding" securely.
    2.  **Verification:** For login, the app collects new sensor data, gets a new
        "verification embedding" from this endpoint, and compares it (e.g., using
        cosine similarity) with the stored enrollment embedding. If the similarity
        is above a certain threshold, access is granted.
    """
    try:
        # 1. Preprocess the input data
        input_tensor = preprocess_input(sensor_data)
        input_tensor = input_tensor.to(DEVICE)

        # 2. Get the model's prediction (no gradients needed for inference)
        with torch.no_grad():
            embedding = model(input_tensor)

        # 3. Convert the output tensor to a standard Python list
        embedding_list = embedding.cpu().numpy().flatten().tolist()
        
        # 4. Return the embedding in the response
        return {"embedding": embedding_list}

    except Exception as e:
        # If anything goes wrong, return a 500 server error
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to the Multimodal Authentication API. Go to /docs for documentation."}

