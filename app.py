# -*- coding: utf-8 -*-


import pickle
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# =========================================================
# FastAPI App
# =========================================================
app = FastAPI()

# =========================================================
# Global Vars and Model Loading
# =========================================================
garch_model = None
lstm_model = None

SEQUENCE_LENGTH = 10   # must match your training window size
NUM_FEATURES = 6       # High, Low, Adj_Close, Volume, SMA, High_Low_Diff



# =========================================================
# Define LSTM Model (must match training code)
# =========================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # last timestep
        return out


# =========================================================
# Instantiate Models
# =========================================================
input_size = NUM_FEATURES
hidden_size = 5
output_size = 1
lstm_model = LSTMModel(input_size, hidden_size, output_size)

try:
    # Load GARCH model
    with open("garch_model_results.pkl", "rb") as f:
        garch_model = pickle.load(f)

    # Load LSTM weights
    lstm_model.load_state_dict(torch.load("lstm_stock_prediction.pt"))
    lstm_model.eval()
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        

    print(" Models and scaler loaded successfully!")

except FileNotFoundError:
    print("Error: Model files not found.")
    garch_model = None
    lstm_model = None
    scaler = None


# =========================================================
# Pydantic Schemas
# =========================================================
class Features(BaseModel):
    High: float
    Low: float
    Adj_Close: float
    Volume: float
    SMA: float
    High_Low_Diff: float


class PredictionRequest(BaseModel):
    features: list[Features]
    last_volatility: float


# =========================================================
# Prediction Endpoint
# =========================================================
@app.post("/predict")
async def get_prediction(request: PredictionRequest):
    if not garch_model or not lstm_model:
        raise HTTPException(status_code=500, detail="Models not loaded.")

    # Ensure sequence length matches training
    if len(request.features) != SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {SEQUENCE_LENGTH} feature rows, got {len(request.features)}"
        )

    # Convert features to numpy array
    features_array = np.array([
        [f.High, f.Low, f.Adj_Close, f.Volume, f.SMA, f.High_Low_Diff]
        for f in request.features
    ], dtype=np.float32)
    
    #  Scale features (same as training)
    features_array = scaler.transform(features_array)

    # Reshape to match LSTM input: (1, seq_len, num_features)
    lstm_input = torch.tensor(features_array).unsqueeze(0)

    # Predict with LSTM
    with torch.no_grad():
        lstm_prediction = lstm_model(lstm_input)
        last_adj_close = request.features[-1].Adj_Close
        price_pred = (lstm_prediction.item())


    # Predict with GARCH
    try:
        garch_forecast = garch_model.forecast(horizon=1)
        forecasted_volatility = np.sqrt(garch_forecast.variance.values[-1, :][0])
    except Exception as e:
        forecasted_volatility = request.last_volatility  # fallback
        print("GARCH forecast failed, using last_volatility:", e)

    # Calculate Value at Risk (VaR)
    confidence_level = 0.95
    z_score = -1.645  # for 95%
    var = z_score * forecasted_volatility

    return {
        "price_prediction": price_pred,
        "volatility_forecast": float(forecasted_volatility),
        "value_at_risk": float(var),
        "num_features_used": features_array.shape[0]
    }


# =========================================================
# Run App
# =========================================================

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)