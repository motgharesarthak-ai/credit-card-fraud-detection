from pydantic import BaseModel, Field
import pandas as pd
from fastapi import FastAPI
import joblib
from pathlib import Path

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for detecting potentially fraudulent credit card transactions",
    version="1.0.0"
)

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load saved model and scaler
model = joblib.load(BASE_DIR / "models" / "fraud_detection_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is running!",
        "status": "success"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


# Request format for a transaction
class Transaction(BaseModel):
    Time: float = Field(ge=0)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0)


@app.post("/predict")
def predict_transaction(transaction: Transaction):
    # Convert the incoming transaction to a DataFrame
    transaction_data = pd.DataFrame(
        [transaction.model_dump()]
    )

    # Apply the same scaler to Time and Amount
    transaction_data[["Time", "Amount"]] = scaler.transform(
        transaction_data[["Time", "Amount"]]
    )

    # Get fraud probability and prediction
    probabilities = model.predict_proba(transaction_data)[0]
    fraud_probability = float(probabilities[1])
    prediction = int(fraud_probability >= 0.5)

    return {
        "prediction": prediction,
        "result": "Fraud" if prediction == 1 else "Legitimate",
        "fraud_probability": round(fraud_probability, 4)
    }
