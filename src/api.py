"""
API FastAPI pour la détection de fraude bancaire.
Usage : uvicorn src.api:app --reload
"""

import json
import pathlib
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

app = FastAPI(
    title="Fraud Detection API",
    description="API de détection de fraude bancaire basée sur XGBoost",
    version="1.0.0",
)

# Chargement du modèle et du seuil au démarrage de l'API
model = joblib.load(MODEL_DIR / "model.joblib")
with open(MODEL_DIR / "threshold.json") as f:
    threshold_data = json.load(f)
THRESHOLD = threshold_data["threshold"]


class Transaction(BaseModel):
    Time: float = Field(..., description="Secondes depuis la première transaction")
    Amount: float = Field(..., ge=0, description="Montant de la transaction")
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

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 50000, "Amount": 850.0,
                "V1": -1.2, "V2": 0.5, "V3": 1.1, "V4": -0.3, "V5": 0.8,
                "V6": -0.5, "V7": 0.2, "V8": -0.1, "V9": 0.4, "V10": -2.1,
                "V11": 1.3, "V12": -1.8, "V13": 0.1, "V14": -3.2, "V15": 0.6,
                "V16": -0.7, "V17": -1.5, "V18": 0.3, "V19": -0.2, "V20": 0.1,
                "V21": 0.2, "V22": 0.5, "V23": -0.1, "V24": 0.3, "V25": 0.1,
                "V26": -0.2, "V27": 0.05, "V28": 0.02,
            }
        }


class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: str
    threshold_used: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    df = pd.DataFrame([transaction.model_dump()])
    proba = float(model.predict_proba(df)[0, 1])
    prediction = "FRAUD" if proba >= THRESHOLD else "LEGIT"

    return PredictionResponse(
        fraud_probability=round(proba, 4),
        prediction=prediction,
        threshold_used=THRESHOLD,
    )