"""
Script d'évaluation du modèle sauvegardé sur le test set.
Usage : python -m src.evaluate
"""

import json
import pathlib
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    confusion_matrix, classification_report,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "donnees" / "creditcard.csv"
MODEL_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns="Class")
    y = df["Class"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = joblib.load(MODEL_DIR / "model.joblib")
    with open(MODEL_DIR / "threshold.json") as f:
        threshold_data = json.load(f)
    threshold = threshold_data["threshold"]

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print(f"Seuil utilisé : {threshold}")
    print(f"PR-AUC  : {average_precision_score(y_test, y_proba):.4f}")
    print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report :")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraude"]))


if __name__ == "__main__":
    main()