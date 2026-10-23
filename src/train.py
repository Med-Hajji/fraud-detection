"""
Script d'entraînement du modèle de détection de fraude.
Usage : python -m src.train
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, average_precision_score
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

from src.features import build_feature_creator, build_column_transformer

# --- Configuration ---
DATA_PATH = "donnees/creditcard.csv"
MODEL_DIR = "models"
RANDOM_STATE = 42
COST_FALSE_NEGATIVE = 100
COST_FALSE_POSITIVE = 5

# Meilleurs hyperparamètres trouvés à l'étape 6 (RandomizedSearchCV)
BEST_PARAMS = {
    "subsample": 0.7,
    "n_estimators": 300,
    "min_child_weight": 1,
    "max_depth": 8,
    "learning_rate": 0.1,
    "colsample_bytree": 0.7,
}


def load_data(path):
    df = pd.read_csv(path)
    return df


def build_model_pipeline(scale_pos_weight):
    return ImbPipeline([
        ("features", build_feature_creator()),
        ("prep", build_column_transformer()),
        ("model", XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **BEST_PARAMS,
        )),
    ])


def find_best_threshold(y_true, y_proba, cost_fn, cost_fp):
    thresholds = np.arange(0.01, 1.0, 0.01)
    costs = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        costs.append(cost_fn * fn + cost_fp * fp)
    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx]), costs[best_idx]


def main():
    print("1/5 - Chargement des données...")
    df = load_data(DATA_PATH)

    print("2/5 - Split train/test...")
    X = df.drop(columns="Class")
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print("3/5 - Entraînement du modèle...")
    pipeline = build_model_pipeline(scale_pos_weight)
    pipeline.fit(X_train, y_train)

    print("4/5 - Choix du seuil (prédictions out-of-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    best_threshold, best_cost = find_best_threshold(
        y_train, oof_proba, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE
    )
    print(f"    Seuil optimal : {best_threshold:.2f} (coût train : {best_cost})")

    print("5/5 - Évaluation finale sur le test set...")
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    pr_auc_test = average_precision_score(y_test, y_test_proba)
    print(f"    PR-AUC (test) : {pr_auc_test:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "model.joblib"))
    with open(os.path.join(MODEL_DIR, "threshold.json"), "w") as f:
        json.dump({
            "threshold": best_threshold,
            "cost_false_negative": COST_FALSE_NEGATIVE,
            "cost_false_positive": COST_FALSE_POSITIVE,
            "pr_auc_test": pr_auc_test,
        }, f, indent=2)

    print("\nModèle et seuil sauvegardés dans", MODEL_DIR)


if __name__ == "__main__":
    main()