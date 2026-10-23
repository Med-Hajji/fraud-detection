"""Feature engineering et preprocessing pour le projet fraud-detection."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer


def add_features(df):
    """Ajoute Hour, is_night, log_amount ; retire Amount brut."""
    df = df.copy()
    df["Hour"] = (df["Time"] // 3600) % 24
    df["is_night"] = df["Hour"].between(0, 5).astype(int)
    df["log_amount"] = np.log1p(df["Amount"])
    df = df.drop(columns=["Amount"])
    return df


def build_feature_creator():
    """Retourne le transformer qui crée Hour, is_night, log_amount."""
    return FunctionTransformer(add_features)


def build_column_transformer():
    """Retourne le ColumnTransformer qui standardise les colonnes numériques."""
    numeric_features = ["log_amount", "Time", "Hour"]
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), numeric_features),
        ],
        remainder="passthrough",
    )


def build_preprocessor():
    """
    Pipeline complet (feature creation + scaling) en un seul objet.
    À utiliser uniquement dans un sklearn.Pipeline classique (ex: étape 4,
    Logistic Regression). Ne PAS l'utiliser à l'intérieur d'un imblearn.Pipeline
    (SMOTE, RandomUnderSampler...) : imblearn interdit d'imbriquer un Pipeline
    dans un autre Pipeline. Dans ce cas, utiliser build_feature_creator() et
    build_column_transformer() séparément, comme deux étapes distinctes.
    """
    return Pipeline([
        ("features", build_feature_creator()),
        ("preprocessing", build_column_transformer()),
    ])