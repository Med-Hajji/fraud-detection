"""Tests unitaires pour le module de feature engineering."""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.features import add_features


def test_add_features_creates_expected_columns():
    """add_features doit créer Hour, is_night, log_amount et retirer Amount."""
    df = pd.DataFrame({
        "Time": [3600 * 2],   # 2 heures
        "Amount": [100.0],
    })
    result = add_features(df)

    assert "Hour" in result.columns
    assert "is_night" in result.columns
    assert "log_amount" in result.columns
    assert "Amount" not in result.columns


def test_hour_is_computed_correctly():
    """Time=3600*2 secondes doit donner Hour=2."""
    df = pd.DataFrame({"Time": [3600 * 2], "Amount": [50.0]})
    result = add_features(df)

    assert result["Hour"].iloc[0] == 2


def test_is_night_flag():
    """is_night doit valoir 1 entre 0h et 5h, 0 sinon."""
    df = pd.DataFrame({
        "Time": [3600 * 2, 3600 * 14],   # 2h (nuit) et 14h (jour)
        "Amount": [50.0, 50.0],
    })
    result = add_features(df)

    assert result["is_night"].iloc[0] == 1   # 2h -> nuit
    assert result["is_night"].iloc[1] == 0   # 14h -> jour


def test_log_amount_is_positive_for_positive_amount():
    """log1p(Amount) doit être positif pour un montant positif."""
    df = pd.DataFrame({"Time": [0], "Amount": [850.0]})
    result = add_features(df)

    assert result["log_amount"].iloc[0] > 0


def test_original_dataframe_is_not_modified():
    """add_features ne doit pas modifier le DataFrame original (copy())."""
    df = pd.DataFrame({"Time": [0], "Amount": [100.0]})
    add_features(df)

    assert "Amount" in df.columns   # toujours présent dans l'original
    assert "Hour" not in df.columns  # pas ajouté à l'original