"""
CamerTrust Lite - Tests de l'encodage categoriel du modele
E2 - Developpeur Backend

Verifie que ModelService encode correctement la colonne 'type' en
colonnes one-hot avant d'appeler le pipeline de E1 (correctif suite a
prepare_data_camertrust.py, etape S3).

Lancer :
  pytest tests/test_ml_encoding.py -v
"""

import pandas as pd
import pytest

from api.ml import ModelService


@pytest.fixture
def service_with_onehot_features():
    """Simule un ModelService avec un model_info.json listant des colonnes one-hot."""
    service = ModelService.__new__(ModelService)  # bypass __init__/_load
    service.pipeline = None
    service.model_type = "FakePipeline"
    service.threshold = 0.5
    service.features = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER",
    ]
    return service


def test_encode_known_type_sets_single_column_to_one(service_with_onehot_features):
    row = pd.DataFrame([{"amount": 1000.0, "type": "TRANSFER"}])
    encoded = service_with_onehot_features._encode_categorical(row)

    assert "type" not in encoded.columns
    assert encoded.at[0, "type_TRANSFER"] == 1
    for col in ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT"]:
        assert encoded.at[0, col] == 0


def test_encode_unknown_type_defaults_to_all_zero(service_with_onehot_features, caplog):
    row = pd.DataFrame([{"amount": 1000.0, "type": "TYPE_INEXISTANT"}])
    encoded = service_with_onehot_features._encode_categorical(row)

    onehot_cols = service_with_onehot_features.onehot_type_columns
    assert all(encoded.at[0, col] == 0 for col in onehot_cols)


def test_encode_without_model_info_leaves_row_unchanged_except_type_dropped():
    service = ModelService.__new__(ModelService)
    service.features = None  # comme le DummyModel sans model_info.json

    row = pd.DataFrame([{"amount": 1000.0, "type": "PAYMENT"}])
    encoded = service._encode_categorical(row)

    assert "type" not in encoded.columns
    assert "amount" in encoded.columns
    assert list(encoded.columns) == ["amount"]


def test_predict_reindexes_columns_in_training_order(service_with_onehot_features, monkeypatch):
    captured = {}

    class FakePipeline:
        def predict_proba(self, X):
            captured["columns"] = list(X.columns)
            return [[0.7, 0.3]]

    service_with_onehot_features.pipeline = FakePipeline()

    score, is_fraud = service_with_onehot_features.predict(
        {
            "amount": 5000.0,
            "type": "CASH_OUT",
            "oldbalanceOrg": 10000.0,
            "newbalanceOrig": 5000.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 5000.0,
        }
    )

    assert captured["columns"] == service_with_onehot_features.features
    assert score == 0.3
    assert is_fraud is False
