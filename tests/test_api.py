"""
CamerTrust Lite - Tests unitaires de l'API
E2 - Developpeur Backend | S2 (1 test), S4 (3 tests), S5 (5 tests total)

Lancer les tests :
  pytest tests/ -v --tb=short

Ces tests utilisent une base SQLite en memoire dediee (isolee de la base
de developpement) pour ne jamais polluer camertrust.db.
"""

import os

os.environ["DATABASE_URL"] = "sqlite:///./test_camertrust.db"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base, get_db
from api.main import app

TEST_DATABASE_URL = "sqlite:///./test_camertrust.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("test_camertrust.db"):
        os.remove("test_camertrust.db")


client = TestClient(app)


# ---------------------------------------------------------------------------
# S2 - Sante de l'API
# ---------------------------------------------------------------------------
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_info():
    response = client.get("/info")
    assert response.status_code == 200
    body = response.json()
    assert "model_loaded" in body


# ---------------------------------------------------------------------------
# S4 - Endpoint /predict
# ---------------------------------------------------------------------------
def test_predict_returns_valid_schema():
    payload = {
        "amount": 150000,
        "type": "TRANSFER",
        "old_balance_org": 200000,
        "new_balance_org": 50000,
        "old_balance_dest": 0,
        "new_balance_dest": 150000,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "is_fraud" in body
    assert "score" in body
    assert 0 <= body["score"] <= 1


def test_predict_rejects_negative_amount():
    payload = {"amount": -100, "type": "TRANSFER"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # erreur de validation Pydantic


def test_predict_rejects_missing_type():
    payload = {"amount": 1000}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# S5 - Endpoints /transactions et /alerts (proteges par JWT)
# ---------------------------------------------------------------------------
def test_transactions_requires_auth():
    response = client.get("/transactions")
    assert response.status_code == 401


def test_alerts_requires_auth():
    response = client.get("/alerts")
    assert response.status_code == 401


def _get_token():
    response = client.post(
        "/auth/token",
        data={"username": "camertrust", "password": "changeme123"},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def test_login_with_valid_credentials():
    token = _get_token()
    assert token is not None


def test_login_with_invalid_credentials():
    response = client.post(
        "/auth/token",
        data={"username": "camertrust", "password": "mauvais_mdp"},
    )
    assert response.status_code == 401


def test_transactions_with_valid_token():
    token = _get_token()
    # On cree d'abord une transaction pour avoir des donnees a lister
    client.post("/predict", json={"amount": 5000, "type": "PAYMENT"})

    response = client.get(
        "/transactions", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_alerts_filter_by_min_amount():
    token = _get_token()
    response = client.get(
        "/alerts?min_amount=1000000",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)
