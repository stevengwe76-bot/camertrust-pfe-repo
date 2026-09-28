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


# ---------------------------------------------------------------------------
# Resolution des alertes (utilisee par la page Alertes du dashboard)
# ---------------------------------------------------------------------------
def _auth_headers():
    return {"Authorization": f"Bearer {_get_token()}"}


def _create_fraud_transaction() -> str:
    """Cree une transaction classee fraude et renvoie son id."""
    from api.ml import model_service

    original = model_service.threshold
    model_service.threshold = 0.0          # tout score >= 0 -> fraude
    try:
        client.post("/predict", json={"amount": 750000, "type": "TRANSFER",
                                      "old_balance_org": 750000, "new_balance_org": 0})
    finally:
        model_service.threshold = original
    alerts = client.get("/alerts?limit=1", headers=_auth_headers()).json()
    return alerts[0]["id"]


def test_alerts_include_status_open_by_default():
    alert_id = _create_fraud_transaction()
    alerts = client.get("/alerts", headers=_auth_headers()).json()
    alert = next(a for a in alerts if a["id"] == alert_id)
    assert alert["status"] == "OPEN"
    assert alert["resolved_by"] is None


def test_resolve_alert_requires_auth():
    response = client.patch("/alerts/x/resolve", json={"status": "RESOLVED", "resolved_by": "a"})
    assert response.status_code == 401


def test_resolve_alert_updates_status_and_filter():
    alert_id = _create_fraud_transaction()
    response = client.patch(
        f"/alerts/{alert_id}/resolve",
        json={"status": "DISMISSED", "resolved_by": "agent_001", "resolution_note": "Client joint"},
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "DISMISSED" and body["resolved_by"] == "agent_001"

    # Une seconde decision remplace la premiere
    client.patch(f"/alerts/{alert_id}/resolve",
                 json={"status": "RESOLVED", "resolved_by": "agent_002"}, headers=_auth_headers())
    resolved = client.get("/alerts?status=RESOLVED", headers=_auth_headers()).json()
    assert [a["id"] for a in resolved].count(alert_id) == 1
    open_ids = [a["id"] for a in client.get("/alerts?status=OPEN", headers=_auth_headers()).json()]
    assert alert_id not in open_ids


def test_resolve_unknown_alert_returns_404():
    response = client.patch("/alerts/inexistant/resolve",
                            json={"status": "RESOLVED", "resolved_by": "a"}, headers=_auth_headers())
    assert response.status_code == 404


def test_resolve_rejects_invalid_status():
    alert_id = _create_fraud_transaction()
    response = client.patch(f"/alerts/{alert_id}/resolve",
                            json={"status": "PEUT_ETRE", "resolved_by": "a"}, headers=_auth_headers())
    assert response.status_code == 422


def test_resolve_non_fraud_transaction_returns_409():
    from api.ml import model_service

    original = model_service.threshold
    model_service.threshold = 1.1          # aucun score ne depasse -> legitime
    try:
        client.post("/predict", json={"amount": 1000, "type": "PAYMENT"})
    finally:
        model_service.threshold = original
    tx = client.get("/transactions?limit=1", headers=_auth_headers()).json()[0]
    assert tx["is_fraud"] is False
    response = client.patch(f"/alerts/{tx['id']}/resolve",
                            json={"status": "RESOLVED", "resolved_by": "a"}, headers=_auth_headers())
    assert response.status_code == 409


def test_info_exposes_model_error_when_dummy():
    body = client.get("/info").json()
    if not body["model_loaded"]:
        assert body["model_error"]
