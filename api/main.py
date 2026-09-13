"""
CamerTrust Lite - API principale
E2 - Developpeur Backend | Semaines S1 a S9

Endpoints :
  GET  /health              - verification que l'API tourne
  GET  /info                - infos sur le modele charge
  POST /auth/token           - obtenir un token JWT (identifiants de demo)
  POST /predict              - analyser une transaction (public, utilise par E3)
  GET  /transactions         - liste paginee des transactions analysees
  GET  /alerts                - transactions marquees fraude, filtrables
  GET  /                      - redirige vers la documentation Swagger

Lancer en local :
  uvicorn api.main:app --reload

Documentation interactive : http://localhost:8000/docs
"""

import logging
import time
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from api import models, schemas
from api.database import Base, SessionLocal, engine, get_db
from api.ml import model_service
from api.security import authenticate_user, create_access_token, get_current_user

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("camertrust.api")

# Cree les tables au demarrage si elles n'existent pas encore
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="CamerTrust API",
    description="API de detection de fraude Mobile Money - Projet PFE SUP'PTIC",
    version="1.0.0",
)

# CORS ouvert : necessaire pour que le dashboard Streamlit (E3, autre domaine
# sur Streamlit Cloud) puisse appeler l'API deployee sur Render.com.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Gestion globale des erreurs -> reponses JSON propres pour E3/E4
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("Erreur non geree : %s", exc)
    return {"detail": "Erreur interne du serveur"}, 500


# ---------------------------------------------------------------------------
# Routes de base
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=schemas.HealthOut, tags=["Monitoring"])
def health():
    """Verifie que l'API repond. Utilise par E3 avant chaque appel du dashboard."""
    return {"status": "ok", "version": app.version}


@app.get("/info", response_model=schemas.InfoOut, tags=["Monitoring"])
def info():
    """Informations sur le modele actuellement charge."""
    return {
        "project": "CamerTrust",
        "model_loaded": model_service.is_real_model,
        "model_type": model_service.model_type,
        "threshold": model_service.threshold,
    }


# ---------------------------------------------------------------------------
# Authentification
# ---------------------------------------------------------------------------
@app.post("/auth/token", response_model=schemas.TokenOut, tags=["Auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Obtenir un token JWT. Utilise les identifiants de demo definis dans .env
    (DEMO_USERNAME / DEMO_PASSWORD). Necessaire pour GET /transactions et /alerts.
    """
    if not authenticate_user(form_data.username, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nom d'utilisateur ou mot de passe incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(subject=form_data.username)
    return {"access_token": token, "token_type": "bearer"}


# ---------------------------------------------------------------------------
# Prediction (endpoint central utilise par le dashboard de E3)
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=schemas.PredictionOut, tags=["Prediction"])
def predict(transaction: schemas.TransactionIn, db: Session = Depends(get_db)):
    """
    Recoit une transaction, renvoie une prediction de fraude, et sauvegarde
    le resultat en base pour alimenter /transactions et /alerts.
    """
    start = time.perf_counter()

    features = transaction.to_feature_dict()
    score, is_fraud = model_service.predict(features)

    latency_ms = (time.perf_counter() - start) * 1000

    db_transaction = models.Transaction(
        amount=transaction.amount,
        type=transaction.type,
        old_balance_org=transaction.old_balance_org,
        new_balance_org=transaction.new_balance_org,
        old_balance_dest=transaction.old_balance_dest,
        new_balance_dest=transaction.new_balance_dest,
        is_fraud=is_fraud,
        score=score,
        latency_ms=latency_ms,
    )
    db.add(db_transaction)
    db.commit()

    return {
        "is_fraud": is_fraud,
        "score": round(score, 4),
        "threshold": model_service.threshold,
        "latency_ms": round(latency_ms, 2),
    }


# ---------------------------------------------------------------------------
# Consultation des transactions (protegee par JWT)
# ---------------------------------------------------------------------------
@app.get("/transactions", response_model=list[schemas.TransactionOut], tags=["Donnees"])
def get_transactions(
    limit: int = Query(default=50, le=500),
    skip: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    user: str = Depends(get_current_user),
):
    """Liste paginee de toutes les transactions analysees, les plus recentes d'abord."""
    return (
        db.query(models.Transaction)
        .order_by(models.Transaction.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@app.get("/alerts", response_model=list[schemas.TransactionOut], tags=["Donnees"])
def get_alerts(
    limit: int = Query(default=50, le=500),
    skip: int = Query(default=0, ge=0),
    date_from: Optional[datetime] = Query(default=None, description="Filtrer depuis cette date (ISO 8601)"),
    min_amount: Optional[float] = Query(default=None, description="Montant minimum"),
    db: Session = Depends(get_db),
    user: str = Depends(get_current_user),
):
    """Transactions marquees comme fraude, filtrables par date et montant."""
    query = db.query(models.Transaction).filter(models.Transaction.is_fraud.is_(True))

    if date_from is not None:
        query = query.filter(models.Transaction.created_at >= date_from)
    if min_amount is not None:
        query = query.filter(models.Transaction.amount >= min_amount)

    return (
        query.order_by(models.Transaction.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
