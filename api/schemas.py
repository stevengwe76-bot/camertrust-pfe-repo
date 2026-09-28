"""
CamerTrust Lite - Schemas Pydantic
E2 - Developpeur Backend

Definissent le format exact des donnees entrantes/sortantes de l'API.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

AlertStatus = Literal["OPEN", "UNDER_REVIEW", "RESOLVED", "DISMISSED"]


class TransactionIn(BaseModel):
    """Donnees envoyees par le dashboard (E3) pour analyser une transaction."""

    amount: float = Field(..., gt=0, description="Montant de la transaction en FCFA")
    type: str = Field(..., description="TRANSFER, CASH_OUT, PAYMENT, CASH_IN, DEBIT")
    old_balance_org: Optional[float] = Field(default=0.0, description="Solde emetteur avant")
    new_balance_org: Optional[float] = Field(default=0.0, description="Solde emetteur apres")
    old_balance_dest: Optional[float] = Field(default=0.0, description="Solde destinataire avant")
    new_balance_dest: Optional[float] = Field(default=0.0, description="Solde destinataire apres")

    def to_feature_dict(self) -> dict:
        """Transforme la requete en dictionnaire de features pour le modele ML."""
        return {
            "amount": self.amount,
            "type": self.type,
            "oldbalanceOrg": self.old_balance_org,
            "newbalanceOrig": self.new_balance_org,
            "oldbalanceDest": self.old_balance_dest,
            "newbalanceDest": self.new_balance_dest,
        }


class PredictionOut(BaseModel):
    """Reponse de l'endpoint /predict."""

    is_fraud: bool
    score: float
    threshold: float
    latency_ms: float


class TransactionOut(BaseModel):
    """Transaction telle que renvoyee par GET /transactions et GET /alerts."""

    id: str
    created_at: datetime
    amount: float
    type: str
    is_fraud: bool
    score: float
    latency_ms: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class AlertOut(TransactionOut):
    """Alerte = transaction classée fraude + suivi de l'analyste."""

    status: AlertStatus = "OPEN"
    resolved_by: Optional[str] = None
    resolution_note: Optional[str] = None
    reviewed_at: Optional[datetime] = None


class AlertResolveIn(BaseModel):
    """Corps de PATCH /alerts/{id}/resolve envoyé par le dashboard."""

    status: AlertStatus = Field(..., description="RESOLVED = fraude confirmée, DISMISSED = faux positif")
    resolved_by: str = Field(..., min_length=1, max_length=100)
    resolution_note: Optional[str] = Field(default="", max_length=1000)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class HealthOut(BaseModel):
    status: str
    version: str


class InfoOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    project: str
    model_loaded: bool
    model_type: Optional[str] = None
    threshold: Optional[float] = None
    model_error: Optional[str] = None
