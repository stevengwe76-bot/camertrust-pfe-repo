"""
CamerTrust Lite - Modeles de base de donnees (ORM)
E2 - Developpeur Backend
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, String

from api.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Transaction(Base):
    """Une transaction analysee par le modele anti-fraude."""

    __tablename__ = "transactions"

    id = Column(String, primary_key=True, default=_uuid)
    created_at = Column(DateTime(timezone=True), default=_utcnow, index=True)

    # Champs correspondant aux features attendues par le pipeline de E1
    amount = Column(Float, nullable=False)
    type = Column(String, nullable=False)  # TRANSFER, CASH_OUT, PAYMENT, ...
    old_balance_org = Column(Float, nullable=True)
    new_balance_org = Column(Float, nullable=True)
    old_balance_dest = Column(Float, nullable=True)
    new_balance_dest = Column(Float, nullable=True)

    # Resultat de la prediction
    is_fraud = Column(Boolean, nullable=False, default=False, index=True)
    score = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=True)
