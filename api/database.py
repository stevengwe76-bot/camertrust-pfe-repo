"""
CamerTrust Lite - Connexion base de donnees
E2 - Developpeur Backend

Fonctionne avec SQLite en local (aucune installation requise) et avec
PostgreSQL sur Render.com (il suffit de changer DATABASE_URL).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from api.config import get_settings

settings = get_settings()

connect_args = {}
if settings.database_url.startswith("sqlite"):
    # Necessaire uniquement pour SQLite avec FastAPI (multi-thread)
    connect_args = {"check_same_thread": False}

engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependance FastAPI : fournit une session DB et la ferme apres la requete."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
