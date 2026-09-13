"""
CamerTrust Lite - Configuration centralisee
E2 - Developpeur Backend

Toutes les variables d'environnement de l'application sont lues ici,
une seule fois, via pydantic-settings. Cela evite de disperser des
os.getenv(...) dans tout le code.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Base de donnees : SQLite en local par defaut, PostgreSQL sur Render
    database_url: str = "sqlite:///./camertrust.db"

    # Authentification JWT
    secret_key: str = "change-moi-avec-une-cle-longue-et-aleatoire"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Identifiants de demo (a adapter/supprimer en production)
    demo_username: str = "camertrust"
    demo_password: str = "changeme123"

    # Modele ML fourni par E1
    model_path: str = "data/models/pipeline_complet.pkl"
    model_info_path: str = "data/models/model_info.json"
    default_threshold: float = 0.5


@lru_cache
def get_settings() -> Settings:
    return Settings()
