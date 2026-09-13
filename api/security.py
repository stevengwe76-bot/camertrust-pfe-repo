"""
CamerTrust Lite - Authentification JWT
E2 - Developpeur Backend

Authentification simple par identifiants de demo (variables d'environnement)
pour proteger les endpoints sensibles. Suffisant pour un prototype academique
(voir DEPLOYMENT.md pour les limites et pistes d'amelioration en production).
"""

from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from api.config import get_settings

settings = get_settings()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)


def authenticate_user(username: str, password: str) -> bool:
    """Verifie les identifiants contre les identifiants de demo configures."""
    return username == settings.demo_username and password == settings.demo_password


def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """
    Dependance FastAPI a utiliser sur les routes protegees :
        @app.get("/route")
        def route(user: str = Depends(get_current_user)): ...
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Identifiants invalides ou token expire",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if token is None:
        raise credentials_exception
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return username
    except JWTError:
        raise credentials_exception
