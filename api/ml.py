"""
CamerTrust Lite - Chargement et inference du modele ML
E2 - Developpeur Backend

Charge le pipeline_complet.pkl produit par E1 (voir data/models/).
Tant que E1 n'a pas encore livre son fichier, un modele factice (dummy)
prend le relais automatiquement pour ne pas bloquer le developpement de
l'API et du dashboard (E3) -> voir points de synchronisation du plan
(Fin S4 / Fin S7).
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from api.config import get_settings

logger = logging.getLogger("camertrust.ml")
settings = get_settings()


class DummyModel:
    """
    Modele de secours utilise uniquement si pipeline_complet.pkl n'existe
    pas encore. Renvoie un score pseudo-aleatoire mais deterministe base
    sur le montant, pour permettre de tester l'API et le dashboard avant
    que E1 ait livre le vrai modele. A ne JAMAIS utiliser pour la demo finale.
    """

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        amounts = X["amount"].to_numpy(dtype=float)
        # Score croissant avec le montant, borne entre 0 et 1 (placeholder uniquement)
        scores = np.clip(amounts / (amounts.max() + 1e-6) * 0.6, 0, 1)
        return np.column_stack([1 - scores, scores])


class ModelService:
    """Encapsule le chargement du pipeline et l'inference pour l'endpoint /predict."""

    def __init__(self) -> None:
        self.pipeline = None
        self.model_type: Optional[str] = None
        self.threshold: float = settings.default_threshold
        self.features: Optional[list] = None
        self._load()

    def _load(self) -> None:
        model_path = Path(settings.model_path)
        info_path = Path(settings.model_info_path)

        if model_path.exists():
            self.pipeline = joblib.load(model_path)
            self.model_type = type(self.pipeline).__name__
            logger.info("Pipeline charge depuis %s", model_path)
        else:
            self.pipeline = DummyModel()
            self.model_type = "DummyModel (placeholder - en attente du fichier de E1)"
            logger.warning(
                "pipeline_complet.pkl introuvable (%s) -> utilisation du DummyModel. "
                "Demande a E1 de deposer le fichier dans data/models/.",
                model_path,
            )

        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            self.threshold = float(info.get("threshold", settings.default_threshold))
            self.features = info.get("features")
        else:
            logger.warning(
                "model_info.json introuvable (%s) -> seuil par defaut = %s",
                info_path,
                self.threshold,
            )

    @property
    def is_real_model(self) -> bool:
        return not isinstance(self.pipeline, DummyModel)

    @property
    def onehot_type_columns(self) -> list[str]:
        """
        Colonnes one-hot 'type_XXX' attendues par le pipeline, deduites de
        model_info.json (self.features). Vide si le modele n'utilise pas
        cet encodage (ex : DummyModel, ou pipeline different).
        """
        if not self.features:
            return []
        return [col for col in self.features if col.startswith("type_")]

    def _encode_categorical(self, row: pd.DataFrame) -> pd.DataFrame:
        """
        Convertit la colonne texte 'type' (ex: "TRANSFER") en colonnes
        one-hot (type_TRANSFER=1, type_CASH_OUT=0, ...), exactement comme
        le fait prepare_data_camertrust.py (E1, etape S3) sur les donnees
        d'entrainement. Sans cette etape, le pipeline recevrait toujours
        des colonnes type_* a zero et ignorerait le type de transaction.
        """
        if "type" not in row.columns:
            return row

        tx_type = row.at[0, "type"]
        row = row.drop(columns=["type"])

        onehot_cols = self.onehot_type_columns
        if not onehot_cols:
            # model_info.json absent ou pipeline sans encodage one-hot connu :
            # on ne peut pas savoir quelles colonnes generer, on laisse tel quel.
            return row

        expected_col = f"type_{tx_type}"
        if expected_col not in onehot_cols:
            logger.warning(
                "Type de transaction inconnu du modele : '%s' (colonnes connues : %s). "
                "Traite comme categorie de reference (toutes les colonnes type_* a 0).",
                tx_type,
                onehot_cols,
            )

        for col in onehot_cols:
            row[col] = 1 if col == expected_col else 0

        return row

    def predict(self, feature_dict: dict) -> tuple[float, bool]:
        """Retourne (score_fraude, is_fraud) pour une transaction donnee."""
        row = pd.DataFrame([feature_dict])
        row = self._encode_categorical(row)

        # Si le pipeline attend des colonnes precises (model_info.json de E1),
        # on aligne les colonnes dans le bon ordre pour eviter les erreurs.
        if self.features:
            for col in self.features:
                if col not in row.columns:
                    row[col] = 0
            row = row[self.features]

        proba = self.pipeline.predict_proba(row)[0][1]
        return float(proba), bool(proba >= self.threshold)


# Instance unique chargee au demarrage de l'application (voir main.py)
model_service = ModelService()
