"""
Chargement et nettoyage initial des données brutes.
"""

import logging

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def load_raw_data(path=None) -> pd.DataFrame:
    """Charge le CSV source et normalise les noms de colonnes."""

    path = path or config.DATA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}. "
            "Vérifie que le CSV est bien dans data/."
        )

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.upper()

    logger.info("Données chargées : %s lignes, %s colonnes", *df.shape)

    return df


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Vérifie que les colonnes nécessaires sont présentes."""

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le CSV : {missing}. "
            f"Colonnes disponibles : {df.columns.tolist()}"
        )
