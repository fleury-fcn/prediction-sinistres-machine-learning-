"""
Nettoyage, feature engineering et préparation X / y.
"""

import logging

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes du CSV source vers les noms internes du pipeline.

    Permet de brancher un nouveau dataset en ne modifiant que
    config.RENAME_MAP, sans toucher au reste du preprocessing.
    """

    applicable = {k: v for k, v in config.RENAME_MAP.items() if k in df.columns}

    if applicable:
        logger.info("Renommage des colonnes : %s", applicable)
        df = df.rename(columns=applicable)

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Crée la variable cible binaire, selon config.TARGET_MODE.

    - "amount_threshold" : 1 si le montant réglé est > 0 (portefeuille avec
      des contrats sans sinistre).
    - "label" : 1 si la colonne cible correspond à l'étiquette positive
      (ex. fraude déclarée "Y"), pour des jeux de données composés
      uniquement de sinistres déjà survenus.
    """

    if config.TARGET_MODE == "amount_threshold":
        df[config.TARGET_NAME] = (
            pd.to_numeric(df[config.TARGET_COLUMN], errors="coerce")
            .fillna(0)
            .gt(0)
            .astype(int)
        )

    elif config.TARGET_MODE == "label":
        df[config.TARGET_NAME] = (
            df[config.TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .eq(config.TARGET_POSITIVE_LABEL)
            .astype(int)
        )

    else:
        raise ValueError(f"TARGET_MODE inconnu : {config.TARGET_MODE}")

    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les colonnes de dates, en loggant les échecs de parsing."""

    for col in config.DATE_COLUMNS:

        if col not in df.columns:
            continue

        before_na = df[col].isna().sum()
        df[col] = pd.to_datetime(df[col], errors="coerce")
        after_na = df[col].isna().sum()

        new_failures = after_na - before_na
        if new_failures > 0:
            logger.warning(
                "%s : %s valeurs n'ont pas pu être converties en date",
                col, new_failures,
            )

    return df


def compute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule l'âge de l'assuré à la date d'effet, avec bornage des aberrations.

    Si AGE_ASSURE existe déjà (ex. fourni directement par le dataset source,
    comme dans insurance_claims.csv), cette étape est ignorée.
    """

    if "AGE_ASSURE" in df.columns:
        return df

    if "DATE_NAISSANCE_CLIENT" not in df.columns or "DATE_EFFET" not in df.columns:
        return df

    age = (df["DATE_EFFET"] - df["DATE_NAISSANCE_CLIENT"]).dt.days / 365.25

    n_invalid = ((age < config.AGE_MIN) | (age > config.AGE_MAX)).sum()
    if n_invalid > 0:
        logger.warning(
            "AGE_ASSURE : %s valeurs hors de [%s, %s] mises à NaN",
            n_invalid, config.AGE_MIN, config.AGE_MAX,
        )
        age = age.where((age >= config.AGE_MIN) & (age <= config.AGE_MAX))

    df["AGE_ASSURE"] = age

    return df


def select_features(df: pd.DataFrame) -> list[str]:
    """Ne garde que les variables candidates réellement présentes."""

    features = [col for col in config.CANDIDATE_FEATURES if col in df.columns]

    missing = set(config.CANDIDATE_FEATURES) - set(features)
    if missing:
        logger.warning("Variables candidates absentes du dataset : %s", missing)

    if not features:
        raise ValueError("Aucune variable explicative disponible après sélection.")

    return features


def build_feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Nettoie et convertit les variables explicatives en numérique."""

    X = df[features].copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Anomalie connue de certains jeux de données (ex. insurance_claims.csv) :
    # des montants de capital négatifs qui n'ont pas de sens métier.
    if "CAPITAL_ASSURE" in X.columns:
        n_negative = (X["CAPITAL_ASSURE"] < 0).sum()
        if n_negative > 0:
            logger.warning(
                "CAPITAL_ASSURE : %s valeurs négatives ramenées à 0", n_negative
            )
            X["CAPITAL_ASSURE"] = X["CAPITAL_ASSURE"].clip(lower=0)

    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    X = X.fillna(0)

    return X


def run_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Enchaîne toutes les étapes de préparation et retourne X, y, features."""

    df = rename_columns(df)
    df = create_target(df)
    df = parse_dates(df)
    df = compute_age(df)

    features = select_features(df)
    X = build_feature_matrix(df, features)
    y = df[config.TARGET_NAME]

    logger.info("Distribution de la cible :\n%s", y.value_counts(normalize=True))

    return X, y, features
