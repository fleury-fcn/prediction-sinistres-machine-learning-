"""
Entraînement du modèle : split, recherche d'hyperparamètres,
validation croisée, sauvegarde.
"""

import logging

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from . import config

logger = logging.getLogger(__name__)


def split_data(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def train_baseline(X_train, y_train) -> LogisticRegression:
    """Modèle de référence simple, pour juger de l'apport du Random Forest."""

    baseline = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=config.RANDOM_STATE,
    )
    baseline.fit(X_train, y_train)
    return baseline


def tune_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Recherche d'hyperparamètres par RandomizedSearchCV avec F1 comme critère."""

    base_model = RandomForestClassifier(
        class_weight="balanced",
        bootstrap=True,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=config.PARAM_DISTRIBUTIONS,
        n_iter=config.N_ITER_SEARCH,
        scoring="f1",
        cv=cv,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    logger.info("Lancement de la recherche d'hyperparamètres (%s itérations)...",
                config.N_ITER_SEARCH)

    search.fit(X_train, y_train)

    logger.info("Meilleurs paramètres : %s", search.best_params_)
    logger.info("Meilleur F1 (CV) : %.3f", search.best_score_)

    return search.best_estimator_


def cross_validate_model(model, X, y) -> dict:
    """Validation croisée sur plusieurs métriques pour une estimation robuste."""

    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    scores = {}
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        result = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
        scores[metric] = (result.mean(), result.std())

    return scores


def save_model(model, path=None) -> None:
    path = path or config.MODEL_PATH
    joblib.dump(model, path)
    logger.info("Modèle sauvegardé : %s", path)


def load_model(path=None):
    path = path or config.MODEL_PATH
    return joblib.load(path)
