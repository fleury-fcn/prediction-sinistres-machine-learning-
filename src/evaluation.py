"""
Évaluation du modèle : métriques, seuil optimal, importance des variables.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def find_optimal_threshold(y_true, y_proba) -> float:
    """
    Cherche le seuil de décision qui maximise le F1-score,
    plutôt que d'utiliser 0.5 par défaut. Pertinent en cas
    de classes déséquilibrées (typique en sinistralité).
    """

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    f1_scores = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0,
    )

    best_idx = np.argmax(f1_scores[:-1])  # dernier point n'a pas de seuil associé
    best_threshold = thresholds[best_idx]

    logger.info(
        "Seuil optimal (max F1) : %.3f (F1 = %.3f, vs F1 @0.5)",
        best_threshold, f1_scores[best_idx],
    )

    return float(best_threshold)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba),
    }


def compute_confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def compute_roc_curve(y_true, y_proba):
    return roc_curve(y_true, y_proba)


def feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    importance = pd.DataFrame({
        "Variable": feature_names,
        "Importance": model.feature_importances_,
    })
    return importance.sort_values("Importance", ascending=False)


def linear_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    coefficients = np.abs(model.coef_).ravel()
    importance = pd.DataFrame({
        "Variable": feature_names,
        "Importance": coefficients,
    })
    return importance.sort_values("Importance", ascending=False)


def prediction_report(y_true, y_proba, y_pred) -> pd.DataFrame:
    results = pd.DataFrame({
        "Valeur_reelle": y_true.values if hasattr(y_true, "values") else y_true,
        "Probabilite_predite": y_proba,
        "Prediction": y_pred,
    })
    results["Erreur"] = results["Valeur_reelle"] - results["Probabilite_predite"]

    r2 = r2_score(results["Valeur_reelle"], results["Probabilite_predite"])
    logger.info("R² sur les probabilités prédites : %.3f", r2)

    return results
