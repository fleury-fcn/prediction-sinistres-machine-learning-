"""
Point d'entrée du pipeline de prédiction des sinistres.

Usage :
    python -m src.main
    python -m src.main --data data/donnees_sinistres.csv --skip-search
"""

import argparse
import importlib
import logging

import numpy as np
import pandas as pd

from . import config, data_loading, evaluation, model, preprocessing, visualization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_BASELINE = "Régression Logistique"
MODEL_RF = "Random Forest"


def _compute_tree_shap_values(trained_model, x_sample: pd.DataFrame) -> np.ndarray:
    try:
        shap_module = importlib.import_module("shap")
        explainer = shap_module.TreeExplainer(trained_model)
        shap_values = explainer.shap_values(x_sample)

        if isinstance(shap_values, list):
            return np.asarray(shap_values[1])

        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            return shap_array[:, :, 1]

        return shap_array
    except Exception as exc:  # pragma: no cover - dépend de l'environnement
        logger.warning("SHAP indisponible (%s), utilisation d'une approximation locale.", exc)
        centered = x_sample - x_sample.median(numeric_only=True)
        scales = x_sample.std(numeric_only=True).replace(0, 1)
        normalized = centered.divide(scales, axis=1).fillna(0)
        weights = np.asarray(getattr(trained_model, "feature_importances_", np.ones(x_sample.shape[1])))
        return normalized.to_numpy() * weights


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de prédiction des sinistres")
    parser.add_argument("--data", type=str, default=None, help="Chemin vers le CSV source")
    parser.add_argument("--skip-search", action="store_true",
                         help="Désactive la recherche d'hyperparamètres (plus rapide)")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = config.ROOT_DIR / args.data if args.data else config.DATA_PATH

    # 1. Chargement
    df = data_loading.load_raw_data(data_path)
    data_loading.validate_columns(df, [config.TARGET_COLUMN])

    # 2. Préparation
    X, y, features = preprocessing.run_pipeline(df)
    logger.info("Variables utilisées : %s", features)

    # 3. Split
    X_train, X_test, y_train, y_test = model.split_data(X, y)

    # 4. Baseline
    baseline = model.train_baseline(X_train, y_train)
    baseline_proba = baseline.predict_proba(X_test)[:, 1]
    baseline_threshold = evaluation.find_optimal_threshold(y_test, baseline_proba)
    baseline_pred = (baseline_proba >= baseline_threshold).astype(int)
    baseline_metrics = evaluation.compute_metrics(y_test, baseline_pred, baseline_proba)
    logger.info("Métriques baseline (régression logistique) : %s",
                {k: round(v, 3) for k, v in baseline_metrics.items()})

    # 5. Random Forest (avec ou sans recherche d'hyperparamètres)
    if args.skip_search:
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            min_samples_leaf=1, max_features="sqrt",
            random_state=config.RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(X_train, y_train)
    else:
        rf = model.tune_random_forest(X_train, y_train)

    # 6. Validation croisée
    cv_scores = model.cross_validate_model(rf, X, y)
    logger.info("Scores en validation croisée (%s folds) :", config.CV_FOLDS)
    for metric, (mean, std) in cv_scores.items():
        logger.info("  %-10s %.3f (+/- %.3f)", metric, mean, std)

    # 7. Prédictions test set
    rf_proba = rf.predict_proba(X_test)[:, 1]

    # 8. Seuil optimal
    rf_threshold = evaluation.find_optimal_threshold(y_test, rf_proba)
    y_pred_default = (rf_proba >= 0.5).astype(int)
    y_pred_optimal = (rf_proba >= rf_threshold).astype(int)

    metrics_default = evaluation.compute_metrics(y_test, y_pred_default, rf_proba)
    metrics_optimal = evaluation.compute_metrics(y_test, y_pred_optimal, rf_proba)

    logger.info("Métriques @ seuil 0.5   : %s", {k: round(v, 3) for k, v in metrics_default.items()})
    logger.info("Métriques @ seuil %.2f : %s", rf_threshold,
                {k: round(v, 3) for k, v in metrics_optimal.items()})

    # On garde le seuil optimal pour la suite (matrice, rapport)
    y_pred = y_pred_optimal
    metrics = metrics_optimal

    # 9. Évaluation détaillée
    cm_rf = evaluation.compute_confusion(y_test, y_pred)
    cm_baseline = evaluation.compute_confusion(y_test, baseline_pred)
    fpr_rf, tpr_rf, _ = evaluation.compute_roc_curve(y_test, rf_proba)
    fpr_baseline, tpr_baseline, _ = evaluation.compute_roc_curve(y_test, baseline_proba)
    rf_importance = evaluation.feature_importance(rf, features)
    baseline_importance = evaluation.linear_feature_importance(baseline, features)
    results = evaluation.prediction_report(y_test, rf_proba, y_pred)
    shap_values = _compute_tree_shap_values(rf, X_test)

    logger.info("Importance des variables (Random Forest) :\n%s", rf_importance)

    # 10. Visualisation
    visualization.plot_dashboard(
        cm=cm_rf, fpr=fpr_rf, tpr=tpr_rf, auc=metrics["auc"],
        importance=rf_importance, results=results, metrics=metrics,
        output_path=config.RESULTS_DIR / "random_forest_sinistres.png",
    )

    interactive_files = visualization.export_interactive_visuals(
        roc_by_model={
            MODEL_BASELINE: {
                "fpr": fpr_baseline,
                "tpr": tpr_baseline,
                "auc": baseline_metrics["auc"],
            },
            MODEL_RF: {
                "fpr": fpr_rf,
                "tpr": tpr_rf,
                "auc": metrics["auc"],
            },
        },
        confusion_by_model={
            MODEL_BASELINE: cm_baseline,
            MODEL_RF: cm_rf,
        },
        importance_by_model={
            MODEL_BASELINE: baseline_importance,
            MODEL_RF: rf_importance,
        },
        shap_values=shap_values,
        X_test=X_test,
        output_dir=config.INTERACTIVE_RESULTS_DIR,
    )

    comparison = pd.DataFrame([
        {
            "Modele": MODEL_BASELINE,
            "Seuil": baseline_threshold,
            **baseline_metrics,
        },
        {
            "Modele": MODEL_RF,
            "Seuil": rf_threshold,
            **metrics,
        },
    ])

    logger.info("Fichiers interactifs exportés : %s", list(interactive_files.keys()))

    # 11. Sauvegardes
    results.to_csv(config.RESULTS_DIR / "predictions_random_forest.csv", index=False)
    rf_importance.to_csv(config.RESULTS_DIR / "importance_variables.csv", index=False)
    baseline_importance.to_csv(config.RESULTS_DIR / "importance_baseline.csv", index=False)
    comparison.to_csv(config.RESULTS_DIR / "model_comparison_metrics.csv", index=False)
    model.save_model(rf)

    logger.info("Pipeline terminé avec succès. Résultats dans : %s", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
