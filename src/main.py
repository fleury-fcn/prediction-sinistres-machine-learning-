"""
Point d'entrée du pipeline de prédiction des sinistres.

Usage :
    python -m src.main
    python -m src.main --data data/donnees_sinistres.csv --skip-search
"""

import argparse
import logging

from . import config, data_loading, evaluation, model, preprocessing, visualization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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
    baseline_auc = evaluation.roc_auc_score(y_test, baseline_proba)
    logger.info("AUC baseline (régression logistique) : %.3f", baseline_auc)

    # 5. Random Forest (avec ou sans recherche d'hyperparamètres)
    if args.skip_search:
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
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
    y_proba = rf.predict_proba(X_test)[:, 1]

    # 8. Seuil optimal
    best_threshold = evaluation.find_optimal_threshold(y_test, y_proba)
    y_pred_default = (y_proba >= 0.5).astype(int)
    y_pred_optimal = (y_proba >= best_threshold).astype(int)

    metrics_default = evaluation.compute_metrics(y_test, y_pred_default, y_proba)
    metrics_optimal = evaluation.compute_metrics(y_test, y_pred_optimal, y_proba)

    logger.info("Métriques @ seuil 0.5   : %s", {k: round(v, 3) for k, v in metrics_default.items()})
    logger.info("Métriques @ seuil %.2f : %s", best_threshold,
                {k: round(v, 3) for k, v in metrics_optimal.items()})

    # On garde le seuil optimal pour la suite (matrice, rapport)
    y_pred = y_pred_optimal
    metrics = metrics_optimal

    # 9. Évaluation détaillée
    cm = evaluation.compute_confusion(y_test, y_pred)
    fpr, tpr, _ = evaluation.compute_roc_curve(y_test, y_proba)
    importance = evaluation.feature_importance(rf, features)
    results = evaluation.prediction_report(y_test, y_proba, y_pred)

    logger.info("Importance des variables :\n%s", importance)

    # 10. Visualisation
    visualization.plot_dashboard(
        cm=cm, fpr=fpr, tpr=tpr, auc=metrics["auc"],
        importance=importance, results=results, metrics=metrics,
        output_path=config.RESULTS_DIR / "random_forest_sinistres.png",
    )

    # 11. Sauvegardes
    results.to_csv(config.RESULTS_DIR / "predictions_random_forest.csv", index=False)
    importance.to_csv(config.RESULTS_DIR / "importance_variables.csv", index=False)
    model.save_model(rf)

    logger.info("Pipeline terminé avec succès. Résultats dans : %s", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
