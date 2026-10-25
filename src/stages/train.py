"""Training stage on the real data (training split only; the test split is not loaded
into any fit here).

1. Optuna tuning of every model with the same trial budget and CV folds.
2. Repeated stratified CV with the tuned parameters -> fold scores + OOF probabilities.
3. Stacking ensemble of the three best models (by CV PR-AUC).
4. Cost-optimal decision threshold per model, chosen on OOF probabilities.
5. Refit on the full training split and save the fitted pipelines.
6. Side experiments: imbalance strategies and feature ablations (CV only).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .. import config, evaluation, tracking, training
from ..config import Settings
from ..features import CLAIM_AMOUNT_FEATURES
from ..models import MODELS, STACKING_LABEL, STACKING_NAME, build
from .common import (
    MODEL_ARTIFACTS,
    TRAIN_ARTIFACTS,
    cost_model,
    real_split,
    restore_params,
    write_json,
)

logger = logging.getLogger(__name__)

IMBALANCE_MODELS = ["logreg_enet", "random_forest", "xgboost", "lightgbm"]
ABLATIONS: dict[str, dict[str, Any]] = {
    "all features (default)": {},
    "+ total_claim_amount (redundant)": {"drop_total_claim_amount": False},
    "- all claim-amount features": {"exclude": tuple(CLAIM_AMOUNT_FEATURES)},
    "- insured_hobbies": {"exclude": ("insured_hobbies",)},
    "- incident_severity": {"exclude": ("incident_severity",)},
}


def _stacking_builder(top: list[str], params: dict[str, dict], seed: int):
    def make() -> StackingClassifier:
        return StackingClassifier(
            estimators=[(name, build(name, params[name], seed)) for name in top],
            final_estimator=LogisticRegression(max_iter=2000),
            cv=StratifiedKFold(5, shuffle=True, random_state=seed),
            stack_method="predict_proba",
        )

    return make


def run(settings: Settings, models: list[str] | None = None) -> dict[str, Any]:
    config.ensure_dirs()
    TRAIN_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MODEL_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    seed = settings.seed
    costs = cost_model(settings)
    cv_cfg, tune_cfg = settings["cv"], settings["tuning"]
    X_train, _, y_train, _ = real_split(settings)
    names = models or list(MODELS)

    best_params: dict[str, dict] = {}
    tuning_seconds: dict[str, float] = {}
    for name in names:
        spec = MODELS[name]
        start = time.perf_counter()
        if spec.tunable:
            logger.info("Tuning %s (%d trials)...", name, tune_cfg["n_trials"])
            best_params[name], trials = training.tune(
                spec,
                X_train,
                y_train,
                tune_cfg["n_trials"],
                cv_cfg["tuning_folds"],
                seed,
                tune_cfg["pruning"],
            )
            trials.to_csv(TRAIN_ARTIFACTS / f"optuna_trials_{name}.csv", index=False)
        else:
            best_params[name] = dict(spec.default_params)
        tuning_seconds[name] = time.perf_counter() - start
    write_json(best_params, config.METRICS_DIR / "real_best_params.json")

    builders = {
        name: (lambda n=name: build(n, restore_params(best_params[n]), seed)) for name in names
    }
    fold_scores, oof, fit_seconds = [], {}, {}
    for name, make in builders.items():
        logger.info("Repeated CV: %s", name)
        res = training.repeated_cv(
            make, X_train, y_train, cv_cfg["eval_folds"], cv_cfg["eval_repeats"], seed
        )
        fold_scores.append(res.fold_scores.assign(model=name))
        oof[name], fit_seconds[name] = res.oof_proba, res.fit_seconds

    labels = {name: MODELS[name].label for name in names}
    cv_means = pd.concat(fold_scores).groupby("model")["pr_auc"].mean()
    top3 = [m for m in cv_means.sort_values(ascending=False).index if m != "logreg"][:3]
    if len(top3) >= 2:
        logger.info("Stacking top models: %s", top3)
        builders[STACKING_NAME] = _stacking_builder(
            top3, {m: restore_params(best_params[m]) for m in top3}, seed
        )
        res = training.repeated_cv(
            builders[STACKING_NAME],
            X_train,
            y_train,
            cv_cfg["eval_folds"],
            cv_cfg["eval_repeats"],
            seed,
        )
        fold_scores.append(res.fold_scores.assign(model=STACKING_NAME))
        oof[STACKING_NAME], fit_seconds[STACKING_NAME] = res.oof_proba, res.fit_seconds
        labels[STACKING_NAME] = STACKING_LABEL
        tuning_seconds[STACKING_NAME] = 0.0

    folds = pd.concat(fold_scores, ignore_index=True)
    folds.to_csv(config.METRICS_DIR / "real_cv_fold_scores.csv", index=False)
    pd.DataFrame(oof, index=X_train.index).assign(y=y_train).to_parquet(
        TRAIN_ARTIFACTS / "oof_proba.parquet"
    )

    thresholds = {
        name: evaluation.cost_optimal_threshold(y_train.to_numpy(), proba, costs)
        for name, proba in oof.items()
    }

    summary_rows = []
    for name, make in builders.items():
        logger.info("Refit on full training split: %s", name)
        model = training.fit_full(make, X_train, y_train)
        joblib.dump(model, MODEL_ARTIFACTS / f"{name}.joblib")
        scores = folds.loc[folds.model == name, ["pr_auc", "roc_auc", "brier"]]
        cv = evaluation.cv_summary(scores)
        row = {
            "model": name,
            "label": labels[name],
            "cv_pr_auc_mean": cv.loc["pr_auc", "mean"],
            "cv_pr_auc_sd": cv.loc["pr_auc", "sd"],
            "cv_roc_auc_mean": cv.loc["roc_auc", "mean"],
            "cv_roc_auc_sd": cv.loc["roc_auc", "sd"],
            "cv_brier_mean": cv.loc["brier", "mean"],
            "oof_cost_threshold": thresholds[name],
            "fit_seconds": fit_seconds[name],
            "tuning_seconds": tuning_seconds[name],
        }
        summary_rows.append(row)
        tracking.log_run(
            "fraud-real-cv",
            name,
            best_params.get(name, {"stack": top3}),
            {k: v for k, v in row.items() if isinstance(v, float)},
        )
    summary = pd.DataFrame(summary_rows).sort_values("cv_pr_auc_mean", ascending=False)
    summary.to_csv(config.METRICS_DIR / "real_cv_summary.csv", index=False)
    write_json(
        {"thresholds": thresholds, "labels": labels, "stacking_members": top3},
        TRAIN_ARTIFACTS / "train_meta.json",
    )

    imbalance = imbalance_experiment(
        settings, X_train, y_train, best_params, [m for m in IMBALANCE_MODELS if m in names]
    )
    ablation = ablation_experiment(settings, X_train, y_train, best_params, summary)
    logger.info("CV summary:\n%s", summary[["model", "cv_pr_auc_mean", "cv_roc_auc_mean"]])
    return {"summary": summary, "imbalance": imbalance, "ablation": ablation}


def imbalance_experiment(
    settings: Settings,
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict[str, dict],
    names: list[str],
) -> pd.DataFrame:
    """Same tuned hyper-parameters, only the imbalance strategy changes."""
    cv_cfg, rows = settings["cv"], []
    for name in names:
        for strategy in ("none", "balanced", "smote"):
            params = {**restore_params(best_params[name]), "imbalance": strategy}
            res = training.repeated_cv(
                lambda p=params, n=name: build(n, p, settings.seed),
                X,
                y,
                cv_cfg["eval_folds"],
                cv_cfg["eval_repeats"],
                settings.seed,
            )
            means = res.fold_scores[["pr_auc", "roc_auc", "brier"]].mean()
            sds = res.fold_scores[["pr_auc", "roc_auc", "brier"]].std()
            rows.append(
                {
                    "model": name,
                    "strategy": strategy,
                    "tuned_choice": best_params[name].get("imbalance") == strategy,
                    **{f"{k}_mean": v for k, v in means.items()},
                    **{f"{k}_sd": v for k, v in sds.items()},
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(config.METRICS_DIR / "real_imbalance_strategies.csv", index=False)
    return out


def ablation_experiment(
    settings: Settings,
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict[str, dict],
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Feature-group ablations on the best single tree model and the linear baseline."""
    cv_cfg, rows = settings["cv"], []
    tree_like = summary[
        summary.model.isin(
            ["catboost", "lightgbm", "xgboost", "gradient_boosting", "random_forest"]
        )
    ]
    targets = [tree_like.iloc[0]["model"]] if len(tree_like) else []
    if "logreg_enet" in best_params:
        targets.append("logreg_enet")
    for name in targets:
        params = restore_params(best_params[name])
        for label, kw in ABLATIONS.items():
            res = training.repeated_cv(
                lambda k=kw, n=name, p=params: build(n, p, settings.seed, **k),
                X,
                y,
                cv_cfg["eval_folds"],
                cv_cfg["eval_repeats"],
                settings.seed,
            )
            fs = res.fold_scores
            rows.append(
                {
                    "model": name,
                    "variant": label,
                    "pr_auc_mean": fs.pr_auc.mean(),
                    "pr_auc_sd": fs.pr_auc.std(),
                    "roc_auc_mean": fs.roc_auc.mean(),
                    "roc_auc_sd": fs.roc_auc.std(),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(config.METRICS_DIR / "real_feature_ablation.csv", index=False)
    return out


def load_fitted(name: str):
    return joblib.load(MODEL_ARTIFACTS / f"{name}.joblib")
