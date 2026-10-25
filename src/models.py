"""Model registry: how to build each pipeline and what Optuna may tune.

Each :class:`ModelSpec` exposes

* ``suggest(trial)`` - samples a hyper-parameter dict (the search space);
* ``build(params, seed)`` - returns an unfitted, leak-free pipeline.

The class-imbalance strategy (``none`` / ``balanced`` weights / ``smote``) is
part of every search space where it applies, so Optuna chooses it like any
other hyper-parameter; :mod:`src.stages.train` also compares the strategies
explicitly at fixed hyper-parameters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .features import BASE_CATEGORICAL, ENGINEERED_CATEGORICAL
from .nn import TabularMLPClassifier
from .pipelines import make_pipeline

Params = dict[str, Any]
N_JOBS = 4


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    build: Callable[..., Any]  # (params, seed, **pipeline_kwargs) -> pipeline
    suggest: Callable[[optuna.Trial], Params] | None = None
    default_params: Params = field(default_factory=dict)
    family: str = "tree"  # "linear" | "tree" | "neural" | "ensemble"

    @property
    def tunable(self) -> bool:
        return self.suggest is not None


def _split(params: Params) -> tuple[str, Params]:
    params = dict(params)
    return params.pop("imbalance", "none"), params


# -- logistic regression ------------------------------------------------------


def _build_logreg_plain(params: Params, seed: int, **kw):
    model = LogisticRegression(penalty=None, max_iter=5000)
    return make_pipeline(model, encoding="onehot", scale=True, seed=seed, **kw)


def _suggest_logreg_enet(trial: optuna.Trial) -> Params:
    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        "imbalance": trial.suggest_categorical("imbalance", ["none", "balanced", "smote"]),
    }


def _build_logreg_enet(params: Params, seed: int, **kw):
    imbalance, p = _split(params)
    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=5000,
        class_weight="balanced" if imbalance == "balanced" else None,
        random_state=seed,
        **p,
    )
    return make_pipeline(model, encoding="onehot", scale=True, imbalance=imbalance, seed=seed, **kw)


# -- random forest ------------------------------------------------------------


def _suggest_rf(trial: optuna.Trial) -> Params:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        "imbalance": trial.suggest_categorical("imbalance", ["none", "balanced", "smote"]),
    }


def _build_rf(params: Params, seed: int, **kw):
    imbalance, p = _split(params)
    model = RandomForestClassifier(
        class_weight="balanced" if imbalance == "balanced" else None,
        random_state=seed,
        n_jobs=N_JOBS,
        **p,
    )
    return make_pipeline(model, encoding="onehot", imbalance=imbalance, seed=seed, **kw)


# -- sklearn gradient boosting -------------------------------------------------


def _suggest_gb(trial: optuna.Trial) -> Params:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 6),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
        # GradientBoostingClassifier has no class_weight argument
        "imbalance": trial.suggest_categorical("imbalance", ["none", "smote"]),
    }


def _build_gb(params: Params, seed: int, **kw):
    imbalance, p = _split(params)
    model = GradientBoostingClassifier(random_state=seed, **p)
    return make_pipeline(model, encoding="onehot", imbalance=imbalance, seed=seed, **kw)


# -- XGBoost ------------------------------------------------------------------


def _suggest_xgb(trial: optuna.Trial) -> Params:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "imbalance": trial.suggest_categorical("imbalance", ["none", "balanced", "smote"]),
    }


def _build_xgb(params: Params, seed: int, **kw):
    imbalance, p = _split(params)
    model = XGBClassifier(
        # negatives / positives in the real data (753 / 247); XGBoost has no
        # "balanced" option, so the ratio is fixed (folds are stratified)
        scale_pos_weight=3.05 if imbalance == "balanced" else 1.0,
        tree_method="hist",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=N_JOBS,
        **p,
    )
    return make_pipeline(model, encoding="onehot", imbalance=imbalance, seed=seed, **kw)


# -- LightGBM -----------------------------------------------------------------


def _suggest_lgbm(trial: optuna.Trial) -> Params:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 64, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "imbalance": trial.suggest_categorical("imbalance", ["none", "balanced", "smote"]),
    }


def _build_lgbm(params: Params, seed: int, **kw):
    imbalance, p = _split(params)
    model = LGBMClassifier(
        class_weight="balanced" if imbalance == "balanced" else None,
        subsample_freq=1,
        random_state=seed,
        n_jobs=N_JOBS,
        verbose=-1,
        **p,
    )
    return make_pipeline(model, encoding="onehot", imbalance=imbalance, seed=seed, **kw)


# -- CatBoost (native categorical handling) --------------------------------------


def _suggest_catboost(trial: optuna.Trial) -> Params:
    return {
        "iterations": trial.suggest_int("iterations", 200, 800, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 2, 7),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-2, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        # SMOTE is not defined on raw categorical columns
        "imbalance": trial.suggest_categorical("imbalance", ["none", "balanced"]),
    }


def _build_catboost(params: Params, seed: int, **kw):
    imbalance, p = _split(params)
    model = CatBoostClassifier(
        cat_features=[
            c for c in BASE_CATEGORICAL + ENGINEERED_CATEGORICAL if c not in kw.get("exclude", ())
        ],
        auto_class_weights="Balanced" if imbalance == "balanced" else None,
        random_seed=seed,
        thread_count=N_JOBS,
        verbose=False,
        allow_writing_files=False,
        **p,
    )
    return make_pipeline(model, encoding="native", seed=seed, **kw)


# -- PyTorch MLP --------------------------------------------------------------


def _suggest_mlp(trial: optuna.Trial) -> Params:
    n_layers = trial.suggest_int("n_layers", 1, 3)
    width = trial.suggest_categorical("width", [32, 64, 128, 256])
    return {
        "hidden_dims": tuple(max(16, width // (2**i)) for i in range(n_layers)),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "imbalance": trial.suggest_categorical("imbalance", ["none", "balanced"]),
    }


def _build_mlp(params: Params, seed: int, **kw):
    imbalance, p = _split(params)
    model = TabularMLPClassifier(
        class_weight="balanced" if imbalance == "balanced" else None,
        random_state=seed,
        **p,
    )
    return make_pipeline(model, encoding="native", seed=seed, **kw)


MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in [
        ModelSpec(
            "logreg", "Logistic regression (no penalty)", _build_logreg_plain, family="linear"
        ),
        ModelSpec(
            "logreg_enet",
            "Logistic regression (elastic net)",
            _build_logreg_enet,
            _suggest_logreg_enet,
            {"C": 0.1, "l1_ratio": 0.5, "imbalance": "none"},
            family="linear",
        ),
        ModelSpec(
            "random_forest",
            "Random forest",
            _build_rf,
            _suggest_rf,
            {
                "n_estimators": 500,
                "max_depth": 10,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "imbalance": "none",
            },
        ),
        ModelSpec(
            "gradient_boosting",
            "Gradient boosting (sklearn)",
            _build_gb,
            _suggest_gb,
            {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 3,
                "subsample": 0.8,
                "min_samples_leaf": 5,
                "imbalance": "none",
            },
        ),
        ModelSpec(
            "xgboost",
            "XGBoost",
            _build_xgb,
            _suggest_xgb,
            {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1.0,
                "reg_lambda": 1.0,
                "reg_alpha": 0.01,
                "imbalance": "none",
            },
        ),
        ModelSpec(
            "lightgbm",
            "LightGBM",
            _build_lgbm,
            _suggest_lgbm,
            {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 8,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "reg_alpha": 0.01,
                "imbalance": "none",
            },
        ),
        ModelSpec(
            "catboost",
            "CatBoost",
            _build_catboost,
            _suggest_catboost,
            {
                "iterations": 500,
                "learning_rate": 0.05,
                "depth": 4,
                "l2_leaf_reg": 3.0,
                "random_strength": 1.0,
                "bagging_temperature": 1.0,
                "imbalance": "none",
            },
        ),
        ModelSpec(
            "mlp",
            "MLP + embeddings (PyTorch)",
            _build_mlp,
            _suggest_mlp,
            {
                "hidden_dims": (128, 64),
                "dropout": 0.2,
                "lr": 1e-3,
                "weight_decay": 1e-5,
                "batch_size": 64,
                "imbalance": "none",
            },
            family="neural",
        ),
    ]
}

STACKING_NAME = "stacking"
STACKING_LABEL = "Stacking (top-3 + logistic meta)"


def build(name: str, params: Params, seed: int, **pipeline_kwargs):
    """Build an unfitted pipeline; ``pipeline_kwargs`` go to :func:`make_pipeline`."""
    return MODELS[name].build(params, seed, **pipeline_kwargs)
