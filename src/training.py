"""Hyper-parameter tuning (Optuna) and repeated cross-validation."""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from .models import ModelSpec

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

Builder = Callable[[], Any]


def _quiet_fit(model, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(X, y)


def tune(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int,
    n_folds: int,
    seed: int,
    pruning: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Maximise mean PR-AUC over a stratified K-fold of the training data.

    The folds are identical for every model (same seed) and every model gets
    the same number of trials, so the comparison is budget-fair. TPE sampling
    is seeded; median pruning stops a trial after a fold when it is clearly
    below the running median.
    """
    folds = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, y))

    def objective(trial: optuna.Trial) -> float:
        params = spec.suggest(trial)
        scores = []
        for i, (tr, va) in enumerate(folds):
            model = _quiet_fit(spec.build(params, seed), X.iloc[tr], y.iloc[tr])
            proba = model.predict_proba(X.iloc[va])[:, 1]
            scores.append(average_precision_score(y.iloc[va], proba))
            trial.report(float(np.mean(scores)), i)
            if pruning and trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(10, n_trials)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        study_name=spec.name,
    )
    # The default parameters are evaluated first so tuning can only improve on them.
    study.enqueue_trial(_as_trial_params(spec))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = spec.suggest(optuna.trial.FixedTrial(study.best_params))
    logger.info("%s: best CV PR-AUC %.4f with %s", spec.name, study.best_value, best)
    return best, study.trials_dataframe()


def _as_trial_params(spec: ModelSpec) -> dict[str, Any]:
    params = dict(spec.default_params)
    if "hidden_dims" in params:  # MLP: map back to the (n_layers, width) search space
        dims = params.pop("hidden_dims")
        params.update(n_layers=len(dims), width=dims[0])
    return params


@dataclass
class CVResult:
    fold_scores: pd.DataFrame  # one row per (repeat, fold)
    oof_proba: np.ndarray  # mean out-of-fold probability across repeats
    fit_seconds: float


def _fit_fold(build: Builder, X, y, tr, va) -> tuple[np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    model = _quiet_fit(build(), X.iloc[tr], y.iloc[tr])
    elapsed = time.perf_counter() - start
    return va, model.predict_proba(X.iloc[va])[:, 1], elapsed


def repeated_cv(
    build: Builder,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int,
    n_repeats: int,
    seed: int,
    n_jobs: int = 3,
) -> CVResult:
    """Repeated stratified K-fold: fold-level scores and averaged OOF probabilities.

    The OOF probabilities are what the decision threshold is tuned on, so the
    threshold never sees the test set.
    """
    splitter = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
    splits = list(splitter.split(X, y))
    outputs = Parallel(n_jobs=n_jobs)(delayed(_fit_fold)(build, X, y, tr, va) for tr, va in splits)

    oof = np.zeros((n_repeats, len(y)))
    rows, total_time = [], 0.0
    y_arr = y.to_numpy()
    for i, (va, proba, elapsed) in enumerate(outputs):
        oof[i // n_folds, va] = proba
        total_time += elapsed
        rows.append(
            {
                "repeat": i // n_folds,
                "fold": i % n_folds,
                "pr_auc": average_precision_score(y_arr[va], proba),
                "roc_auc": roc_auc_score(y_arr[va], proba),
                "brier": brier_score_loss(y_arr[va], proba),
            }
        )
    return CVResult(pd.DataFrame(rows), oof.mean(axis=0), total_time / len(splits))


def fit_full(build: Builder, X: pd.DataFrame, y: pd.Series):
    return _quiet_fit(build(), X, y)
