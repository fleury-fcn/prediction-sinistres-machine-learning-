"""Model explanations: SHAP (global + local) and permutation importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from . import config


def shap_explanation(
    pipeline: Pipeline, X: pd.DataFrame
) -> tuple[shap.Explanation, dict, pd.DataFrame]:
    """SHAP values (log-odds scale) of a fitted tree pipeline on ``X``.

    Returns the explanation over model inputs, a mapping
    ``model input -> source feature`` used to aggregate one-hot columns, and the
    human-readable feature values (for local explanations).
    """
    features = pipeline.named_steps["features"]
    model = pipeline.named_steps["model"]
    F = features.transform(X)

    if type(model).__name__ == "CatBoostClassifier":
        from catboost import Pool

        cat_cols = features.categorical_features_
        raw = model.get_feature_importance(Pool(F, cat_features=cat_cols), type="ShapValues")
        values, base = raw[:, :-1], raw[:, -1]
        display = F.copy()
        display[cat_cols] = np.nan  # no meaningful colour scale for categories: grey dots
        exp = shap.Explanation(
            values,
            base_values=base,
            data=display.to_numpy(dtype=float),
            feature_names=list(F.columns),
        )
        return exp, {c: c for c in F.columns}, F

    preprocess = pipeline.named_steps["preprocess"]
    Z = preprocess.transform(F)
    names = list(preprocess.get_feature_names_out())
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(Z)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    base = np.ravel(explainer.expected_value)[-1]
    exp = shap.Explanation(
        values, base_values=np.full(len(Z), base), data=np.asarray(Z), feature_names=names
    )
    cats = features.categorical_features_
    source = {n: next((c for c in cats if n.startswith(c + "_")), n) for n in names}
    return exp, source, pd.DataFrame(np.asarray(Z), columns=names, index=X.index)


def grouped_mean_abs_shap(exp: shap.Explanation, source: dict[str, str]) -> pd.Series:
    """Mean |SHAP| per source feature (one-hot columns summed per row first)."""
    df = pd.DataFrame(exp.values, columns=exp.feature_names)
    grouped = df.T.groupby(pd.Series(source)).sum().T
    return grouped.abs().mean().sort_values(ascending=False)


def permutation_table(
    pipeline, X: pd.DataFrame, y: pd.Series, n_repeats: int, seed: int
) -> pd.DataFrame:
    """Drop in PR-AUC when one source column is shuffled (on held-out data)."""
    cols = [c for c in X.columns if c not in config.IDENTIFIER_COLUMNS]
    result = permutation_importance(
        pipeline,
        X[cols],
        y,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )
    return pd.DataFrame(
        {"importance_mean": result.importances_mean, "importance_sd": result.importances_std},
        index=cols,
    ).sort_values("importance_mean", ascending=False)
