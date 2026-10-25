"""Leak-free preprocessing pipelines.

Every statistic (medians, category levels, scaling parameters, SMOTE
neighbours) is learned inside ``Pipeline.fit``, i.e. on the training fold only.
"""

from __future__ import annotations

from typing import Literal

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import ClaimFeatureBuilder

Imbalance = Literal["none", "balanced", "smote"]


def make_feature_builder(drop_total_claim_amount: bool = True, exclude: tuple[str, ...] = ()):
    return ClaimFeatureBuilder(drop_total_claim_amount=drop_total_claim_amount, exclude=exclude)


def make_onehot_preprocessor(
    builder: ClaimFeatureBuilder, scale: bool, min_frequency: int = 10
) -> ColumnTransformer:
    """Median imputation (+ optional scaling) for numerics, one-hot for categoricals.

    Rare levels (fewer than ``min_frequency`` occurrences in the training fold)
    and levels unseen at fit time are mapped to a shared "infrequent" column.
    """
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    categorical = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=min_frequency,
        sparse_output=False,
    )
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), builder.numeric_features_),
            ("cat", categorical, builder.categorical_features_),
        ],
        verbose_feature_names_out=False,
    )


def make_pipeline(
    estimator: BaseEstimator,
    *,
    encoding: Literal["onehot", "native"],
    scale: bool = False,
    imbalance: Imbalance = "none",
    seed: int = 42,
    drop_total_claim_amount: bool = True,
    exclude: tuple[str, ...] = (),
    min_frequency: int = 10,
) -> Pipeline | ImbPipeline:
    """Assemble ``features -> [preprocess] -> [SMOTE] -> estimator``.

    ``encoding="native"`` is for estimators that handle raw categorical columns
    themselves (CatBoost, the PyTorch MLP with embeddings). SMOTE is only
    available after one-hot encoding and lives in an imblearn pipeline, so it
    is applied when fitting and never at prediction time.
    """
    builder = make_feature_builder(drop_total_claim_amount, exclude)
    steps: list[tuple[str, object]] = [("features", builder)]
    if encoding == "onehot":
        steps.append(("preprocess", make_onehot_preprocessor(builder, scale, min_frequency)))
    if imbalance == "smote":
        if encoding != "onehot":
            raise ValueError("SMOTE requires one-hot encoded inputs")
        steps.append(("smote", SMOTE(random_state=seed)))
        steps.append(("model", estimator))
        return ImbPipeline(steps)
    steps.append(("model", estimator))
    return Pipeline(steps)
