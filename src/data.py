"""Loading, deterministic cleaning and train/test split of the claims data.

Everything in this module is a row-wise, rule-based transformation: nothing is
estimated from the data, so it can safely run before the train/test split.
Anything that *learns* from data (imputation, encoding, scaling, resampling)
lives in the sklearn pipelines of :mod:`src.pipelines`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

logger = logging.getLogger(__name__)


def load_raw(path: Path | str) -> pd.DataFrame:
    """Read the source CSV without pandas' default NA parsing.

    ``keep_default_na=False`` matters: the source uses the literal string
    ``"None"`` as a legitimate ``authorities_contacted`` level, which pandas
    would otherwise turn into NaN.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df.columns = df.columns.str.strip()
    # Some copies of the dataset carry an empty trailing column ("_c39").
    df = df.loc[:, ~df.columns.str.startswith("_c")]
    logger.info("Loaded %s rows x %s columns from %s", *df.shape, path.name)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, type them, and encode missing markers explicitly.

    * ``?`` in categorical columns becomes the explicit level ``UNKNOWN``:
      missingness here is structural (e.g. no collision type for a vehicle
      theft) and potentially informative, so it is not imputed away.
    * A negative ``umbrella_limit`` (one row) is a data-entry error and is set to
      NaN; it is imputed later inside the training pipeline.
    * The target becomes ``is_fraud`` in {0, 1}.
    """
    missing = [c for c in config.RENAME_MAP if c not in df.columns]
    if missing:
        raise ValueError(f"Missing source columns: {missing}")
    out = coerce_types(df[list(config.RENAME_MAP)].rename(columns=config.RENAME_MAP))
    out[config.TARGET] = (
        out.pop(config.RAW_TARGET).astype(str).str.strip().eq(config.TARGET_POSITIVE_LABEL)
    ).astype(int)
    return out


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Type the internal-name columns present in ``df`` (shared with the API)."""
    out = df.copy()
    for col in config.RAW_CATEGORICAL_COLUMNS:
        if col in out:
            out[col] = (
                out[col]
                .astype(str)
                .str.strip()
                .replace({config.MISSING_TOKEN: config.UNKNOWN_CATEGORY})
            )
    for col in config.RAW_NUMERIC_COLUMNS:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    for col in config.DATE_COLUMNS:
        if col in out:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    if "umbrella_limit" in out:
        negative = out["umbrella_limit"] < 0
        if negative.any():
            logger.warning("umbrella_limit: %d negative value(s) set to NaN", int(negative.sum()))
            out.loc[negative, "umbrella_limit"] = np.nan
    return out


def load_clean(path: Path | str) -> pd.DataFrame:
    return clean(load_raw(path))


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[config.TARGET]), df[config.TARGET]


def train_test(
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified hold-out split. The test part is only touched at evaluation."""
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
