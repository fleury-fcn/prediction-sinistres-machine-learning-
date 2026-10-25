"""Feature engineering on the cleaned claims table.

:class:`ClaimFeatureBuilder` is stateless (no ``fit`` statistics), so it cannot
leak information across folds; it is nevertheless the first step of every model
pipeline so that the exported model accepts a raw, cleaned claim as input.

Each engineered feature and the reason for it:

* ``policy_tenure_days`` - days between policy bind date and incident. Frauds
  are often suspected on young policies; the raw dates are not usable as is.
* ``incident_weekday`` - day of week of the incident (categorical). The data
  only covers January-March 2015, so month/year features are not meaningful.
* ``csl_per_person`` / ``csl_per_accident`` - the combined single limit
  ``"250/500"`` is two numeric limits packed into a string.
* ``vehicle_age`` - vehicle age at the incident, more interpretable than the
  model year.
* ``log_total_claim`` - log of the summed claim; amounts are right-skewed.
* ``injury_share`` / ``property_share`` - composition of the claim (the vehicle
  share is their complement and is omitted to avoid an exact linear
  dependency).
* ``claim_to_premium`` - claim size relative to the annual premium.
* ``has_umbrella`` - 80 % of policies have no umbrella cover; the flag helps
  linear models separate "none" from "amount".

Dropped: identifiers (``policy_number``, ``insured_zip``, ``incident_location``),
raw dates, ``policy_csl`` (replaced), ``auto_year`` (replaced) and, by default,
``total_claim_amount`` which is exactly the sum of its three components.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config

ENGINEERED_NUMERIC = [
    "policy_tenure_days",
    "csl_per_person",
    "csl_per_accident",
    "vehicle_age",
    "log_total_claim",
    "injury_share",
    "property_share",
    "claim_to_premium",
    "has_umbrella",
]
ENGINEERED_CATEGORICAL = ["incident_weekday"]

BASE_NUMERIC = [
    "months_as_customer",
    "age",
    "policy_deductible",
    "policy_annual_premium",
    "umbrella_limit",
    "capital_gains",
    "capital_loss",
    "incident_hour",
    "number_of_vehicles_involved",
    "bodily_injuries",
    "witnesses",
    "injury_claim",
    "property_claim",
    "vehicle_claim",
]
BASE_CATEGORICAL = [c for c in config.RAW_CATEGORICAL_COLUMNS if c != "policy_csl"]

CLAIM_AMOUNT_FEATURES = [
    "injury_claim",
    "property_claim",
    "vehicle_claim",
    "total_claim_amount",
    "log_total_claim",
    "injury_share",
    "property_share",
    "claim_to_premium",
]


class ClaimFeatureBuilder(BaseEstimator, TransformerMixin):
    """Build the model feature table from a cleaned claims DataFrame.

    Parameters
    ----------
    drop_total_claim_amount:
        Drop ``total_claim_amount`` (redundant with its components).
    exclude:
        Extra features to remove, used for ablation studies.
    """

    def __init__(self, drop_total_claim_amount: bool = True, exclude: tuple[str, ...] = ()):
        self.drop_total_claim_amount = drop_total_claim_amount
        self.exclude = exclude

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ClaimFeatureBuilder:
        self.feature_names_out_ = np.array(self.numeric_features_ + self.categorical_features_)
        return self

    @property
    def numeric_features_(self) -> list[str]:
        cols = BASE_NUMERIC + ENGINEERED_NUMERIC
        if not self.drop_total_claim_amount:
            cols = cols + ["total_claim_amount"]
        return [c for c in cols if c not in self.exclude]

    @property
    def categorical_features_(self) -> list[str]:
        return [c for c in BASE_CATEGORICAL + ENGINEERED_CATEGORICAL if c not in self.exclude]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        bind = pd.to_datetime(df["policy_bind_date"], errors="coerce")
        incident = pd.to_datetime(df["incident_date"], errors="coerce")
        df["policy_tenure_days"] = (incident - bind).dt.days.astype(float)
        df["incident_weekday"] = incident.dt.day_name().fillna(config.UNKNOWN_CATEGORY)

        csl = df["policy_csl"].astype(str).str.split("/", expand=True)
        df["csl_per_person"] = pd.to_numeric(csl[0], errors="coerce")
        df["csl_per_accident"] = pd.to_numeric(csl.get(1), errors="coerce")

        df["vehicle_age"] = incident.dt.year - pd.to_numeric(df["auto_year"], errors="coerce")

        total = df[config.CLAIM_COMPONENTS].sum(axis=1, min_count=1)
        safe_total = total.where(total > 0)
        df["log_total_claim"] = np.log1p(total.clip(lower=0))
        df["injury_share"] = df["injury_claim"] / safe_total
        df["property_share"] = df["property_claim"] / safe_total
        df["claim_to_premium"] = total / df["policy_annual_premium"].where(
            df["policy_annual_premium"] > 0
        )
        df["has_umbrella"] = (df["umbrella_limit"].fillna(0) > 0).astype(float)
        if "total_claim_amount" not in df:
            df["total_claim_amount"] = total

        out = df[self.numeric_features_ + self.categorical_features_].copy()
        out[self.numeric_features_] = out[self.numeric_features_].astype(float)
        for col in self.categorical_features_:
            out[col] = out[col].astype(str)
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.numeric_features_ + self.categorical_features_)
