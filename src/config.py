"""Central configuration: paths, column schema and experiment settings.

The column schema of the public dataset lives here so that a new dataset can be
plugged in by editing ``RENAME_MAP`` and the feature lists only.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "default.yaml"

DATA_DIR = ROOT_DIR / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"  # large intermediate files, git-ignored

# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------

# Source column -> internal name. Every column of the source CSV is listed so
# that nothing is silently dropped. Internal names are plain English snake_case;
# the former French aliases were removed because some were misleading
# (``umbrella_limit`` is an umbrella liability limit, not an insured capital;
# ``total_claim_amount`` is the amount claimed, not a settled amount).
RENAME_MAP: dict[str, str] = {
    "months_as_customer": "months_as_customer",
    "age": "age",
    "policy_number": "policy_number",
    "policy_bind_date": "policy_bind_date",
    "policy_state": "policy_state",
    "policy_csl": "policy_csl",
    "policy_deductable": "policy_deductible",  # typo in the source
    "policy_annual_premium": "policy_annual_premium",
    "umbrella_limit": "umbrella_limit",
    "insured_zip": "insured_zip",
    "insured_sex": "insured_sex",
    "insured_education_level": "insured_education_level",
    "insured_occupation": "insured_occupation",
    "insured_hobbies": "insured_hobbies",
    "insured_relationship": "insured_relationship",
    "capital-gains": "capital_gains",
    "capital-loss": "capital_loss",
    "incident_date": "incident_date",
    "incident_type": "incident_type",
    "collision_type": "collision_type",
    "incident_severity": "incident_severity",
    "authorities_contacted": "authorities_contacted",
    "incident_state": "incident_state",
    "incident_city": "incident_city",
    "incident_location": "incident_location",
    "incident_hour_of_the_day": "incident_hour",
    "number_of_vehicles_involved": "number_of_vehicles_involved",
    "property_damage": "property_damage",
    "bodily_injuries": "bodily_injuries",
    "witnesses": "witnesses",
    "police_report_available": "police_report_available",
    "total_claim_amount": "total_claim_amount",
    "injury_claim": "injury_claim",
    "property_claim": "property_claim",
    "vehicle_claim": "vehicle_claim",
    "auto_make": "auto_make",
    "auto_model": "auto_model",
    "auto_year": "auto_year",
    "fraud_reported": "fraud_reported",
}

RAW_TARGET = "fraud_reported"
TARGET_POSITIVE_LABEL = "Y"
TARGET = "is_fraud"

# Marker used by the source for missing categorical values.
MISSING_TOKEN = "?"
UNKNOWN_CATEGORY = "UNKNOWN"

DATE_COLUMNS = ["policy_bind_date", "incident_date"]

# Columns that identify a record rather than describe it. They are kept in the
# cleaned table (and reproduced by the synthetic generator to preserve the
# schema) but never used as model inputs.
IDENTIFIER_COLUMNS = ["policy_number", "insured_zip", "incident_location"]

RAW_NUMERIC_COLUMNS = [
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
    "total_claim_amount",
    "auto_year",
]

RAW_CATEGORICAL_COLUMNS = [
    "policy_state",
    "policy_csl",
    "insured_sex",
    "insured_education_level",
    "insured_occupation",
    "insured_hobbies",
    "insured_relationship",
    "incident_type",
    "collision_type",
    "incident_severity",
    "authorities_contacted",
    "incident_state",
    "incident_city",
    "property_damage",
    "police_report_available",
    "auto_make",
    "auto_model",
]

CLAIM_COMPONENTS = ["injury_claim", "property_claim", "vehicle_claim"]


# ---------------------------------------------------------------------------
# Experiment settings (configs/default.yaml)
# ---------------------------------------------------------------------------


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class Settings:
    """Experiment settings resolved from the YAML file (and ``--fast``)."""

    raw: dict[str, Any]
    fast: bool = False

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def raw_data_path(self) -> Path:
        return ROOT_DIR / self.raw["data"]["raw_path"]


def load_settings(
    path: Path | None = None, fast: bool = False, overrides: dict[str, Any] | None = None
) -> Settings:
    """Load the YAML config, apply the ``fast`` block and explicit overrides."""
    with open(path or CONFIG_PATH, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    fast_block = raw.pop("fast", {})
    if fast:
        raw = _deep_update(raw, fast_block)
    if overrides:
        raw = _deep_update(raw, overrides)
    return Settings(raw=raw, fast=fast)


def ensure_dirs() -> None:
    for directory in (
        SYNTHETIC_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        MODELS_DIR,
        ARTIFACTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
