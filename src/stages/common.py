"""Shared helpers for the pipeline stages: data split, artefact paths, JSON I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import config, data
from ..config import Settings
from ..evaluation import CostModel

TRAIN_ARTIFACTS = config.ARTIFACTS_DIR / "train"
MODEL_ARTIFACTS = config.ARTIFACTS_DIR / "models"


def real_split(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = data.load_clean(settings.raw_data_path)
    return data.train_test(df, settings["data"]["test_size"], settings.seed)


def cost_model(settings: Settings) -> CostModel:
    c = settings["costs"]
    return CostModel(investigation=float(c["investigation"]), missed_fraud=float(c["missed_fraud"]))


def _default(o: Any) -> Any:
    if isinstance(o, np.integer | np.floating):
        return o.item()
    if isinstance(o, np.ndarray | tuple):
        return list(o)
    raise TypeError(type(o))


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_default)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def restore_params(params: dict[str, Any]) -> dict[str, Any]:
    """JSON turns tuples into lists; the MLP expects a tuple of layer widths."""
    params = dict(params)
    if "hidden_dims" in params:
        params["hidden_dims"] = tuple(params["hidden_dims"])
    return params
