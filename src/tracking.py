"""Optional MLflow tracking (local ``mlruns/``). Versioned CSV/JSON files in
``results/metrics`` remain the source of truth; MLflow is a convenience UI."""

from __future__ import annotations

import logging
from typing import Any

from . import config

logger = logging.getLogger(__name__)

try:
    import mlflow
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None


def log_run(
    experiment: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
) -> None:
    if mlflow is None:
        return
    try:
        mlflow.set_tracking_uri((config.ROOT_DIR / "mlruns").as_uri())
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            if tags:
                mlflow.set_tags(tags)
    except Exception as exc:  # tracking must never break an experiment
        logger.warning("MLflow logging skipped (%s)", exc)
