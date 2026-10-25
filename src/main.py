"""Command-line entry point.

Usage::

    python -m src.main --stage all            # full run (numbers in the README)
    python -m src.main --stage train --fast   # quick smoke run
    python -m src.main --stage evaluate --cost-fn 20000

Stages: ``eda`` -> ``train`` -> ``evaluate`` (real data) and ``synth``
(synthetic data generation, validation, TSTR and large-scale benchmark; it
reuses the hyper-parameters tuned on the real data when available).
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time

import numpy as np

from . import config

STAGES = ["eda", "train", "evaluate", "synth"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insurance claim fraud detection pipeline")
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="small budgets for a quick end-to-end check (not for reporting)",
    )
    parser.add_argument("--config", type=str, default=None, help="path to a YAML config")
    parser.add_argument(
        "--models", type=str, default=None, help="comma-separated subset of models to train"
    )
    parser.add_argument(
        "--cost-fn",
        type=float,
        default=None,
        help="cost of a missed fraud (USD), overrides the config",
    )
    parser.add_argument(
        "--cost-investigation",
        type=float,
        default=None,
        help="cost of investigating a claim (USD), overrides the config",
    )
    parser.add_argument(
        "--synthetic-rows",
        type=int,
        default=None,
        help="size of the large synthetic dataset, overrides the config",
    )
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    """Seed Python and NumPy. Torch is seeded inside the MLP and SDV wrappers: importing
    it here, before XGBoost/LightGBM, makes their OpenMP runtimes clash on macOS."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("sdv", "copulas", "rdt", "SingleTableSynthesizer", "MultiTableSynthesizer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    overrides: dict = {}
    if args.cost_fn is not None:
        overrides.setdefault("costs", {})["missed_fraud"] = args.cost_fn
    if args.cost_investigation is not None:
        overrides.setdefault("costs", {})["investigation"] = args.cost_investigation
    if args.synthetic_rows is not None:
        overrides["synthetic"] = {"n_rows": args.synthetic_rows}
    settings = config.load_settings(args.config and config.Path(args.config), args.fast, overrides)
    seed_everything(settings.seed)

    stages = STAGES if args.stage == "all" else [args.stage]
    for stage in stages:
        start = time.perf_counter()
        logging.info("=== stage: %s%s ===", stage, " (fast)" if args.fast else "")
        if stage == "eda":
            from .stages import eda

            eda.run(settings)
        elif stage == "train":
            from .stages import train

            train.run(settings, args.models.split(",") if args.models else None)
        elif stage == "evaluate":
            from .stages import evaluate

            evaluate.run(settings)
        elif stage == "synth":
            from .stages import synth

            synth.run(settings)
        logging.info("=== stage %s done in %.1f min ===", stage, (time.perf_counter() - start) / 60)


if __name__ == "__main__":
    main()
