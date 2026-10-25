"""Metrics, confidence intervals, cost-based thresholds and model comparison tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Business costs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """Per-claim costs (USD) of each decision outcome.

    Every claim flagged by the model is investigated (``investigation`` is paid
    for true and false positives). A fraudulent claim that is not flagged is
    paid out (``missed_fraud``). Correctly passing an honest claim costs 0.
    """

    investigation: float = 1000.0
    missed_fraud: float = 10000.0

    def total(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        flagged = y_pred == 1
        missed = (y_true == 1) & ~flagged
        return float(flagged.sum() * self.investigation + missed.sum() * self.missed_fraud)

    def per_claim(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.total(y_true, y_pred) / len(y_true)

    @property
    def bayes_threshold(self) -> float:
        """Optimal threshold for perfectly calibrated probabilities.

        Flag when ``p * missed_fraud > investigation``.
        """
        return self.investigation / self.missed_fraud


THRESHOLD_GRID = np.round(np.linspace(0.01, 0.99, 99), 2)


def cost_curve(
    y_true: np.ndarray, proba: np.ndarray, costs: CostModel, grid: np.ndarray = THRESHOLD_GRID
) -> pd.DataFrame:
    rows = [
        {"threshold": t, "cost_per_claim": costs.per_claim(y_true, (proba >= t).astype(int))}
        for t in grid
    ]
    return pd.DataFrame(rows)


def cost_optimal_threshold(y_true: np.ndarray, proba: np.ndarray, costs: CostModel) -> float:
    """Threshold minimising the average cost. Must be called on validation/OOF data."""
    curve = cost_curve(y_true, proba, costs)
    return float(curve.loc[curve["cost_per_claim"].idxmin(), "threshold"])


def trivial_policy_costs(y_true: np.ndarray, costs: CostModel) -> dict[str, float]:
    y_true = np.asarray(y_true)
    return {
        "investigate_none": costs.per_claim(y_true, np.zeros_like(y_true)),
        "investigate_all": costs.per_claim(y_true, np.ones_like(y_true)),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

RANKING_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "pr_auc": average_precision_score,
    "roc_auc": roc_auc_score,
    "brier": brier_score_loss,
}
THRESHOLD_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "f1": lambda y, p: f1_score(y, p, zero_division=0),
    "precision": lambda y, p: precision_score(y, p, zero_division=0),
    "recall": recall_score,
    "mcc": matthews_corrcoef,
}
METRIC_ORDER = ["pr_auc", "roc_auc", "f1", "precision", "recall", "mcc", "brier", "cost_per_claim"]


def compute_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float, costs: CostModel
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = (proba >= threshold).astype(int)
    out = {name: float(fn(y_true, proba)) for name, fn in RANKING_METRICS.items()}
    out.update({name: float(fn(y_true, y_pred)) for name, fn in THRESHOLD_METRICS.items()})
    out["cost_per_claim"] = costs.per_claim(y_true, y_pred)
    return out


def bootstrap_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    costs: CostModel,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Point estimate + percentile bootstrap CI for every metric.

    Resampling is stratified (positives and negatives resampled separately) so
    that every replicate keeps the observed prevalence.
    """
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    point = compute_metrics(y_true, proba, threshold, costs)
    reps = _stratified_bootstrap_indices(y_true, n_boot, seed)
    samples = pd.DataFrame(
        [compute_metrics(y_true[idx], proba[idx], threshold, costs) for idx in reps]
    )
    alpha = (1 - ci_level) / 2
    return pd.DataFrame(
        {
            "estimate": pd.Series(point),
            "ci_low": samples.quantile(alpha),
            "ci_high": samples.quantile(1 - alpha),
        }
    ).loc[METRIC_ORDER]


def _stratified_bootstrap_indices(y: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return [
        np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for _ in range(n_boot)
    ]


# ---------------------------------------------------------------------------
# Model comparison tests
# ---------------------------------------------------------------------------


def paired_bootstrap_diff(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = average_precision_score,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap the difference ``metric(a) - metric(b)`` on the same resamples.

    The two-sided p-value is the bootstrap estimate of P(diff has the other
    sign), doubled.
    """
    y_true = np.asarray(y_true)
    diffs = np.array(
        [
            metric(y_true[idx], proba_a[idx]) - metric(y_true[idx], proba_b[idx])
            for idx in _stratified_bootstrap_indices(y_true, n_boot, seed)
        ]
    )
    alpha = (1 - ci_level) / 2
    observed = metric(y_true, proba_a) - metric(y_true, proba_b)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "diff": float(observed),
        "ci_low": float(np.quantile(diffs, alpha)),
        "ci_high": float(np.quantile(diffs, 1 - alpha)),
        "p_value": float(min(p, 1.0)),
    }


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test on the discordant pairs of two classifiers."""
    correct_a = np.asarray(pred_a) == np.asarray(y_true)
    correct_b = np.asarray(pred_b) == np.asarray(y_true)
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    p = 1.0 if b + c == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": p}


def corrected_resampled_ttest(
    scores_a: np.ndarray, scores_b: np.ndarray, n_train: int, n_test: int
) -> dict[str, float]:
    """Nadeau & Bengio (2003) corrected t-test for repeated k-fold CV scores.

    Fold scores overlap in their training sets, so the naive t-test is far too
    optimistic; the variance is inflated by ``1/k + n_test/n_train``.
    """
    d = np.asarray(scores_a) - np.asarray(scores_b)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return {"mean_diff": float(d.mean()), "t": 0.0, "p_value": 1.0}
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return {"mean_diff": float(d.mean()), "t": float(t), "p_value": float(p)}


def holm_correction(p_values: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    order = p_values.sort_values().index
    m = len(p_values)
    adjusted = pd.Series(index=order, dtype=float)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p_values[idx]))
        adjusted[idx] = running
    return adjusted.reindex(p_values.index)


def cv_summary(fold_scores: pd.DataFrame, ci_level: float = 0.95) -> pd.DataFrame:
    """Mean and t-based interval of repeated-CV fold scores (one row per metric).

    The interval ignores fold dependence and is therefore optimistic; it is
    reported as a spread indicator, the test-set bootstrap CI is the headline.
    """
    n = len(fold_scores)
    mean, sd = fold_scores.mean(), fold_scores.std(ddof=1)
    half = stats.t.ppf(0.5 + ci_level / 2, n - 1) * sd / np.sqrt(n)
    return pd.DataFrame({"mean": mean, "sd": sd, "ci_low": mean - half, "ci_high": mean + half})
