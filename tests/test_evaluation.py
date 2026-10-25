import numpy as np
import pandas as pd

from src import evaluation
from src.evaluation import CostModel


def test_cost_model():
    c = CostModel(investigation=100, missed_fraud=1000)
    y = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 1, 0])
    # 2 flagged -> 200, 1 missed fraud -> 1000
    assert c.total(y, pred) == 1200
    assert c.bayes_threshold == 0.1


def test_cost_optimal_threshold_on_separable_scores():
    y = np.array([0] * 80 + [1] * 20)
    p = np.r_[np.linspace(0.0, 0.3, 80), np.linspace(0.6, 0.9, 20)]
    t = evaluation.cost_optimal_threshold(y, p, CostModel(100, 1000))
    assert 0.3 < t <= 0.6
    assert CostModel(100, 1000).per_claim(y, (p >= t).astype(int)) == 20.0


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 300), 0, 1)
    t = evaluation.bootstrap_metrics(y, p, 0.5, CostModel(), n_boot=200)
    assert (t["ci_low"] <= t["estimate"] + 1e-12).all()
    assert (t["estimate"] <= t["ci_high"] + 1e-12).all()


def test_mcnemar_identical_classifiers():
    y = np.array([0, 1, 1, 0])
    assert evaluation.mcnemar_test(y, y, y)["p_value"] == 1.0


def test_holm_is_monotone_and_bounded():
    p = pd.Series([0.01, 0.04, 0.03, 0.5], index=list("abcd"))
    adj = evaluation.holm_correction(p)
    assert (adj >= p).all() and (adj <= 1).all()
    assert adj["a"] == 0.04


def test_corrected_ttest_is_more_conservative_than_naive():
    from scipy import stats

    rng = np.random.default_rng(1)
    a = 0.7 + rng.normal(0, 0.02, 25)
    b = a - 0.01 + rng.normal(0, 0.02, 25)
    corrected = evaluation.corrected_resampled_ttest(a, b, n_train=640, n_test=160)["p_value"]
    naive = stats.ttest_rel(a, b).pvalue
    assert corrected > naive


def test_paired_bootstrap_same_model_is_zero():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200)
    p = rng.uniform(size=200)
    out = evaluation.paired_bootstrap_diff(y, p, p, n_boot=50)
    assert out["diff"] == 0 and out["p_value"] == 1.0
