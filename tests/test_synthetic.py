import numpy as np
import pandas as pd
import pytest

from src import config, synthetic


@pytest.fixture(scope="module")
def real_train(split):
    X_train, _, y_train, _ = split
    return X_train.assign(**{config.TARGET: y_train})


@pytest.fixture(scope="module")
def sample(real_train):
    gen = synthetic.ClassConditionalGaussianCopula(seed=0).fit(real_train)
    return synthetic.generate(gen, 4000, real_train, seed=0, rules=True)


def test_schema_matches_real(sample, real_train):
    assert list(sample.columns) == list(real_train.columns)
    assert len(sample) == 4000


def test_fraud_rate_preserved(sample, real_train):
    assert abs(sample[config.TARGET].mean() - real_train[config.TARGET].mean()) < 0.001


def test_values_stay_in_real_support(sample, real_train):
    for col in config.RAW_CATEGORICAL_COLUMNS:
        assert set(sample[col]) <= set(real_train[col])
    for col in ["age", "policy_annual_premium", "vehicle_claim"]:
        assert sample[col].between(real_train[col].min(), real_train[col].max()).all()


def test_derived_columns_consistent(sample):
    np.testing.assert_allclose(
        sample["total_claim_amount"], sample[config.CLAIM_COMPONENTS].sum(axis=1)
    )
    assert synthetic.rule_violation_rate(sample) == 0.0
    assert sample["policy_number"].is_unique


def test_no_real_row_copied(sample, real_train, split):
    _, X_test, _, y_test = split
    scores = synthetic.privacy_scores(real_train, X_test.assign(**{config.TARGET: y_test}), sample)
    assert scores["exact_copies"] == 0
    assert scores["dcr_synth_median"] > 0


def test_deterministic_given_seed(real_train):
    gen = synthetic.ClassConditionalGaussianCopula(seed=3).fit(real_train)
    a = gen.sample(200, seed=7)
    b = gen.sample(200, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_marginal_and_correlation_metrics(sample, real_train):
    marg = synthetic.marginal_scores(real_train, sample)
    assert marg["distance"].between(0, 1).all()
    gap = synthetic.correlation_gap(real_train, sample)
    assert 0 <= gap["spearman_mean_abs_diff"] < 0.2
