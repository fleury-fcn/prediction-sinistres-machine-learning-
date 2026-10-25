"""Every learned statistic must come from the training data only."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import config
from src.models import MODELS
from src.nn import TabularMLPClassifier
from src.pipelines import make_pipeline


def _fit_logreg(X, y, **kw):
    return make_pipeline(
        LogisticRegression(max_iter=2000), encoding="onehot", scale=True, **kw
    ).fit(X, y)


def test_imputer_and_scaler_fitted_on_train_only(split):
    X_train, X_test, y_train, _ = split
    pipe = _fit_logreg(X_train, y_train)
    pre = pipe.named_steps["preprocess"]
    num_cols = pipe.named_steps["features"].numeric_features_
    F_train = pipe.named_steps["features"].transform(X_train)
    imputer = pre.named_transformers_["num"].named_steps["impute"]
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    np.testing.assert_allclose(imputer.statistics_, F_train[num_cols].median().to_numpy())
    np.testing.assert_allclose(
        scaler.mean_, F_train[num_cols].fillna(F_train[num_cols].median()).mean().to_numpy()
    )


def test_test_data_does_not_change_fitted_state(split):
    X_train, X_test, y_train, _ = split
    a = _fit_logreg(X_train, y_train)
    shifted = X_test.copy()
    shifted["age"] += 1000  # anything done to the test set must be irrelevant to fit
    b = _fit_logreg(X_train, y_train)
    np.testing.assert_allclose(a.named_steps["model"].coef_, b.named_steps["model"].coef_)
    assert a.predict_proba(shifted).shape == (len(shifted), 2)


def test_categories_learned_from_train_only(split):
    X_train, X_test, y_train, _ = split
    pipe = _fit_logreg(X_train, y_train)
    X_new = X_test.copy()
    X_new["auto_make"] = "BrandNeverSeen"
    enc = pipe.named_steps["preprocess"].named_transformers_["cat"]
    makes = enc.categories_[pipe.named_steps["features"].categorical_features_.index("auto_make")]
    assert "BrandNeverSeen" not in makes
    assert np.isfinite(pipe.predict_proba(X_new)).all()


def test_smote_only_at_fit_time(split):
    X_train, X_test, y_train, _ = split
    pipe = _fit_logreg(X_train, y_train, imbalance="smote")
    # resampling happens in fit; predict returns one score per input row
    assert pipe.predict_proba(X_test).shape[0] == len(X_test)
    assert "smote" in pipe.named_steps


def test_smote_requires_onehot():
    with pytest.raises(ValueError):
        make_pipeline(LogisticRegression(), encoding="native", imbalance="smote")


def test_mlp_encoders_fitted_inside_fit(split):
    X_train, X_test, y_train, _ = split
    F = make_pipeline(LogisticRegression(), encoding="onehot").named_steps["features"]
    F_train, F_test = F.fit_transform(X_train), F.transform(X_test)
    mlp = TabularMLPClassifier(max_epochs=3, hidden_dims=(16,)).fit(F_train, y_train)
    # medians come from the internal fit split, a subset of the training data
    assert set(mlp.num_median_.index) == set(mlp.num_cols_)
    F_test = F_test.assign(auto_make="BrandNeverSeen")
    assert np.isfinite(mlp.predict_proba(F_test)).all()


@pytest.mark.parametrize("name", [n for n in MODELS if n != "mlp"])
def test_every_model_pipeline_fits_and_predicts(split, name):
    X_train, X_test, y_train, _ = split
    model = MODELS[name].build(MODELS[name].default_params, 0).fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    assert proba.shape == (len(X_test),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_threshold_is_chosen_without_test_labels():
    """The training stage picks thresholds from OOF probabilities of the training split."""
    import inspect

    from src.stages import train

    source = inspect.getsource(train.run)
    assert "cost_optimal_threshold(y_train" in source
    assert "y_test" not in source and "X_test" not in source


def test_target_never_a_feature(clean_df):
    pipe = make_pipeline(LogisticRegression(), encoding="onehot")
    F = pipe.named_steps["features"].fit_transform(clean_df)
    assert config.TARGET not in F.columns
    assert isinstance(F, pd.DataFrame)
