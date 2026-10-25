import numpy as np

from src import config, data


def test_all_source_columns_mapped(clean_df):
    assert len(clean_df) == 1000
    assert config.TARGET in clean_df
    assert config.RAW_TARGET not in clean_df
    assert set(config.RENAME_MAP.values()) - {config.RAW_TARGET} <= set(clean_df.columns)


def test_question_marks_become_explicit_category(clean_df):
    for col in config.RAW_CATEGORICAL_COLUMNS:
        assert not (clean_df[col] == config.MISSING_TOKEN).any()
    assert (clean_df["collision_type"] == config.UNKNOWN_CATEGORY).sum() == 178


def test_literal_none_is_a_level_not_missing(clean_df):
    # pandas would parse "None" as NaN without keep_default_na=False
    assert (clean_df["authorities_contacted"] == "None").sum() == 91
    assert clean_df["authorities_contacted"].notna().all()


def test_target_and_types(clean_df):
    assert set(clean_df[config.TARGET].unique()) == {0, 1}
    assert abs(clean_df[config.TARGET].mean() - 0.247) < 1e-9
    assert np.issubdtype(clean_df["incident_date"].dtype, np.datetime64)


def test_negative_umbrella_limit_set_to_nan(clean_df):
    assert clean_df["umbrella_limit"].isna().sum() == 1
    assert (clean_df["umbrella_limit"].dropna() >= 0).all()


def test_split_is_stratified_and_disjoint(split):
    X_train, X_test, y_train, y_test = split
    assert len(X_train) == 800 and len(X_test) == 200
    assert not set(X_train.index) & set(X_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_coerce_types_shared_with_api():
    import pandas as pd

    out = data.coerce_types(pd.DataFrame({"collision_type": ["?"], "umbrella_limit": [-5]}))
    assert out.loc[0, "collision_type"] == config.UNKNOWN_CATEGORY
    assert np.isnan(out.loc[0, "umbrella_limit"])
