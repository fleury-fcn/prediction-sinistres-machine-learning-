import numpy as np

from src import config
from src.features import CLAIM_AMOUNT_FEATURES, ClaimFeatureBuilder


def test_builder_is_stateless_and_row_wise(clean_df):
    X = clean_df.drop(columns=[config.TARGET])
    full = ClaimFeatureBuilder().fit(X).transform(X)
    part = ClaimFeatureBuilder().fit(X.iloc[:10]).transform(X.iloc[:10])
    assert full.iloc[:10].equals(part)


def test_identifiers_and_redundant_total_dropped(clean_df):
    F = ClaimFeatureBuilder().fit_transform(clean_df)
    for col in [*config.IDENTIFIER_COLUMNS, "total_claim_amount", "policy_csl", config.TARGET]:
        assert col not in F
    assert "total_claim_amount" in ClaimFeatureBuilder(drop_total_claim_amount=False).fit_transform(
        clean_df
    )


def test_engineered_values(clean_df):
    F = ClaimFeatureBuilder().fit_transform(clean_df)
    row = clean_df.iloc[0]
    assert F.loc[0, "policy_tenure_days"] == (row.incident_date - row.policy_bind_date).days
    assert F.loc[0, "csl_per_person"] == 250 and F.loc[0, "csl_per_accident"] == 500
    shares = F[["injury_share", "property_share"]].dropna()
    assert ((shares >= 0) & (shares <= 1)).all().all()
    np.testing.assert_allclose(np.expm1(F["log_total_claim"]), clean_df["total_claim_amount"])


def test_exclude_for_ablation(clean_df):
    F = ClaimFeatureBuilder(exclude=tuple(CLAIM_AMOUNT_FEATURES)).fit_transform(clean_df)
    assert not set(CLAIM_AMOUNT_FEATURES) & set(F.columns)
