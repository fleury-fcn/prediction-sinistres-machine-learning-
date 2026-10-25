from __future__ import annotations

import pandas as pd
import pytest

from src import config, data


@pytest.fixture(scope="session")
def clean_df() -> pd.DataFrame:
    return data.load_clean(config.DATA_DIR / "insurance_claims.csv")


@pytest.fixture(scope="session")
def split(clean_df):
    return data.train_test(clean_df, test_size=0.2, seed=42)
