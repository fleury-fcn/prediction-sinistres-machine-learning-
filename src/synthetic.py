"""Synthetic claims: generators and quality checks.

Generators are fitted on the **real training split only**, so that the real
test split can serve for "train on synthetic, test on real" (TSTR) without
leakage. A generator fitted on 800 rows cannot add information about fraud:
the synthetic data is used for scale and robustness tests, never to report a
real-world performance.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

logger = logging.getLogger(__name__)

EPOCH = pd.Timestamp("1970-01-01")
INTEGER_COLUMNS = [
    "months_as_customer",
    "age",
    "policy_deductible",
    "umbrella_limit",
    "capital_gains",
    "capital_loss",
    "incident_hour",
    "number_of_vehicles_involved",
    "bodily_injuries",
    "witnesses",
    "injury_claim",
    "property_claim",
    "vehicle_claim",
    "auto_year",
    "insured_zip",
]
# Generated as columns of the copula (total_claim_amount is re-derived as a sum).
COPULA_NUMERIC = [c for c in config.RAW_NUMERIC_COLUMNS if c != "total_claim_amount"] + [
    "insured_zip",
    *config.DATE_COLUMNS,
]
COPULA_CATEGORICAL = list(config.RAW_CATEGORICAL_COLUMNS)
NO_COLLISION_INCIDENTS = {"Vehicle Theft", "Parked Car"}


# ---------------------------------------------------------------------------
# Shared pre/post-processing
# ---------------------------------------------------------------------------


def to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in config.DATE_COLUMNS:
        out[c] = (pd.to_datetime(out[c]) - EPOCH).dt.days.astype(float)
    out["insured_zip"] = out["insured_zip"].astype(float)
    return out


def finalize(df: pd.DataFrame, rng: np.random.Generator, ref: pd.DataFrame) -> pd.DataFrame:
    """Restore types, derived columns and identifiers so the schema matches the real data."""
    out = df.copy()
    for c in config.DATE_COLUMNS:
        days = pd.to_numeric(out[c], errors="coerce").round()
        lo, hi = (ref[c] - EPOCH).dt.days.min(), (ref[c] - EPOCH).dt.days.max()
        out[c] = EPOCH + pd.to_timedelta(days.clip(lo, hi), unit="D")
    for c in INTEGER_COLUMNS:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce").round()
            out[c] = out[c].clip(ref[c].min(), ref[c].max())
    out["total_claim_amount"] = out[config.CLAIM_COMPONENTS].sum(axis=1)
    # identifiers: fresh values, never copied from real rows
    out["policy_number"] = np.arange(10**6, 10**6 + len(out))
    out["incident_location"] = [f"{n} Synthetic Ave" for n in rng.integers(1, 9999, len(out))]
    for c in COPULA_CATEGORICAL:
        out[c] = out[c].astype(str)
    out[config.TARGET] = out[config.TARGET].astype(int)
    return out[list(ref.columns)]


def enforce_rules(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Business rule observed in 100 % of real rows: a collision type exists iff the
    incident is a collision. Violating rows get a value consistent with the rule."""
    out = df.copy()
    no_coll = out["incident_type"].isin(NO_COLLISION_INCIDENTS)
    out.loc[no_coll, "collision_type"] = config.UNKNOWN_CATEGORY
    bad = ~no_coll & (out["collision_type"] == config.UNKNOWN_CATEGORY)
    levels = ["Rear Collision", "Side Collision", "Front Collision"]
    out.loc[bad, "collision_type"] = rng.choice(levels, int(bad.sum()))
    return out


def rule_violation_rate(df: pd.DataFrame) -> float:
    no_coll = df["incident_type"].isin(NO_COLLISION_INCIDENTS)
    unknown = df["collision_type"] == config.UNKNOWN_CATEGORY
    return float((no_coll != unknown).mean())


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


class ClassConditionalGaussianCopula:
    """Gaussian copula fitted separately for each class of the target.

    * numeric columns: empirical marginal, inverted by linear interpolation of
      the order statistics (new values stay inside the observed range);
    * categorical columns: each level owns an interval of [0, 1] proportional
      to its frequency; rows are mapped to a uniform draw inside their interval;
    * dependence: correlation of the normal scores, shrunk towards the identity
      to stay positive definite (n = 600 / 200 rows per class).

    Sampling per class reproduces the training fraud rate exactly.
    """

    name = "gaussian_copula_custom"

    def __init__(self, shrinkage: float = 0.05, seed: int = 42):
        self.shrinkage = shrinkage
        self.seed = seed

    def fit(self, df: pd.DataFrame) -> ClassConditionalGaussianCopula:
        num = to_numeric_frame(df)
        num[COPULA_NUMERIC] = num[COPULA_NUMERIC].fillna(num[COPULA_NUMERIC].median())
        self.class_share_ = df[config.TARGET].value_counts(normalize=True).sort_index()
        self.models_ = {}
        rng = np.random.default_rng(self.seed)
        for cls, part in num.groupby(config.TARGET):
            n = len(part)
            z_cols, model = [], {"numeric": {}, "categorical": {}}
            for c in COPULA_NUMERIC:
                values = part[c].to_numpy(dtype=float)
                model["numeric"][c] = np.sort(values)
                ranks = stats.rankdata(values, method="average")
                z_cols.append(stats.norm.ppf(ranks / (n + 1)))
            for c in COPULA_CATEGORICAL:
                freq = part[c].value_counts(normalize=True)
                edges = np.concatenate([[0.0], freq.cumsum().to_numpy()])
                model["categorical"][c] = (freq.index.to_numpy(), edges)
                idx = pd.Categorical(part[c], categories=freq.index).codes
                u = edges[idx] + rng.uniform(0, 1, n) * (edges[idx + 1] - edges[idx])
                z_cols.append(stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6)))
            Z = np.column_stack(z_cols)
            corr = np.corrcoef(Z, rowvar=False)
            model["corr"] = (1 - self.shrinkage) * corr + self.shrinkage * np.eye(len(corr))
            self.models_[cls] = model
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed if seed is None else seed)
        counts = np.floor(self.class_share_.to_numpy() * n).astype(int)
        counts[-1] = n - counts[:-1].sum()
        parts = []
        for (cls, model), k in zip(self.models_.items(), counts, strict=False):
            Z = rng.multivariate_normal(
                np.zeros(len(model["corr"])), model["corr"], size=k, method="cholesky"
            )
            U = stats.norm.cdf(Z)
            part = {}
            for j, c in enumerate(COPULA_NUMERIC):
                sorted_vals = model["numeric"][c]
                grid = (np.arange(1, len(sorted_vals) + 1)) / (len(sorted_vals) + 1)
                part[c] = np.interp(U[:, j], grid, sorted_vals)
            offset = len(COPULA_NUMERIC)
            for j, c in enumerate(COPULA_CATEGORICAL):
                levels, edges = model["categorical"][c]
                idx = np.searchsorted(edges, U[:, offset + j], side="right") - 1
                part[c] = levels[np.clip(idx, 0, len(levels) - 1)]
            frame = pd.DataFrame(part)
            frame[config.TARGET] = cls
            parts.append(frame)
        return (
            pd.concat(parts, ignore_index=True)
            .sample(frac=1, random_state=int(rng.integers(1 << 31)))
            .reset_index(drop=True)
        )


class SDVGenerator:
    """Thin wrapper around SDV single-table synthesizers (GaussianCopula, CTGAN)."""

    def __init__(self, kind: str, seed: int = 42, epochs: int = 300):
        self.kind = kind
        self.seed = seed
        self.epochs = epochs
        self.name = f"sdv_{kind}"

    def fit(self, df: pd.DataFrame) -> SDVGenerator:
        import torch
        from sdv.metadata import Metadata
        from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer

        torch.manual_seed(self.seed)
        torch.set_num_threads(1)  # avoids the torch / LightGBM OpenMP clash (see src.nn)
        np.random.seed(self.seed)
        frame = self._prepare(df)
        meta = Metadata.detect_from_dataframe(frame, table_name="claims")
        for c in COPULA_CATEGORICAL + [config.TARGET]:
            meta.update_column(column_name=c, sdtype="categorical", table_name="claims")
        for c in COPULA_NUMERIC:
            meta.update_column(column_name=c, sdtype="numerical", table_name="claims")
        if self.kind == "gaussian_copula":
            self.model_ = GaussianCopulaSynthesizer(meta, enforce_min_max_values=True)
        elif self.kind == "ctgan":
            self.model_ = CTGANSynthesizer(
                meta, epochs=self.epochs, enforce_min_max_values=True, verbose=False, cuda=False
            )
        else:
            raise ValueError(self.kind)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(frame)
        return self

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        frame = to_numeric_frame(df)[COPULA_NUMERIC + COPULA_CATEGORICAL + [config.TARGET]]
        frame[COPULA_NUMERIC] = frame[COPULA_NUMERIC].fillna(frame[COPULA_NUMERIC].median())
        frame[config.TARGET] = frame[config.TARGET].astype(str)
        return frame

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        import torch

        torch.manual_seed(self.seed if seed is None else seed)
        torch.set_num_threads(1)
        np.random.seed(self.seed if seed is None else seed)
        out = self.model_.sample(num_rows=n)
        out[config.TARGET] = out[config.TARGET].astype(int)
        return out


def make_generator(name: str, seed: int, ctgan_epochs: int = 300):
    if name == "gaussian_copula_custom":
        return ClassConditionalGaussianCopula(seed=seed)
    if name == "sdv_gaussian_copula":
        return SDVGenerator("gaussian_copula", seed)
    if name == "sdv_ctgan":
        return SDVGenerator("ctgan", seed, epochs=ctgan_epochs)
    raise ValueError(f"Unknown generator {name}")


def generate(
    generator, n: int, ref: pd.DataFrame, seed: int, chunk: int = 100_000, rules: bool = False
) -> pd.DataFrame:
    """Sample ``n`` rows in chunks, finalise the schema, optionally enforce rules."""
    rng = np.random.default_rng(seed)
    parts = []
    for i, start in enumerate(range(0, n, chunk)):
        part = generator.sample(min(chunk, n - start), seed=seed + i)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out = finalize(out, rng, ref)
    if rules:
        out = enforce_rules(out, rng)
    return out


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------


def marginal_scores(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    """KS statistic (numeric) / total variation distance (categorical); 0 = identical."""
    rows = []
    r, s = to_numeric_frame(real), to_numeric_frame(synth)
    for c in COPULA_NUMERIC + ["total_claim_amount"]:
        rows.append(
            {
                "column": c,
                "type": "numeric",
                "distance": stats.ks_2samp(r[c].dropna(), s[c].dropna()).statistic,
            }
        )
    for c in COPULA_CATEGORICAL + [config.TARGET]:
        p = real[c].value_counts(normalize=True)
        q = synth[c].value_counts(normalize=True)
        rows.append(
            {
                "column": c,
                "type": "categorical",
                "distance": 0.5 * p.subtract(q, fill_value=0).abs().sum(),
            }
        )
    return pd.DataFrame(rows)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y)
    if min(table.shape) < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def association_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    num = to_numeric_frame(df)[COPULA_NUMERIC + ["total_claim_amount", config.TARGET]]
    spearman = num.corr(method="spearman")
    cats = COPULA_CATEGORICAL + [config.TARGET]
    cv = pd.DataFrame(np.eye(len(cats)), index=cats, columns=cats)
    for i, a in enumerate(cats):
        for b in cats[i + 1 :]:
            cv.loc[a, b] = cv.loc[b, a] = cramers_v(df[a], df[b])
    return spearman, cv


def correlation_gap(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, float]:
    rs, rc = association_matrices(real)
    ss, sc = association_matrices(synth)
    mask_s = ~np.eye(len(rs), dtype=bool)
    mask_c = ~np.eye(len(rc), dtype=bool)
    return {
        "spearman_mean_abs_diff": float(np.abs(rs - ss).to_numpy()[mask_s].mean()),
        "cramers_v_mean_abs_diff": float(np.abs(rc - sc).to_numpy()[mask_c].mean()),
        "target_assoc_mean_abs_diff": float(
            np.abs(rc[config.TARGET] - sc[config.TARGET]).drop(config.TARGET).mean()
        ),
    }


def _encode_for_distance(train: pd.DataFrame, *others: pd.DataFrame) -> list[np.ndarray]:
    num_cols = COPULA_NUMERIC + ["total_claim_amount"]
    scaler = StandardScaler().fit(to_numeric_frame(train)[num_cols].fillna(0))
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[COPULA_CATEGORICAL])
    out = []
    for df in (train, *others):
        n = scaler.transform(to_numeric_frame(df)[num_cols].fillna(0))
        c = enc.transform(df[COPULA_CATEGORICAL]) * np.sqrt(0.5)  # a level change counts 1
        out.append(np.hstack([n, c]))
    return out


def privacy_scores(
    real_train: pd.DataFrame, real_holdout: pd.DataFrame, synth: pd.DataFrame
) -> dict[str, float]:
    """Exact copies and distance to closest real record (DCR).

    DCR of synthetic rows is compared to DCR of real hold-out rows (rows the
    generator never saw): a generator that memorises would produce synthetic
    rows much closer to the training rows than genuinely new real rows are.
    """
    cols = COPULA_NUMERIC + COPULA_CATEGORICAL
    key_real = set(map(tuple, to_numeric_frame(real_train)[cols].astype(str).to_numpy()))
    key_syn = map(tuple, to_numeric_frame(synth)[cols].astype(str).to_numpy())
    exact = sum(k in key_real for k in key_syn)
    tr, ho, sy = _encode_for_distance(real_train, real_holdout, synth)
    tree = cKDTree(tr)
    d_holdout = tree.query(ho, k=1)[0]
    d_synth = tree.query(sy, k=1)[0]
    p5 = np.quantile(d_holdout, 0.05)
    return {
        "exact_copies": int(exact),
        "dcr_synth_median": float(np.median(d_synth)),
        "dcr_real_holdout_median": float(np.median(d_holdout)),
        "dcr_ratio": float(np.median(d_synth) / np.median(d_holdout)),
        "share_closer_than_holdout_p5": float((d_synth < p5).mean()),
    }


def detection_auc(real: pd.DataFrame, synth: pd.DataFrame, seed: int) -> float:
    """CV ROC-AUC of a classifier separating real rows from synthetic rows (0.5 = ideal)."""
    from lightgbm import LGBMClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    n = min(len(real), len(synth))
    both = pd.concat(
        [real.sample(n, random_state=seed), synth.sample(n, random_state=seed)], ignore_index=True
    )
    label = pd.Series(np.r_[np.zeros(n), np.ones(n)].astype(int))
    # all generated columns, target included (the joint distribution must match too)
    numeric = COPULA_NUMERIC + ["total_claim_amount", config.TARGET]
    features = to_numeric_frame(both)[numeric + COPULA_CATEGORICAL]
    model = Pipeline(
        [
            (
                "encode",
                ColumnTransformer(
                    [
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=5),
                            COPULA_CATEGORICAL,
                        ),
                    ],
                    remainder="passthrough",
                ),
            ),
            (
                "model",
                LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=15,
                    min_child_samples=10,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    random_state=seed,
                    n_jobs=4,
                    verbose=-1,
                ),
            ),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = cross_val_predict(
            model,
            features,
            label,
            cv=StratifiedKFold(5, shuffle=True, random_state=seed),
            method="predict_proba",
        )[:, 1]
    return float(roc_auc_score(label, proba))


def tstr_scores(
    build_model, synth: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Train on synthetic, test on the real hold-out split."""
    X, y = synth.drop(columns=[config.TARGET]), synth[config.TARGET]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_model().fit(X, y)
    p = model.predict_proba(X_test)[:, 1]
    return {
        "pr_auc": float(average_precision_score(y_test, p)),
        "roc_auc": float(roc_auc_score(y_test, p)),
    }
