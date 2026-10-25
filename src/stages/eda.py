"""Exploratory data analysis of the real dataset (descriptive only).

EDA looks at the full 1,000 claims to describe the data. None of its outputs
feed the models: every learned transformation is fitted inside the training
pipelines.
"""

from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .. import config, data, plots
from ..config import Settings
from ..features import ClaimFeatureBuilder

logger = logging.getLogger(__name__)

KEY_NUMERIC = [
    "age",
    "months_as_customer",
    "policy_annual_premium",
    "policy_tenure_days",
    "incident_hour",
    "witnesses",
    "total_claim_amount",
    "vehicle_age",
]
KEY_CATEGORICAL = [
    "incident_severity",
    "insured_hobbies",
    "collision_type",
    "incident_type",
    "authorities_contacted",
    "property_damage",
]


def wilson_interval(k: np.ndarray, n: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, ...]:
    z = stats.norm.ppf(0.5 + level / 2)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def fraud_rate_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col)[config.TARGET].agg(["sum", "count"])
    g["rate"] = g["sum"] / g["count"]
    g["lo"], g["hi"] = wilson_interval(g["sum"].to_numpy(), g["count"].to_numpy())
    return g.sort_values("rate")


def summary(df: pd.DataFrame) -> dict:
    numeric = df[config.RAW_NUMERIC_COLUMNS]
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    outliers = ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum()
    unknown = (df[config.RAW_CATEGORICAL_COLUMNS] == config.UNKNOWN_CATEGORY).sum()
    total_gap = (df["total_claim_amount"] - df[config.CLAIM_COMPONENTS].sum(axis=1)).abs()
    return {
        "n_rows": int(len(df)),
        "n_source_columns": len(config.RENAME_MAP),
        "fraud_rate": float(df[config.TARGET].mean()),
        "n_fraud": int(df[config.TARGET].sum()),
        "duplicated_rows": int(df.duplicated().sum()),
        "unknown_category_counts": {k: int(v) for k, v in unknown[unknown > 0].items()},
        "numeric_nan_counts": {k: int(v) for k, v in numeric.isna().sum().items() if v},
        "iqr_outlier_counts": {k: int(v) for k, v in outliers[outliers > 0].items()},
        "total_claim_equals_sum_of_components": bool((total_gap < 1e-6).all()),
        "incident_date_range": [
            str(df.incident_date.min().date()),
            str(df.incident_date.max().date()),
        ],
        "univariate_roc_auc_claim_amounts": {
            c: float(_auc(df[config.TARGET], df[c]))
            for c in ["total_claim_amount", *config.CLAIM_COMPONENTS]
        },
    }


def _auc(y: pd.Series, x: pd.Series) -> float:
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(y, x)


def fig_class_balance(df: pd.DataFrame) -> plt.Figure:
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    ax.barh(
        [plots.CLASS_LABELS[i] for i in counts.index],
        counts.values,
        color=[plots.CLASS_COLORS[i] for i in counts.index],
        height=0.6,
    )
    for i, v in enumerate(counts.values):
        ax.text(v, i, f"  {v} ({v / counts.sum():.1%})", va="center", color=plots.INK_2)
    ax.set_xlim(0, counts.max() * 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Number of claims")
    ax.set_title("Class balance (real data, n = 1,000)")
    fig.tight_layout()
    return fig


def fig_numeric_distributions(feat: pd.DataFrame, y: pd.Series) -> plt.Figure:
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, col in zip(axes.ravel(), KEY_NUMERIC, strict=False):
        values = feat[col]
        bins = np.histogram_bin_edges(values.dropna(), bins=20)
        for cls in (0, 1):
            ax.hist(
                values[y == cls].dropna(),
                bins=bins,
                density=True,
                alpha=0.45,
                color=plots.CLASS_COLORS[cls],
                label=plots.CLASS_LABELS[cls],
                edgecolor=plots.SURFACE,
                linewidth=0.5,
            )
        ks = stats.ks_2samp(values[y == 0].dropna(), values[y == 1].dropna())
        ax.set_title(f"{col}\nKS p = {ks.pvalue:.2g}", fontsize=9.5)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Key numeric features by class (density; KS test legit vs fraud)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def fig_fraud_rate_by_category(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    overall = df[config.TARGET].mean()
    for ax, col in zip(axes.ravel(), KEY_CATEGORICAL, strict=False):
        t = fraud_rate_table(df, col)
        y = np.arange(len(t))
        ax.barh(y, t["rate"], color=plots.ORANGE, alpha=0.85, height=0.62)
        ax.errorbar(
            t["rate"],
            y,
            xerr=[t["rate"] - t["lo"], t["hi"] - t["rate"]],
            fmt="none",
            ecolor=plots.INK,
            elinewidth=0.9,
            capsize=2,
        )
        ax.set_yticks(
            y, [f"{idx} (n={n})" for idx, n in zip(t.index, t["count"], strict=False)], fontsize=8
        )
        ax.axvline(overall, color=plots.INK_2, lw=1, linestyle="--")
        ax.set_xlim(0, 1)
        ax.grid(axis="y", visible=False)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("Fraud rate (95% Wilson CI)")
    fig.suptitle(
        f"Fraud rate by category (dashed line = overall rate {overall:.1%})",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def fig_correlation(feat: pd.DataFrame, y: pd.Series) -> plt.Figure:
    num = feat.select_dtypes("number").copy()
    num["is_fraud"] = y.values
    corr = num.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(corr, cmap=plots.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            if i != j and abs(corr.iat[i, j]) >= 0.3:
                ax.text(
                    j,
                    i,
                    f"{corr.iat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if abs(corr.iat[i, j]) > 0.7 else plots.INK,
                )
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman correlation")
    ax.set_title("Spearman correlation of numeric features and target (|r| >= 0.3 annotated)")
    fig.tight_layout()
    return fig


def fig_data_quality(df: pd.DataFrame, info: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    unk = pd.Series(info["unknown_category_counts"]).sort_values()
    ax1.barh(unk.index, unk.values / len(df), color=plots.BLUE, height=0.55)
    for i, v in enumerate(unk.values):
        ax1.text(v / len(df), i, f"  {v}", va="center", fontsize=8, color=plots.INK_2)
    ax1.set_xlim(0, 0.5)
    ax1.set_xlabel("Share of rows with '?' (kept as category UNKNOWN)")
    ax1.set_title("Missing values ('?' markers)")
    ax1.grid(axis="y", visible=False)
    out = pd.Series(info["iqr_outlier_counts"]).sort_values()
    ax2.barh(out.index, out.values, color=plots.YELLOW, height=0.55)
    for i, v in enumerate(out.values):
        ax2.text(v, i, f"  {v}", va="center", fontsize=8, color=plots.INK_2)
    ax2.set_xlabel("Rows outside [Q1 - 1.5 IQR, Q3 + 1.5 IQR]")
    ax2.set_title("Outliers by IQR rule (kept; tree models are robust to them)")
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def run(settings: Settings) -> dict:
    plots.apply_style()
    config.ensure_dirs()
    df = data.load_clean(settings.raw_data_path)
    feat = ClaimFeatureBuilder(drop_total_claim_amount=False).fit(df).transform(df)
    y = df[config.TARGET]

    info = summary(df)
    with open(config.METRICS_DIR / "eda_summary.json", "w") as fh:
        json.dump(info, fh, indent=2)

    figs = {
        "eda_class_balance.png": fig_class_balance(df),
        "eda_numeric_distributions.png": fig_numeric_distributions(feat, y),
        "eda_fraud_rate_by_category.png": fig_fraud_rate_by_category(df),
        "eda_correlation_matrix.png": fig_correlation(feat, y),
        "eda_data_quality.png": fig_data_quality(df, info),
    }
    for name, fig in figs.items():
        plots.save(fig, config.FIGURES_DIR / name)
    logger.info("EDA: %d figures written to %s", len(figs), config.FIGURES_DIR)
    return info
