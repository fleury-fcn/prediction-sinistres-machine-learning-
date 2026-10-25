"""Synthetic data stage. Everything produced here is labelled SYNTHETIC.

1. Fit each generator on the real *training* split.
2. Quality report per generator: marginals, associations, fraud rate, business
   rule violations, real-vs-synthetic detection AUC, privacy (copies, DCR) and
   TSTR (train on synthetic, test on the real test split) against TRTR.
3. Pick the generator with the lowest detection AUC (a train-only criterion; the
   real test split plays no role in the choice) and generate the large dataset.
4. Scale benchmark: fit time and performance vs training size, evaluated on a
   synthetic hold-out and on the real test split.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .. import config, plots, synthetic
from ..config import Settings
from ..models import MODELS, build
from .common import read_json, real_split, restore_params, write_json

logger = logging.getLogger(__name__)

SCALE_MODELS = [
    "logreg_enet",
    "lightgbm",
    "xgboost",
    "catboost",
    "mlp",
    "random_forest",
    "gradient_boosting",
]
# sklearn RandomForest / GradientBoosting become impractically slow above this size
SIZE_CAPS = {"random_forest": 100_000, "gradient_boosting": 100_000}
SYNTH_TEST_ROWS = 50_000
GENERATOR_LABELS = {
    "gaussian_copula_custom": "Gaussian copula (custom, per class)",
    "sdv_gaussian_copula": "SDV GaussianCopula",
    "sdv_ctgan": "SDV CTGAN",
}
GENERATOR_COLORS = {
    "gaussian_copula_custom": plots.BLUE,
    "sdv_gaussian_copula": plots.ORANGE,
    "sdv_ctgan": plots.AQUA,
}


def _params() -> dict[str, dict]:
    path = config.METRICS_DIR / "real_best_params.json"
    tuned = read_json(path) if path.exists() else {}
    return {n: restore_params(tuned.get(n, MODELS[n].default_params)) for n in MODELS}


def run(settings: Settings) -> dict[str, Any]:
    plots.apply_style()
    config.ensure_dirs()
    seed, cfg = settings.seed, settings["synthetic"]
    X_train, X_test, y_train, y_test = real_split(settings)
    real_train = X_train.assign(**{config.TARGET: y_train})
    real_test = X_test.assign(**{config.TARGET: y_test})
    params = _params()

    def lgbm():
        return build("lightgbm", params["lightgbm"], seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trtr_model = lgbm().fit(X_train, y_train)
    p = trtr_model.predict_proba(X_test)[:, 1]
    trtr = {"pr_auc": average_precision_score(y_test, p), "roc_auc": roc_auc_score(y_test, p)}

    rows, samples, generators = [], {}, {}
    for name in cfg["generators"]:
        logger.info("Fitting generator %s on %d real training rows", name, len(real_train))
        start = time.perf_counter()
        gen = synthetic.make_generator(name, seed, cfg["ctgan_epochs"]).fit(real_train)
        fit_s = time.perf_counter() - start
        start = time.perf_counter()
        raw = synthetic.generate(gen, cfg["validation_rows"], real_train, seed)
        sample_s = time.perf_counter() - start
        # the same rule post-processing is applied to every generator (and to the
        # large dataset); the raw violation rate is reported separately
        sample = synthetic.enforce_rules(raw, np.random.default_rng(seed))
        generators[name], samples[name] = gen, sample

        marg = synthetic.marginal_scores(real_train, sample)
        marg.assign(generator=name).to_csv(
            config.METRICS_DIR / f"synthetic_marginals_{name}.csv", index=False
        )
        small = sample.sample(len(real_train), random_state=seed)
        row = {
            "generator": name,
            "fit_seconds": fit_s,
            "sample_seconds_per_100k": sample_s * 100_000 / cfg["validation_rows"],
            "fraud_rate": sample[config.TARGET].mean(),
            "ks_numeric_mean": marg.loc[marg.type == "numeric", "distance"].mean(),
            "tvd_categorical_mean": marg.loc[marg.type == "categorical", "distance"].mean(),
            **synthetic.correlation_gap(real_train, sample),
            "rule_violation_rate_raw": synthetic.rule_violation_rate(raw),
            "detection_auc": synthetic.detection_auc(real_train, small, seed),
            **synthetic.privacy_scores(real_train, real_test, sample),
        }
        tstr_small = synthetic.tstr_scores(lgbm, small, X_test, y_test)
        tstr_big = synthetic.tstr_scores(lgbm, sample, X_test, y_test)
        row.update(
            {
                "tstr_pr_auc_n800": tstr_small["pr_auc"],
                "tstr_roc_auc_n800": tstr_small["roc_auc"],
                f"tstr_pr_auc_n{cfg['validation_rows']}": tstr_big["pr_auc"],
                f"tstr_roc_auc_n{cfg['validation_rows']}": tstr_big["roc_auc"],
                "trtr_pr_auc_n800": trtr["pr_auc"],
                "trtr_roc_auc_n800": trtr["roc_auc"],
            }
        )
        rows.append(row)
        logger.info(
            "%s: %s", name, {k: round(v, 3) for k, v in row.items() if isinstance(v, float)}
        )
    quality = pd.DataFrame(rows)
    quality.to_csv(config.METRICS_DIR / "synthetic_quality.csv", index=False)

    chosen = quality.sort_values("detection_auc").iloc[0]["generator"]
    logger.info("Generator selected on detection AUC (train-only criterion): %s", chosen)
    _fig("synthetic_marginals.png", fig_marginals(real_train, samples))
    _fig("synthetic_correlations.png", fig_correlations(real_train, samples[chosen], chosen))
    _fig("synthetic_quality.png", fig_quality(quality, cfg["validation_rows"]))

    n_rows = int(cfg["n_rows"])
    logger.info("Generating %d synthetic rows with %s", n_rows, chosen)
    large = synthetic.generate(
        generators[chosen], n_rows + SYNTH_TEST_ROWS, real_train, seed + 1, rules=True
    )
    path = config.SYNTHETIC_DIR / f"SYNTHETIC_claims_{chosen}_{n_rows}.parquet"
    large.iloc[:n_rows].to_parquet(path, index=False)
    large.iloc[:1000].to_csv(config.SYNTHETIC_DIR / "SYNTHETIC_sample_1000.csv", index=False)
    write_json(
        {
            "generator": chosen,
            "n_rows": n_rows,
            "path": str(path.relative_to(config.ROOT_DIR)),
            "fitted_on": "real training split (800 rows)",
            "rules_enforced": ["collision_type UNKNOWN iff incident is theft / parked car"],
            "fraud_rate": float(large[config.TARGET].mean()),
        },
        config.METRICS_DIR / "synthetic_dataset.json",
    )

    scale = scale_benchmark(large, n_rows, cfg["scale_sizes"], params, X_test, y_test, seed)
    scale.to_csv(config.METRICS_DIR / "synthetic_scale_benchmark.csv", index=False)
    _fig("synthetic_scale.png", fig_scale(scale, trtr))
    return {"quality": quality, "scale": scale, "chosen": chosen}


def _fig(name: str, fig: plt.Figure) -> None:
    plots.save(fig, config.FIGURES_DIR / name)


def _scale_params(name: str, params: dict, n: int) -> dict:
    """The MLP's tuned batch size (<= 128) is meant for 640 rows; on large data the
    batch is enlarged and the epoch budget reduced so one fit stays tractable."""
    p = dict(params)
    if name == "mlp" and n > 10_000:
        p.update(batch_size=max(p.get("batch_size", 64), 1024), max_epochs=30, patience=4)
    return p


def scale_benchmark(
    large: pd.DataFrame,
    n_rows: int,
    sizes: list[int],
    params: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
) -> pd.DataFrame:
    train, hold = large.iloc[:n_rows], large.iloc[n_rows:]
    X_hold, y_hold = hold.drop(columns=[config.TARGET]), hold[config.TARGET]
    rows = []
    for size in sizes:
        sub = train.sample(min(size, len(train)), random_state=seed)
        X, y = sub.drop(columns=[config.TARGET]), sub[config.TARGET]
        for name in SCALE_MODELS:
            if size > SIZE_CAPS.get(name, np.inf):
                continue
            model = build(name, _scale_params(name, params[name], size), seed)
            start = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
            fit_s = time.perf_counter() - start
            p_hold = model.predict_proba(X_hold)[:, 1]
            p_real = model.predict_proba(X_test)[:, 1]
            rows.append(
                {
                    "model": name,
                    "n_train": len(sub),
                    "fit_seconds": fit_s,
                    "synthetic_test_pr_auc": average_precision_score(y_hold, p_hold),
                    "synthetic_test_roc_auc": roc_auc_score(y_hold, p_hold),
                    "real_test_pr_auc_tstr": average_precision_score(y_test, p_real),
                    "real_test_roc_auc_tstr": roc_auc_score(y_test, p_real),
                }
            )
            logger.info(
                "scale %s n=%d: %.1fs, synth PR-AUC %.3f, real PR-AUC %.3f",
                name,
                len(sub),
                fit_s,
                rows[-1]["synthetic_test_pr_auc"],
                rows[-1]["real_test_pr_auc_tstr"],
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

MARGINAL_NUMERIC = [
    "age",
    "months_as_customer",
    "policy_annual_premium",
    "vehicle_claim",
    "injury_claim",
    "policy_bind_date",
]
MARGINAL_CATEGORICAL = ["incident_severity", "collision_type", "insured_hobbies"]


def fig_marginals(real: pd.DataFrame, samples: dict[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    r = synthetic.to_numeric_frame(real)
    for ax, col in zip(axes.ravel()[:6], MARGINAL_NUMERIC, strict=False):
        bins = np.histogram_bin_edges(r[col].dropna(), bins=25)
        ax.hist(
            r[col].dropna(),
            bins=bins,
            density=True,
            color=plots.MUTED,
            alpha=0.45,
            label="Real (train split)",
        )
        for name, s in samples.items():
            sv = synthetic.to_numeric_frame(s)[col]
            ax.hist(
                sv,
                bins=bins,
                density=True,
                histtype="step",
                lw=1.8,
                color=GENERATOR_COLORS[name],
                label=GENERATOR_LABELS[name],
            )
        ax.set_title(col + (" (days since 1970)" if col.endswith("date") else ""), fontsize=10)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    axes[0, 0].legend(fontsize=7.5)
    for ax, col in zip(axes.ravel()[6:], MARGINAL_CATEGORICAL, strict=False):
        levels = real[col].value_counts().index[:10]
        x = np.arange(len(levels))
        width = 0.8 / (1 + len(samples))
        ax.bar(
            x,
            real[col].value_counts(normalize=True).reindex(levels),
            width,
            color=plots.MUTED,
            alpha=0.6,
            label="Real",
        )
        for k, (name, s) in enumerate(samples.items(), start=1):
            ax.bar(
                x + k * width,
                s[col].value_counts(normalize=True).reindex(levels).fillna(0),
                width,
                color=GENERATOR_COLORS[name],
                label=GENERATOR_LABELS[name],
            )
        ax.set_xticks(x + width * len(samples) / 2, levels, rotation=40, ha="right", fontsize=7.5)
        ax.set_title(f"{col} (top levels, share)", fontsize=10)
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        f"{plots.SYNTHETIC_TAG}: marginal distributions, real training split vs " "each generator",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def fig_correlations(real: pd.DataFrame, synth: pd.DataFrame, name: str) -> plt.Figure:
    rs, _ = synthetic.association_matrices(real)
    ss, _ = synthetic.association_matrices(synth)
    cols = rs.columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    for ax, mat, title in zip(
        axes,
        [rs, ss, ss - rs],
        [
            "Real (train split)",
            f"Synthetic: {GENERATOR_LABELS[name]}",
            "Difference (synthetic - real)",
        ],
        strict=False,
    ):
        im = ax.imshow(mat, cmap=plots.DIVERGING, vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
        ax.set_yticks(range(len(cols)), cols if ax is axes[0] else [], fontsize=7)
        ax.grid(False)
        ax.set_title(title, fontsize=10.5)
    fig.colorbar(im, ax=axes, shrink=0.75, label="Spearman correlation")
    fig.suptitle(
        f"{plots.SYNTHETIC_TAG}: Spearman correlations of numeric columns and target",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    return fig


def fig_quality(q: pd.DataFrame, n_val: int) -> plt.Figure:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.2))
    labels = [GENERATOR_LABELS[g] for g in q.generator]
    colors = [GENERATOR_COLORS[g] for g in q.generator]
    y = np.arange(len(q))
    ax1.barh(y, q.detection_auc, color=colors, height=0.6)
    ax1.axvline(0.5, color=plots.INK_2, lw=1, linestyle="--")
    ax1.set_xlim(0.4, 1.0)
    ax1.set_yticks(y, labels)
    ax1.set_title("Real-vs-synthetic detection ROC-AUC\n(0.5 = indistinguishable)", fontsize=10)
    ax2.barh(y, q.ks_numeric_mean, color=colors, height=0.6)
    ax2.set_yticks(y, [])
    ax2.set_title("Mean KS distance of numeric marginals\n(0 = identical)", fontsize=10)
    width = 0.38
    ax3.barh(
        y - width / 2,
        q.tstr_pr_auc_n800,
        width,
        color=colors,
        alpha=0.55,
        label="TSTR, 800 synthetic rows",
    )
    ax3.barh(
        y + width / 2,
        q[f"tstr_pr_auc_n{n_val}"],
        width,
        color=colors,
        label=f"TSTR, {n_val:,} synthetic rows",
    )
    ax3.axvline(q.trtr_pr_auc_n800.iloc[0], color=plots.INK, lw=1.2, linestyle="--")
    ax3.text(
        q.trtr_pr_auc_n800.iloc[0],
        -0.62,
        " TRTR (800 real rows)",
        fontsize=8,
        color=plots.INK,
    )
    ax3.set_yticks(y, [])
    ax3.set_xlim(0, 1)
    ax3.set_title("LightGBM PR-AUC on the REAL test split", fontsize=10)
    ax3.legend(fontsize=7.5, loc="lower right")
    for a in (ax1, ax2, ax3):
        a.grid(axis="y", visible=False)
    fig.suptitle(
        f"{plots.SYNTHETIC_TAG}: generator quality (all generators fitted on the "
        "800-row real training split)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def fig_scale(scale: pd.DataFrame, trtr: dict[str, float]) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for name, g in scale.groupby("model"):
        style = {
            "color": plots.model_color(name),
            "marker": "o",
            "ms": 4,
            "label": MODELS[name].label,
        }
        axes[0].plot(g.n_train, g.fit_seconds, **style)
        axes[1].plot(g.n_train, g.synthetic_test_pr_auc, **style)
        axes[2].plot(g.n_train, g.real_test_pr_auc_tstr, **style)
    axes[0].set(
        xscale="log", yscale="log", xlabel="Training rows (synthetic)", ylabel="Fit time (s)"
    )
    axes[0].set_title("Fit time vs training size")
    axes[1].set(xscale="log", xlabel="Training rows (synthetic)", ylabel="PR-AUC")
    axes[1].set_title(f"PR-AUC on a {SYNTH_TEST_ROWS:,}-row SYNTHETIC hold-out")
    axes[2].axhline(trtr["pr_auc"], color=plots.INK, lw=1.2, linestyle="--")
    axes[2].text(
        0.01,
        trtr["pr_auc"],
        " LightGBM trained on 800 real rows (TRTR)",
        va="bottom",
        fontsize=8,
        transform=axes[2].get_yaxis_transform(),  # x in axes units: safe on a log axis
    )
    axes[2].set(xscale="log", xlabel="Training rows (synthetic)", ylabel="PR-AUC")
    axes[2].set_title("TSTR: PR-AUC on the REAL test split (n = 200)")
    axes[0].legend(fontsize=7.5)
    fig.suptitle(
        f"{plots.SYNTHETIC_TAG}: scaling benchmark (not a measure of real-world " "performance)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig
