"""Evaluation stage on the real hold-out test set (opened here for the first time).

* Every model is scored once on the test split at the threshold chosen on its
  training OOF probabilities; 95 % CIs come from a stratified bootstrap.
* The reference model is chosen on cross-validation (not on the test set).
* Comparisons vs the reference: corrected resampled t-test on CV folds,
  paired bootstrap of PR-AUC/ROC-AUC differences and McNemar on the test set,
  Holm-adjusted.
* Calibration, cost analysis, SHAP, permutation importance, learning curves.
* The deployable model is exported to ``results/models``.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, learning_curve

from .. import config, evaluation, interpret, plots
from ..config import Settings
from ..evaluation import CostModel
from ..models import MODELS, build
from .common import (
    MODEL_ARTIFACTS,
    TRAIN_ARTIFACTS,
    cost_model,
    read_json,
    real_split,
    restore_params,
    write_json,
)

logger = logging.getLogger(__name__)

TREE_MODELS = ["catboost", "lightgbm", "xgboost", "gradient_boosting", "random_forest"]
TORCH_MODELS = {"mlp"}


def _fig(name: str, fig: plt.Figure) -> None:
    plots.save(fig, config.FIGURES_DIR / name)


def run(settings: Settings) -> dict[str, Any]:
    plots.apply_style()
    config.ensure_dirs()
    seed, costs = settings.seed, cost_model(settings)
    n_boot, ci = settings["evaluation"]["n_bootstrap"], settings["evaluation"]["ci_level"]
    X_train, X_test, y_train, y_test = real_split(settings)
    y_te = y_test.to_numpy()

    meta = read_json(TRAIN_ARTIFACTS / "train_meta.json")
    thresholds, labels = meta["thresholds"], meta["labels"]
    cv_summary = pd.read_csv(config.METRICS_DIR / "real_cv_summary.csv").set_index("model")
    folds = pd.read_csv(config.METRICS_DIR / "real_cv_fold_scores.csv")
    best_params = read_json(config.METRICS_DIR / "real_best_params.json")
    names = cv_summary.index.tolist()
    fitted = {n: joblib.load(MODEL_ARTIFACTS / f"{n}.joblib") for n in names}
    proba = {n: m.predict_proba(X_test)[:, 1] for n, m in fitted.items()}

    # -- test metrics with bootstrap CIs ------------------------------------------
    tables = []
    for n in names:
        t = evaluation.bootstrap_metrics(y_te, proba[n], thresholds[n], costs, n_boot, ci, seed)
        tables.append(t.assign(model=n).rename_axis("metric").reset_index())
    test_table = pd.concat(tables).set_index(["model", "metric"])
    test_table.to_csv(config.METRICS_DIR / "real_test_metrics.csv")
    point = test_table["estimate"].unstack()

    reference = cv_summary.index[0]  # best mean CV PR-AUC: selected without the test set
    comparisons = compare_models(
        reference, names, folds, proba, thresholds, y_te, len(X_train), settings
    )
    comparisons.to_csv(config.METRICS_DIR / "real_model_comparison_tests.csv", index=False)

    trivial = evaluation.trivial_policy_costs(y_te, costs)
    write_json(
        {
            "reference_model": reference,
            "trivial_policies_test": trivial,
            "costs": {"investigation": costs.investigation, "missed_fraud": costs.missed_fraud},
            "bayes_threshold": costs.bayes_threshold,
        },
        config.METRICS_DIR / "real_reference.json",
    )

    # -- figures: discrimination ------------------------------------------------
    curves = {}
    for n in names:
        fpr, tpr, _ = roc_curve(y_te, proba[n])
        prec, rec, _ = precision_recall_curve(y_te, proba[n])
        curves[n] = {"fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec}
    _fig("real_roc_pr_curves.png", plots.roc_pr_curves(curves, labels, point, float(y_te.mean())))
    _fig(
        "real_metrics_ci.png",
        plots.metric_bars_with_ci(
            test_table,
            labels,
            ["pr_auc", "roc_auc", "mcc", "brier"],
            "Hold-out test metrics with 95% bootstrap CIs (real data, n = 200)",
        ),
    )
    _fig("real_cv_pr_auc_folds.png", plots.cv_boxplot(folds, labels))
    _fig(
        "real_confusion_matrices.png",
        plots.confusion_grid(
            {n: confusion_matrix(y_te, (proba[n] >= thresholds[n]).astype(int)) for n in names},
            labels,
            thresholds,
        ),
    )

    focus = list(dict.fromkeys([reference, *names[:4], "logreg"]))
    focus = [n for n in focus if n in names]
    _fig(
        "real_cost_vs_threshold.png",
        plots.cost_vs_threshold(
            {n: evaluation.cost_curve(y_te, proba[n], costs) for n in focus},
            labels,
            {n: thresholds[n] for n in focus},
            trivial,
            f"Assumed costs: investigation = ${costs.investigation:,.0f} per flagged claim\n"
            f"missed fraud = ${costs.missed_fraud:,.0f} per undetected fraudulent claim",
        ),
    )
    sensitivity = cost_sensitivity(reference, y_train, y_te, proba[reference], costs)
    sensitivity.to_csv(config.METRICS_DIR / "real_cost_sensitivity.csv", index=False)
    _fig("real_cost_sensitivity.png", fig_cost_sensitivity(sensitivity, labels[reference]))

    # -- calibration --------------------------------------------------------------
    calib = calibration_analysis(
        reference, best_params, X_train, y_train, X_test, y_te, proba, focus, labels, seed
    )
    calib.to_csv(config.METRICS_DIR / "real_calibration.csv", index=False)

    # -- interpretability -----------------------------------------------------------
    explained = next((n for n in names if n in TREE_MODELS), None)
    shap_top = explain(explained, fitted[explained], X_test, y_te, proba[explained], labels)
    perm = interpret.permutation_table(
        fitted[reference], X_test, y_test, n_repeats=10 if settings.fast else 30, seed=seed
    )
    perm.to_csv(config.METRICS_DIR / "real_permutation_importance.csv")
    _fig(
        "real_permutation_importance.png",
        plots.importance_bars(
            perm,
            "importance_mean",
            "importance_sd",
            f"Permutation importance on the test set: {labels[reference]}",
            "Drop in PR-AUC when the column is shuffled (mean +/- sd)",
        ),
    )

    # -- learning curves & NN training ----------------------------------------------
    lc_models = list(dict.fromkeys([reference, "logreg_enet"]))
    lcs = {
        n: learning_curve_table(n, best_params, X_train, y_train, settings)
        for n in lc_models
        if n in best_params
    }
    _fig("real_learning_curves.png", plots.learning_curve_plot(lcs, labels))
    if "mlp" in fitted:
        mlp = fitted["mlp"].named_steps["model"]
        _fig("real_mlp_loss_curves.png", plots.nn_loss_curves(mlp.history_, mlp.best_epoch_))

    deployable = export_model(
        cv_summary,
        fitted,
        thresholds,
        labels,
        test_table,
        costs,
        meta["stacking_members"],
        best_params,
    )
    write_results_markdown(
        cv_summary, test_table, comparisons, labels, reference, trivial, calib, deployable
    )
    logger.info("Evaluation done. Reference model: %s; deployable: %s", reference, deployable)
    return {"reference": reference, "test_table": test_table, "shap_top": shap_top}


# ---------------------------------------------------------------------------


def compare_models(
    reference: str,
    names: list[str],
    folds: pd.DataFrame,
    proba: dict[str, np.ndarray],
    thresholds: dict[str, float],
    y_te: np.ndarray,
    n_train: int,
    settings: Settings,
) -> pd.DataFrame:
    n_boot, seed = settings["evaluation"]["n_bootstrap"], settings.seed
    k = settings["cv"]["eval_folds"]
    ref_folds = folds[folds.model == reference].sort_values(["repeat", "fold"])
    rows = []
    for n in names:
        if n == reference:
            continue
        other = folds[folds.model == n].sort_values(["repeat", "fold"])
        cv_t = evaluation.corrected_resampled_ttest(
            ref_folds.pr_auc.to_numpy(),
            other.pr_auc.to_numpy(),
            n_train=int(n_train * (k - 1) / k),
            n_test=int(n_train / k),
        )
        pr = evaluation.paired_bootstrap_diff(
            y_te, proba[reference], proba[n], average_precision_score, n_boot, seed=seed
        )
        roc = evaluation.paired_bootstrap_diff(
            y_te, proba[reference], proba[n], roc_auc_score, n_boot, seed=seed
        )
        mc = evaluation.mcnemar_test(
            y_te,
            (proba[reference] >= thresholds[reference]).astype(int),
            (proba[n] >= thresholds[n]).astype(int),
        )
        rows.append(
            {
                "reference": reference,
                "model": n,
                "cv_pr_auc_diff": cv_t["mean_diff"],
                "cv_ttest_p": cv_t["p_value"],
                "test_pr_auc_diff": pr["diff"],
                "test_pr_auc_diff_ci_low": pr["ci_low"],
                "test_pr_auc_diff_ci_high": pr["ci_high"],
                "test_pr_auc_boot_p": pr["p_value"],
                "test_roc_auc_diff": roc["diff"],
                "test_roc_auc_boot_p": roc["p_value"],
                "mcnemar_ref_right_other_wrong": mc["a_right_b_wrong"],
                "mcnemar_ref_wrong_other_right": mc["a_wrong_b_right"],
                "mcnemar_p": mc["p_value"],
            }
        )
    out = pd.DataFrame(rows)
    for col in ["cv_ttest_p", "test_pr_auc_boot_p", "test_roc_auc_boot_p", "mcnemar_p"]:
        out[col + "_holm"] = evaluation.holm_correction(out[col]).to_numpy()
    return out


def cost_sensitivity(
    reference: str, y_train: pd.Series, y_te: np.ndarray, test_proba: np.ndarray, base: CostModel
) -> pd.DataFrame:
    """How the chosen threshold and the savings move with the FN/FP cost ratio."""
    oof = pd.read_parquet(TRAIN_ARTIFACTS / "oof_proba.parquet")
    rows = []
    for ratio in [2, 3, 5, 7.5, 10, 15, 20, 30, 50]:
        c = CostModel(base.investigation, base.investigation * ratio)
        t = evaluation.cost_optimal_threshold(y_train.to_numpy(), oof[reference].to_numpy(), c)
        model_cost = c.per_claim(y_te, (test_proba >= t).astype(int))
        trivial = evaluation.trivial_policy_costs(y_te, c)
        best_trivial = min(trivial.values())
        rows.append(
            {
                "cost_ratio": ratio,
                "threshold": t,
                "model_cost": model_cost,
                "best_trivial_cost": best_trivial,
                "best_trivial_policy": min(trivial, key=trivial.get),
                "savings_pct": 100 * (best_trivial - model_cost) / best_trivial,
            }
        )
    return pd.DataFrame(rows)


def fig_cost_sensitivity(df: pd.DataFrame, label: str) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(df.cost_ratio, df.threshold, marker="o", color=plots.BLUE)
    ax1.set(
        xscale="log",
        xlabel="Cost ratio missed fraud / investigation",
        ylabel="Threshold chosen on train OOF",
    )
    ax1.set_title("Decision threshold vs cost ratio")
    ax2.plot(df.cost_ratio, df.savings_pct, marker="o", color=plots.GREEN)
    ax2.axhline(0, color=plots.AXIS, lw=1)
    ax2.set(
        xscale="log",
        xlabel="Cost ratio missed fraud / investigation",
        ylabel="Savings vs best trivial policy (%)",
    )
    ax2.set_title("Test-set savings vs the cheaper of 'investigate all / none'")
    for a in (ax1, ax2):
        a.set_xticks(df.cost_ratio, [f"{r:g}" for r in df.cost_ratio])
        a.minorticks_off()
    fig.suptitle(f"Cost sensitivity: {label} (real test set)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def calibration_analysis(
    reference, best_params, X_train, y_train, X_test, y_te, proba, focus, labels, seed
) -> pd.DataFrame:
    """Reliability of raw probabilities, and Platt / isotonic recalibration of the
    reference model (fitted by internal 5-fold CV on the training split)."""
    rows, curves, brier = [], {}, {}
    for n in focus:
        mp, fp = calibration_curve(y_te, proba[n], n_bins=8, strategy="quantile")
        curves[n], brier[n] = (mp, fp), brier_score_loss(y_te, proba[n])
        rows.append(
            {
                "model": n,
                "method": "uncalibrated",
                "brier": brier[n],
                "pr_auc": average_precision_score(y_te, proba[n]),
            }
        )
    if reference in MODELS:
        for method in ("sigmoid", "isotonic"):
            est = CalibratedClassifierCV(
                build(reference, restore_params(best_params[reference]), seed), method=method, cv=5
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(X_train, y_train)
            p = est.predict_proba(X_test)[:, 1]
            key = f"{reference}::{method}"
            labels[key] = f"{labels[reference]} + {'Platt' if method == 'sigmoid' else 'isotonic'}"
            mp, fp = calibration_curve(y_te, p, n_bins=8, strategy="quantile")
            curves[key], brier[key] = (mp, fp), brier_score_loss(y_te, p)
            rows.append(
                {
                    "model": reference,
                    "method": method,
                    "brier": brier[key],
                    "pr_auc": average_precision_score(y_te, p),
                }
            )
    _fig(
        "real_calibration.png",
        plots.calibration_plot(
            curves, labels, brier, "Reliability diagram (real test set, quantile bins)"
        ),
    )
    return pd.DataFrame(rows)


def explain(
    name: str,
    pipeline,
    X_test: pd.DataFrame,
    y_te: np.ndarray,
    proba: np.ndarray,
    labels: dict[str, str],
) -> list[str]:
    exp, source, shown = interpret.shap_explanation(pipeline, X_test)
    grouped = interpret.grouped_mean_abs_shap(exp, source)
    grouped.to_csv(
        config.METRICS_DIR / "real_shap_mean_abs_by_feature.csv", header=["mean_abs_shap"]
    )

    plt.figure()
    shap.plots.beeswarm(exp, max_display=15, show=False, plot_size=(9, 6.5))
    fig = plt.gcf()
    fig.axes[0].set_title(f"SHAP summary: {labels[name]} (real test set, log-odds)", loc="left")
    _fig("real_shap_beeswarm.png", fig)

    _fig(
        "real_shap_by_feature.png",
        plots.importance_bars(
            grouped.to_frame("mean_abs_shap"),
            "mean_abs_shap",
            None,
            f"Mean |SHAP| per source feature: {labels[name]} (one-hot levels summed)",
            "Mean |SHAP value| (log-odds)",
        ),
    )

    numeric_top = [f for f in exp.feature_names if source[f] == f and f in grouped.index]
    numeric_top = [
        f for f in grouped.index if f in numeric_top and np.unique(exp[:, f].data).size > 5
    ][:2]
    if numeric_top:
        fig, axes = plt.subplots(1, len(numeric_top), figsize=(6 * len(numeric_top), 4.2))
        for ax, f in zip(np.atleast_1d(axes), numeric_top, strict=False):
            col = exp[:, f]
            ax.scatter(
                col.data,
                col.values,
                s=14,
                alpha=0.7,
                c=y_te,
                cmap=plt.matplotlib.colors.ListedColormap([plots.BLUE, plots.ORANGE]),
            )
            ax.axhline(0, color=plots.AXIS, lw=1)
            ax.set(xlabel=f, ylabel="SHAP value (log-odds)")
            ax.set_title(f"SHAP dependence: {f}")
        np.atleast_1d(axes)[0].text(
            0.01,
            0.98,
            "blue = legitimate, orange = fraud",
            transform=np.atleast_1d(axes)[0].transAxes,
            va="top",
            fontsize=8,
            color=plots.INK_2,
        )
        fig.tight_layout()
        _fig("real_shap_dependence.png", fig)

    # local explanations: highest-scored true fraud, a false alarm, a missed fraud
    order = np.argsort(-proba)
    picks = {
        "Detected fraud (highest score)": next(i for i in order if y_te[i] == 1),
        "False alarm (highest-scored legit claim)": next(i for i in order if y_te[i] == 0),
        "Missed fraud (lowest-scored fraud)": next(i for i in order[::-1] if y_te[i] == 1),
    }
    fig, axes = plt.subplots(3, 1, figsize=(9, 11))
    for ax, (title, i) in zip(axes, picks.items(), strict=False):
        plots.local_explanation(
            ax,
            pd.Series(exp.values[i], index=exp.feature_names),
            shown.iloc[i],
            float(np.ravel(exp.base_values)[i]),
            f"{title}: P(fraud) = {proba[i]:.2f}",
        )
    fig.suptitle(
        f"Local SHAP explanations: {labels[name]} (real test claims)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    _fig("real_shap_local.png", fig)
    return grouped.head(10).index.tolist()


def learning_curve_table(name: str, best_params, X, y, settings: Settings) -> pd.DataFrame:
    cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=1 if settings.fast else 2, random_state=settings.seed
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sizes, tr, va = learning_curve(
            build(name, restore_params(best_params[name]), settings.seed),
            X,
            y,
            train_sizes=[0.15, 0.3, 0.5, 0.75, 1.0],
            cv=cv,
            scoring="average_precision",
            n_jobs=3,
            shuffle=True,
            random_state=settings.seed,
        )
    return pd.DataFrame(
        {"train_size": sizes, "train_mean": tr.mean(1), "val_mean": va.mean(1), "val_sd": va.std(1)}
    )


def export_model(
    cv_summary, fitted, thresholds, labels, test_table, costs, stack_members, best_params
) -> str:
    """Export the best-CV model that does not need PyTorch at inference time."""

    def needs_torch(n: str) -> bool:
        return n in TORCH_MODELS or (n == "stacking" and bool(TORCH_MODELS & set(stack_members)))

    name = next(n for n in cv_summary.index if not needs_torch(n))
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[name], config.MODELS_DIR / "fraud_model.joblib", compress=3)
    card = {
        "model": name,
        "label": labels[name],
        "params": best_params.get(name, {"stacking_members": stack_members}),
        "decision_threshold": thresholds[name],
        "threshold_rule": "minimises expected cost on training out-of-fold predictions",
        "costs_usd": {"investigation": costs.investigation, "missed_fraud": costs.missed_fraud},
        "cv": cv_summary.loc[name].to_dict(),
        "test_metrics": test_table.loc[name].to_dict(orient="index"),
        "trained_on": "real data, training split (800 claims)",
        "sklearn_version": sklearn.__version__,
        "input_columns": [
            c
            for c in config.RENAME_MAP.values()
            if c not in (*config.IDENTIFIER_COLUMNS, config.RAW_TARGET, "total_claim_amount")
        ],
    }
    write_json(card, config.MODELS_DIR / "model_card.json")
    return name


def _fmt(est: float, lo: float, hi: float, money: bool = False) -> str:
    if money:
        return f"{est:,.0f} [{lo:,.0f}; {hi:,.0f}]"
    return f"{est:.3f} [{lo:.3f}; {hi:.3f}]"


def write_results_markdown(
    cv_summary, test_table, comparisons, labels, reference, trivial, calib, deployable
) -> None:
    lines = [
        "| Model | CV PR-AUC (mean ± sd) | Test PR-AUC [95% CI] | Test ROC-AUC [95% CI] "
        "| F1 | Recall | Precision | MCC | Brier | Cost / claim (USD) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in cv_summary.index:
        t = test_table.loc[n]
        name = f"**{labels[n]}**" if n == reference else labels[n]
        lines.append(
            f"| {name} | {cv_summary.loc[n, 'cv_pr_auc_mean']:.3f} ± "
            f"{cv_summary.loc[n, 'cv_pr_auc_sd']:.3f} "
            f"| {_fmt(*t.loc['pr_auc'])} | {_fmt(*t.loc['roc_auc'])} "
            f"| {t.loc['f1', 'estimate']:.3f} | {t.loc['recall', 'estimate']:.3f} "
            f"| {t.loc['precision', 'estimate']:.3f} | {t.loc['mcc', 'estimate']:.3f} "
            f"| {t.loc['brier', 'estimate']:.3f} | {t.loc['cost_per_claim', 'estimate']:,.0f} |"
        )
    lines += [
        "",
        f"Trivial policies on the test set: investigate nothing = "
        f"{trivial['investigate_none']:,.0f} USD/claim, investigate all = "
        f"{trivial['investigate_all']:,.0f} USD/claim.",
        "",
        f"Reference model (best mean CV PR-AUC): {labels[reference]}. "
        f"Exported model: {labels[deployable]}.",
        "",
        "| Model vs reference | Δ CV PR-AUC | corrected t-test p (Holm) "
        "| Δ test PR-AUC [95% CI] | bootstrap p (Holm) | McNemar p (Holm) |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in comparisons.iterrows():
        lines.append(
            f"| {labels[r.model]} | {-r.cv_pr_auc_diff:+.3f} | {r.cv_ttest_p_holm:.3f} "
            f"| {-r.test_pr_auc_diff:+.3f} [{-r.test_pr_auc_diff_ci_high:+.3f}; "
            f"{-r.test_pr_auc_diff_ci_low:+.3f}] | {r.test_pr_auc_boot_p_holm:.3f} "
            f"| {r.mcnemar_p_holm:.3f} |"
        )
    lines += ["", "| Calibration (test) | Brier | PR-AUC |", "|---|---|---|"]
    for _, r in calib.iterrows():
        lines.append(f"| {labels[r.model]} ({r.method}) | {r.brier:.4f} | {r.pr_auc:.3f} |")
    (config.METRICS_DIR / "real_results.md").write_text("\n".join(lines) + "\n")
