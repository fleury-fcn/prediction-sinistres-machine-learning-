"""Figure style and plotting helpers (matplotlib, static PNG for the README).

One colour per model, fixed for the whole project (colour follows the entity,
never its rank). Baseline logistic regression is drawn in neutral grey.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)

MODEL_COLORS = {
    "catboost": BLUE,
    "lightgbm": ORANGE,
    "xgboost": AQUA,
    "gradient_boosting": YELLOW,
    "random_forest": MAGENTA,
    "logreg_enet": GREEN,
    "mlp": VIOLET,
    "stacking": RED,
    "logreg": MUTED,
}

# Two-class encoding used in EDA plots
CLASS_COLORS = {0: BLUE, 1: ORANGE}
CLASS_LABELS = {0: "Legitimate", 1: "Fraud"}

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f0efec", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#1c5cab", "#86b6ef", "#f0efec", "#f19a99", "#b42e2e"]
)

SYNTHETIC_TAG = "SYNTHETIC DATA"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def model_color(name: str) -> str:
    return MODEL_COLORS.get(name, INK_2)


def _line_style(name: str) -> dict:
    return {"color": model_color(name), "linestyle": "--" if name == "logreg" else "-"}


# ---------------------------------------------------------------------------
# Model comparison figures
# ---------------------------------------------------------------------------


def roc_pr_curves(
    curves: dict[str, dict[str, np.ndarray]],
    labels: dict[str, str],
    metrics: pd.DataFrame,
    prevalence: float,
    title_suffix: str = "real data, hold-out test set",
) -> plt.Figure:
    """Two panels: ROC and Precision-Recall, all models overlaid."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7.4))
    order = metrics.sort_values("pr_auc", ascending=False).index
    for name in order:
        c = curves[name]
        ax1.plot(
            c["fpr"],
            c["tpr"],
            lw=1.6,
            alpha=0.9,
            **_line_style(name),
            label=f"{labels[name]}  {metrics.loc[name, 'roc_auc']:.3f}",
        )
        ax2.plot(
            c["recall"],
            c["precision"],
            lw=1.6,
            alpha=0.9,
            **_line_style(name),
            label=f"{labels[name]}  {metrics.loc[name, 'pr_auc']:.3f}",
        )
    ax1.plot([0, 1], [0, 1], color=AXIS, lw=1, linestyle=":")
    ax2.axhline(prevalence, color=AXIS, lw=1, linestyle=":")
    ax2.text(
        1.0, prevalence + 0.015, f"no-skill = {prevalence:.2f}", ha="right", color=MUTED, fontsize=8
    )
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
    ax2.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01))
    ax1.set_title(f"ROC curves ({title_suffix})")
    ax2.set_title(f"Precision-Recall curves ({title_suffix})")
    ax1.legend(
        title="ROC-AUC", loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, title_fontsize=9
    )
    ax2.legend(
        title="PR-AUC (average precision)",
        loc="upper left",
        bbox_to_anchor=(0, -0.13),
        ncol=2,
        title_fontsize=9,
    )
    fig.tight_layout()
    return fig


def metric_bars_with_ci(
    table: pd.DataFrame, labels: dict[str, str], metrics: Iterable[str], title: str
) -> plt.Figure:
    """Small multiples: horizontal bars with bootstrap CI whiskers, one panel per metric.

    ``table`` has a MultiIndex (model, metric) and columns estimate/ci_low/ci_high.
    """
    metrics = list(metrics)
    pretty = {
        "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC",
        "mcc": "MCC",
        "f1": "F1",
        "brier": "Brier score (lower is better)",
        "cost_per_claim": "Cost per claim, USD (lower is better)",
        "recall": "Recall",
        "precision": "Precision",
    }
    models = (
        table.xs("pr_auc", level="metric")["estimate"].sort_values(ascending=True).index.tolist()
    )
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(3.6 * len(metrics), 0.42 * len(models) + 1.6), sharey=True
    )
    for ax, metric in zip(np.atleast_1d(axes), metrics, strict=False):
        sub = table.xs(metric, level="metric").loc[models]
        y = np.arange(len(models))
        ax.barh(
            y,
            sub["estimate"],
            color=[model_color(m) for m in models],
            height=0.62,
            edgecolor=SURFACE,
            linewidth=2,
        )
        ax.errorbar(
            sub["estimate"],
            y,
            xerr=[sub["estimate"] - sub["ci_low"], sub["ci_high"] - sub["estimate"]],
            fmt="none",
            ecolor=INK,
            elinewidth=1,
            capsize=2.5,
        )
        for yi, (est, hi) in enumerate(zip(sub["estimate"], sub["ci_high"], strict=False)):
            fmt = f"{est:,.0f}" if metric == "cost_per_claim" else f"{est:.3f}"
            ax.text(hi, yi, "  " + fmt, va="center", fontsize=8, color=INK_2)
        ax.set_title(pretty.get(metric, metric), fontsize=10.5)
        ax.grid(axis="y", visible=False)
        lo = 0 if metric != "mcc" else min(0, sub["ci_low"].min())
        ax.set_xlim(lo, sub["ci_high"].max() * 1.25)
    axes = np.atleast_1d(axes)
    axes[0].set_yticks(np.arange(len(models)), [labels[m] for m in models])
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig


def cv_boxplot(
    fold_scores: pd.DataFrame, labels: dict[str, str], metric: str = "pr_auc"
) -> plt.Figure:
    """Distribution of repeated-CV fold scores per model."""
    order = fold_scores.groupby("model")[metric].median().sort_values().index.tolist()
    data = [fold_scores.loc[fold_scores.model == m, metric] for m in order]
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(order) + 1.4))
    bp = ax.boxplot(
        data,
        vert=False,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.4},
    )
    for patch, m in zip(bp["boxes"], order, strict=False):
        patch.set(facecolor=model_color(m), alpha=0.35, edgecolor=model_color(m))
    rng = np.random.default_rng(0)
    for i, (m, d) in enumerate(zip(order, data, strict=False), start=1):
        ax.scatter(d, i + rng.uniform(-0.15, 0.15, len(d)), s=10, color=model_color(m), zorder=3)
    ax.set_yticks(range(1, len(order) + 1), [labels[m] for m in order])
    ax.grid(axis="y", visible=False)
    name = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"}.get(metric, metric)
    ax.set_xlabel(f"{name} per fold")
    ax.set_title(f"Repeated stratified CV on the training set: {name} per fold (real data)")
    fig.tight_layout()
    return fig


def calibration_plot(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    labels: dict[str, str],
    brier: dict[str, float],
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, linestyle=":", label="Perfect calibration")
    for name, (mean_pred, frac_pos) in curves.items():
        style = _line_style(name.split("::")[0])
        if "::" in name:
            style["linestyle"] = {"sigmoid": "-.", "isotonic": ":"}.get(name.split("::")[1], "-")
        ax.plot(
            mean_pred,
            frac_pos,
            marker="o",
            ms=4,
            lw=1.6,
            **style,
            label=f"{labels[name]} (Brier {brier[name]:.3f})",
        )
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed fraud frequency",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def confusion_grid(
    matrices: dict[str, np.ndarray], labels: dict[str, str], thresholds: dict[str, float]
) -> plt.Figure:
    n = len(matrices)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, cm) in zip(axes, matrices.items(), strict=False):
        ax.imshow(cm, cmap=SEQUENTIAL, vmin=0, vmax=cm.sum())
        for i in range(2):
            for j in range(2):
                share = cm[i, j] / cm.sum()
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color="white" if share > 0.35 else INK,
                )
        ax.set_xticks([0, 1], ["Pred. legit", "Pred. fraud"], fontsize=8)
        ax.set_yticks([0, 1], ["Legit", "Fraud"], fontsize=8)
        ax.grid(False)
        ax.set_title(f"{labels[name]}\nthreshold = {thresholds[name]:.2f}", fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(
        "Confusion matrices at the cost-optimal threshold (chosen on train OOF; " "real test set)",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def cost_vs_threshold(
    curves: dict[str, pd.DataFrame],
    labels: dict[str, str],
    chosen: dict[str, float],
    trivial: dict[str, float],
    costs_text: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name, curve in curves.items():
        ax.plot(
            curve["threshold"],
            curve["cost_per_claim"],
            lw=1.8,
            **_line_style(name),
            label=labels[name],
        )
        t = chosen[name]
        y = np.interp(t, curve["threshold"], curve["cost_per_claim"])
        ax.scatter(
            [t], [y], s=46, color=model_color(name), edgecolor=SURFACE, linewidth=2, zorder=4
        )
    ax.axhline(trivial["investigate_all"], color=INK_2, lw=1, linestyle="--")
    ax.text(
        0.99,
        trivial["investigate_all"],
        "investigate every claim",
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK_2,
        transform=ax.get_yaxis_transform(),
    )
    ax.axhline(trivial["investigate_none"], color=INK_2, lw=1, linestyle=":")
    ax.text(
        0.99,
        trivial["investigate_none"],
        "investigate nothing",
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK_2,
        transform=ax.get_yaxis_transform(),
    )
    ax.set(
        xlabel="Decision threshold on P(fraud)", ylabel="Average cost per claim (USD)", xlim=(0, 1)
    )
    ax.set_title(
        "Expected cost vs decision threshold (real test set); "
        "dots = threshold chosen on train OOF"
    )
    ax.text(
        0.01,
        0.9,
        costs_text,
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        color=INK_2,
        bbox={"facecolor": SURFACE, "edgecolor": GRID, "boxstyle": "round"},
    )
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def nn_loss_curves(history: dict[str, list[float]], best_epoch: int) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], color=VIOLET, label="Training loss")
    ax1.plot(epochs, history["val_loss"], color=ORANGE, label="Validation loss")
    ax1.axvline(best_epoch, color=INK_2, lw=1, linestyle=":")
    ax1.text(
        best_epoch,
        ax1.get_ylim()[1],
        f" best epoch {best_epoch}",
        va="top",
        fontsize=8,
        color=INK_2,
    )
    ax1.set(xlabel="Epoch", ylabel="Binary cross-entropy")
    ax1.set_title("MLP training: loss (early stopping on validation loss)")
    ax1.legend()
    ax2.plot(epochs, history["val_auc"], color=ORANGE)
    ax2.axvline(best_epoch, color=INK_2, lw=1, linestyle=":")
    ax2.set(xlabel="Epoch", ylabel="ROC-AUC")
    ax2.set_title("MLP training: validation ROC-AUC")
    fig.tight_layout()
    return fig


def learning_curve_plot(curves: dict[str, pd.DataFrame], labels: dict[str, str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, lc in curves.items():
        color = model_color(name)
        ax.plot(
            lc["train_size"],
            lc["val_mean"],
            color=color,
            marker="o",
            ms=4,
            label=f"{labels[name]} (validation)",
        )
        ax.fill_between(
            lc["train_size"],
            lc["val_mean"] - lc["val_sd"],
            lc["val_mean"] + lc["val_sd"],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            lc["train_size"],
            lc["train_mean"],
            color=color,
            lw=1.2,
            linestyle="--",
            label=f"{labels[name]} (training)",
        )
    ax.set(xlabel="Number of training claims", ylabel="PR-AUC")
    ax.set_title("Learning curves (repeated CV on real training data)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def importance_bars(
    df: pd.DataFrame,
    value: str,
    err: str | None,
    title: str,
    xlabel: str,
    color: str = BLUE,
    top: int = 15,
) -> plt.Figure:
    sub = df.sort_values(value, ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(sub) + 1.2))
    ax.barh(sub.index, sub[value], color=color, height=0.65)
    if err:
        ax.errorbar(
            sub[value],
            np.arange(len(sub)),
            xerr=sub[err],
            fmt="none",
            ecolor=INK,
            elinewidth=1,
            capsize=2,
        )
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def local_explanation(
    ax: plt.Axes, contributions: pd.Series, values: pd.Series, base: float, title: str, top: int = 8
) -> None:
    """Horizontal bars of the largest SHAP contributions for one claim."""
    order = contributions.abs().sort_values(ascending=False).index
    shown = contributions[order[:top]]
    rest = contributions[order[top:]].sum()
    labels = [f"{f} = {_fmt_value(values[f])}" for f in shown.index]
    if len(order) > top:
        shown = pd.concat([shown, pd.Series({"other": rest})])
        labels.append(f"{len(order) - top} other features")
    y = np.arange(len(shown))[::-1]
    ax.barh(y, shown.values, color=[ORANGE if v > 0 else BLUE for v in shown.values], height=0.62)
    for yi, v in zip(y, shown.values, strict=False):
        ax.text(
            v,
            yi,
            f" {v:+.2f} " if v > 0 else f" {v:+.2f} ",
            va="center",
            ha="left" if v > 0 else "right",
            fontsize=8,
            color=INK_2,
        )
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    lo, hi = min(shown.min(), 0), max(shown.max(), 0)
    pad = 0.25 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel(
        f"SHAP contribution (log-odds); base value {base:.2f}, " f"orange pushes towards fraud"
    )
    ax.set_title(title, fontsize=10)


def _fmt_value(v) -> str:
    if isinstance(v, int | float | np.floating) and not isinstance(v, bool):
        if pd.isna(v):
            return "NaN"
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3g}"
    return str(v)
