"""
Génération du tableau de bord graphique.
"""

import logging
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

LABEL_NEGATIVE = "Pas de sinistre"
LABEL_POSITIVE = "Sinistre"
LABEL_REAL = "Valeur réelle"


def _save_plotly_figure(fig: go.Figure, output_dir: Path, file_stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{file_stem}.html"
    png_path = output_dir / f"{file_stem}.png"

    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)

    try:
        fig.write_image(str(png_path), scale=2)
    except Exception as exc:  # pragma: no cover - dépend de l'environnement local
        logger.warning("Export PNG impossible pour %s (%s)", file_stem, exc)

    logger.info("Graphique interactif exporté : %s", html_path)
    return {"html": str(html_path), "png": str(png_path)}


def _slugify(label: str) -> str:
    normalized = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode("ascii")
    return normalized.lower().replace(" ", "_")


def build_comparative_roc_figure(roc_by_model: dict[str, dict]) -> go.Figure:
    fig = go.Figure()

    for model_name, roc_data in roc_by_model.items():
        fig.add_trace(
            go.Scatter(
                x=roc_data["fpr"],
                y=roc_data["tpr"],
                mode="lines",
                name=f"{model_name} (AUC={roc_data['auc']:.3f})",
                hovertemplate=(
                    "Modèle: " + model_name
                    + "<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}"
                    + f"<br>AUC: {roc_data['auc']:.3f}<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="Aléatoire",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="Comparaison des courbes ROC",
        xaxis_title="Taux de faux positifs",
        yaxis_title="Taux de vrais positifs",
        template="plotly_white",
        legend_title="Modèles",
    )
    return fig


def build_confusion_figure(cm: np.ndarray, model_name: str) -> go.Figure:
    total = cm.sum() if cm.sum() else 1
    cm_percent = (cm / total) * 100
    annotation = np.array(
        [
            [
                f"{cm[i, j]}<br>({cm_percent[i, j]:.1f}%)"
                for j in range(cm.shape[1])
            ]
            for i in range(cm.shape[0])
        ]
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[LABEL_NEGATIVE, LABEL_POSITIVE],
            y=[LABEL_NEGATIVE, LABEL_POSITIVE],
            text=annotation,
            texttemplate="%{text}",
            colorscale="Blues",
            hovertemplate=(
                "Réel: %{y}<br>Prédit: %{x}<br>Observations: %{z}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Matrice de confusion — {model_name}",
        xaxis_title="Prédiction",
        yaxis_title=LABEL_REAL,
        template="plotly_white",
    )
    return fig


def build_importance_figure(importance_df: pd.DataFrame, model_name: str) -> go.Figure:
    ordered = importance_df.sort_values("Importance", ascending=True)

    fig = px.bar(
        ordered,
        x="Importance",
        y="Variable",
        orientation="h",
        title=f"Importance des variables — {model_name}",
        template="plotly_white",
    )
    fig.update_traces(hovertemplate="%{y}: %{x:.4f}<extra></extra>")
    return fig


def build_shap_scatter_figure(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    model_name: str,
) -> go.Figure:
    mean_abs = np.abs(shap_values).mean(axis=0)
    feature_index = int(np.argmax(mean_abs))
    feature_name = X_test.columns[feature_index]

    df_plot = pd.DataFrame({
        "feature_value": X_test.iloc[:, feature_index],
        "shap_value": shap_values[:, feature_index],
        "feature": feature_name,
    })

    fig = px.scatter(
        df_plot,
        x="feature_value",
        y="shap_value",
        color="feature_value",
        color_continuous_scale="RdBu",
        title=f"SHAP Scatter — {model_name} ({feature_name})",
        labels={
            "feature_value": f"Valeur de {feature_name}",
            "shap_value": "Impact SHAP",
        },
        template="plotly_white",
    )
    fig.update_traces(
        mode="markers",
        marker={"size": 8, "opacity": 0.75},
        hovertemplate=(
            f"{feature_name}: %{{x:.3f}}<br>Impact SHAP: %{{y:.4f}}<extra></extra>"
        ),
    )
    return fig


def export_interactive_visuals(
    roc_by_model: dict[str, dict],
    confusion_by_model: dict[str, np.ndarray],
    importance_by_model: dict[str, pd.DataFrame],
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    output_dir: Path,
) -> dict[str, dict[str, str]]:
    files = {}

    roc_fig = build_comparative_roc_figure(roc_by_model)
    files["roc_comparative"] = _save_plotly_figure(roc_fig, output_dir, "roc_comparative")

    for model_name, cm in confusion_by_model.items():
        safe_name = _slugify(model_name)
        cm_fig = build_confusion_figure(cm, model_name)
        files[f"confusion_{safe_name}"] = _save_plotly_figure(
            cm_fig, output_dir, f"confusion_{safe_name}"
        )

    for model_name, importance_df in importance_by_model.items():
        safe_name = _slugify(model_name)
        imp_fig = build_importance_figure(importance_df, model_name)
        files[f"importance_{safe_name}"] = _save_plotly_figure(
            imp_fig, output_dir, f"importance_{safe_name}"
        )

    shap_fig = build_shap_scatter_figure(shap_values, X_test, "Random Forest")
    files["shap_scatter_random_forest"] = _save_plotly_figure(
        shap_fig,
        output_dir,
        "shap_scatter_random_forest",
    )

    return files


def plot_dashboard(cm, fpr, tpr, auc, importance, results, metrics, output_path):

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        "PRÉDICTION DES SINISTRES — MODÈLE RANDOM FOREST",
        fontsize=22, fontweight="bold",
    )

    # 1. Matrice de confusion
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(cm, interpolation="nearest")
    ax1.set_title("Matrice de confusion", fontweight="bold")
    ax1.set_xlabel("Prédiction")
    ax1.set_ylabel(LABEL_REAL)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels([LABEL_NEGATIVE, LABEL_POSITIVE])
    ax1.set_yticklabels([LABEL_NEGATIVE, LABEL_POSITIVE])
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, cm[i, j], ha="center", va="center",
                      fontsize=15, fontweight="bold")

    # 2. ROC
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    ax2.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax2.set_title("Courbe ROC", fontweight="bold")
    ax2.set_xlabel("Taux de faux positifs")
    ax2.set_ylabel("Taux de vrais positifs")
    ax2.legend()

    # 3. Importance des variables
    ax3 = plt.subplot(2, 3, 3)
    importance_sorted = importance.sort_values("Importance", ascending=True)
    ax3.barh(importance_sorted["Variable"], importance_sorted["Importance"])
    ax3.set_title("Importance des variables", fontweight="bold")
    ax3.set_xlabel("Importance")

    # 4. Réel vs prédit
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(results["Valeur_reelle"], results["Probabilite_predite"], alpha=0.5)
    ax4.plot([0, 1], [0, 1], linestyle="--")
    ax4.set_title("Valeurs réelles vs probabilités prédites", fontweight="bold")
    ax4.set_xlabel(LABEL_REAL)
    ax4.set_ylabel("Probabilité prédite")

    # 5. Résidus
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(results["Probabilite_predite"], results["Erreur"], alpha=0.5)
    ax5.axhline(0, linestyle="--", linewidth=1)
    ax5.set_title("Résidus du modèle", fontweight="bold")
    ax5.set_xlabel("Probabilité prédite")
    ax5.set_ylabel("Résidu")

    # 6. Performances
    ax6 = plt.subplot(2, 3, 6)
    names = list(metrics.keys())
    values = list(metrics.values())
    bars = ax6.bar(names, values)
    ax6.set_ylim(0, 1)
    ax6.set_title("Performances du modèle", fontweight="bold")
    ax6.set_ylabel("Score")
    for bar, value in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                  f"{value:.2f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Graphique sauvegardé : %s", output_path)
