"""
Génération du tableau de bord graphique.
"""

import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


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
    ax1.set_ylabel("Valeur réelle")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Pas de sinistre", "Sinistre"])
    ax1.set_yticklabels(["Pas de sinistre", "Sinistre"])
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
    ax4.set_xlabel("Valeur réelle")
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
