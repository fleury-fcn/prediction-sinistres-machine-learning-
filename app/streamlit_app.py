"""Interactive results explorer and claim scorer (French / English).

    streamlit run app/streamlit_app.py

Reads only versioned files (results/, data/insurance_claims.csv, examples/), so it runs
on Streamlit Community Cloud with the light dependencies of ``app/requirements.txt``.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data import coerce_types, load_clean  # noqa: E402

FIG = config.FIGURES_DIR
MET = config.METRICS_DIR
REPO_URL = "https://github.com/fleury-fcn/prediction-sinistres-machine-learning-"

MODEL_COLORS = {
    "catboost": "#2a78d6",
    "lightgbm": "#eb6834",
    "xgboost": "#1baf7a",
    "gradient_boosting": "#eda100",
    "random_forest": "#e87ba4",
    "logreg_enet": "#008300",
    "mlp": "#6250d6",
    "stacking": "#e34948",
    "logreg": "#898781",
}
FRAUD, LEGIT = "#eb6834", "#2a78d6"

st.set_page_config(page_title="Claim fraud detection", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------

lang = st.sidebar.radio("Langue / Language", ["Français", "English"], horizontal=True)
FR = lang == "Français"


def t(fr: str, en: str) -> str:
    return fr if FR else en


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_data
def read_csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(MET / name, **kw)


@st.cache_data
def read_json(name: str) -> dict:
    return json.loads((MET / name).read_text())


@st.cache_data
def claims() -> pd.DataFrame:
    return load_clean(config.DATA_DIR / "insurance_claims.csv")


@st.cache_resource
def model_and_card():
    card = json.loads((config.MODELS_DIR / "model_card.json").read_text())
    return joblib.load(config.MODELS_DIR / "fraud_model.joblib"), card


def labels() -> dict[str, str]:
    cv = read_csv("real_cv_summary.csv")
    return dict(zip(cv.model, cv.label, strict=True))


def figure(name: str, caption: str | None = None) -> None:
    path = FIG / f"{name}.png"
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


def test_table() -> pd.DataFrame:
    return read_csv("real_test_metrics.csv").set_index(["model", "metric"])


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_overview() -> None:
    ref = read_json("real_reference.json")
    tt = test_table()
    name = ref["reference_model"]
    cost = tt.loc[(name, "cost_per_claim"), "estimate"]
    best_trivial = min(ref["trivial_policies_test"].values())

    st.title(
        t("🛡️ Détection de fraude sur sinistres automobiles", "🛡️ Insurance claim fraud detection")
    )
    st.markdown(
        t(
            "Benchmark **sans fuite de données** et **statistiquement honnête** sur le jeu public de "
            "1 000 sinistres (Derrick Mwiti) : 9 modèles optimisés avec le même budget, validation "
            "croisée répétée, intervalles de confiance, seuil de décision fondé sur le coût métier, "
            "SHAP, et un jeu synthétique de 500 000 lignes pour les tests de montée en charge.",
            "A **leak-free**, **statistically honest** benchmark on the public 1,000-claim dataset "
            "(Derrick Mwiti): 9 models tuned with the same budget, repeated cross-validation, "
            "confidence intervals, cost-based decision threshold, SHAP, and a 500,000-row synthetic "
            "dataset for scale tests.",
        )
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        t("ROC-AUC (test)", "ROC-AUC (test)"),
        f"{tt.loc[(name, 'roc_auc'), 'estimate']:.2f}",
        t("+0,34 vs ancienne version (0,50)", "+0.34 vs previous version (0.50)"),
    )
    c2.metric(
        t("PR-AUC (test)", "PR-AUC (test)"),
        f"{tt.loc[(name, 'pr_auc'), 'estimate']:.2f}",
        t("hasard = 0,25", "chance = 0.25"),
        delta_color="off",
    )
    c3.metric(
        t("Coût par sinistre", "Cost per claim"),
        f"{cost:,.0f} $",
        f"{100 * (cost - best_trivial) / best_trivial:.0f} % "
        + t("vs tout enquêter", "vs investigate all"),
        delta_color="inverse",
    )
    c4.metric(
        t("Recall / précision", "Recall / precision"),
        f"{tt.loc[(name, 'recall'), 'estimate']:.0%} / "
        f"{tt.loc[(name, 'precision'), 'estimate']:.0%}",
    )
    st.caption(
        t(
            f"Modèle de référence : {labels()[name]}, choisi sur la validation croisée. Test : 200 "
            "sinistres (49 fraudes) jamais vus pendant l'entraînement.",
            f"Reference model: {labels()[name]}, selected on cross-validation. Test: 200 claims "
            "(49 frauds) never seen during training.",
        )
    )

    st.subheader(t("Ce qu'il faut retenir", "Key findings"))
    st.markdown(
        t(
            "1. Utiliser toutes les variables fait passer la ROC-AUC de **0,50 à 0,84**.\n"
            "2. **Aucun modèle n'est significativement meilleur** que les autres après correction de "
            "Holm : les écarts sont plus petits que l'incertitude.\n"
            "3. Le signal repose presque entièrement sur **deux variables** : la gravité de "
            "l'incident et les loisirs de l'assuré (un artefact probable du jeu de données).\n"
            "4. Les montants réclamés n'apportent rien : **pas de fuite** de la cible.\n"
            "5. Les **données synthétiques** reproduisent le signal mais n'apportent aucune "
            "information nouvelle sur la fraude réelle.",
            "1. Using every relevant column lifts ROC-AUC from **0.50 to 0.84**.\n"
            "2. **No model is significantly better** than the others after Holm correction: the "
            "differences are smaller than the uncertainty.\n"
            "3. The signal rests almost entirely on **two features**: incident severity and the "
            "insured's hobbies (likely an artefact of the dataset).\n"
            "4. Claim amounts add nothing: **no target leakage**.\n"
            "5. **Synthetic data** reproduces the signal but adds no information about real fraud.",
        )
    )
    figure("real_roc_pr_curves")
    st.markdown(
        t(
            f"Code, méthode et limites : [dépôt GitHub]({REPO_URL})",
            f"Code, methodology and limitations: [GitHub repository]({REPO_URL})",
        )
    )


def page_data() -> None:
    st.title(t("📊 Données et analyse exploratoire", "📊 Data and exploratory analysis"))
    df = claims()
    info = read_json("eda_summary.json")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("Sinistres", "Claims"), f"{info['n_rows']:,}")
    c2.metric(t("Colonnes", "Columns"), info["n_source_columns"])
    c3.metric(t("Taux de fraude", "Fraud rate"), f"{info['fraud_rate']:.1%}")
    c4.metric(t("Doublons", "Duplicates"), info["duplicated_rows"])

    st.subheader(t("Taux de fraude par catégorie", "Fraud rate by category"))
    col = st.selectbox(
        t("Variable", "Feature"),
        config.RAW_CATEGORICAL_COLUMNS,
        index=config.RAW_CATEGORICAL_COLUMNS.index("incident_severity"),
    )
    g = df.groupby(col)[config.TARGET].agg(rate="mean", n="size").reset_index()
    overall = df[config.TARGET].mean()
    bars = (
        alt.Chart(g)
        .mark_bar(color=FRAUD, cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "rate:Q",
                title=t("Taux de fraude", "Fraud rate"),
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(f"{col}:N", sort="-x", title=None),
            tooltip=[col, alt.Tooltip("rate:Q", format=".1%"), "n"],
        )
    )
    rule = alt.Chart(pd.DataFrame({"x": [overall]})).mark_rule(strokeDash=[4, 4]).encode(x="x")
    st.altair_chart(bars + rule, use_container_width=True)
    st.caption(
        t(
            "Ligne pointillée : taux global (24,7 %). Survolez les barres pour les effectifs.",
            "Dashed line: overall rate (24.7%). Hover the bars for counts.",
        )
    )

    with st.expander(
        t("Qualité des données et audit de fuite", "Data quality and leakage audit"), expanded=True
    ):
        st.markdown(
            t(
                "* Les `?` (collision_type, property_damage, police_report_available) sont gardés "
                "comme catégorie `UNKNOWN` : pour collision_type, ils correspondent exactement aux "
                "vols et véhicules stationnés.\n"
                '* La valeur `"None"` d\'authorities_contacted est un vrai niveau (pandas la '
                "transformait en NaN par défaut).\n"
                "* `total_claim_amount` = somme exacte des trois montants : supprimé (redondant). "
                "Retirer **tous** les montants ne dégrade pas le modèle : aucune fuite.",
                "* `?` markers (collision_type, property_damage, police_report_available) are kept "
                "as an `UNKNOWN` level: for collision_type they match thefts and parked cars "
                "exactly.\n"
                '* The value `"None"` of authorities_contacted is a real level (pandas turned it '
                "into NaN by default).\n"
                "* `total_claim_amount` is the exact sum of the three amounts: dropped (redundant). "
                "Removing **every** amount does not hurt the model: no leakage.",
            )
        )
        figure("eda_data_quality")
    tab1, tab2, tab3 = st.tabs(
        [
            t("Distributions", "Distributions"),
            t("Corrélations", "Correlations"),
            t("Aperçu des données", "Data preview"),
        ]
    )
    with tab1:
        figure("eda_numeric_distributions")
    with tab2:
        figure("eda_correlation_matrix")
    with tab3:
        st.dataframe(df.head(200), use_container_width=True)


def page_models() -> None:
    st.title(t("🏁 Comparaison des modèles", "🏁 Model comparison"))
    st.markdown(
        t(
            "Optuna (40 essais par modèle, mêmes plis) → validation croisée répétée 5×5 sur 800 "
            "sinistres → **une seule** évaluation sur les 200 sinistres de test. Intervalles : "
            "bootstrap stratifié à 95 % (2 000 rééchantillonnages).",
            "Optuna (40 trials per model, same folds) → 5×5 repeated cross-validation on 800 claims "
            "→ **a single** evaluation on the 200 test claims. Intervals: 95% stratified bootstrap "
            "(2,000 resamples).",
        )
    )
    names = labels()
    cv = read_csv("real_cv_summary.csv").set_index("model")
    tt = test_table()
    metric_names = {
        "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC",
        "f1": "F1",
        "recall": "Recall",
        "precision": t("Précision", "Precision"),
        "mcc": "MCC",
        "brier": t("Brier (plus bas = mieux)", "Brier (lower is better)"),
        "cost_per_claim": t("Coût / sinistre ($)", "Cost / claim ($)"),
    }

    rows = []
    for m in cv.index:
        row = {
            t("Modèle", "Model"): names[m],
            "CV PR-AUC": f"{cv.loc[m, 'cv_pr_auc_mean']:.3f} ± {cv.loc[m, 'cv_pr_auc_sd']:.3f}",
        }
        for k in ["pr_auc", "roc_auc"]:
            e, lo, hi = tt.loc[(m, k)]
            row[f"Test {metric_names[k]}"] = f"{e:.3f} [{lo:.3f}; {hi:.3f}]"
        for k in ["f1", "recall", "precision", "mcc", "brier"]:
            row[metric_names[k]] = round(tt.loc[(m, k), "estimate"], 3)
        row[metric_names["cost_per_claim"]] = int(tt.loc[(m, "cost_per_claim"), "estimate"])
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        t(
            "Trié par PR-AUC en validation croisée (critère de sélection).",
            "Sorted by cross-validated PR-AUC (the selection criterion).",
        )
    )

    metric = st.selectbox(
        t("Métrique à afficher", "Metric to plot"), list(metric_names), format_func=metric_names.get
    )
    sub = tt.xs(metric, level="metric").reset_index()
    sub["label"] = sub.model.map(names)
    order = sub.sort_values("estimate", ascending=metric in ("brier", "cost_per_claim")).label
    base = alt.Chart(sub).encode(y=alt.Y("label:N", sort=list(order), title=None))
    chart = base.mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("estimate:Q", title=metric_names[metric]),
        color=alt.Color(
            "model:N",
            scale=alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values())),
            legend=None,
        ),
        tooltip=[
            "label",
            alt.Tooltip("estimate:Q", format=".3f"),
            alt.Tooltip("ci_low:Q", format=".3f"),
            alt.Tooltip("ci_high:Q", format=".3f"),
        ],
    ) + base.mark_rule(color="#0b0b0b").encode(x="ci_low:Q", x2="ci_high:Q")
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        t(
            "Barre noire : intervalle de confiance à 95 %.",
            "Black line: 95% confidence " "interval.",
        )
    )

    st.subheader(t("Les écarts sont-ils significatifs ?", "Are the differences significant?"))
    tests = read_csv("real_model_comparison_tests.csv")
    tests = pd.DataFrame(
        {
            t("Modèle", "Model"): tests.model.map(names),
            t("Δ PR-AUC CV vs référence", "Δ CV PR-AUC vs reference"): (
                -tests.cv_pr_auc_diff
            ).round(3),
            t("p t-test corrigé (Holm)", "corrected t-test p (Holm)"): tests.cv_ttest_p_holm.round(
                3
            ),
            t("Δ PR-AUC test", "Δ test PR-AUC"): (-tests.test_pr_auc_diff).round(3),
            t("p bootstrap (Holm)", "bootstrap p (Holm)"): tests.test_pr_auc_boot_p_holm.round(3),
            t("p McNemar (Holm)", "McNemar p (Holm)"): tests.mcnemar_p_holm.round(4),
        }
    )
    st.dataframe(tests, hide_index=True, use_container_width=True)
    st.info(
        t(
            "Aucune p-value du t-test corrigé (Nadeau-Bengio) ni du bootstrap apparié n'est sous "
            "0,05 après correction de Holm : on ne peut pas départager les meilleurs modèles. "
            "McNemar compare les erreurs aux seuils choisis : il montre que les modèles linéaires "
            "et le MLP signalent plus de sinistres honnêtes à leur propre seuil.",
            "No corrected t-test (Nadeau-Bengio) or paired-bootstrap p-value is below 0.05 after "
            "Holm correction: the best models cannot be told apart. McNemar compares errors at the "
            "chosen thresholds: it shows that the linear models and the MLP flag more honest "
            "claims at their own threshold.",
        )
    )
    with st.expander(t("Plus de graphiques", "More charts")):
        for name in [
            "real_metrics_ci",
            "real_cv_pr_auc_folds",
            "real_confusion_matrices",
            "real_calibration",
            "real_learning_curves",
            "real_mlp_loss_curves",
        ]:
            figure(name)
    with st.expander(
        t("Stratégies de déséquilibre et ablations", "Imbalance strategies and " "ablations")
    ):
        st.dataframe(read_csv("real_imbalance_strategies.csv").round(3), hide_index=True)
        st.dataframe(read_csv("real_feature_ablation.csv").round(3), hide_index=True)


def page_costs() -> None:
    st.title(t("💰 Simulateur de coût métier", "💰 Business cost simulator"))
    st.markdown(
        t(
            "Chaque sinistre signalé est envoyé en enquête (coût fixe) ; une fraude non détectée "
            "est payée. Modifiez les hypothèses et le seuil pour voir l'effet sur les 200 sinistres "
            "de test.",
            "Every flagged claim is investigated (fixed cost); an undetected fraud is paid out. "
            "Change the assumptions and the threshold to see the effect on the 200 test claims.",
        )
    )
    preds = read_csv("real_test_predictions.csv")
    names = labels()
    cv = read_csv("real_cv_summary.csv").set_index("model")
    ref = read_json("real_reference.json")["reference_model"]

    c1, c2, c3 = st.columns(3)
    inv = c1.number_input(
        t("Coût d'une enquête ($)", "Investigation cost ($)"), 50, 20000, 1000, step=50
    )
    miss = c2.number_input(
        t("Coût d'une fraude manquée ($)", "Missed-fraud cost ($)"), 100, 200000, 10000, step=500
    )
    model = c3.selectbox(
        t("Modèle", "Model"), list(cv.index), format_func=names.get, index=list(cv.index).index(ref)
    )
    y, p = preds["is_fraud"].to_numpy(), preds[model].to_numpy()

    grid = np.round(np.linspace(0.01, 0.99, 99), 2)

    def cost(th: float) -> float:
        flagged = p >= th
        return (flagged.sum() * inv + ((y == 1) & ~flagged).sum() * miss) / len(y)

    curve = pd.DataFrame({"threshold": grid, "cost": [cost(g) for g in grid]})
    official = float(cv.loc[model, "oof_cost_threshold"])
    th = st.slider(
        t("Seuil de décision sur P(fraude)", "Decision threshold on P(fraud)"),
        0.01,
        0.99,
        official,
        0.01,
    )
    all_cost, none_cost = inv, y.mean() * miss
    best_trivial = min(all_cost, none_cost)
    now = cost(th)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t("Coût / sinistre", "Cost / claim"), f"{now:,.0f} $")
    m2.metric(t("Tout enquêter", "Investigate all"), f"{all_cost:,.0f} $")
    m3.metric(t("Ne rien enquêter", "Investigate none"), f"{none_cost:,.0f} $")
    m4.metric(
        t("Économie vs meilleure règle simple", "Savings vs best simple rule"),
        f"{100 * (best_trivial - now) / best_trivial:.0f} %",
    )

    line = (
        alt.Chart(curve)
        .mark_line(color=MODEL_COLORS.get(model, "#2a78d6"))
        .encode(
            x=alt.X("threshold:Q", title=t("Seuil", "Threshold")),
            y=alt.Y("cost:Q", title=t("Coût moyen par sinistre ($)", "Average cost per claim ($)")),
            tooltip=[
                alt.Tooltip("threshold:Q", format=".2f"),
                alt.Tooltip("cost:Q", format=",.0f"),
            ],
        )
    )
    refs = pd.DataFrame(
        {
            "y": [all_cost, none_cost],
            "what": [
                t("tout enquêter", "investigate all"),
                t("ne rien enquêter", "investigate none"),
            ],
        }
    )
    hrules = (
        alt.Chart(refs)
        .mark_rule(strokeDash=[4, 4], color="#52514e")
        .encode(y="y:Q", tooltip=["what"])
    )
    vrule = alt.Chart(pd.DataFrame({"x": [th]})).mark_rule(color="#e34948").encode(x="x:Q")
    st.altair_chart(line + hrules + vrule, use_container_width=True)

    flagged = p >= th
    cm = pd.DataFrame(
        [
            [int(((y == 0) & ~flagged).sum()), int(((y == 0) & flagged).sum())],
            [int(((y == 1) & ~flagged).sum()), int(((y == 1) & flagged).sum())],
        ],
        index=[t("Réel : honnête", "Actual: legit"), t("Réel : fraude", "Actual: fraud")],
        columns=[t("Non signalé", "Not flagged"), t("Signalé (enquête)", "Flagged (investigate)")],
    )
    c1, c2 = st.columns([1, 1])
    c1.dataframe(cm)
    tp, fp = cm.iloc[1, 1], cm.iloc[0, 1]
    c2.markdown(
        t(
            f"**{tp}/49** fraudes détectées, **{fp}** sinistres honnêtes envoyés en enquête "
            f"(précision {tp / max(tp + fp, 1):.0%}).",
            f"**{tp}/49** frauds detected, **{fp}** honest claims sent to investigation "
            f"(precision {tp / max(tp + fp, 1):.0%}).",
        )
    )
    st.warning(
        t(
            f"Le seuil officiel ({official:.2f}) a été choisi sur les prédictions hors-pli du jeu "
            "d'entraînement, pour 1 000 $ / 10 000 $. Choisir le seuil ici, sur le jeu de test, "
            "sert à explorer : cela donne une vision optimiste.",
            f"The official threshold ({official:.2f}) was chosen on training out-of-fold "
            "predictions, for $1,000 / $10,000. Picking the threshold here, on the test set, is "
            "for exploration only and gives an optimistic view.",
        )
    )


def page_interpret() -> None:
    st.title(t("🔍 Interprétabilité", "🔍 Interpretability"))
    shap_df = read_csv("real_shap_mean_abs_by_feature.csv", index_col=0).reset_index()
    shap_df.columns = ["feature", "mean_abs_shap"]
    top = st.slider(t("Nombre de variables", "Number of features"), 5, len(shap_df), 15)
    chart = (
        alt.Chart(shap_df.head(top))
        .mark_bar(color="#2a78d6", cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "mean_abs_shap:Q", title=t("|SHAP| moyen (log-odds)", "Mean |SHAP| (log-odds)")
            ),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("mean_abs_shap:Q", format=".3f")],
        )
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown(
        t(
            "* **incident_severity** : les sinistres *Major Damage* ont ~60 % de fraude contre "
            "7-13 % ailleurs.\n"
            "* **insured_hobbies** : *chess* et *cross-fit* ont 83 % et 74 % de fraude. Aucun "
            "mécanisme causal plausible : très probablement un artefact du jeu de données.\n"
            "* Tout le reste pèse un ordre de grandeur de moins. SHAP décrit le modèle, pas une "
            "causalité.",
            "* **incident_severity**: *Major Damage* claims carry ~60% fraud vs 7-13% elsewhere.\n"
            "* **insured_hobbies**: *chess* and *cross-fit* show 83% and 74% fraud. No plausible "
            "causal mechanism: most likely an artefact of the dataset.\n"
            "* Everything else weighs an order of magnitude less. SHAP describes the model, not "
            "causality.",
        )
    )
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Beeswarm",
            t("Explications locales", "Local explanations"),
            t("Dépendance", "Dependence"),
            t("Importance par permutation", "Permutation importance"),
        ]
    )
    with tab1:
        figure("real_shap_beeswarm")
    with tab2:
        figure("real_shap_local")
    with tab3:
        figure("real_shap_dependence")
    with tab4:
        figure("real_permutation_importance")


def page_synthetic() -> None:
    st.title(t("🧪 Données synthétiques", "🧪 Synthetic data"))
    st.error(
        t(
            "⚠️ DONNÉES SYNTHÉTIQUES : rien sur cette page n'est une performance réelle. Elles "
            "servent aux tests de montée en charge et de robustesse.",
            "⚠️ SYNTHETIC DATA: nothing on this page is a real-world performance. It is used for "
            "scale and robustness tests only.",
        )
    )
    q = read_csv("synthetic_quality.csv").set_index("generator")
    rows = {
        "fraud_rate": t("Taux de fraude (réel 0,248)", "Fraud rate (real 0.248)"),
        "ks_numeric_mean": t("Distance KS moyenne (numériques)", "Mean KS distance (numeric)"),
        "tvd_categorical_mean": t(
            "Distance VT moyenne (catégorielles)", "Mean TV distance " "(categorical)"
        ),
        "spearman_mean_abs_diff": t("Écart de corrélation de Spearman", "Spearman gap"),
        "rule_violation_rate_raw": t(
            "Violations de règles métier (brut)", "Business-rule " "violations (raw)"
        ),
        "detection_auc": t(
            "AUC de détection réel/synthétique (0,5 = idéal)",
            "Real-vs-synthetic " "detection AUC (0.5 = ideal)",
        ),
        "exact_copies": t("Copies exactes de lignes réelles", "Exact copies of real rows"),
        "tstr_pr_auc_n20000": t("TSTR : PR-AUC sur le test réel", "TSTR: PR-AUC on real test"),
        "trtr_pr_auc_n800": t(
            "Référence TRTR (800 vraies lignes)", "TRTR reference (800 real " "rows)"
        ),
    }
    table = q[list(rows)].T.rename(index=rows).round(3)
    table.columns = [
        {
            "gaussian_copula_custom": t("Copule gaussienne (maison)", "Gaussian copula (custom)"),
            "sdv_gaussian_copula": "SDV GaussianCopula",
            "sdv_ctgan": "SDV CTGAN",
        }[c]
        for c in table.columns
    ]
    st.dataframe(table, use_container_width=True)
    st.markdown(
        t(
            "La copule maison est la meilleure sur tous les critères et a été retenue sur l'AUC de "
            "détection (critère calculé sans le jeu de test). Un modèle entraîné sur 20 000 lignes "
            "synthétiques fait **aussi bien, mais pas mieux**, que le même modèle entraîné sur les "
            "800 vraies lignes.",
            "The custom copula is best on every check and was selected on the detection AUC (a "
            "criterion computed without the test set). A model trained on 20,000 synthetic rows "
            "does **as well as, but no better than**, the same model trained on the 800 real rows.",
        )
    )
    figure("synthetic_quality")

    st.subheader(t("Montée en charge : 1 000 → 500 000 lignes", "Scaling: 1,000 → 500,000 rows"))
    scale = read_csv("synthetic_scale_benchmark.csv")
    names = labels()
    scale["label"] = scale.model.map(names)
    choice = st.radio(
        t("Mesure", "Measure"),
        [
            t("PR-AUC sur test SYNTHÉTIQUE", "PR-AUC on SYNTHETIC hold-out"),
            t("PR-AUC sur test RÉEL (TSTR)", "PR-AUC on REAL test (TSTR)"),
            t("Temps d'entraînement (s)", "Fit time (s)"),
        ],
        horizontal=True,
    )
    col = {0: "synthetic_test_pr_auc", 1: "real_test_pr_auc_tstr", 2: "fit_seconds"}[
        [
            t("PR-AUC sur test SYNTHÉTIQUE", "PR-AUC on SYNTHETIC hold-out"),
            t("PR-AUC sur test RÉEL (TSTR)", "PR-AUC on REAL test (TSTR)"),
            t("Temps d'entraînement (s)", "Fit time (s)"),
        ].index(choice)
    ]
    chart = (
        alt.Chart(scale)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "n_train:Q",
                scale=alt.Scale(type="log"),
                title=t("Lignes d'entraînement", "Training rows"),
            ),
            y=alt.Y(
                f"{col}:Q",
                scale=alt.Scale(type="log" if col == "fit_seconds" else "linear", zero=False),
                title=choice,
            ),
            color=alt.Color(
                "model:N",
                scale=alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values())),
                legend=alt.Legend(title=None),
            ),
            tooltip=["label", "n_train", alt.Tooltip(f"{col}:Q", format=".3f")],
        )
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown(
        t(
            "Sur le test synthétique, tous les modèles progressent avec la taille (0,80 → 0,93) ; "
            "sur les **vrais** sinistres, plus de données synthétiques n'aident pas (variations "
            "dans l'incertitude de ±0,1). La tâche synthétique est plus facile que la vraie.",
            "On the synthetic hold-out every model improves with size (0.80 → 0.93); on **real** "
            "claims, more synthetic data does not help (changes within the ±0.1 uncertainty). The "
            "synthetic task is easier than the real one.",
        )
    )
    with st.expander(t("Distributions et corrélations", "Distributions and correlations")):
        figure("synthetic_marginals")
        figure("synthetic_correlations")


def page_score() -> None:
    st.title(t("🧾 Évaluer un sinistre", "🧾 Score a claim"))
    model, card = model_and_card()
    st.caption(
        t(
            f"Modèle : {card['label']} · seuil {card['decision_threshold']:.2f} (coût minimal pour "
            "1 000 $ par enquête et 10 000 $ par fraude manquée). Démonstration sur 800 sinistres "
            "publics : pas un score de production.",
            f"Model: {card['label']} · threshold {card['decision_threshold']:.2f} (minimum cost for "
            "$1,000 per investigation and $10,000 per missed fraud). Demo trained on 800 public "
            "claims: not a production score.",
        )
    )
    example = json.loads((ROOT / "examples" / "claim.json").read_text())
    df = claims()
    levels = {c: sorted(df[c].unique()) for c in config.RAW_CATEGORICAL_COLUMNS}
    main = [
        "incident_severity",
        "insured_hobbies",
        "incident_type",
        "collision_type",
        "authorities_contacted",
        "witnesses",
        "injury_claim",
        "property_claim",
        "vehicle_claim",
    ]

    def widget(container, key, default):
        if key in levels:
            return container.selectbox(key, levels[key], index=levels[key].index(default))
        if key.endswith("_date"):
            return container.date_input(key, date.fromisoformat(default)).isoformat()
        return container.number_input(key, value=float(default))

    with st.form("claim"):
        st.markdown(t("**Variables principales**", "**Main features**"))
        cols = st.columns(3)
        values = {k: widget(cols[i % 3], k, example[k]) for i, k in enumerate(main)}
        with st.expander(t("Autres variables (pré-remplies)", "Other features (pre-filled)")):
            cols = st.columns(3)
            others = [k for k in example if k not in main]
            values.update({k: widget(cols[i % 3], k, example[k]) for i, k in enumerate(others)})
        submitted = st.form_submit_button(t("Évaluer", "Score"), type="primary")
    if not submitted:
        st.info(
            t(
                "Exemple pré-rempli : un vrai sinistre frauduleux (Major Damage). Essayez de "
                "passer la gravité à Minor Damage.",
                "Pre-filled example: a real fraudulent claim (Major Damage). Try switching "
                "severity to Minor Damage.",
            )
        )
        return

    frame = coerce_types(pd.DataFrame([values]))
    proba = float(model.predict_proba(frame)[:, 1][0])
    threshold = float(card["decision_threshold"])
    c1, c2 = st.columns([1, 2])
    c1.metric(t("Probabilité de fraude", "Fraud probability"), f"{proba:.1%}")
    c1.progress(min(proba, 1.0))
    if proba >= threshold:
        c1.warning(
            t("Au-dessus du seuil : envoyer en enquête", "Above threshold: send to investigation")
        )
    else:
        c1.success(
            t("Sous le seuil : traitement standard", "Below threshold: standard " "processing")
        )
    contrib = local_contributions(model, frame)
    if contrib is not None:
        chart = (
            alt.Chart(contrib)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                x=alt.X(
                    "shap:Q",
                    title=t("Contribution SHAP (log-odds)", "SHAP contribution (log-odds)"),
                ),
                y=alt.Y(
                    "label:N", sort=alt.EncodingSortField("abs", order="descending"), title=None
                ),
                color=alt.condition("datum.shap > 0", alt.value(FRAUD), alt.value(LEGIT)),
                tooltip=["label", alt.Tooltip("shap:Q", format="+.3f")],
            )
        )
        c2.markdown(
            t(
                "**Pourquoi ce score ?** (orange = pousse vers la fraude)",
                "**Why this score?** (orange = pushes towards fraud)",
            )
        )
        c2.altair_chart(chart, use_container_width=True)


def local_contributions(model, frame: pd.DataFrame) -> pd.DataFrame | None:
    """SHAP contributions of one claim (CatBoost exports only)."""
    est = model.named_steps.get("model")
    if type(est).__name__ != "CatBoostClassifier":
        return None
    from catboost import Pool

    builder = model.named_steps["features"]
    F = builder.transform(frame)
    shap = est.get_feature_importance(
        Pool(F, cat_features=builder.categorical_features_), type="ShapValues"
    )[0, :-1]
    out = pd.DataFrame({"feature": F.columns, "shap": shap, "value": F.iloc[0].to_numpy()})
    out["abs"] = out.shap.abs()
    out = out.sort_values("abs", ascending=False).head(10)
    out["label"] = [
        f"{f} = {v:,.4g}" if isinstance(v, int | float) else f"{f} = {v}"
        for f, v in zip(out.feature, out.value, strict=True)
    ]
    return out


PAGES = {
    t("Vue d'ensemble", "Overview"): page_overview,
    t("Données", "Data"): page_data,
    t("Modèles", "Models"): page_models,
    t("Simulateur de coût", "Cost simulator"): page_costs,
    t("Interprétabilité", "Interpretability"): page_interpret,
    t("Données synthétiques", "Synthetic data"): page_synthetic,
    t("Évaluer un sinistre", "Score a claim"): page_score,
}
choice = st.sidebar.radio(t("Navigation", "Navigation"), list(PAGES))
st.sidebar.markdown("---")
st.sidebar.markdown(
    t(f"[Code source sur GitHub]({REPO_URL})", f"[Source code on GitHub]({REPO_URL})")
)
st.sidebar.caption("Fleury Niyokwizera · Master MIASHS, Université de Lille")
PAGES[choice]()
