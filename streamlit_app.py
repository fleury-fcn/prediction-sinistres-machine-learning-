"""
Application Streamlit pour explorer les performances et prédictions du modèle.
"""

from __future__ import annotations

import importlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

from src import config, data_loading, evaluation, model, preprocessing, visualization

MODEL_BASELINE = "Régression Logistique"
MODEL_RF = "Random Forest"
SHAP_IMPACT_LABEL = "Impact SHAP"

st.set_page_config(page_title="Prédiction des sinistres", page_icon="🛡️", layout="wide")


@st.cache_data
def load_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = data_loading.load_raw_data(config.DATA_PATH)
    data_loading.validate_columns(df, [config.TARGET_COLUMN])
    return preprocessing.run_pipeline(df)


@st.cache_resource
def load_models(X_train: pd.DataFrame, y_train: pd.Series):
    baseline_model = model.train_baseline(X_train, y_train)

    try:
        rf_model = model.load_model()
    except FileNotFoundError:
        rf_model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)

    return baseline_model, rf_model


def compute_shap_values(rf_model, X_test: pd.DataFrame) -> np.ndarray:
    try:
        shap_module = importlib.import_module("shap")
        explainer = shap_module.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            return np.asarray(shap_values[1])

        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            return shap_array[:, :, 1]

        return shap_array
    except Exception:
        centered = X_test - X_test.median(numeric_only=True)
        scales = X_test.std(numeric_only=True).replace(0, 1)
        normalized = centered.divide(scales, axis=1).fillna(0)
        weights = np.asarray(getattr(rf_model, "feature_importances_", np.ones(X_test.shape[1])))
        return normalized.to_numpy() * weights


def compute_single_shap_vector(rf_model, input_df: pd.DataFrame, reference_x: pd.DataFrame) -> np.ndarray:
    try:
        shap_module = importlib.import_module("shap")
        explainer = shap_module.TreeExplainer(rf_model)
        shap_single = explainer.shap_values(input_df)

        if isinstance(shap_single, list):
            return np.asarray(shap_single[1]).ravel()

        shap_array = np.asarray(shap_single)
        if shap_array.ndim == 3:
            return shap_array[0, :, 1]

        return shap_array.ravel()
    except Exception:
        centered = input_df - reference_x.median(numeric_only=True)
        scales = reference_x.std(numeric_only=True).replace(0, 1)
        normalized = centered.divide(scales, axis=1).fillna(0)
        weights = np.asarray(getattr(rf_model, "feature_importances_", np.ones(input_df.shape[1])))
        return (normalized.to_numpy() * weights).ravel()


@st.cache_data
def compute_artifacts():
    X, y, features = load_dataset()
    X_train, X_test, y_train, y_test = model.split_data(X, y)
    baseline_model, rf_model = load_models(X_train, y_train)

    baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
    rf_proba = rf_model.predict_proba(X_test)[:, 1]

    baseline_threshold = evaluation.find_optimal_threshold(y_test, baseline_proba)
    rf_threshold = evaluation.find_optimal_threshold(y_test, rf_proba)

    baseline_pred = (baseline_proba >= baseline_threshold).astype(int)
    rf_pred = (rf_proba >= rf_threshold).astype(int)

    comparison_df = pd.DataFrame(
        [
            {
                "Modele": MODEL_BASELINE,
                "Seuil": baseline_threshold,
                **evaluation.compute_metrics(y_test, baseline_pred, baseline_proba),
            },
            {
                "Modele": MODEL_RF,
                "Seuil": rf_threshold,
                **evaluation.compute_metrics(y_test, rf_pred, rf_proba),
            },
        ]
    )

    roc_by_model = {
        MODEL_BASELINE: {
            "fpr": evaluation.compute_roc_curve(y_test, baseline_proba)[0],
            "tpr": evaluation.compute_roc_curve(y_test, baseline_proba)[1],
            "auc": evaluation.roc_auc_score(y_test, baseline_proba),
        },
        MODEL_RF: {
            "fpr": evaluation.compute_roc_curve(y_test, rf_proba)[0],
            "tpr": evaluation.compute_roc_curve(y_test, rf_proba)[1],
            "auc": evaluation.roc_auc_score(y_test, rf_proba),
        },
    }

    confusion_by_model = {
        MODEL_BASELINE: evaluation.compute_confusion(y_test, baseline_pred),
        MODEL_RF: evaluation.compute_confusion(y_test, rf_pred),
    }

    importance_by_model = {
        MODEL_BASELINE: evaluation.linear_feature_importance(baseline_model, features),
        MODEL_RF: evaluation.feature_importance(rf_model, features),
    }

    shap_values = compute_shap_values(rf_model, X_test)

    return {
        "models": {
            MODEL_BASELINE: baseline_model,
            MODEL_RF: rf_model,
        },
        "comparison_df": comparison_df,
        "roc_by_model": roc_by_model,
        "confusion_by_model": confusion_by_model,
        "importance_by_model": importance_by_model,
        "shap_values": shap_values,
        "X": X,
        "X_test": X_test,
        "features": features,
        "thresholds": {
            MODEL_BASELINE: baseline_threshold,
            MODEL_RF: rf_threshold,
        },
    }


artifacts = compute_artifacts()

st.title("🛡️ Dashboard — Prédiction des sinistres")
st.caption("Analyse interactive des modèles et exploration des prédictions individuelles")

comparison_tab, explorer_tab = st.tabs(["Comparaison des modèles", "Explorer une prédiction"])

with comparison_tab:
    st.subheader("Comparaison des performances")

    search = st.text_input("Filtrer le tableau (nom de modèle)", value="")
    comparison_df = artifacts["comparison_df"].copy()
    if search:
        comparison_df = comparison_df[
            comparison_df["Modele"].str.contains(search, case=False, na=False)
        ]

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    selected_model = st.selectbox(
        "Sélectionner un modèle pour mettre à jour les graphiques",
        [MODEL_RF, MODEL_BASELINE],
    )

    roc_fig = visualization.build_comparative_roc_figure(artifacts["roc_by_model"])
    for trace in roc_fig.data:
        if selected_model in trace.name:
            trace.update(line={"width": 4}, opacity=1.0)
        elif "Aléatoire" not in trace.name:
            trace.update(opacity=0.35)

    st.plotly_chart(roc_fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        cm_fig = visualization.build_confusion_figure(
            artifacts["confusion_by_model"][selected_model],
            selected_model,
        )
        st.plotly_chart(cm_fig, use_container_width=True)

    with col_b:
        imp_fig = visualization.build_importance_figure(
            artifacts["importance_by_model"][selected_model],
            selected_model,
        )
        st.plotly_chart(imp_fig, use_container_width=True)

    if selected_model == MODEL_RF:
        shap_fig = visualization.build_shap_scatter_figure(
            artifacts["shap_values"],
            artifacts["X_test"],
            selected_model,
        )
        st.plotly_chart(shap_fig, use_container_width=True)

with explorer_tab:
    st.subheader("Simuler une nouvelle observation")

    X = artifacts["X"]
    features = artifacts["features"]

    col_left, col_right = st.columns(2)
    input_values = {}

    for idx, feature in enumerate(features):
        lower = float(X[feature].quantile(0.01))
        upper = float(X[feature].quantile(0.99))
        default = float(X[feature].median())
        step = max((upper - lower) / 200, 0.01)

        target_col = col_left if idx % 2 == 0 else col_right
        with target_col:
            input_values[feature] = st.number_input(
                feature,
                min_value=lower,
                max_value=upper,
                value=default,
                step=float(step),
            )

    model_name = st.selectbox("Modèle utilisé", [MODEL_RF, MODEL_BASELINE], key="explorer_model")

    input_df = pd.DataFrame([input_values])
    selected_model = artifacts["models"][model_name]
    probability = float(selected_model.predict_proba(input_df)[0, 1])
    threshold = float(artifacts["thresholds"][model_name])
    predicted_class = int(probability >= threshold)

    st.metric("Probabilité prédite de fraude", f"{probability:.2%}")
    st.metric("Classe prédite", "Fraude" if predicted_class == 1 else "Non fraude")
    st.caption(f"Seuil de décision appliqué: {threshold:.3f}")

    if model_name == MODEL_RF:
        shap_vector = compute_single_shap_vector(selected_model, input_df, X)

        shap_df = pd.DataFrame({
            "Variable": features,
            SHAP_IMPACT_LABEL: shap_vector,
        }).sort_values(SHAP_IMPACT_LABEL, key=np.abs, ascending=True)

        shap_local_fig = go.Figure(
            go.Bar(
                x=shap_df[SHAP_IMPACT_LABEL],
                y=shap_df["Variable"],
                orientation="h",
                marker_color=np.where(shap_df[SHAP_IMPACT_LABEL] >= 0, "#E74C3C", "#3498DB"),
            )
        )
        shap_local_fig.update_layout(
            title="Explication SHAP locale de la prédiction",
            xaxis_title="Contribution à la probabilité de fraude",
            yaxis_title="Variable",
            template="plotly_white",
        )
        st.plotly_chart(shap_local_fig, use_container_width=True)
    else:
        st.info("L'explication SHAP détaillée est disponible pour le modèle Random Forest.")

st.markdown("---")
st.markdown(
    "Les visualisations HTML exportées sont disponibles dans `results/interactive/` "
    "et publiées automatiquement via GitHub Pages."
)
