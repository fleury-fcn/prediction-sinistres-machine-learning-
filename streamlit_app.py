"""Streamlit demo: score a claim with the exported model and browse the results.

    streamlit run streamlit_app.py

Requires ``results/models`` (produced by ``python -m src.main --stage evaluate``).
"""

from __future__ import annotations

import json
from datetime import date

import joblib
import pandas as pd
import streamlit as st

from src import config
from src.data import coerce_types

st.set_page_config(page_title="Claim fraud scoring", page_icon="🛡️", layout="wide")

EXAMPLE = json.loads((config.ROOT_DIR / "examples" / "claim.json").read_text())


@st.cache_resource
def load_model():
    card = json.loads((config.MODELS_DIR / "model_card.json").read_text())
    return joblib.load(config.MODELS_DIR / "fraud_model.joblib"), card


@st.cache_data
def reference_levels() -> dict[str, list[str]]:
    from src.data import load_clean

    df = load_clean(config.DATA_DIR / "insurance_claims.csv")
    return {c: sorted(df[c].unique()) for c in config.RAW_CATEGORICAL_COLUMNS}


def claim_form(levels: dict[str, list[str]]) -> dict:
    values: dict = {}
    cols = st.columns(3)
    for i, (key, default) in enumerate(EXAMPLE.items()):
        col = cols[i % 3]
        if key in levels:
            options = levels[key]
            values[key] = col.selectbox(key, options, index=options.index(default))
        elif key.endswith("_date"):
            values[key] = col.date_input(key, date.fromisoformat(default)).isoformat()
        else:
            values[key] = col.number_input(key, value=float(default))
    return values


st.title("🛡️ Insurance claim fraud scoring")
st.caption(
    "Demonstration model trained on 800 public claims (Derrick Mwiti dataset). "
    "Not a production fraud score: see the README for limitations."
)

try:
    model, card = load_model()
except FileNotFoundError:
    st.error("No exported model found. Run `make train evaluate` first.")
    st.stop()

tab_score, tab_results = st.tabs(["Score a claim", "Results on the real test set"])

with tab_score:
    st.write(
        f"Model: **{card['label']}** · decision threshold **{card['decision_threshold']:.2f}** "
        f"(minimises expected cost with investigation = ${card['costs_usd']['investigation']:,.0f}"
        f", missed fraud = ${card['costs_usd']['missed_fraud']:,.0f})"
    )
    with st.form("claim"):
        values = claim_form(reference_levels())
        submitted = st.form_submit_button("Score")
    if submitted:
        proba = float(model.predict_proba(coerce_types(pd.DataFrame([values])))[:, 1][0])
        flag = proba >= card["decision_threshold"]
        st.metric("P(fraud)", f"{proba:.1%}")
        if flag:
            st.warning("Above threshold: send to investigation")
        else:
            st.success("Below threshold: standard processing")

with tab_results:
    results_md = config.METRICS_DIR / "real_results.md"
    if results_md.exists():
        st.markdown(results_md.read_text())
    for name in ["real_roc_pr_curves", "real_metrics_ci", "real_cost_vs_threshold",
                 "real_shap_by_feature"]:
        path = config.FIGURES_DIR / f"{name}.png"
        if path.exists():
            st.image(str(path))
