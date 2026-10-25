"""Minimal fraud-scoring API.

Run locally::

    uvicorn api.app:app --port 8000
    curl -X POST localhost:8000/predict -H "Content-Type: application/json" \\
         -d @examples/claim.json

The model is the pipeline exported by ``python -m src.main --stage evaluate``
(``results/models/fraud_model.joblib``); set ``FRAUD_MODEL_DIR`` to load it
from elsewhere. Scores come from a model trained on 800 public claims: they
are a demonstration, not a production fraud score.
"""

from __future__ import annotations

import json
import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config
from src.data import coerce_types

MODEL_DIR = Path(os.environ.get("FRAUD_MODEL_DIR", config.MODELS_DIR))

YesNo = Literal["YES", "NO", "?", "UNKNOWN"]


class Claim(BaseModel):
    """One claim, in the internal column names of the cleaned dataset.

    Categorical values follow the public dataset; unseen levels are accepted and
    handled by the encoders. ``?`` / ``UNKNOWN`` mark a missing value.
    """

    months_as_customer: float = Field(ge=0)
    age: float = Field(ge=16, le=120)
    policy_bind_date: date
    policy_state: str
    policy_csl: str = Field(pattern=r"^\d+/\d+$", examples=["250/500"])
    policy_deductible: float = Field(ge=0)
    policy_annual_premium: float = Field(gt=0)
    umbrella_limit: float = Field(ge=0)
    insured_sex: str
    insured_education_level: str
    insured_occupation: str
    insured_hobbies: str
    insured_relationship: str
    capital_gains: float = Field(ge=0)
    capital_loss: float = Field(le=0)
    incident_date: date
    incident_type: str
    collision_type: str
    incident_severity: str
    authorities_contacted: str
    incident_state: str
    incident_city: str
    incident_hour: int = Field(ge=0, le=23)
    number_of_vehicles_involved: int = Field(ge=1)
    property_damage: YesNo
    bodily_injuries: int = Field(ge=0)
    witnesses: int = Field(ge=0)
    police_report_available: YesNo
    injury_claim: float = Field(ge=0)
    property_claim: float = Field(ge=0)
    vehicle_claim: float = Field(ge=0)
    auto_make: str
    auto_model: str
    auto_year: int = Field(ge=1900, le=2100)


class Prediction(BaseModel):
    fraud_probability: float
    flag_for_investigation: bool
    decision_threshold: float
    model: str


@lru_cache(maxsize=1)
def load_model() -> tuple[object, dict]:
    model_path = MODEL_DIR / "fraud_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found: run `make train evaluate` first")
    with open(MODEL_DIR / "model_card.json", encoding="utf-8") as fh:
        card = json.load(fh)
    return joblib.load(model_path), card


app = FastAPI(title="Insurance claim fraud scoring", version="1.0.0")


@app.get("/health")
def health() -> dict:
    try:
        _, card = load_model()
    except FileNotFoundError as exc:
        return {"status": "degraded", "detail": str(exc)}
    return {"status": "ok", "model": card["model"], "threshold": card["decision_threshold"]}


@app.post("/predict", response_model=Prediction)
def predict(claim: Claim) -> Prediction:
    try:
        model, card = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    frame = coerce_types(pd.DataFrame([claim.model_dump()]))
    proba = float(model.predict_proba(frame)[:, 1][0])
    threshold = float(card["decision_threshold"])
    return Prediction(
        fraud_probability=proba,
        flag_for_investigation=proba >= threshold,
        decision_threshold=threshold,
        model=card["label"],
    )
