import json

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src import config
from src.pipelines import make_pipeline


@pytest.fixture(scope="module")
def client(split, tmp_path_factory):
    X_train, _, y_train, _ = split
    model_dir = tmp_path_factory.mktemp("model")
    pipe = make_pipeline(LogisticRegression(max_iter=2000), encoding="onehot", scale=True)
    joblib.dump(pipe.fit(X_train, y_train), model_dir / "fraud_model.joblib")
    (model_dir / "model_card.json").write_text(
        json.dumps({"model": "logreg", "label": "test model", "decision_threshold": 0.3})
    )
    import api.app as app_module

    app_module.MODEL_DIR = model_dir
    app_module.load_model.cache_clear()
    yield TestClient(app_module.app)
    app_module.load_model.cache_clear()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_example(client):
    claim = json.loads((config.ROOT_DIR / "examples" / "claim.json").read_text())
    res = client.post("/predict", json=claim)
    assert res.status_code == 200
    body = res.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["flag_for_investigation"] == (body["fraud_probability"] >= 0.3)


def test_predict_rejects_invalid_input(client):
    claim = json.loads((config.ROOT_DIR / "examples" / "claim.json").read_text())
    claim["incident_hour"] = 30
    assert client.post("/predict", json=claim).status_code == 422
