"""Smoke test of the Streamlit app: every page, both languages, plus claim scoring."""

import pytest

from src import config

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
APP = str(config.ROOT_DIR / "app" / "streamlit_app.py")

pytestmark = pytest.mark.skipif(
    not (config.MODELS_DIR / "fraud_model.joblib").exists(), reason="no exported model"
)


@pytest.mark.parametrize("language", ["Français", "English"])
def test_every_page_renders(language):
    at = streamlit_testing.AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value(language).run()
    pages = at.sidebar.radio[1].options
    assert len(pages) == 7
    for page in pages:
        at.sidebar.radio[1].set_value(page).run()
        assert not at.exception, f"{language} / {page}: {at.exception}"


def test_score_a_claim():
    at = streamlit_testing.AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value("English").run()
    at.sidebar.radio[1].set_value("Score a claim").run()
    at.button[0].click().run()
    assert not at.exception
    assert at.metric[0].value.endswith("%")
