PYTHON ?= python
NB_TIMEOUT ?= 1800

.PHONY: help install eda train evaluate synth all fast notebooks test lint format api app docker-build docker-run clean

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:  ## install pinned dependencies
	$(PYTHON) -m pip install -r requirements.txt

eda:  ## exploratory analysis figures
	$(PYTHON) -m src.main --stage eda

train:  ## tune + cross-validate every model on the real training split
	$(PYTHON) -m src.main --stage train

evaluate:  ## test-set evaluation, figures, SHAP, model export
	$(PYTHON) -m src.main --stage evaluate

synth:  ## synthetic generators, validation, TSTR, 500k dataset, scale benchmark
	$(PYTHON) -m src.main --stage synth

all:  ## full pipeline (numbers in the README) then execute the notebooks
	$(PYTHON) -m src.main --stage all
	$(MAKE) notebooks

fast:  ## quick end-to-end smoke run (small budgets, not for reporting)
	$(PYTHON) -m src.main --stage all --fast

notebooks:  ## execute the notebooks in place
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=$(NB_TIMEOUT) notebooks/*.ipynb

test:  ## unit tests
	$(PYTHON) -m pytest -q

lint:  ## ruff + black check
	$(PYTHON) -m ruff check src tests api streamlit_app.py
	$(PYTHON) -m black --check src tests api streamlit_app.py

format:  ## auto-format
	$(PYTHON) -m black src tests api streamlit_app.py
	$(PYTHON) -m ruff check --fix src tests api streamlit_app.py

api:  ## serve the exported model on :8000
	$(PYTHON) -m uvicorn api.app:app --host 0.0.0.0 --port 8000

app:  ## Streamlit demo
	$(PYTHON) -m streamlit run streamlit_app.py

docker-build:  ## build the API image (needs results/models from `make evaluate`)
	docker build -t fraud-api .

docker-run:
	docker run --rm -p 8000:8000 fraud-api

clean:  ## remove intermediate artefacts (keeps versioned results)
	rm -rf artifacts mlruns logs .pytest_cache
