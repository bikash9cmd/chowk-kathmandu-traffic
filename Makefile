PY ?= python

.PHONY: setup data quality evaluate robustness train replay serve test lint validate-infra cost

setup:            ## create a virtualenv with dev dependencies
	$(PY) -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

data:             ## download the dataset and detector weights, verify checksums
	$(PY) -m chowk download

quality: data     ## data quality report (sessions, duplicates, sizes)
	$(PY) -m chowk quality

evaluate: data    ## session-grouped CV, baselines, charts in docs/img (about 4 minutes, first run 5)
	$(PY) -m chowk evaluate

robustness: evaluate  ## dusk / far away / dusty lens / low bitrate experiments (about 10 minutes)
	$(PY) -m chowk robustness

train: data       ## train with the promotion gate -> artifacts/models/champion
	$(PY) -m chowk train

replay: train     ## run every recorded session through the counting code as 15-minute flows
	$(PY) -m chowk replay

serve:            ## HTTP API on :8000
	$(PY) -m chowk serve

cost:             ## monthly AWS cost for 1, 20 and 200 junctions
	$(PY) scripts/cost_model.py

test:
	$(PY) -m pytest -q

lint:
	ruff check .

validate-infra:   ## lint the CloudFormation/SAM template
	cfn-lint infra/template.yaml
