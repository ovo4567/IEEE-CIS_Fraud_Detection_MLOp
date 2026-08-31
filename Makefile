#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = IEEE-CIS_Fraud_Detection_MLOp
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Verify the committed seed artifact's feature contract (CI step, ticket 10)
.PHONY: contract
contract:
	python -m ieee_cis_fraud_detection.deployment.contract_check


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Build the processed features from the raw CSVs in data/raw
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m ieee_cis_fraud_detection.features


## Seed the committed champion model (re-fit finetuned_lgbm on the 70/15/15 split, register champion v1)
.PHONY: seed
seed:
	$(PYTHON_INTERPRETER) -m ieee_cis_fraud_detection.modeling.train


## Run the retraining flow once (trigger -> corpus -> challenger -> promotion gate)
.PHONY: retrain
retrain:
	$(PYTHON_INTERPRETER) -m ieee_cis_fraud_detection.orchestration.retraining


## Replay the production stream through the real-time API (live demo)
.PHONY: simulate
simulate:
	$(PYTHON_INTERPRETER) -m ieee_cis_fraud_detection.orchestration.monitoring simulate


## Run one scheduled drift-monitoring pass (batch-score chunk -> store -> Evidently report -> alarm)
.PHONY: monitor
monitor:
	$(PYTHON_INTERPRETER) -m ieee_cis_fraud_detection.orchestration.monitoring monitor


#################################################################################
# DEMO (ticket 09): the self-contained Docker Compose stack (ADR-0001)         #
#################################################################################

## Bring up the self-contained demo stack (MLflow, API, Prefect, worker) from the committed seed, offline
.PHONY: demo
demo:
	@docker info >/dev/null 2>&1 || { echo "ERROR: Docker is not running (start Docker Desktop first)"; exit 1; }
	@test -f data/processed/train_transaction_filtered.parquet || { echo "ERROR: processed features missing — copy the Kaggle CSVs into data/raw and run 'make data'"; exit 1; }
	@test -f models/seed/mlflow.db || { echo "ERROR: committed seed missing — run 'make seed' first"; exit 1; }
	docker compose -f deploy/compose.yaml up --build -d
	@echo ""
	@echo "Demo stack is up — no training, no cloud:"
	@echo "  MLflow UI   http://localhost:5001   (champion v1 seeded from the committed artifact)"
	@echo "  Prefect UI  http://localhost:4200   (scheduled simulator + monitoring deployments)"
	@echo "  API         http://localhost:8000   (POST /predict)"
	@echo ""
	@echo "Watch live scoring + monitoring passes:  make demo-logs"
	@echo "Stop the stack:                          make demo-down"


## Tail the demo stack logs (live scoring + monitoring passes)
.PHONY: demo-logs
demo-logs:
	docker compose -f deploy/compose.yaml logs -f --tail=100


## Stop the demo stack
.PHONY: demo-down
demo-down:
	docker compose -f deploy/compose.yaml down


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
