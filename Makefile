#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = IEEE-CIS_Fraud_Detection_MLOp
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#                                                                               #
# Every target below is a thin wrapper around `scripts/dev.py <name>` — the     #
# cross-platform runner that also works on Windows, where `make` is not native. #
# Run any target from any OS with:                                              #
#     python scripts/dev.py <name>                                              #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) scripts/dev.py requirements



## Delete all compiled Python files
.PHONY: clean
clean:
	$(PYTHON_INTERPRETER) scripts/dev.py clean


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	$(PYTHON_INTERPRETER) scripts/dev.py lint

## Format source code with ruff
.PHONY: format
format:
	$(PYTHON_INTERPRETER) scripts/dev.py format



## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) scripts/dev.py test


## Verify the committed seed artifact's feature contract (CI step, ticket 10)
.PHONY: contract
contract:
	$(PYTHON_INTERPRETER) scripts/dev.py contract


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	$(PYTHON_INTERPRETER) scripts/dev.py create_environment
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Build the processed features from the raw CSVs in data/raw
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) scripts/dev.py data


## Seed the committed champion model (re-fit finetuned_lgbm on the 70/15/15 split, register champion v1)
.PHONY: seed
seed:
	$(PYTHON_INTERPRETER) scripts/dev.py seed


## Run the retraining flow once (trigger -> corpus -> challenger -> promotion gate)
.PHONY: retrain
retrain:
	$(PYTHON_INTERPRETER) scripts/dev.py retrain


## Replay the production stream through the real-time API (live demo)
.PHONY: simulate
simulate:
	$(PYTHON_INTERPRETER) scripts/dev.py simulate


## Run one scheduled drift-monitoring pass (batch-score chunk -> store -> Evidently report -> alarm)
.PHONY: monitor
monitor:
	$(PYTHON_INTERPRETER) scripts/dev.py monitor


#################################################################################
# DEMO (ticket 09): the self-contained Docker Compose stack (ADR-0001)         #
#################################################################################

## Bring up the self-contained demo stack (MLflow, API, Prefect, worker) from the committed seed, offline
.PHONY: demo
demo:
	$(PYTHON_INTERPRETER) scripts/dev.py demo


## Tail the demo stack logs (live scoring + monitoring passes)
.PHONY: demo-logs
demo-logs:
	$(PYTHON_INTERPRETER) scripts/dev.py logs


## Stop the demo stack
.PHONY: demo-down
demo-down:
	$(PYTHON_INTERPRETER) scripts/dev.py down


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
