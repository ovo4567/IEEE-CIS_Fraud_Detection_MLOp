# IEEE-CIS Fraud Detection MLOps

> An end-to-end fraud-detection platform that demonstrates how a model moves
> from training to deployment, monitoring, and controlled retraining.

[![CI](https://github.com/ovo4567/IEEE-CIS_Fraud_Detection_MLOp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ovo4567/IEEE-CIS_Fraud_Detection_MLOp/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Docker Compose](https://img.shields.io/badge/deployment-Docker%20Compose-2496ED?logo=docker&logoColor=white)](deploy/README.md)

This is a hands-on project for learning the complete machine-learning
operations (MLOps) lifecycle with LightGBM, MLflow, FastAPI, Prefect, Evidently,
DVC, and Docker Compose.

The goal of version 1 is not to find the perfect fraud-detection algorithm. It
is to understand how a model moves from data to a maintained, monitored service.
The current implementation uses LightGBM as the v1 model. Future versions can
replace or compare the model algorithm without changing the surrounding MLOps
workflow.

## Run the demo

The fastest way to see the project in action is to run the complete local stack:

```bash
# one-time data setup — download from Kaggle first (see Quickstart → Step 2)
python -m ieee_cis_fraud_detection.features
make demo
```

Once the stack is running, open these interfaces:

| Interface | URL | What to inspect |
| --- | --- | --- |
| FastAPI | http://localhost:8000/docs | Real-time scoring endpoint |
| MLflow | http://localhost:5001 | Runs, artifacts, and model registry |
| Prefect | http://localhost:4200 | Scheduled simulation and monitoring flows |

> **Screenshot placeholder:** After running the demo, capture a screenshot of
> the MLflow or Prefect dashboard. Save it as `docs/figures/demo-dashboard.png`,
> then replace this note with:
> `![MLflow and Prefect dashboards](docs/figures/demo-dashboard.png)`.

## Project status

| Item | Status |
| --- | --- |
| Project version | v1 |
| Learning focus | End-to-end MLOps lifecycle |
| Current model | LightGBM |
| Deployment style | Local Docker Compose |
| Future direction | Compare and improve ML algorithms |

## MLOps workflow

This project demonstrates the following closed loop:

```mermaid
flowchart LR
    A[Data acquisition] --> B[Data versioning]
    B --> C[Data preparation]
    C --> D[Feature engineering]
    D --> E[Model training]
    E --> F[Evaluation]
    F --> G[MLflow registry]
    G --> H[Deployment]
    H --> I[Real-time and batch scoring]
    I --> J[Drift monitoring]
    J --> K{Retraining trigger}
    K -->|No| J
    K -->|Yes| L[Train challenger]
    L --> M[Promotion gate]
    M -->|Promote| G
    M -->|Reject| J
```

### 1. Data acquisition and versioning

The dataset is obtained outside the repository and tracked with DVC. Raw data
files are kept out of Git, while DVC pointer files make the data dependency
reproducible.

> **No DVC remote:** this repository does not configure a DVC remote, so a
> fresh clone cannot `dvc pull`. Download the raw CSVs from Kaggle and place
> them in `data/raw/` instead — see Quickstart → Step 2.

**Dataset source:** [`IEEE-CIS Fraud Detection`](https://www.kaggle.com/competitions/ieee-fraud-detection/overview)

The current data represents online transactions and includes a binary
`isFraud` target. The data contains class imbalance, missing values, and a
temporal ordering that is useful for simulating production traffic.

### 2. Data preparation and feature engineering

The source tables are joined and transformed into processed feature files.
Feature preparation is kept separate from model serving so that the same
transformation is used consistently during training and inference.

The deployed model has a strict feature contract: exactly 218 input features,
including 9 categorical features with the expected dtypes. Invalid payloads
are rejected instead of being silently scored.

### 3. Training and evaluation

The current v1 model is a LightGBM classifier. The data is split chronologically
into:

- 70% training data
- 15% test data
- 15% production stream used to simulate unseen traffic

MLflow records training runs, parameters, metrics, and model artifacts. The
operating threshold is selected during evaluation and stored with the model.

### 4. Model registration

Models are packaged as MLflow `pyfunc` artifacts containing the feature
transformation, model, and operating threshold.

- **Champion**: the model currently served by the application.
- **Challenger**: a newly retrained model waiting for evaluation.
- **Promotion**: making a challenger the champion.

This keeps the model registry, the served artifact, and the prediction
interfaces aligned.

### 5. Deployment and serving

Docker Compose runs the local MLOps stack:

- MLflow provides experiment tracking and the model registry.
- FastAPI exposes the real-time prediction API.
- Prefect provides workflow orchestration and scheduling.
- A worker runs the stream simulator, monitoring, and retraining flows.

The API and worker share the same model store. When a challenger is promoted,
the API can serve the new champion without rebuilding or redeploying the
application.

The project supports two inference surfaces:

- **Real-time scoring**: score one transaction with `POST /predict`.
- **Batch scoring**: score a CSV and write results for monitoring.

### 6. Monitoring

The stream simulator replays the chronological production-stream slice through
the API. The monitoring flow batch-scores unseen chunks, stores the results, and
uses Evidently to compare the current window with the training reference.

Monitoring produces HTML and JSON drift reports and raises an alarm when the
configured drift rule is met.

### 7. Retraining and promotion

Retraining can be triggered by a drift alarm or by accumulated scored volume.
The retraining flow:

1. Collects historical data and transactions whose labels have been revealed.
2. Builds a new retraining corpus.
3. Trains a challenger model.
4. Evaluates the challenger against the champion.
5. Promotes the challenger only when the promotion gate is satisfied.

The simulated reveal lag models the fact that production labels are often
available after the original prediction, not at scoring time.

### 8. CI/CD

GitHub Actions validates every push and pull request with:

1. Locked dependency installation.
2. Ruff formatting and lint checks.
3. The model feature-contract check.
4. The pytest suite.
5. Docker Compose configuration validation.

After CI succeeds on the default branch, the serving image is built and
published to GitHub Container Registry.

## Engineering highlights

- A strict 218-feature contract rejects invalid real-time payloads.
- A chronological 70/15/15 split separates training, evaluation, and simulated production traffic.
- MLflow keeps the registry and served champion model aligned through promotion.
- Evidently reports feature and score drift in the simulated production stream.
- Hermetic tests validate serving, monitoring, retraining, and deployment behavior.

## Technology stack

| Area | Technologies |
| --- | --- |
| Language | Python 3.12 |
| Dependency management | uv, `pyproject.toml`, `uv.lock` |
| Data processing | pandas, NumPy, PyArrow |
| Current ML algorithm | LightGBM |
| Alternative ML libraries | scikit-learn, XGBoost, CatBoost |
| Experiment tracking and registry | MLflow |
| Data versioning | DVC |
| Workflow orchestration | Prefect |
| Real-time serving | FastAPI, Uvicorn |
| Batch serving | Python CLI |
| Drift monitoring | Evidently |
| Containers | Docker, Docker Compose |
| CI/CD | GitHub Actions, GitHub Container Registry |
| Testing and quality | pytest, Ruff |
| Exploration and visualization | JupyterLab, Matplotlib, Seaborn |

## Quickstart

Run the whole stack on a fresh clone in four steps. No training is required —
the v1 champion model is already committed to the repo.

```bash
# 1. Install the project (requires Python 3.12 and uv)
uv sync

# 2. Get the dataset — download it from Kaggle, then copy the two training
#    tables into data/raw/ (there is no DVC remote to pull from)
mkdir -p data/raw
cp path/to/train_transaction.csv data/raw/
cp path/to/train_identity.csv   data/raw/

# 3. Build the processed feature files from those CSVs
python -m ieee_cis_fraud_detection.features

# 4. Start the full MLOps stack (requires Docker Desktop, ≥ 12 GB memory)
make demo
```

When the stack is up, open <http://localhost:8000/docs> and try `POST /predict`.
Each step is explained below.

### Step 1 — Install dependencies

Prerequisites: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

You will also need [Docker Desktop](https://www.docker.com/products/docker-desktop/)
with at least 12 GB of memory allocated (used in step 4).

### Step 2 — Get the dataset (from Kaggle, no DVC remote)

This repository tracks raw data with DVC but does **not** configure a DVC
remote, so `dvc pull` is not available on a fresh clone. Get the data directly
from Kaggle instead:

1. Download the data from the
   [IEEE-CIS Fraud Detection competition](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
   (a free Kaggle account is required).
2. Unzip the download.
3. Copy the two **training** tables into `data/raw/`:

   ```bash
   mkdir -p data/raw
   cp path/to/train_transaction.csv data/raw/
   cp path/to/train_identity.csv   data/raw/
   ```

Only these two files are required by the pipeline. `test_transaction.csv`,
`test_identity.csv`, and `sample_submission.csv` are only used by the
exploration notebooks, so they can be skipped.

### Step 3 — Build the processed features

```bash
python -m ieee_cis_fraud_detection.features
```

This reads the raw CSVs from step 2 and writes
`data/processed/train_transaction_filtered.parquet` (and
`train_identity_filtered.parquet`). These generated files are intentionally
not committed to Git. The same command is available as `make data`.

### Step 4 — Deploy locally with Docker Compose

Make sure Docker Desktop is running and has at least 12 GB of memory allocated,
then start the complete local stack:

```bash
make demo
```

The committed seed model is used automatically, so a fresh deployment does not
need to retrain a model. When the command completes, verify that the services
are available by opening the FastAPI documentation at
`http://localhost:8000/docs`.

To watch the simulator and monitoring activity, or to stop the deployment:

```bash
make demo-logs
make demo-down
```

For Docker Compose configuration, resource requirements, environment variables,
and troubleshooting, see [`deploy/README.md`](deploy/README.md).

### Local service URLs

```bash
make demo
```

The stack provides:

| Service | URL | Purpose |
| --- | --- | --- |
| FastAPI | http://localhost:8000 | Real-time prediction API |
| MLflow | http://localhost:5001 | Experiments and model registry |
| Prefect | http://localhost:4200 | Flow scheduling and monitoring |
| Worker | No public URL | Simulator, monitoring, and retraining flows |

Useful commands:

```bash
make demo-logs       # follow service logs
make demo-down       # stop the Docker Compose stack
```

## MLOps commands without Docker

```bash
make data            # build processed features from the raw CSVs in data/raw
make seed            # create or refresh the v1 champion artifact
make simulate        # replay the production stream through the API
make monitor         # run one drift-monitoring pass
make retrain         # run one retraining and promotion pass
```

The demo alarms on drift but does not automatically retrain by default. To
enable automatic drift-to-retraining behavior:

```bash
MONITOR_TRIGGER_RETRAINING=true make demo
```

## Quality checks

```bash
make lint            # Ruff format and lint checks
make contract        # validate the model feature contract
make test            # run the test suite
```

Tests are designed to be hermetic and run without a network connection or the
raw dataset when the committed seed artifact is available.

## Repository layout

```text
ieee_cis_fraud_detection/
├── modeling/        # splitting, thresholds, training, and model packaging
├── serving/         # FastAPI, batch scoring, and model loading
├── monitoring/      # drift detection and monitoring storage
├── orchestration/   # simulation, monitoring, and retraining flows
└── deployment/      # model seeding and feature-contract checks

data/                # DVC-tracked raw data and generated processed features
models/seed/         # committed v1 MLflow champion artifact
deploy/              # Dockerfile, Compose stack, and container scripts
notebooks/            # exploratory analysis and modeling notebooks
tests/                # automated tests
docs/                 # project documentation and architecture decisions
references/           # dataset and supporting references
```

## Version 1 limitations and future work

This project intentionally prioritizes understanding the MLOps lifecycle over
production-scale infrastructure. Planned improvements include:

- Comparing LightGBM with XGBoost, CatBoost, and other algorithms.
- Improving feature engineering and model calibration.
- Adding richer experiment-comparison dashboards.
- Adding production-grade metrics and observability.
- Evaluating cloud deployment options.
- Replacing the simulated stream and label reveal process with real data
  integrations.

The algorithm is expected to evolve. The reusable part of the project is the
workflow that makes each new model reproducible, testable, deployable,
monitorable, and replaceable.

## Related documentation

- [`deploy/README.md`](deploy/README.md) - detailed Docker deployment guide
- [`CONTEXT.md`](CONTEXT.md) - project vocabulary and lifecycle terminology
- [`docs/`](docs/) - documentation and architecture decisions
- [`references/data-description.md`](references/data-description.md) - current
  data description
