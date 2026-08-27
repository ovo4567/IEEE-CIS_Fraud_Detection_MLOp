#!/usr/bin/env sh
set -eu

# MLflow service entrypoint (ticket 09): seed the shared model store from the
# committed seed artifact on first boot (idempotent — never clobbers a live
# registry that may hold challengers), then run the tracking/registry server
# against the same seeded store the worker writes to and the API serves from.
# --require: on a fresh clone with no committed seed the container must fail
# fast instead of silently booting an empty registry the API would serve from.
python -m ieee_cis_fraud_detection.deployment.seed \
  --source /seed-src \
  --target /app/models \
  --require

exec mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:////app/models/seed/mlflow.db \
  --default-artifact-root /app/models/seed
