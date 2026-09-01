# The single image for the whole self-contained demo stack (ADR-0001): the
# real-time API, the batch scorer, the Prefect worker, and MLflow all run this
# image with different entrypoints. Only the *runtime* dependencies are
# installed (no notebooks / docs / dev tooling) to keep the build lean.
#
# The committed seed model and the processed data are NOT baked in — they are
# mounted at runtime (models via a seeded named volume, data read-only), so the
# image is small and the demo serves the committed champion with no re-training.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    MLFLOW_HOME=/app

WORKDIR /app

# Build tools: some runtime deps (e.g. statsmodels, pulled in by evidently)
# fall back to source builds, which need a C compiler + Python headers. curl is
# used by the Compose healthchecks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Runtime dependencies first so code changes reuse the cached layer.
COPY deploy/requirements.txt /app/deploy/requirements.txt
RUN pip install --no-cache-dir -r /app/deploy/requirements.txt

# The package source + metadata (flit_core needs pyproject, readme, license).
COPY pyproject.toml README.md LICENSE /app/
COPY ieee_cis_fraud_detection /app/ieee_cis_fraud_detection
RUN pip install --no-cache-dir --no-deps -e /app

# Container entrypoints (seed MLflow store, Prefect worker, wait-for-http).
COPY deploy/scripts /app/deploy/scripts
# Normalize line endings: a Windows checkout (core.autocrlf=true) can inject
# CRLF into these scripts, which breaks the shebang (`env: 'sh\r': ...`). Strip
# any trailing CR so the image works regardless of the host's line endings.
RUN sed -i 's/\r$//' /app/deploy/scripts/*.sh /app/deploy/scripts/*.py \
    && chmod +x /app/deploy/scripts/*.sh

# Default: the real-time API (overridden per service in compose.yaml).
CMD ["uvicorn", "ieee_cis_fraud_detection.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
