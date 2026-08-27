#!/usr/bin/env sh
set -eu

# Prefect worker entrypoint (ticket 09): the Compose healthchecks already order
# startup, but wait for the Prefect server (to register deployments) and the
# API (the simulator posts to it) before serving the scheduled flows.
python /app/deploy/scripts/wait_for_http.py http://prefect-server:4200/api/health 120
python /app/deploy/scripts/wait_for_http.py http://api:8000/openapi.json 120

exec python /app/deploy/scripts/worker.py
