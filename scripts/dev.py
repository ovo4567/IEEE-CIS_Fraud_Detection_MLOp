#!/usr/bin/env python3
"""Cross-platform task runner for this repo — no `make` needed.

`make` is not native to Windows, so this script is the portable equivalent of
every Makefile target. Python is a hard prerequisite of this project, so it
runs on macOS, Linux, and Windows with nothing extra to install.

Each subcommand mirrors a Makefile target (run `make <name>` or
`python scripts/dev.py <name>` — identical behavior):

    requirements        uv sync
    clean               remove __pycache__ / *.pyc
    lint                ruff format --check + ruff check
    format              ruff check --fix + ruff format
    test                pytest tests
    contract            feature-contract check on the committed seed
    create_environment  uv venv --python 3.12
    data                build processed features from data/raw
    seed                re-fit + seed the champion
    retrain             one retraining pass (trigger -> challenger -> promotion)
    simulate            replay the production stream through the API
    monitor             one drift-monitoring pass
    demo                build + start the Docker stack (== make demo)
    logs                tail the stack logs (== make demo-logs)
    down                stop the stack (== make demo-down)

Tuning env vars (same as `make demo`):
    SIMULATOR_MAX_TRANSACTIONS, SIMULATOR_CADENCE_SECONDS,
    SIMULATOR_INTERVAL_SECONDS, MONITOR_INTERVAL_SECONDS, MONITOR_CHUNK_ROWS,
    MONITOR_TRIGGER_RETRAINING
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = REPO_ROOT / "deploy" / "compose.yaml"
PROCESSED_FEATURES = REPO_ROOT / "data" / "processed" / "train_transaction_filtered.parquet"
SEED_DB = REPO_ROOT / "models" / "seed" / "mlflow.db"
PYTHON_VERSION = "3.12"

URLS = """\
Demo stack is up — no training, no cloud:
  MLflow UI   http://localhost:5001   (champion v1 seeded from the committed artifact)
  Prefect UI  http://localhost:4200   (scheduled simulator + monitoring deployments)
  API         http://localhost:8000   (POST /predict)

Watch live scoring + monitoring passes:  python scripts/dev.py logs
Stop the stack:                          python scripts/dev.py down
"""


def _run(*cmd: str) -> int:
    return subprocess.run([*cmd], check=False).returncode


def _tool(name: str) -> str:
    """Resolve a CLI tool, preferring the active venv's copy when present."""
    if name == "ruff":
        candidate = Path(sys.executable).parent / ("ruff.exe" if os.name == "nt" else "ruff")
        if candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    if found is None:
        sys.exit(f"ERROR: '{name}' not found on PATH — run 'uv sync' first (or install it).")
    return found


def _py(*args: str) -> int:
    return _run(sys.executable, *args)


def cmd_requirements() -> int:
    return _run(_tool("uv"), "sync")


def cmd_clean() -> int:
    removed = 0
    for path in REPO_ROOT.rglob("__pycache__"):
        if ".venv" in path.parts:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    for path in REPO_ROOT.rglob("*.pyc"):
        if ".venv" in path.parts:
            continue
        path.unlink(missing_ok=True)
        removed += 1
    print(f"Removed {removed} bytecode paths.")
    return 0


def cmd_lint() -> int:
    ruff = _tool("ruff")
    if _run(ruff, "format", "--check") != 0:
        return 1
    return _run(ruff, "check")


def cmd_format() -> int:
    ruff = _tool("ruff")
    if _run(ruff, "check", "--fix") != 0:
        return 1
    return _run(ruff, "format")


def cmd_test() -> int:
    return _py("-m", "pytest", "tests")


def cmd_contract() -> int:
    return _py("-m", "ieee_cis_fraud_detection.deployment.contract_check")


def cmd_create_environment() -> int:
    code = _run(_tool("uv"), "venv", "--python", PYTHON_VERSION)
    if code == 0:
        print(">>> New uv virtual environment created. Activate with:")
        print(">>> Windows:      .venv\\Scripts\\activate")
        print(">>> Unix/macOS:   source ./.venv/bin/activate")
    return code


def cmd_data() -> int:
    return _py("-m", "ieee_cis_fraud_detection.features")


def cmd_seed() -> int:
    return _py("-m", "ieee_cis_fraud_detection.modeling.train")


def cmd_retrain() -> int:
    return _py("-m", "ieee_cis_fraud_detection.orchestration.retraining")


def cmd_simulate() -> int:
    return _py("-m", "ieee_cis_fraud_detection.orchestration.monitoring", "simulate")


def cmd_monitor() -> int:
    return _py("-m", "ieee_cis_fraud_detection.orchestration.monitoring", "monitor")


def _docker() -> str:
    return _tool("docker")


def _compose(*args: str) -> int:
    return _run(_docker(), "compose", "-f", str(COMPOSE_FILE), *args)


def preflight() -> None:
    """Fail fast with a friendly message if a demo prerequisite is missing."""
    # `docker info` fails when the daemon is not running.
    result = subprocess.run([_docker(), "info"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.exit("ERROR: Docker is not running (start Docker Desktop first).")
    if not PROCESSED_FEATURES.exists():
        sys.exit(
            "ERROR: processed features missing — copy the Kaggle CSVs into "
            "data/raw, then run:\n"
            "  macOS/Linux:  .venv/bin/python -m ieee_cis_fraud_detection.features\n"
            "  Windows:      .venv\\Scripts\\python -m ieee_cis_fraud_detection.features"
        )
    if not SEED_DB.exists():
        sys.exit(
            "ERROR: committed seed missing — run "
            "'.venv/bin/python -m ieee_cis_fraud_detection.modeling.train' first."
        )


def cmd_demo() -> int:
    preflight()
    code = _compose("up", "--build", "-d")
    if code != 0:
        return code
    print(URLS)
    return 0


def cmd_logs() -> int:
    return _compose("logs", "-f", "--tail=100")


def cmd_down() -> int:
    return _compose("down")


COMMANDS = {
    "requirements": cmd_requirements,
    "clean": cmd_clean,
    "lint": cmd_lint,
    "format": cmd_format,
    "test": cmd_test,
    "contract": cmd_contract,
    "create_environment": cmd_create_environment,
    "data": cmd_data,
    "seed": cmd_seed,
    "retrain": cmd_retrain,
    "simulate": cmd_simulate,
    "monitor": cmd_monitor,
    "demo": cmd_demo,
    "logs": cmd_logs,
    "down": cmd_down,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-platform task runner for this repo (no make needed)."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        sub.add_parser(name, help=f"== make {name}")
    args = parser.parse_args()
    return COMMANDS[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
