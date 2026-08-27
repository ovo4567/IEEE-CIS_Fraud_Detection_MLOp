"""Seeding the MLflow store from the committed seed (ticket 09).

The self-contained demo stack (ADR-0001) runs MLflow on a named volume that
is seeded on first boot from the committed seed artifact (``models/seed``), so
a fresh clone serves the champion with no re-training and no cloud. This is the
one pure, unit-testable seam in ticket 09 (Compose / Prefect correctness is
verified by running the stack, per the master spec's testing decision).
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from ieee_cis_fraud_detection.deployment.seed import app, seed_mlflow_store


def _committed_seed(tmp_path: Path) -> Path:
    """A committed ``models/seed``-shaped directory (db + pyfunc artifact)."""
    seed = tmp_path / "models" / "seed"
    (seed / "champion_model").mkdir(parents=True)
    (seed / "mlflow.db").write_text("seed-db")
    (seed / "champion_model" / "MLmodel").write_text("pyfunc")
    return seed


def test_seeds_an_empty_store(tmp_path: Path) -> None:
    source = _committed_seed(tmp_path)
    target = tmp_path / "store" / "models"

    assert seed_mlflow_store(source, target) is True

    marker = target / "seed" / "mlflow.db"
    assert marker.exists()
    assert marker.read_text() == "seed-db"
    assert (target / "seed" / "champion_model" / "MLmodel").read_text() == "pyfunc"


def test_is_idempotent_and_never_clobbers_a_live_registry(tmp_path: Path) -> None:
    source = _committed_seed(tmp_path)
    target = tmp_path / "store" / "models"

    seed_mlflow_store(source, target)
    # A live registry accumulates challengers after seeding.
    (target / "seed" / "mlflow.db").write_text("live-registry")

    assert seed_mlflow_store(source, target) is False
    assert (target / "seed" / "mlflow.db").read_text() == "live-registry"


def test_missing_source_leaves_store_untouched(tmp_path: Path) -> None:
    target = tmp_path / "store" / "models"
    missing = tmp_path / "no-seed"

    assert seed_mlflow_store(missing, target) is False
    assert not (target / "seed").exists()


def test_cli_require_fails_fast_when_no_committed_seed(tmp_path: Path) -> None:
    """The container entrypoint passes --require: an empty store with no
    committed seed must fail fast instead of booting an empty registry."""
    runner = CliRunner()
    target = tmp_path / "store" / "models"
    missing = tmp_path / "no-seed"

    result = runner.invoke(
        app,
        ["--source", str(missing), "--target", str(target), "--require"],
    )

    assert result.exit_code == 1
    assert "run `make seed`" in result.output
    assert not (target / "seed").exists()


def test_cli_require_succeeds_when_already_seeded(tmp_path: Path) -> None:
    """--require on an already-seeded store is a no-op (idempotent), not a
    failure — a restart of the stack must not trip the guard."""
    source = _committed_seed(tmp_path)
    target = tmp_path / "store" / "models"
    seed_mlflow_store(source, target)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["--source", str(source), "--target", str(target), "--require"],
    )

    assert result.exit_code == 0
    assert (target / "seed" / "mlflow.db").read_text() == "seed-db"
