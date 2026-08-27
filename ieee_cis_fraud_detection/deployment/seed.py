"""Seeding the MLflow store from the committed seed (ticket 09).

The self-contained demo stack (ADR-0001) runs MLflow on a named volume that is
seeded on first boot from the committed seed artifact (``models/seed``), so a
fresh clone serves the champion with no re-training and no cloud. This module
holds the idempotent seeding rule the ``mlflow`` container entrypoint applies:
copy the committed seed into an empty store exactly once, and never clobber a
store that already has a registry (a live store may have accumulated
challengers from retraining).
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Annotated

from loguru import logger
import typer

# The file whose presence marks a seeded store: the champion registry DB at
# ``<store>/seed/mlflow.db`` (mirrors ``SEED_REGISTRY_DIR`` in the container).
_MARKER = "seed/mlflow.db"


def seed_mlflow_store(source: Path, target: Path) -> bool:
    """Copy the committed seed into an empty model store; no-op when seeded.

    ``source`` is the committed ``models/seed`` directory (``mlflow.db`` +
    ``champion_model/``); ``target`` is the mounted model store (``models`` in
    the container) that MLflow, the API, and the worker share. The store is
    seeded exactly once: when ``target/seed/mlflow.db`` is absent. A store that
    already has the marker is left untouched — a live registry may hold
    challengers from retraining and must never be clobbered.

    Returns ``True`` when it seeded, ``False`` when the store was already
    seeded or the source is missing (nothing to seed from).
    """
    source = Path(source)
    target = Path(target)
    marker = target / _MARKER
    if marker.exists():
        logger.info(f"Model store already seeded ({marker}); skipping seed")
        return False

    seed_registry = source / "mlflow.db"
    if not seed_registry.exists():
        # No committed seed to copy (e.g. `make seed` has not run on this
        # clone). Do not create a half-seeded store; leave it for `make seed`.
        logger.warning(f"Committed seed not found at {source}; nothing to seed")
        return False

    logger.info(f"Seeding model store {target} from committed seed {source}")
    (target / "seed").mkdir(parents=True, exist_ok=True)
    _copy_tree(source, target / "seed")
    return True


def _copy_tree(source: Path, target: Path) -> None:
    """Recursively copy ``source`` into ``target`` (files and subdirectories)."""
    for child in source.iterdir():
        dest = target / child.name
        if child.is_dir():
            shutil.copytree(child, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(child, dest)


app = typer.Typer()


@app.command()
def main(
    source: Annotated[Path, typer.Option(help="Committed seed dir (models/seed)")],
    target: Annotated[Path, typer.Option(help="Model store to seed (models, in the container)")],
    require: Annotated[
        bool,
        typer.Option(help="Exit non-zero when the store is unseeded and no committed seed exists"),
    ] = False,
) -> None:
    """Seed the MLflow model store from the committed seed (idempotent).

    With ``--require`` (the container entrypoint), an unseeded store with no
    committed seed to copy fails fast instead of booting an empty registry.
    """
    seeded = seed_mlflow_store(source, target)
    if seeded:
        logger.success(f"Seeded {target} from {source}")
        return
    if require and not (target / _MARKER).exists():
        logger.error(
            "Committed seed not found and the store is empty — run `make seed` "
            "first (or mount models/seed) so the stack has a registry to serve."
        )
        raise typer.Exit(1)
    logger.info(f"Store {target} left as-is")


if __name__ == "__main__":
    app()
