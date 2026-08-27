from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --------------------------------------------------------------------------- #
# MLflow registries
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# MLflow registries
# --------------------------------------------------------------------------- #

MLRUNS_DIR = PROJ_ROOT / "mlruns"
SEED_REGISTRY_DIR = MODELS_DIR / "seed"
SEED_MODEL_PATH = SEED_REGISTRY_DIR / "champion_model"

# --------------------------------------------------------------------------- #
# Monitoring (drift current-window store + Evidently reports)
# --------------------------------------------------------------------------- #

# Every batch-scored transaction accumulates here (TransactionID, score,
# decision); ticket 08's drift monitor time-slices this store into the "current"
# window and compares it to the training reference. Lives under data/ (DVC-
# tracked, gitignored) because it is runtime data, not a committed artifact.
MONITORING_DIR = DATA_DIR / "monitoring"
DRIFT_STORE_PATH = MONITORING_DIR / "current_window.csv"

# The Evidently drift report artifacts the monitoring flow writes (ticket 08):
# the current window vs the training reference, feature + score drift. HTML is
# the human-readable report; JSON is the machine-readable snapshot.
DRIFT_REPORTS_DIR = MONITORING_DIR / "reports"
DRIFT_REPORT_PATH = DRIFT_REPORTS_DIR / "latest_drift_report.html"
DRIFT_REPORT_JSON_PATH = DRIFT_REPORTS_DIR / "latest_drift_report.json"

# --------------------------------------------------------------------------- #
# Retraining flow + served model (ticket 07)
# --------------------------------------------------------------------------- #

# A promoted challenger is published here so the serving surfaces pick up the
# update; it is runtime data (gitignored), distinct from the committed seed.
SERVING_MODEL_PATH = MODELS_DIR / "serving" / "champion_model"

# Retrain bookkeeping: the drift-store row count at the last retrain.
RETRAIN_STATE_PATH = MONITORING_DIR / "retrain_state.json"


def tracking_uri_for(db_dir: Path) -> str:
    """Local SQLite MLflow tracking URI for a registry directory."""
    return f"sqlite:///{db_dir / 'mlflow.db'}"


# Legacy notebook store (read-only). It holds the `finetuned_lgbm` recipe the
# seed pipeline re-fits; its artifacts are split-brain archaeology and are
# never reused (see .scratch/mlops-deployment/spec.md).
LEGACY_TRACKING_URI = tracking_uri_for(MLRUNS_DIR)

# Clean deployment registry (ticket 02): the committed seed store, distinct
# from the legacy notebook store. The seed pyfunc is saved to SEED_MODEL_PATH
# so `make demo` can serve it offline on a fresh clone.
SEED_TRACKING_URI = tracking_uri_for(SEED_REGISTRY_DIR)

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
