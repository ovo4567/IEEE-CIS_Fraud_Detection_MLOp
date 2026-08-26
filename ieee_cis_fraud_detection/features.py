from pathlib import Path

import pandas as pd
from loguru import logger
import typer

from ieee_cis_fraud_detection.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# --------------------------------------------------------------------------- #
# Column definitions (from the competition data description)
# --------------------------------------------------------------------------- #

TRANSACTION_CATEGORICAL_COLS = (
    ["ProductCD", "addr1", "addr2", "P_emaildomain", "R_emaildomain"]
    + [f"card{i}" for i in range(1, 7)]
    + [f"M{i}" for i in range(1, 10)]
)

IDENTITY_CATEGORICAL_COLS = (
    ["DeviceType", "DeviceInfo"] + [f"id_{i:02d}" for i in range(12, 39)]
)

# Drop columns whose missing-value percentage exceeds this threshold.
MISSING_PERCENT_THRESHOLD = 50.0

# Output files (parquet preserves the `category` dtype).
TRANSACTION_FEATURES_PATH = PROCESSED_DATA_DIR / "train_transaction_filtered.parquet"
IDENTITY_FEATURES_PATH = PROCESSED_DATA_DIR / "train_identity_filtered.parquet"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_raw_transaction() -> pd.DataFrame:
    """Load the raw `train_transaction.csv` table."""
    return pd.read_csv(RAW_DATA_DIR / "train_transaction.csv")


def load_raw_identity() -> pd.DataFrame:
    """Load the raw `train_identity.csv` table."""
    return pd.read_csv(RAW_DATA_DIR / "train_identity.csv")


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def filter_columns_by_missingness(
    df: pd.DataFrame, threshold: float = MISSING_PERCENT_THRESHOLD
) -> pd.DataFrame:
    """Drop columns whose missing-value percentage exceeds ``threshold``.

    A column with more than ``threshold`` % missing values is considered
    uninformative and removed (matching the exploration step in the notebook).
    """
    null_percent = (df.isnull().sum() / len(df)) * 100
    cols_to_keep = null_percent[null_percent <= threshold].index
    dropped = df.columns.difference(cols_to_keep).tolist()
    if dropped:
        logger.info(
            f"Dropped {len(dropped)} columns with > {threshold}% missing values"
        )
    return df[cols_to_keep]


def convert_categoricals(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """Convert the given columns to the pandas ``category`` dtype.

    Only columns present in ``df`` are converted, and a copy is returned so the
    caller's frame is not mutated.
    """
    existing = [col for col in categorical_cols if col in df.columns]
    out = df.copy()
    out[existing] = out[existing].astype("category")
    return out


# --------------------------------------------------------------------------- #
# Build pipeline
# --------------------------------------------------------------------------- #


def build_train_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full raw -> processed features pipeline for the training tables.

    Loads the raw CSVs, drops columns with > 50% missing values, converts the
    categorical columns to the ``category`` dtype, and returns the two filtered
    DataFrames (transaction and identity).
    """
    logger.info("Loading raw train_transaction.csv ...")
    transaction = load_raw_transaction()
    logger.info("Loading raw train_identity.csv ...")
    identity = load_raw_identity()

    logger.info("Dropping columns with > 50% missing values ...")
    transaction = filter_columns_by_missingness(transaction)
    identity = filter_columns_by_missingness(identity)

    logger.info("Converting categorical columns to 'category' dtype ...")
    transaction = convert_categoricals(transaction, TRANSACTION_CATEGORICAL_COLS)
    identity = convert_categoricals(identity, IDENTITY_CATEGORICAL_COLS)

    logger.info(
        f"train_transaction: {transaction.shape} "
        f"({transaction.memory_usage().sum() / 1e6:.1f} MB)"
    )
    logger.info(
        f"train_identity:   {identity.shape} "
        f"({identity.memory_usage().sum() / 1e6:.1f} MB)"
    )
    return transaction, identity


def save_train_features() -> None:
    """Build and persist the processed training features as parquet.

    Parquet is used instead of CSV so that the ``category`` dtype (and all
    other dtypes) survive round-trips.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    transaction, identity = build_train_features()

    transaction.to_parquet(TRANSACTION_FEATURES_PATH, index=False)
    identity.to_parquet(IDENTITY_FEATURES_PATH, index=False)
    logger.success(f"Saved: {TRANSACTION_FEATURES_PATH}")
    logger.success(f"Saved: {IDENTITY_FEATURES_PATH}")


@app.command()
def main(
    output_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    """CLI entry point: build and save the processed training features."""
    save_train_features()


if __name__ == "__main__":
    app()
