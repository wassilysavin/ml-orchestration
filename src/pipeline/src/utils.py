import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    COMBINED_PROCESSED_FILE,
    CURRENT_PROCESSED_FILE,
    PROCESSED_DIR,
    RAW_DIR,
    REFERENCE_PROCESSED_FILE,
    SUMMARY_FILE,
    TEST_RAW_FILE,
    TRAIN_RAW_FILE,
    TRAINING_PROCESSED_FILE,
)


def ensure_directories() -> None:
    """Create deterministic data directories if they do not already exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file with pandas."""
    return pd.read_csv(path)


def write_csv(dataframe: pd.DataFrame, path: Path) -> None:
    """Serialize a dataframe as CSV without an index."""
    dataframe.to_csv(path, index=False)


def write_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    """Serialize a dataframe as Parquet without an index."""
    dataframe.to_parquet(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Write a JSON file with stable formatting."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_processed_reference() -> pd.DataFrame:
    """Load the frozen baseline split (drift baseline + training seed)."""
    return pd.read_parquet(REFERENCE_PROCESSED_FILE)


def load_processed_current() -> pd.DataFrame:
    """Load the current/evaluation split (unseen segment)."""
    return pd.read_parquet(CURRENT_PROCESSED_FILE)


def load_processed_training() -> pd.DataFrame:
    """Load the set the model fits: reference plus any folded-in incoming data."""
    return pd.read_parquet(TRAINING_PROCESSED_FILE)


def processed_outputs_exist() -> bool:
    """Return True when all processed artifacts exist."""
    required_paths = (
        COMBINED_PROCESSED_FILE,
        CURRENT_PROCESSED_FILE,
        REFERENCE_PROCESSED_FILE,
        TRAINING_PROCESSED_FILE,
        SUMMARY_FILE,
    )
    return all(path.exists() for path in required_paths)


def raw_outputs_exist() -> bool:
    """Return True when canonical raw artifacts exist."""
    required_paths = (TRAIN_RAW_FILE, TEST_RAW_FILE)
    return all(path.exists() for path in required_paths)
