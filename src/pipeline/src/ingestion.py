import os
from dataclasses import dataclass

from src.config import INCOMING_DATASET_ENV


def incoming_dataset_path() -> str | None:
    """Return the dropped dataset path from the environment, or None if unset."""
    return os.environ.get(INCOMING_DATASET_ENV) or None


@dataclass(frozen=True)
class IngestionPlan:
    """Which raw inputs a prepare step should (re)create this run."""

    need_kaggle_train: bool
    need_kaggle_test: bool
    use_incoming: bool
    rebuild_test_parquet: bool


def plan_ingestion(
    *,
    has_incoming: bool,
    train_exists: bool,
    test_exists: bool,
    test_parquet_exists: bool,
    overwrite: bool,
) -> IngestionPlan:
    """Decide the raw-input actions from what exists and whether a file was dropped."""
    need_kaggle_train = overwrite or not train_exists
    need_kaggle_test = (not has_incoming) and (overwrite or not test_exists)
    rebuild_test_parquet = (
        has_incoming or need_kaggle_test or overwrite or not test_parquet_exists
    )
    return IngestionPlan(
        need_kaggle_train=need_kaggle_train,
        need_kaggle_test=need_kaggle_test,
        use_incoming=has_incoming,
        rebuild_test_parquet=rebuild_test_parquet,
    )
