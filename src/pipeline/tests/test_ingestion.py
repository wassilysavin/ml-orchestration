import pytest

from src.config import INCOMING_DATASET_ENV
from src.ingestion import incoming_dataset_path, plan_ingestion


@pytest.mark.flow_versioning
def test_fresh_pulls_both_splits_from_kaggle() -> None:
    """With nothing on disk and no drop, both splits come from Kaggle."""
    plan = plan_ingestion(
        has_incoming=False,
        train_exists=False,
        test_exists=False,
        test_parquet_exists=False,
        overwrite=False,
    )
    assert plan.need_kaggle_train and plan.need_kaggle_test
    assert not plan.use_incoming
    assert plan.rebuild_test_parquet


@pytest.mark.flow_versioning
def test_drop_replaces_current_but_preserves_reference() -> None:
    """A dropped file becomes current; the existing reference is left untouched."""
    plan = plan_ingestion(
        has_incoming=True,
        train_exists=True,
        test_exists=True,
        test_parquet_exists=True,
        overwrite=False,
    )
    assert not plan.need_kaggle_train
    assert not plan.need_kaggle_test
    assert plan.use_incoming
    assert plan.rebuild_test_parquet


@pytest.mark.flow_versioning
def test_drop_on_empty_disk_still_bootstraps_reference() -> None:
    """A drop with no reference yet still pulls the reference baseline from Kaggle."""
    plan = plan_ingestion(
        has_incoming=True,
        train_exists=False,
        test_exists=False,
        test_parquet_exists=False,
        overwrite=False,
    )
    assert plan.need_kaggle_train
    assert not plan.need_kaggle_test
    assert plan.use_incoming


@pytest.mark.flow_versioning
def test_cached_run_does_nothing() -> None:
    """All splits present, no drop, no overwrite -> no work."""
    plan = plan_ingestion(
        has_incoming=False,
        train_exists=True,
        test_exists=True,
        test_parquet_exists=True,
        overwrite=False,
    )
    assert not any(
        (plan.need_kaggle_train, plan.need_kaggle_test, plan.use_incoming, plan.rebuild_test_parquet)
    )


@pytest.mark.flow_versioning
def test_incoming_dataset_path_reads_env(monkeypatch) -> None:
    """incoming_dataset_path reflects the env var, or None when unset/empty."""
    monkeypatch.delenv(INCOMING_DATASET_ENV, raising=False)
    assert incoming_dataset_path() is None
    monkeypatch.setenv(INCOMING_DATASET_ENV, "/app/data/incoming/new.csv")
    assert incoming_dataset_path() == "/app/data/incoming/new.csv"
