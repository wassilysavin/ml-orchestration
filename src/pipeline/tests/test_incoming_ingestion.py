import pandas as pd
import pytest

from src import config
from src.config import INCOMING_DATASET_ENV


@pytest.fixture()
def fake_raw_dirs(tmp_path, monkeypatch):
    """Redirect raw-file paths to a tmp dir and stub the Kaggle download."""
    raw = tmp_path / "raw"
    raw.mkdir()
    train = raw / "train.csv"
    test = raw / "test.csv"
    test_parquet = raw / "test.parquet"

    import src.download_data as dl

    monkeypatch.setattr(config, "TRAIN_RAW_FILE", train, raising=False)
    monkeypatch.setattr(config, "TEST_RAW_FILE", test, raising=False)
    monkeypatch.setattr(dl, "TRAIN_RAW_FILE", train)
    monkeypatch.setattr(dl, "TEST_RAW_FILE", test)
    monkeypatch.setattr(dl, "TEST_RAW_PARQUET_FILE", test_parquet)
    monkeypatch.setattr(dl, "ensure_directories", lambda: None)
    monkeypatch.setattr(dl, "_maybe_track_dataset_pull", lambda: None)

    kaggle = tmp_path / "kaggle"
    kaggle.mkdir()
    pd.DataFrame({"rating": [1, 2]}).to_csv(kaggle / config.KAGGLE_TRAIN_FILENAME, index=False)
    pd.DataFrame({"rating": [9, 10]}).to_csv(kaggle / config.KAGGLE_TEST_FILENAME, index=False)
    monkeypatch.setattr(dl, "_kaggle_download_dir", lambda: kaggle)

    monkeypatch.delenv(INCOMING_DATASET_ENV, raising=False)
    return {"train": train, "test": test, "test_parquet": test_parquet, "tmp": tmp_path}


def test_incoming_overrides_current_keeps_reference(fake_raw_dirs, monkeypatch) -> None:
    """A dropped file becomes the current split; the reference stays the Kaggle one."""
    import src.download_data as dl

    dl.prepare_raw_dataset()
    reference_before = fake_raw_dirs["train"].read_text()
    assert pd.read_csv(fake_raw_dirs["test"])["rating"].tolist() == [9, 10]

    dropped = fake_raw_dirs["tmp"] / "incoming.csv"
    pd.DataFrame({"rating": [3, 4, 5]}).to_csv(dropped, index=False)
    monkeypatch.setenv(INCOMING_DATASET_ENV, str(dropped))

    dl.prepare_raw_dataset()

    assert pd.read_csv(fake_raw_dirs["test"])["rating"].tolist() == [3, 4, 5]
    assert fake_raw_dirs["train"].read_text() == reference_before
    assert pd.read_parquet(fake_raw_dirs["test_parquet"])["rating"].tolist() == [3, 4, 5]
