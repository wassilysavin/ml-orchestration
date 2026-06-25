import pytest

from src.train import InsufficientTrainingDataError, train


@pytest.mark.robustness
def test_training_below_min_rows_raises(prepared_splits, monkeypatch) -> None:
    """The training step must refuse to fit when the post-filter set is too small."""
    tiny = prepared_splits["reference"].head(50)
    monkeypatch.setattr("src.train.load_processed_training", lambda: tiny)
    monkeypatch.setattr("src.train.load_processed_current", lambda: tiny)
    with pytest.raises(InsufficientTrainingDataError):
        train()
