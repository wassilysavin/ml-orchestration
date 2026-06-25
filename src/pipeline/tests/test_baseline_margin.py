import pytest

from src.model_checks import assert_baseline_margin
from src.sentiment_data import build_xy, filter_for_binary_sentiment


@pytest.mark.robustness
def test_macro_f1_beats_majority_baseline_by_margin(prepared_splits, trained_model) -> None:
    """The model must beat the majority-class baseline by the configured margin."""
    current = filter_for_binary_sentiment(prepared_splits["current"])
    if len(current) > 3000:
        current = current.sample(n=3000, random_state=0).reset_index(drop=True)
    X_eval, y_eval = build_xy(current)
    margin = assert_baseline_margin(trained_model["pipeline"], X_eval, y_eval)
    assert margin >= 0
