"""Data-quality check: the condition column must remain ≤1% null in the current split."""
import pytest

from src.config import MAX_CONDITION_MISSINGNESS
from src.quality_checks import build_condition_missingness_schema, condition_missing_rate


@pytest.mark.data_quality
def test_condition_missingness_stays_below_one_percent(prepared_splits) -> None:
    """The current batch should preserve the historically low missingness in condition."""
    current = prepared_splits["current"]

    schema = build_condition_missingness_schema()
    schema.validate(current[["condition"]])

    missing_rate = condition_missing_rate(current)
    assert missing_rate <= MAX_CONDITION_MISSINGNESS
