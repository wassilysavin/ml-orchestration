"""Data-quality check: condition column distribution must stay close to the reference split."""
import pytest

from src.config import MAX_CONDITION_TVD
from src.quality_checks import assert_condition_distribution


@pytest.mark.data_quality
def test_condition_distribution_matches_reference_split(prepared_splits) -> None:
    """condition should stay close to the historical baseline."""
    reference = prepared_splits["reference"]
    current = prepared_splits["current"]

    distance = assert_condition_distribution(reference["condition"], current["condition"])
    assert distance <= MAX_CONDITION_TVD
