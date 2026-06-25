import pytest

from src.config import MAX_RATING_KS_STATISTIC
from src.quality_checks import assert_rating_distribution


@pytest.mark.data_quality
def test_rating_distribution_matches_reference_split(prepared_splits) -> None:
    """rating should remain close to the historical train split."""
    reference = prepared_splits["reference"]
    current = prepared_splits["current"]

    statistic = assert_rating_distribution(reference["rating"], current["rating"])
    assert statistic <= MAX_RATING_KS_STATISTIC
