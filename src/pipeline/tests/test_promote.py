import pytest

from src.config import PROMOTION_MIN_MACRO_F1_MARGIN
from promote import should_promote


@pytest.mark.flow_versioning
def test_promotes_on_sufficient_margin() -> None:
    """A margin at or above the threshold promotes."""
    assert should_promote(PROMOTION_MIN_MACRO_F1_MARGIN)
    assert should_promote(PROMOTION_MIN_MACRO_F1_MARGIN + 0.1)


@pytest.mark.flow_versioning
def test_keeps_champion_below_margin() -> None:
    """A margin below the threshold (or negative) keeps the champion."""
    assert not should_promote(PROMOTION_MIN_MACRO_F1_MARGIN - 1e-6)
    assert not should_promote(-0.05)
