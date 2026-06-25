import pytest

from src.adaptation_policy import Action, decide, in_cooldown


@pytest.mark.adaptation
def test_band_maps_to_expected_action() -> None:
    """Each PSI band selects its intended response."""
    assert decide("none") is Action.NONE
    assert decide("moderate") is Action.ALERT
    assert decide("significant") is Action.RETRAIN


@pytest.mark.adaptation
def test_unknown_band_is_treated_as_no_action() -> None:
    """An unrecognized band fails safe to NONE rather than retraining."""
    assert decide("") is Action.NONE
    assert decide("weird") is Action.NONE


@pytest.mark.adaptation
def test_no_prior_retrain_is_never_in_cooldown() -> None:
    """With no recorded retrain, a retrain is always allowed."""
    assert in_cooldown(None, now=1000.0, cooldown_seconds=3600) is False


@pytest.mark.adaptation
def test_cooldown_window_boundaries() -> None:
    """A retrain inside the window is suppressed; at/after the window it is allowed."""
    assert in_cooldown(1000.0, now=1000.0 + 1, cooldown_seconds=3600) is True
    assert in_cooldown(1000.0, now=1000.0 + 3600, cooldown_seconds=3600) is False
    assert in_cooldown(1000.0, now=1000.0 + 7200, cooldown_seconds=3600) is False
