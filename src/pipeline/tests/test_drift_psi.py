"""Monitoring check: PSI behaves as a drift distance and flags real shifts."""
import numpy as np
import pandas as pd
import pytest

from src.config import DRIFT_FEATURE, PSI_SIGNIFICANT_THRESHOLD
from src.drift_checks import assert_no_significant_drift, population_stability_index


@pytest.mark.monitoring
def test_psi_is_zero_for_identical_distributions() -> None:
    """A distribution compared against itself must have ~zero PSI."""
    values = pd.Series(np.linspace(0, 100, 5000))
    assert population_stability_index(values, values) < 1e-9


@pytest.mark.monitoring
def test_psi_grows_when_distribution_shifts() -> None:
    """Shifting the current distribution must increase PSI monotonically."""
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(100, 15, 20000))
    small_shift = pd.Series(rng.normal(105, 15, 20000))
    large_shift = pd.Series(rng.normal(160, 15, 20000))

    psi_small = population_stability_index(reference, small_shift)
    psi_large = population_stability_index(reference, large_shift)

    assert 0 < psi_small < psi_large
    assert psi_large > PSI_SIGNIFICANT_THRESHOLD


@pytest.mark.monitoring
def test_unseen_segment_has_no_significant_drift(prepared_splits) -> None:
    """The unseen (current) split must not significantly drift from the reference."""
    reference = prepared_splits["reference"][DRIFT_FEATURE]
    current = prepared_splits["current"][DRIFT_FEATURE]
    psi = assert_no_significant_drift(reference, current)
    assert psi <= PSI_SIGNIFICANT_THRESHOLD
