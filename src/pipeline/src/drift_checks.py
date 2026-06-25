
import numpy as np
import pandas as pd

from src.config import PSI_BINS, PSI_SIGNIFICANT_THRESHOLD


def _quantile_bin_edges(reference: pd.Series, bins: int) -> np.ndarray:
    """Return reference-quantile bin edges, de-duplicated and open at both ends."""
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference.astype(float), quantiles))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _binned_shares(values: pd.Series, edges: np.ndarray, epsilon: float) -> np.ndarray:
    """Fraction of ``values`` falling in each bin, floored at ``epsilon``."""
    counts, _ = np.histogram(values.astype(float), bins=edges)
    shares = counts / max(counts.sum(), 1)
    return np.clip(shares, epsilon, None)


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = PSI_BINS,
    epsilon: float = 1e-6,
) -> float:
    """Return the PSI between a reference and a current distribution."""
    edges = _quantile_bin_edges(reference, bins)
    expected = _binned_shares(reference, edges, epsilon)
    actual = _binned_shares(current, edges, epsilon)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def assert_no_significant_drift(
    reference: pd.Series,
    current: pd.Series,
    threshold: float = PSI_SIGNIFICANT_THRESHOLD,
    bins: int = PSI_BINS,
) -> float:
    """Return the PSI and raise if it exceeds the significant-drift threshold."""
    psi = population_stability_index(reference, current, bins=bins)
    if psi > threshold:
        raise AssertionError(
            f"Population Stability Index {psi:.6f} exceeds the significant-drift "
            f"threshold {threshold:.6f}."
        )
    return psi
