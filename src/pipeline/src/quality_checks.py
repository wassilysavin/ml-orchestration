"""Data-quality primitives: missingness, KS, TVD, and threshold-enforcing assertions."""
import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema
from scipy.stats import ks_2samp

from src.config import (
    MAX_CONDITION_MISSINGNESS,
    MAX_CONDITION_TVD,
    MAX_RATING_KS_STATISTIC,
)

MISSING_CATEGORY_LABEL = "__MISSING__"


def condition_missing_rate(dataframe: pd.DataFrame) -> float:
    """Return the null rate for the condition column."""
    return float(dataframe["condition"].isna().mean())


def build_condition_missingness_schema(
    max_missing_rate: float = MAX_CONDITION_MISSINGNESS,
) -> DataFrameSchema:
    """Create a Pandera schema enforcing the allowed null rate for condition."""
    return DataFrameSchema(
        columns={"condition": Column(str, nullable=True, coerce=True)},
        checks=[
            Check(
                lambda df: condition_missing_rate(df) <= max_missing_rate,
                error=(
                    "The condition column exceeds the allowed missingness threshold "
                    f"of {max_missing_rate:.2%}."
                ),
            )
        ],
    )


def rating_ks_statistic(reference: pd.Series, current: pd.Series) -> float:
    """Compute the Kolmogorov-Smirnov statistic for the numeric `rating` column."""
    statistic, _ = ks_2samp(reference.astype(float), current.astype(float))
    return float(statistic)


def categorical_total_variation_distance(
    reference: pd.Series,
    current: pd.Series,
    missing_label: str = MISSING_CATEGORY_LABEL,
) -> float:
    """Compute total variation distance between two categorical distributions."""
    reference_values = reference.astype("object").where(reference.notna(), missing_label)
    current_values = current.astype("object").where(current.notna(), missing_label)

    reference_distribution = reference_values.value_counts(normalize=True)
    current_distribution = current_values.value_counts(normalize=True)

    full_index = reference_distribution.index.union(current_distribution.index)
    reference_distribution = reference_distribution.reindex(full_index, fill_value=0.0)
    current_distribution = current_distribution.reindex(full_index, fill_value=0.0)

    return float(0.5 * (reference_distribution - current_distribution).abs().sum())


def assert_rating_distribution(
    reference: pd.Series,
    current: pd.Series,
    threshold: float = MAX_RATING_KS_STATISTIC,
) -> float:
    """Return the KS statistic and raise if it exceeds the configured limit."""
    statistic = rating_ks_statistic(reference, current)
    if statistic > threshold:
        raise AssertionError(
            f"Rating KS statistic {statistic:.6f} exceeds the threshold {threshold:.6f}."
        )
    return statistic


def assert_condition_distribution(
    reference: pd.Series,
    current: pd.Series,
    threshold: float = MAX_CONDITION_TVD,
) -> float:
    """Return the TVD and raise if it exceeds the configured limit."""
    distance = categorical_total_variation_distance(reference, current)
    if distance > threshold:
        raise AssertionError(
            f"Condition TVD {distance:.6f} exceeds the threshold {threshold:.6f}."
        )
    return distance
