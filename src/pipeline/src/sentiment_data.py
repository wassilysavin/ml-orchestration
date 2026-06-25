import pandas as pd

from src.config import (
    NEGATIVE_LABEL,
    NEGATIVE_RATING_CEILING,
    POSITIVE_LABEL,
    POSITIVE_RATING_FLOOR,
)


def filter_for_binary_sentiment(
    dataframe: pd.DataFrame,
    positive_floor: int = POSITIVE_RATING_FLOOR,
    negative_ceiling: int = NEGATIVE_RATING_CEILING,
) -> pd.DataFrame:
    """Drop ambiguous middle ratings and add a binary sentiment label."""
    rating = dataframe["rating"]
    mask = (rating >= positive_floor) | (rating <= negative_ceiling)
    filtered = dataframe.loc[mask].copy()
    filtered["sentiment"] = (
        (filtered["rating"] >= positive_floor)
        .map({True: POSITIVE_LABEL, False: NEGATIVE_LABEL})
        .astype("int8")
    )
    return filtered


def build_xy(dataframe: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the input text series and the integer sentiment label."""
    return dataframe["review_text"].astype("string"), dataframe["sentiment"]
