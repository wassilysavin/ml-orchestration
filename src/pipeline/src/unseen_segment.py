
import pandas as pd

from src.utils import load_processed_current, load_processed_reference


def load_unseen_segment() -> pd.DataFrame:
    """Return the never-trained-on segment (the current/test split)."""
    return load_processed_current()


def load_reference_segment() -> pd.DataFrame:
    """Return the training/reference split used as the drift baseline."""
    return load_processed_reference()
