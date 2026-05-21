"""Robustness check: prepending negation phrases must lower the predicted-class probability."""
import numpy as np
import pandas as pd
import pytest

from src.config import NEGATION_SAMPLE_SIZE
from src.model_checks import assert_negation_shift
from src.sentiment_data import filter_for_binary_sentiment


@pytest.mark.robustness
def test_negation_prefix_pulls_probability_toward_the_other_class(
    prepared_splits, trained_model
) -> None:
    """Prepending negation phrases must reduce the original-class probability."""
    pipeline = trained_model["pipeline"]
    current = filter_for_binary_sentiment(prepared_splits["current"])
    texts = current["review_text"].astype("string").reset_index(drop=True)

    confident_texts_parts: list[pd.Series] = []
    confident_classes_parts: list[np.ndarray] = []
    collected = 0
    for start in range(0, len(texts), 2000):
        if collected >= NEGATION_SAMPLE_SIZE:
            break
        chunk = texts.iloc[start:start + 2000]
        probs = pipeline.predict_proba(chunk)
        pred = probs.argmax(axis=1)
        mask = probs.max(axis=1) > 0.9
        confident_texts_parts.append(chunk[mask])
        confident_classes_parts.append(pred[mask])
        collected += int(mask.sum())

    confident_texts = pd.concat(confident_texts_parts).head(NEGATION_SAMPLE_SIZE)
    confident_classes = np.concatenate(confident_classes_parts)[:NEGATION_SAMPLE_SIZE]

    if len(confident_texts) < NEGATION_SAMPLE_SIZE:
        pytest.skip(
            f"Only {len(confident_texts)} confident reviews available, "
            f"need {NEGATION_SAMPLE_SIZE}."
        )

    drop = assert_negation_shift(pipeline, confident_texts, confident_classes)
    assert drop >= 0
