from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src.config import (
    MIN_INVARIANCE_AGREEMENT,
    MIN_MACRO_F1_MARGIN_OVER_BASELINE,
    MIN_NEGATION_PROBABILITY_SHIFT,
    NEGATION_PREFIXES,
)

INVARIANCE_TRANSFORMS = ("whitespace", "trailing_period", "lowercase", "char_swap")


def majority_class_macro_f1(y_true: pd.Series) -> float:
    """Macro-F1 of a model that always predicts the most common class."""
    majority = int(pd.Series(y_true).mode().iloc[0])
    y_pred = np.full(len(y_true), majority)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def macro_f1(model: Pipeline, X: pd.Series, y_true: pd.Series) -> float:
    """Compute macro-F1 of the model on the provided inputs."""
    return float(f1_score(y_true, model.predict(X), average="macro", zero_division=0))


def assert_baseline_margin(
    model: Pipeline,
    X_eval: pd.Series,
    y_eval: pd.Series,
    threshold: float = MIN_MACRO_F1_MARGIN_OVER_BASELINE,
) -> float:
    """Return the macro-F1 margin and raise if it is below the floor."""
    model_score = macro_f1(model, X_eval, y_eval)
    baseline = majority_class_macro_f1(y_eval)
    margin = model_score - baseline
    if margin < threshold:
        raise AssertionError(
            f"Macro-F1 margin {margin:.4f} (model {model_score:.4f} - "
            f"baseline {baseline:.4f}) is below the threshold {threshold:.4f}."
        )
    return margin


def _apply_transform(text: str, transform: str, rng: np.random.Generator) -> str:
    """Apply one of the named benign text transforms used by the invariance check."""
    if transform == "whitespace":
        return " ".join(text.split())
    if transform == "trailing_period":
        return text.rstrip(".") + "."
    if transform == "lowercase":
        return text.lower()
    if transform == "char_swap":
        chars = list(text)
        if len(chars) < 2:
            return text
        for _ in range(max(1, int(0.01 * len(chars)))):
            i = int(rng.integers(0, len(chars) - 1))
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    raise ValueError(f"Unknown transform: {transform}")


def assert_negation_shift(
    model: Pipeline,
    confident_samples: pd.Series,
    confident_predicted_class: np.ndarray,
    threshold: float = MIN_NEGATION_PROBABILITY_SHIFT,
    prefixes: Iterable[str] = NEGATION_PREFIXES,
) -> float:
    """Return the mean original-class probability drop after prepending negation."""
    prefix_list = list(prefixes)
    base_texts = confident_samples.reset_index(drop=True)
    prefixed = pd.Series(
        [
            prefix_list[i % len(prefix_list)] + base_texts.iloc[i]
            for i in range(len(base_texts))
        ],
        dtype="string",
    )

    original_probs = model.predict_proba(base_texts)
    perturbed_probs = model.predict_proba(prefixed)
    rows = np.arange(len(base_texts))
    drops = (
        original_probs[rows, confident_predicted_class]
        - perturbed_probs[rows, confident_predicted_class]
    )
    mean_drop = float(np.mean(drops))
    if mean_drop < threshold:
        raise AssertionError(
            f"Mean negation probability drop {mean_drop:.4f} "
            f"is below the threshold {threshold:.4f}."
        )
    return mean_drop
