
from typing import Mapping


def pick_winner(results: Mapping[str, Mapping[str, float]]) -> str:
    """Return the candidate with the highest macro-F1 (ties broken by name).

    `results` maps each candidate name to a stats dict carrying a ``macro_f1``.
    Sorting the names first makes the tie-break deterministic and order-independent.
    """
    if not results:
        raise ValueError("no candidates to select from")
    return max(sorted(results), key=lambda name: results[name]["macro_f1"])
