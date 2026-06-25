import pandas as pd
import pytest

from src import prepare_dataset


def test_no_incoming_passes_through() -> None:
    """Without a drop, training = reference and current = the full current split."""
    reference = pd.DataFrame({"unique_id": [1, 2], "rating": [9, 1]})
    current = pd.DataFrame({"unique_id": [3, 4], "rating": [10, 2]})
    training, out_current = prepare_dataset._assemble_splits(
        reference, current, has_incoming=False
    )
    assert training.equals(reference)
    assert out_current.equals(current)


def test_incoming_folds_train_and_holds_out_eval(monkeypatch) -> None:
    """A drop folds the train portion into training and holds out the eval portion."""
    monkeypatch.setattr(prepare_dataset, "is_incoming_eval", lambda uid: uid % 2 == 0)
    reference = pd.DataFrame({"unique_id": [1], "rating": [9]})
    incoming = pd.DataFrame({"unique_id": [10, 11, 12, 13], "rating": [9, 1, 10, 2]})

    training, current = prepare_dataset._assemble_splits(
        reference, incoming, has_incoming=True
    )

    assert sorted(current["unique_id"]) == [10, 12]
    assert sorted(training["unique_id"]) == [1, 11, 13]
    assert set(current["unique_id"]).isdisjoint(set(training["unique_id"]))
