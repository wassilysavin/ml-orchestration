import pytest

from src.flow_config import CANDIDATE_FLOW_CONFIGS
from src.selection import pick_winner


@pytest.mark.selection
def test_pick_winner_returns_highest_macro_f1() -> None:
    """The selection step picks the candidate with the best shared-eval macro-F1."""
    results = {
        "baseline": {"macro_f1": 0.70},
        "challenger": {"macro_f1": 0.81},
        "c03": {"macro_f1": 0.66},
    }
    assert pick_winner(results) == "challenger"


@pytest.mark.selection
def test_pick_winner_breaks_ties_deterministically_by_name() -> None:
    """Equal macro-F1 resolves to the first candidate by name, regardless of order."""
    forward = {"baseline": {"macro_f1": 0.5}, "challenger": {"macro_f1": 0.5}}
    reversed_order = {"challenger": {"macro_f1": 0.5}, "baseline": {"macro_f1": 0.5}}
    assert pick_winner(forward) == "baseline"
    assert pick_winner(reversed_order) == "baseline"


@pytest.mark.selection
def test_pick_winner_rejects_empty_results() -> None:
    """Selecting from no candidates is a programming error, not a silent default."""
    with pytest.raises(ValueError):
        pick_winner({})


@pytest.mark.selection
def test_candidate_configs_have_distinct_versions() -> None:
    """Each candidate must hash to a distinct flow_version_id so models don't collide."""
    versions = {name: cfg.flow_version_id() for name, cfg in CANDIDATE_FLOW_CONFIGS.items()}
    assert len(set(versions.values())) == len(versions)
