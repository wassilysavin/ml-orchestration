import pytest

from src.splits import is_incoming_eval


@pytest.mark.flow_versioning
def test_assignment_is_deterministic() -> None:
    """The same id always lands on the same side of the split."""
    assert is_incoming_eval(12345) == is_incoming_eval(12345)


@pytest.mark.flow_versioning
def test_fraction_is_approximately_honoured() -> None:
    """Roughly `eval_fraction` of ids are held out as eval over a large sample."""
    n = 5000
    held_out = sum(is_incoming_eval(i, eval_fraction=0.3) for i in range(n))
    assert 0.25 * n < held_out < 0.35 * n


@pytest.mark.flow_versioning
def test_boundary_fractions() -> None:
    """fraction 0.0 holds out nothing; 1.0 holds out everything."""
    assert not any(is_incoming_eval(i, eval_fraction=0.0) for i in range(200))
    assert all(is_incoming_eval(i, eval_fraction=1.0) for i in range(200))


@pytest.mark.flow_versioning
def test_salt_changes_the_partition() -> None:
    """A different salt reshuffles which ids are held out."""
    a = [is_incoming_eval(i, salt="salt-a") for i in range(500)]
    b = [is_incoming_eval(i, salt="salt-b") for i in range(500)]
    assert a != b
