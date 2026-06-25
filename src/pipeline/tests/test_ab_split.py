import pytest

from src.ab_split import assign_variant
from src.config import AB_VARIANT_A, AB_VARIANT_B

_AB_TEST_ID = "unit-test-ab"
_IDS = range(20000)


@pytest.mark.abtest
def test_assignment_is_reproducible() -> None:
    """The same id + test must always route to the same variant."""
    assignments = [assign_variant(i, _AB_TEST_ID) for i in _IDS]
    again = [assign_variant(i, _AB_TEST_ID) for i in _IDS]
    assert assignments == again


@pytest.mark.abtest
def test_split_is_roughly_even() -> None:
    """A good hash must split the population close to 50/50."""
    share_a = sum(
        assign_variant(i, _AB_TEST_ID) == AB_VARIANT_A for i in _IDS
    ) / len(_IDS)
    assert 0.47 <= share_a <= 0.53


@pytest.mark.abtest
def test_only_two_variants_are_produced() -> None:
    """Routing must only ever yield variant A or B."""
    seen = {assign_variant(i, _AB_TEST_ID) for i in _IDS}
    assert seen <= {AB_VARIANT_A, AB_VARIANT_B}


@pytest.mark.abtest
def test_different_test_ids_reroute_some_users() -> None:
    """Salting by ab_test_id must reassign a meaningful fraction of ids."""
    moved = sum(
        assign_variant(i, "test-one") != assign_variant(i, "test-two") for i in _IDS
    ) / len(_IDS)
    assert 0.4 <= moved <= 0.6
