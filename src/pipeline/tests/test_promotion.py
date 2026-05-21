"""Champion-pointer checks: promote, retain a rollback target, and revert."""
import pytest

from src import promotion


@pytest.mark.adaptation
def test_no_champion_initially(tmp_path) -> None:
    """A fresh store has no champion."""
    assert promotion.get_champion(models_dir=tmp_path) is None


@pytest.mark.adaptation
def test_set_then_get_roundtrips(tmp_path) -> None:
    """Setting a champion makes it the live record (with no prior to roll back to)."""
    promotion.set_champion("v1", "run-1", models_dir=tmp_path)
    champ = promotion.get_champion(models_dir=tmp_path)
    assert champ["flow_version_id"] == "v1"
    assert champ["run_id"] == "run-1"
    assert champ["previous"] is None


@pytest.mark.adaptation
def test_promotion_retains_previous_as_rollback_target(tmp_path) -> None:
    """Promoting a second model keeps the first as the immediate rollback target."""
    promotion.set_champion("v1", "run-1", models_dir=tmp_path)
    promotion.set_champion("v2", "run-2", models_dir=tmp_path)
    champ = promotion.get_champion(models_dir=tmp_path)
    assert champ["flow_version_id"] == "v2"
    assert champ["previous"]["flow_version_id"] == "v1"


@pytest.mark.adaptation
def test_previous_is_only_one_level_deep(tmp_path) -> None:
    """The store keeps a single rollback target, not an unbounded history chain."""
    promotion.set_champion("v1", "run-1", models_dir=tmp_path)
    promotion.set_champion("v2", "run-2", models_dir=tmp_path)
    promotion.set_champion("v3", "run-3", models_dir=tmp_path)
    champ = promotion.get_champion(models_dir=tmp_path)
    assert champ["previous"]["flow_version_id"] == "v2"
    assert champ["previous"]["previous"] is None


@pytest.mark.adaptation
def test_rollback_reverts_to_previous(tmp_path) -> None:
    """Rollback makes the previous champion live again."""
    promotion.set_champion("v1", "run-1", models_dir=tmp_path)
    promotion.set_champion("v2", "run-2", models_dir=tmp_path)
    reverted = promotion.rollback(models_dir=tmp_path)
    assert reverted["flow_version_id"] == "v1"
    assert promotion.get_champion(models_dir=tmp_path)["flow_version_id"] == "v1"


@pytest.mark.adaptation
def test_rollback_without_history_is_a_noop(tmp_path) -> None:
    """Rollback with no prior champion returns None and leaves state unchanged."""
    promotion.set_champion("v1", "run-1", models_dir=tmp_path)
    assert promotion.rollback(models_dir=tmp_path) is None
    assert promotion.get_champion(models_dir=tmp_path)["flow_version_id"] == "v1"
