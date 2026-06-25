import json

import pytest

from serve import NoChampionError, format_prediction, resolve_champion

from src import promotion


def test_resolve_champion_raises_without_one(tmp_path) -> None:
    """With no champion.json, resolution raises NoChampionError."""
    with pytest.raises(NoChampionError):
        resolve_champion(models_dir=tmp_path)


def test_resolve_champion_returns_promoted(tmp_path) -> None:
    """resolve_champion reads back the pointer set by promotion.set_champion."""
    promotion.set_champion("ver-abc", "run-xyz", models_dir=tmp_path)
    champion = resolve_champion(models_dir=tmp_path)
    assert champion["flow_version_id"] == "ver-abc"
    assert champion["run_id"] == "run-xyz"


def test_resolve_champion_tracks_latest_promotion(tmp_path) -> None:
    """After a second promotion, resolution returns the new champion."""
    promotion.set_champion("ver-1", "run-1", models_dir=tmp_path)
    promotion.set_champion("ver-2", "run-2", models_dir=tmp_path)
    assert resolve_champion(models_dir=tmp_path)["run_id"] == "run-2"


def test_format_prediction_renders_label_and_champion() -> None:
    """The output line carries label, sentiment, probability, and champion ids."""
    champion = {"run_id": "run-xyz", "flow_version_id": "ver-abc"}
    line = format_prediction("great drug", 1, 0.9312, champion)
    assert "label=1 (positive)" in line
    assert "prob=0.9312" in line
    assert "champion=run-xyz" in line
    assert "version=ver-abc" in line


def test_champion_pointer_is_json_on_disk(tmp_path) -> None:
    """The champion pointer resolve reads is the JSON promotion wrote."""
    promotion.set_champion("ver-abc", "run-xyz", models_dir=tmp_path)
    on_disk = json.loads((tmp_path / "champion.json").read_text())
    assert resolve_champion(models_dir=tmp_path)["run_id"] == on_disk["run_id"]
