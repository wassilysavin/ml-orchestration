
import json
from pathlib import Path
from typing import Any

from src.config import CHAMPION_FILENAME, MODELS_DIR


def _champion_path(models_dir: Path = MODELS_DIR) -> Path:
    """Resolve the champion-pointer file path under ``models_dir``."""
    return models_dir / CHAMPION_FILENAME


def get_champion(models_dir: Path = MODELS_DIR) -> dict[str, Any] | None:
    """Return the live champion record, or None if none has been set."""
    path = _champion_path(models_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def set_champion(
    flow_version_id: str, run_id: str, models_dir: Path = MODELS_DIR
) -> dict[str, Any]:
    """Promote a model to champion, retaining the prior one (rollback target)."""
    models_dir.mkdir(parents=True, exist_ok=True)
    prior = get_champion(models_dir)
    if prior is not None:
        prior = {**prior, "previous": None}
    record = {"flow_version_id": flow_version_id, "run_id": run_id, "previous": prior}
    _champion_path(models_dir).write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    return record


def rollback(models_dir: Path = MODELS_DIR) -> dict[str, Any] | None:
    """Revert to the previous champion; return the new (reverted) record or None."""
    current = get_champion(models_dir)
    if current is None or current.get("previous") is None:
        return None
    prev = current["previous"]
    return set_champion(prev["flow_version_id"], prev["run_id"], models_dir=models_dir)
