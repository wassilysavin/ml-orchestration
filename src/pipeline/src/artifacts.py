"""Filesystem artifact passing between orchestrated steps."""

import json
import os
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT


def artifacts_dir() -> Path:
    """Return the shared artifact directory, creating it if needed."""
    root = Path(os.environ.get("PIPELINE_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def artifact_path(name: str) -> Path:
    """Resolve an artifact name (e.g. ``ab/model_ids``) to its JSON path."""
    path = artifacts_dir() / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(name: str, obj: Any) -> Path:
    """Persist ``obj`` as the named JSON artifact; return its path."""
    path = artifact_path(name)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_json(name: str) -> Any:
    """Read back the named JSON artifact written by an upstream step."""
    path = artifact_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing artifact {name!r} at {path}. "
            "Did the upstream step run and share the artifact directory?"
        )
    return json.loads(path.read_text(encoding="utf-8"))
