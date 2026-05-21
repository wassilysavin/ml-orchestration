"""Pytest fixtures and conditional collection rules for the pipeline test suite."""
import importlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prepare_dataset import ensure_processed_datasets
from src.utils import load_processed_current, load_processed_reference

collect_ignore_glob: list[str] = []
for _mod, _files in (
    ("sklearn", ["test_baseline_margin.py", "test_invariance.py", "test_negation.py"]),
    ("mlflow", ["test_min_training_rows.py"]),
    ("onnxruntime", []),
):
    if importlib.util.find_spec(_mod) is None:
        collect_ignore_glob.extend(_files)


@pytest.fixture(scope="session")
def prepared_splits():
    """Session fixture: ensure processed splits exist and return them as a dict."""
    ensure_processed_datasets()
    return {
        "reference": load_processed_reference(),
        "current": load_processed_current(),
    }


@pytest.fixture(scope="session")
def trained_model():
    """Session fixture: ensure a trained model exists and return its run id + pipeline."""
    import os

    from src.registry import load_model
    from src.train import ensure_trained_model

    ensure_processed_datasets()
    run_id = os.environ.get("PIPELINE_MODEL_RUN_ID") or ensure_trained_model()
    return {"run_id": run_id, "pipeline": load_model(run_id)}
