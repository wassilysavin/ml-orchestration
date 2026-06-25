
import os
import uuid

import mlflow

from src.config import MLRUNS_DIR

EXPERIMENT_ID_ENV = "EXPERIMENT_ID"
EXPERIMENTS_STORE_DIRNAME = "mlruns_experiments"
FLOW_RUN_ID_ENV = "FLOW_RUN_ID"


def current_experiment_id() -> str | None:
    """Return the active experiment id from the environment, or None."""
    return os.environ.get(EXPERIMENT_ID_ENV) or None


def current_flow_run_id() -> str | None:
    """Return the active flow run id from the environment, or None."""
    return os.environ.get(FLOW_RUN_ID_ENV) or None


def ensure_flow_run_id() -> str:
    """Return the active flow run id, minting and exporting one if unset."""
    existing = current_flow_run_id()
    if existing:
        return existing
    new_id = uuid.uuid4().hex[:12]
    os.environ[FLOW_RUN_ID_ENV] = new_id
    return new_id


def experiment_name(base_name: str) -> str:
    """Namespace a base MLflow experiment name by the active experiment id."""
    eid = current_experiment_id()
    return f"{eid}:{base_name}" if eid else base_name


def experiment_tags() -> dict[str, str]:
    """Tags stamped on every run: its experiment id and its flow run id (when set)."""
    tags: dict[str, str] = {}
    eid = current_experiment_id()
    if eid:
        tags["experiment_id"] = eid
    fid = current_flow_run_id()
    if fid:
        tags["flow_run_id"] = fid
    return tags


def using_remote_tracking() -> bool:
    """True when an MLflow tracking server URI is configured in the environment."""
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def _local_tracking_uri() -> str:
    """Local file store: per-experiment when an experiment id is active, else default."""
    eid = current_experiment_id()
    if eid:
        store = MLRUNS_DIR.parent / EXPERIMENTS_STORE_DIRNAME / eid
        store.mkdir(parents=True, exist_ok=True)
        return f"file:{store}"
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    return f"file:{MLRUNS_DIR}"


def configure_tracking(base_experiment_name: str | None = None) -> None:
    """Point MLflow at the right store and select the namespaced experiment."""
    if not using_remote_tracking():
        mlflow.set_tracking_uri(_local_tracking_uri())
    if base_experiment_name:
        mlflow.set_experiment(experiment_name(base_experiment_name))
