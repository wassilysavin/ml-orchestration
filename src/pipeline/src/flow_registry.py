"""Resolve a versioned training flow back to its concrete model."""

from mlflow.tracking import MlflowClient

from src.config import (
    FLOW_VERSION_TAG,
    MLFLOW_EXPERIMENT_NAME,
)
from src.mlflow_setup import configure_tracking, experiment_name


def _client() -> MlflowClient:
    """Return an MlflowClient pointed at the local store or the remote server."""
    configure_tracking()
    return MlflowClient()


def resolve_flow_version_to_run_id(flow_version_id: str) -> str:
    """Return the most recent model run id produced by the given flow version."""
    client = _client()
    scoped_name = experiment_name(MLFLOW_EXPERIMENT_NAME)
    experiment = client.get_experiment_by_name(scoped_name)
    if experiment is None:
        raise LookupError(f"MLflow experiment {scoped_name!r} not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.{FLOW_VERSION_TAG} = '{flow_version_id}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise LookupError(
            f"No training run tagged {FLOW_VERSION_TAG}={flow_version_id!r}. "
            "Run the training flow with that configuration first."
        )
    return runs[0].info.run_id
