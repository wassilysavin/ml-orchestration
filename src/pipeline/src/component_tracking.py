import os
from typing import Any, Mapping

import mlflow

from src.config import (
    COMPONENT_NAME_TAG,
    COMPONENT_VERSION_TAG,
    COMPONENTS_EXPERIMENT_NAME,
)
from src.mlflow_setup import configure_tracking, current_flow_run_id, experiment_tags

TRACK_COMPONENTS_ENV = "TRACK_COMPONENTS"


def should_track() -> bool:
    """Track when inside an orchestrated flow run, or when explicitly opted in."""
    return bool(current_flow_run_id() or os.environ.get(TRACK_COMPONENTS_ENV))


def log_component(
    component: str,
    version_id: str,
    params: Mapping[str, Any],
    metrics: Mapping[str, float] | None = None,
    tags: Mapping[str, str] | None = None,
) -> str:
    """Record one component version as an MLflow run; return its run id."""
    configure_tracking(COMPONENTS_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{component}:{version_id}") as run:
        mlflow.set_tags(
            {
                COMPONENT_NAME_TAG: component,
                COMPONENT_VERSION_TAG: version_id,
                **experiment_tags(),
                **(dict(tags) if tags else {}),
            }
        )
        mlflow.log_params(dict(params))
        if metrics:
            mlflow.log_metrics(dict(metrics))
        return run.info.run_id
