from pathlib import Path
from orchestration import Flow, Step, Resources


MLOPS_ROOT = (Path(__file__).resolve().parent.parent / "src" / "pipeline").resolve()

for sub in ("data", "models", "mlruns", "artifacts"):
    (MLOPS_ROOT / sub).mkdir(parents=True, exist_ok=True)

SHARED_VOLUMES: dict[str, str] = {
    str(MLOPS_ROOT / "data"):      "/app/data",
    str(MLOPS_ROOT / "models"):    "/app/models",
    str(MLOPS_ROOT / "mlruns"):    "/app/mlruns",
    str(MLOPS_ROOT / "artifacts"): "/app/artifacts",
}

AB_TEST_ID = "drug-sentiment-ab-001"


class MeasureDrift(Step):
    """Compute PSI drift on review_length; persist the measurement artifact."""

    image = "mlops-train:latest"
    command = ["python", "monitoring_flow.py", "measure"]
    workdir = "/app"
    volumes = SHARED_VOLUMES
    resources = Resources(cpu=2, memory_gb=2)


class EvaluateDrift(Step):
    """Read the measurement, compare to the PSI band, and record the verdict to MLflow."""

    image = "mlops-train:latest"
    command = ["python", "monitoring_flow.py", "evaluate", "--gate"]
    workdir = "/app"
    volumes = SHARED_VOLUMES
    resources = Resources(cpu=2, memory_gb=2)


monitoring_flow = Flow("drug-review-monitoring")
_measure = monitoring_flow.add(MeasureDrift())
monitoring_flow.add(EvaluateDrift(), after=_measure)


class DataPrep(Step):
    """Download the raw dataset and build the prepared parquet under /app/data."""

    image = "mlops-data:latest"
    command = None
    workdir = "/app"
    volumes = SHARED_VOLUMES
    resources = Resources(cpu=2, memory_gb=2)


class TrainVersioned(Step):
    """Train the challenger flow version; persist its run id for robustness."""

    image = "mlops-train:latest"
    command = ["python", "training_flow.py", "--variant", "challenger", "--step", "train"]
    workdir = "/app"
    volumes = SHARED_VOLUMES
    resources = Resources(cpu=2, memory_gb=5)


class RobustnessVersioned(Step):
    """Run the robustness suite pinned to the run id the train step produced."""

    image = "mlops-train:latest"
    command = ["python", "training_flow.py", "--step", "robustness"]
    workdir = "/app"
    volumes = SHARED_VOLUMES
    resources = Resources(cpu=2, memory_gb=5)


versioned_training_flow = Flow("drug-review-training-versioned")
_prep = versioned_training_flow.add(DataPrep())
_train = versioned_training_flow.add(TrainVersioned(), after=_prep)
versioned_training_flow.add(RobustnessVersioned(), after=_train)


class ResolveVersions(Step):
    """Resolve the two flow_version_ids to model run ids; persist them."""

    image = "mlops-train:latest"
    command = ["python", "ab_flow.py", "resolve"]
    workdir = "/app"
    volumes = SHARED_VOLUMES
    resources = Resources(cpu=1, memory_gb=2)


class PredictA(Step):
    """Fork branch A: score the baseline model on its routed half of the segment."""

    image = "mlops-train:latest"
    command = ["python", "ab_flow.py", "predict", "--variant", "A", "--ab-test-id", AB_TEST_ID]
    workdir = "/app"
    volumes = SHARED_VOLUMES
    resources = Resources(cpu=2, memory_gb=4)


class PredictB(Step):
    """Fork branch B: score the challenger model on its routed half of the segment."""

    image = "mlops-train:latest"
    command = ["python", "ab_flow.py", "predict", "--variant", "B", "--ab-test-id", AB_TEST_ID]
    workdir = "/app"
    volumes = SHARED_VOLUMES
    resources = Resources(cpu=2, memory_gb=4)


class MeasureAB(Step):
    """Join both branches' stats and log the per-variant comparison to MLflow."""

    image = "mlops-train:latest"
    command = ["python", "ab_flow.py", "measure", "--ab-test-id", AB_TEST_ID]
    workdir = "/app"
    volumes = SHARED_VOLUMES
    resources = Resources(cpu=1, memory_gb=2)


ab_flow = Flow("drug-review-abtest")
_resolve = ab_flow.add(ResolveVersions())
_predict_a = ab_flow.add(PredictA(), after=_resolve)
_predict_b = ab_flow.add(PredictB(), after=_resolve)
ab_flow.add(MeasureAB(), after=[_predict_a, _predict_b])
