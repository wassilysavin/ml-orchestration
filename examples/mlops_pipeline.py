"""Example flow: orchestrates src/pipeline/ as three sequential containers."""
from pathlib import Path
from orchestration import Flow, Step, Resources


MLOPS_ROOT = (Path(__file__).resolve().parent.parent / "src" / "pipeline").resolve()

for sub in ("data", "models", "mlruns"):
    (MLOPS_ROOT / sub).mkdir(parents=True, exist_ok=True)

SHARED_VOLUMES: dict[str, str] = {
    str(MLOPS_ROOT / "data"):   "/app/data",
    str(MLOPS_ROOT / "models"): "/app/models",
    str(MLOPS_ROOT / "mlruns"): "/app/mlruns",
}


class DataPrep(Step):
    """Download the raw dataset and build the prepared parquet under /app/data."""

    image = "mlops-data:latest"
    command = None
    workdir = "/app"
    volumes = SHARED_VOLUMES
    resources = Resources(cpu=2, memory_gb=2)


class Train(Step):
    """Train the sentiment model and emit a versioned artifact under /app/models."""

    image = "mlops-train:latest"
    command = [
        "python", "-c",
        "from src.train import ensure_trained_model; "
        "rid = ensure_trained_model(); print('model run id:', rid)",
    ]
    workdir = "/app"
    volumes = SHARED_VOLUMES
    resources = Resources(cpu=2, memory_gb=4)


class Robustness(Step):
    """Run the robustness/behavioral test suite against the trained model."""

    image = "mlops-train:latest"
    command = ["pytest", "-m", "robustness", "-q", "-p", "no:warnings", "tests"]
    workdir = "/app"
    volumes = SHARED_VOLUMES
    resources = Resources(cpu=2, memory_gb=5)


flow = Flow("drug-review-sentiment")
data_prep = flow.add(DataPrep())
train = flow.add(Train(), after=data_prep)
flow.add(Robustness(), after=train)
