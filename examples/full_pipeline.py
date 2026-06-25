import asyncio
import csv
import io
import os
from pathlib import Path

from orchestration import (
    Flow,
    Resources,
    Step,
    SubFlow,
    Trigger,
    directory_source,
    exited_with,
)
from orchestration.agent import LocalDockerAgent
from orchestration.control_plane import ControlPlane
from orchestration.monitoring import MonitoringService

MLOPS_ROOT = (Path(__file__).resolve().parent.parent / "src" / "pipeline").resolve()
INCOMING_DIR = MLOPS_ROOT / "data" / "incoming"

for sub in ("data", "models", "mlruns", "artifacts"):
    (MLOPS_ROOT / sub).mkdir(parents=True, exist_ok=True)
INCOMING_DIR.mkdir(parents=True, exist_ok=True)

SHARED_VOLUMES: dict[str, str] = {
    str(MLOPS_ROOT / "data"): "/app/data",
    str(MLOPS_ROOT / "models"): "/app/models",
    str(MLOPS_ROOT / "mlruns"): "/app/mlruns",
    str(MLOPS_ROOT / "artifacts"): "/app/artifacts",
}

NO_DRIFT_EXIT_CODE = 64

INCOMING_DATASET_ENV = "INCOMING_DATASET"
INCOMING_CONTAINER_DIR = "/app/data/incoming"

# Set DRIFTED_DATASET to a filename under INCOMING_DIR to seed the one-shot CLI run
# (`orchestration examples/full_pipeline.py:flow`) with that dataset.
DRIFTED_DATASET = os.environ.get("DRIFTED_DATASET")

TRAINING_VARIANTS: list[tuple[str, list[str]]] = [
    ("baseline", ["--variant", "baseline"]),
    ("challenger", ["--variant", "challenger"]),
    ("c03", ["--C", "0.3"]),
]

REQUIRED_COLUMNS = {
    "uniqueID",
    "drugName",
    "condition",
    "review",
    "rating",
    "date",
    "usefulCount",
}


def validate_csv_header(raw: bytes) -> None:
    """Raise ValueError unless the CSV header contains every required column."""
    text = raw.decode("utf-8", errors="replace")
    try:
        header = next(csv.reader(io.StringIO(text)))
    except StopIteration:
        raise ValueError("empty dataset: no CSV header found")
    columns = {cell.strip() for cell in header}
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")


class Ingest(Step):
    """Get the dataset and build the reference/training/current splits."""

    image = "mlops-data:latest"
    command = ["sh", "-c", "python -m src.download_data && python -m src.prepare_dataset"]
    workdir = "/app"
    volumes = SHARED_VOLUMES
    resources = Resources(cpu=2, memory_gb=2)


class DataQualityTests(Step):
    """Pre-training data-quality tests (Pandera/pytest) — gates the pipeline."""

    image = "mlops-data:latest"
    command = ["pytest", "-m", "data_quality", "-q", "tests"]
    workdir = "/app"
    volumes = SHARED_VOLUMES
    resources = Resources(cpu=1, memory_gb=2)


class DriftBranch(Step):
    """Covariate drift (PSI). Exit 0 = drift detected (retrain), 64 = no drift (skip)."""

    image = "mlops-train:latest"
    command = ["python", "monitoring_flow.py", "--branch"]
    workdir = "/app"
    volumes = SHARED_VOLUMES
    resources = Resources(cpu=1, memory_gb=2)
    expected_exit_codes = frozenset({0, NO_DRIFT_EXIT_CODE})


class TrainVariant(Step):
    """Train one retrain candidate and persist its run id for robustness."""

    image = "mlops-train:latest"
    workdir = "/app"
    volumes = SHARED_VOLUMES
    resources = Resources(cpu=2, memory_gb=2)

    def __init__(self, name: str, args: list[str]) -> None:
        """Pin the command to the versioned train step for this config."""
        self.command = ["python", "training_flow.py", *args, "--step", "train"]
        super().__init__(name=f"train-{name}")


class RobustnessVariant(Step):
    """Run robustness pinned to the run id the train step persisted (non-fatal)."""

    image = "mlops-train:latest"
    command = ["python", "training_flow.py", "--step", "robustness"]
    workdir = "/app"
    volumes = SHARED_VOLUMES
    resources = Resources(cpu=2, memory_gb=2)

    def __init__(self, name: str) -> None:
        """Name the step per variant so the DAG nodes stay distinct."""
        super().__init__(name=f"robustness-{name}")


class ABTest(Step):
    """Resolve baseline vs challenger to model ids; compare macro-F1; record the winner."""

    image = "mlops-train:latest"
    command = ["python", "ab_flow.py", "--ab-test-id", "full-demo"]
    workdir = "/app"
    volumes = SHARED_VOLUMES
    resources = Resources(cpu=2, memory_gb=3)


class Promote(Step):
    """Promote the A/B winner to champion (prior champion kept as rollback)."""

    image = "mlops-train:latest"
    command = ["python", "promote.py"]
    workdir = "/app"
    volumes = SHARED_VOLUMES
    resources = Resources(cpu=2, memory_gb=3)


def _training_subflow(name: str, args: list[str]) -> SubFlow:
    """A nested train → robustness flow for one candidate (artifact-isolated per run)."""

    def build() -> Flow:
        flow = Flow(f"train-{name}")
        train = flow.add(TrainVariant(name, args))
        flow.add(RobustnessVariant(name), after=train)
        return flow

    return SubFlow(build, name=f"hp-{name}")


def build_flow(incoming_filename: str | None = None) -> Flow:
    """The end-to-end DAG, optionally seeded with a freshly-arrived dataset.

    Ingest → DataQuality → DriftBranch → (3 train/robustness subflows when drift) →
    ABTest → Promote. When `incoming_filename` is given, Ingest reads that file from
    the incoming drop directory instead of re-pulling the reference dataset.
    """
    ingest_step = Ingest()
    if incoming_filename is not None:
        ingest_step.env = {
            INCOMING_DATASET_ENV: f"{INCOMING_CONTAINER_DIR}/{incoming_filename}"
        }

    flow = Flow("full-mlops-demo")
    ingest = flow.add(ingest_step)
    dq = flow.add(DataQualityTests(), after=ingest)
    drift = flow.add(DriftBranch(), after=dq)

    subflows = {
        name: flow.add(
            _training_subflow(name, args), after=drift, when=exited_with(drift, 0)
        )
        for name, args in TRAINING_VARIANTS
    }

    ab = flow.add(ABTest(), after=[subflows["baseline"], subflows["challenger"]])
    flow.add(Promote(), after=ab)
    return flow


# One-shot entry point loaded by the CLI: `orchestration examples/full_pipeline.py:flow`.
flow = build_flow(DRIFTED_DATASET)


async def watch_and_run(*, interval: float = 2.0) -> None:
    """Watch the drop dir; run the full DAG once per newly-arrived *.csv file."""
    agent = LocalDockerAgent(host_id="host-local")
    cp = ControlPlane(agent)
    monitor = MonitoringService()
    cp.bus.subscribe(monitor.handle)

    async def on_new_dataset(filename: str) -> None:
        """Validate the arrived dataset, then run the DAG and print its state tree."""
        try:
            validate_csv_header((INCOMING_DIR / filename).read_bytes())
        except (ValueError, OSError) as exc:
            print(f"\n>>> skipping {filename}: {exc}")
            return
        print(f"\n>>> new dataset detected: {filename} -> running full pipeline")
        run = await cp.run_flow(build_flow(incoming_filename=filename))
        print(monitor.render_tree(run.id))

    trigger = Trigger(directory_source(INCOMING_DIR, suffix=".csv"), on_new_dataset)
    trigger.baseline()
    print(f"watching {INCOMING_DIR} for new *.csv (Ctrl-C to stop)")
    try:
        await trigger.watch(interval=interval)
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(watch_and_run())
