"""Single-process pipeline driver: data quality → train → robustness."""
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import ensure_trained_model


def _banner(name: str) -> None:
    """Print a visual step separator to stdout."""
    print(f"\n=== step: {name} ===")


def _run_pytest(marker: str) -> None:
    """Invoke pytest restricted to a given marker and raise SystemExit on failure."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", marker, "-q", "tests"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        raise SystemExit(f"step {marker} failed (pytest exit {result.returncode})")


def data_quality_step() -> None:
    """Run the `data_quality`-marked tests."""
    _banner("data-quality")
    _run_pytest("data_quality")


def train_step() -> str:
    """Ensure a trained model exists and return its run id."""
    _banner("train")
    run_id = ensure_trained_model()
    print(f"model run id: {run_id}")
    return run_id


def robustness_step(run_id: str) -> None:
    """Run the `robustness`-marked tests against the trained model."""
    _banner(f"robustness (run_id={run_id})")
    _run_pytest("robustness")


def pipeline() -> None:
    """Run the three pipeline stages end-to-end and report total elapsed time."""
    started = time.time()
    data_quality_step()
    run_id = train_step()
    robustness_step(run_id)
    print(f"\npipeline completed in {time.time() - started:.1f}s")


if __name__ == "__main__":
    pipeline()
