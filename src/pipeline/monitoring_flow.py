import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow

from src.artifacts import read_json, write_json
from src.config import (
    DRIFT_FEATURE,
    MONITORING_EXPERIMENT_NAME,
    PSI_MODERATE_THRESHOLD,
    PSI_SIGNIFICANT_THRESHOLD,
)
from src.drift_checks import population_stability_index
from src.mlflow_setup import configure_tracking, ensure_flow_run_id, experiment_tags
from src.unseen_segment import (
    load_reference_segment,
    load_unseen_segment,
)

_MEASUREMENT_ARTIFACT = "monitoring/measurement"

NO_DRIFT_EXIT_CODE = 64


def _configure_mlflow() -> None:
    """Point MLflow at the local store (or remote server); select the experiment."""
    configure_tracking(MONITORING_EXPERIMENT_NAME)


def measure_step() -> dict[str, float]:
    """Step 1: compute the PSI drift statistic on the unseen segment."""
    print(f"\n=== step: measure-drift (feature={DRIFT_FEATURE}) ===")
    reference = load_reference_segment()[DRIFT_FEATURE]
    current = load_unseen_segment()[DRIFT_FEATURE]
    psi = population_stability_index(reference, current)
    print(
        f"reference rows={len(reference)} current rows={len(current)} "
        f"PSI({DRIFT_FEATURE})={psi:.6f}"
    )
    return {"psi": psi, "reference_rows": len(reference), "current_rows": len(current)}


def band_of(psi: float) -> str:
    """Map a PSI value to its decision band (none / moderate / significant)."""
    if psi > PSI_SIGNIFICANT_THRESHOLD:
        return "significant"
    if psi > PSI_MODERATE_THRESHOLD:
        return "moderate"
    return "none"


def evaluate_step(measurement: dict[str, float]) -> str:
    """Step 2: compare PSI to the threshold, log to MLflow, return the verdict."""
    print("\n=== step: evaluate-drift ===")
    psi = measurement["psi"]
    band = band_of(psi)
    passed = psi <= PSI_SIGNIFICANT_THRESHOLD
    verdict = "PASS" if passed else "FAIL"

    _configure_mlflow()
    with mlflow.start_run(run_name="drift-check") as run:
        mlflow.set_tags(
            {
                "flow_name": "drug-review-monitoring",
                "drift_band": band,
                **experiment_tags(),
            }
        )
        mlflow.log_params(
            {
                "drift_test": "population_stability_index",
                "feature": DRIFT_FEATURE,
                "expected_source": "training_reference_split",
                "significant_threshold": PSI_SIGNIFICANT_THRESHOLD,
            }
        )
        mlflow.log_metrics(
            {
                "psi": psi,
                "reference_rows": measurement["reference_rows"],
                "current_rows": measurement["current_rows"],
                "passed": float(passed),
            }
        )
        print(
            f"PSI={psi:.6f} band={band} threshold={PSI_SIGNIFICANT_THRESHOLD} "
            f"-> {verdict}  (mlflow run {run.info.run_id})"
        )
    return verdict


def _maybe_gate(verdict: str, gate: bool) -> None:
    """Exit non-zero on detected drift when gating, so the orchestrated DAG fails."""
    if gate and verdict == "FAIL":
        print("gate: significant drift detected -> exiting non-zero to fail the flow")
        raise SystemExit(1)


def _maybe_branch(verdict: str, branch: bool) -> None:
    """Signal the orchestrator whether downstream retraining should run."""
    if not branch:
        return
    if verdict == "PASS":
        print(f"branch: no significant drift -> exit {NO_DRIFT_EXIT_CODE} (skip retrain)")
        raise SystemExit(NO_DRIFT_EXIT_CODE)
    print("branch: drift detected -> exit 0 (run retrain)")


def monitoring_flow(gate: bool = False, branch: bool = False) -> str:
    """Run measure -> evaluate in-process and return the drift verdict."""
    started = time.time()
    print(f"\n=== monitoring flow (flow_run_id={ensure_flow_run_id()}) ===")
    measurement = measure_step()
    verdict = evaluate_step(measurement)
    print(f"\nmonitoring flow completed in {time.time() - started:.1f}s -> {verdict}")
    _maybe_gate(verdict, gate)
    _maybe_branch(verdict, branch)
    return verdict


def measure_command() -> None:
    """DAG step: compute drift and persist the measurement for the evaluate step."""
    write_json(_MEASUREMENT_ARTIFACT, measure_step())


def evaluate_command(gate: bool = False, branch: bool = False) -> str:
    """DAG step: read the upstream measurement artifact and record the verdict."""
    verdict = evaluate_step(read_json(_MEASUREMENT_ARTIFACT))
    _maybe_gate(verdict, gate)
    _maybe_branch(verdict, branch)
    return verdict


def main() -> None:
    """CLI entry point for running the drift-monitoring flow or its individual steps."""
    parser = argparse.ArgumentParser(description="Run the drift-monitoring flow.")
    parser.add_argument(
        "step",
        nargs="?",
        default="run",
        choices=["run", "measure", "evaluate"],
        help="Which step to run (default: run the whole flow in-process).",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Exit non-zero when significant drift is detected (fails the DAG).",
    )
    parser.add_argument(
        "--branch",
        action="store_true",
        help=(
            f"Exit 0 on drift / {NO_DRIFT_EXIT_CODE} on no-drift so the orchestrator "
            "can skip the retrain subtree with when=exited_with(drift, 0)."
        ),
    )
    args = parser.parse_args()
    if args.step == "measure":
        measure_command()
    elif args.step == "evaluate":
        evaluate_command(gate=args.gate, branch=args.branch)
    else:
        monitoring_flow(gate=args.gate, branch=args.branch)


if __name__ == "__main__":
    main()
