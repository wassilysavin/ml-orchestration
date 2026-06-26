import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flow import data_quality_step, robustness_step
from src.artifacts import read_json, write_json
from src.mlflow_setup import ensure_flow_run_id
from src.flow_config import (
    CANDIDATE_FLOW_CONFIGS,
    TrainingFlowConfig,
)
from src.train import train

VARIANTS = CANDIDATE_FLOW_CONFIGS

_RUN_ID_ARTIFACT = "training/run_id"


def _run_robustness(run_id: str) -> None:
    """Run robustness against a specific run id, reported but kept non-fatal."""
    os.environ["PIPELINE_MODEL_RUN_ID"] = run_id
    try:
        robustness_step(run_id)
        print(f"robustness: PASSED for {run_id}")
    except SystemExit as exc:
        print(f"robustness: FAILED for {run_id} ({exc}); continuing (non-fatal).")
    finally:
        os.environ.pop("PIPELINE_MODEL_RUN_ID", None)


def run_training_flow(config: TrainingFlowConfig) -> str:
    """Run data-quality → train(config) → robustness; return the model run id."""
    started = time.time()
    version = config.flow_version_id()
    print(
        f"\n=== training flow (flow_version_id={version}, "
        f"flow_run_id={ensure_flow_run_id()}) ==="
    )
    print(f"config: {config.as_params()}")

    data_quality_step()

    print(f"\n=== step: train (flow_version_id={version}) ===")
    run_id = train(config)
    print(f"model run id: {run_id}  (flow_version_id={version})")

    _run_robustness(run_id)

    print(
        f"\ntraining flow {version} completed in {time.time() - started:.1f}s "
        f"-> model {run_id}"
    )
    return run_id


def train_command(config: TrainingFlowConfig) -> None:
    """DAG step: train(config) and persist the run id for the robustness step."""
    version = config.flow_version_id()
    print(f"\n=== step: train (flow_version_id={version}) ===")
    print(f"config: {config.as_params()}")
    run_id = train(config)
    write_json(_RUN_ID_ARTIFACT, {"run_id": run_id, "flow_version_id": version})
    print(f"model run id: {run_id}  (flow_version_id={version})")


def robustness_command() -> None:
    """DAG step: read the trained run id and run robustness pinned to it."""
    run_id = read_json(_RUN_ID_ARTIFACT)["run_id"]
    _run_robustness(run_id)


def _config_from_args(args: argparse.Namespace) -> TrainingFlowConfig:
    """Build the flow config from a named variant or explicit overrides."""
    if args.variant:
        return VARIANTS[args.variant]
    return TrainingFlowConfig(C=args.C, ngram_max=args.ngram_max, min_df=args.min_df)


def main() -> None:
    """CLI entry point for running one versioned training flow."""
    parser = argparse.ArgumentParser(description="Run a versioned training flow.")
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        help="Use a named preset configuration (overrides the individual flags).",
    )
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--ngram-max", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument(
        "--step",
        default="all",
        choices=["all", "data-quality", "train", "robustness"],
        help="Which step to run (default: the whole flow in-process).",
    )
    args = parser.parse_args()
    if args.step == "data-quality":
        data_quality_step()
    elif args.step == "train":
        train_command(_config_from_args(args))
    elif args.step == "robustness":
        robustness_command()
    else:
        run_training_flow(_config_from_args(args))


if __name__ == "__main__":
    main()
