import argparse
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment import EXPERIMENTS_OUT_DIR, ExperimentSpec
from src.flow_config import TrainingFlowConfig
from src.mlflow_setup import EXPERIMENT_ID_ENV

DEMO_BATCH: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        experiment_id="exp-reg-low",
        train_configs=(
            TrainingFlowConfig(C=1.0, ngram_max=1, min_df=50),
            TrainingFlowConfig(C=0.1, ngram_max=1, min_df=50),
        ),
        ab_pairs=((0, 1),),
    ),
    ExperimentSpec(
        experiment_id="exp-reg-high",
        train_configs=(
            TrainingFlowConfig(C=2.0, ngram_max=1, min_df=50),
            TrainingFlowConfig(C=0.05, ngram_max=1, min_df=50),
        ),
        ab_pairs=((0, 1),),
    ),
    ExperimentSpec(
        experiment_id="exp-ngram",
        train_configs=(
            TrainingFlowConfig(C=1.0, ngram_max=1, min_df=50),
            TrainingFlowConfig(C=1.0, ngram_max=2, min_df=50),
        ),
        ab_pairs=((0, 1),),
    ),
)


@dataclass(frozen=True)
class RuntimeOptions:
    """How to launch each experiment: as a subprocess or its own docker container."""

    runtime: str = "subprocess"
    image: str = "mlops-flow:latest"
    mlflow_uri: str | None = None
    network: str | None = None
    data_dir: Path = PROJECT_ROOT / "data"
    dry_run: bool = False


_FORWARDED_ENV = (
    "MLFLOW_S3_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)


def _subprocess_command(spec: ExperimentSpec) -> list[str]:
    """The local subprocess invocation for one experiment."""
    return [sys.executable, "-m", "src.experiment", "--spec", json.dumps(spec.to_dict())]


def _docker_command(spec: ExperimentSpec, opts: RuntimeOptions) -> list[str]:
    """Build the `docker run` invocation that runs one experiment in its own container."""
    cmd = ["docker", "run", "--rm", "-e", f"{EXPERIMENT_ID_ENV}={spec.experiment_id}"]
    if opts.mlflow_uri:
        cmd += ["-e", f"MLFLOW_TRACKING_URI={opts.mlflow_uri}"]
        for var in _FORWARDED_ENV:
            if os.environ.get(var):
                cmd += ["-e", f"{var}={os.environ[var]}"]
    if opts.network:
        cmd += ["--network", opts.network]
    cmd += [
        "-v", f"{opts.data_dir.resolve()}:/app/data:ro",
        "-v", f"{EXPERIMENTS_OUT_DIR.resolve()}:/app/experiments_out",
        opts.image,
        "python", "-m", "src.experiment", "--spec", json.dumps(spec.to_dict()),
    ]
    return cmd


def _run_one(spec: ExperimentSpec, opts: RuntimeOptions) -> tuple[str, int]:
    """Launch one experiment (subprocess or container); tee its log. Returns (id, rc)."""
    EXPERIMENTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENTS_OUT_DIR / f"{spec.experiment_id}.log"

    if opts.runtime == "docker":
        command = _docker_command(spec, opts)
        env = os.environ.copy()
    else:
        command = _subprocess_command(spec)
        env = os.environ.copy()
        env[EXPERIMENT_ID_ENV] = spec.experiment_id

    if opts.dry_run:
        print(f"[dry-run] {spec.experiment_id}:\n  {shlex.join(command)}")
        return spec.experiment_id, 0

    proc = subprocess.run(command, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    log_path.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    return spec.experiment_id, proc.returncode


def _summarize(specs: tuple[ExperimentSpec, ...]) -> None:
    """Print a per-experiment summary table from the result JSON files."""
    print("\n================ experiment batch summary ================")
    for spec in specs:
        out = EXPERIMENTS_OUT_DIR / f"{spec.experiment_id}.json"
        if not out.exists():
            print(f"{spec.experiment_id:<14}  (no result — see {spec.experiment_id}.log)")
            continue
        r = json.loads(out.read_text(encoding="utf-8"))
        versions = ", ".join(
            f"{t['flow_version_id']}={t['eval_macro_f1']:.4f}"
            for t in r["trained"]
            if t["eval_macro_f1"] is not None
        )
        print(f"\n{spec.experiment_id}")
        print(f"  trained : {versions}")
        print(f"  drift   : {r['monitoring_verdict']}")
        for ab in r["ab_results"]:
            scores = "  ".join(f"{v}={s:.4f}" for v, s in sorted(ab["macro_f1"].items()))
            print(f"  A/B {ab['ab_test_id']}: {scores}  -> winner {ab['winner']}")
    print("\n==========================================================")


def main() -> None:
    """Launch a batch of experiments concurrently and summarize the results."""
    parser = argparse.ArgumentParser(description="Run multiple experiments at once.")
    parser.add_argument(
        "--specs-file",
        help="JSON file: a list of experiment specs (defaults to the demo batch).",
    )
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument(
        "--runtime",
        choices=["subprocess", "docker"],
        default="subprocess",
        help="subprocess (default): one process per experiment. "
        "docker: one container per experiment (image must be built).",
    )
    parser.add_argument("--image", default="mlops-flow:latest")
    parser.add_argument(
        "--mlflow-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="Remote MLflow server for docker runs (recommended). "
        "Defaults to $MLFLOW_TRACKING_URI.",
    )
    parser.add_argument(
        "--network",
        help="Docker network to join (e.g. the deploy/ compose network) so the "
        "container can reach the MLflow server by name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the launch command for each experiment without running it.",
    )
    args = parser.parse_args()

    if args.specs_file:
        data = json.loads(Path(args.specs_file).read_text(encoding="utf-8"))
        specs = tuple(ExperimentSpec.from_dict(d) for d in data)
    else:
        specs = DEMO_BATCH

    opts = RuntimeOptions(
        runtime=args.runtime,
        image=args.image,
        mlflow_uri=args.mlflow_uri,
        network=args.network,
        dry_run=args.dry_run,
    )
    if opts.runtime == "docker" and not opts.mlflow_uri:
        print(
            "warning: docker runtime without --mlflow-uri — each container will "
            "track to its own ephemeral in-container store; results still surface "
            "via experiments_out/, but runs won't land in a shared MLflow."
        )

    print(
        f"launching {len(specs)} experiments via {opts.runtime}, "
        f"up to {args.max_concurrency} at once: "
        f"{', '.join(s.experiment_id for s in specs)}"
    )
    with ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
        futures = {pool.submit(_run_one, s, opts): s for s in specs}
        for fut in as_completed(futures):
            eid, rc = fut.result()
            print(f"  finished {eid} (exit {rc})")

    if not opts.dry_run:
        _summarize(specs)


if __name__ == "__main__":
    main()
