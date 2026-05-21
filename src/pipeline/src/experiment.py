"""Run one self-contained experiment of versioned training, monitoring, and A/B."""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, MODEL_METADATA_FILENAME  # noqa: E402
from src.flow_config import TrainingFlowConfig  # noqa: E402
from src.mlflow_setup import EXPERIMENT_ID_ENV  # noqa: E402

EXPERIMENTS_OUT_DIR = PROJECT_ROOT / "experiments_out"


@dataclass(frozen=True)
class ExperimentSpec:
    """A batch-launchable experiment: configs to train + A/B pairs to compare."""

    experiment_id: str
    train_configs: tuple[TrainingFlowConfig, ...]
    ab_pairs: tuple[tuple[int, int], ...] = ()
    run_monitoring: bool = True

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form (passed to child processes by the launcher)."""
        return {
            "experiment_id": self.experiment_id,
            "train_configs": [c.as_params() for c in self.train_configs],
            "ab_pairs": [list(p) for p in self.ab_pairs],
            "run_monitoring": self.run_monitoring,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentSpec":
        """Rebuild a spec from its JSON form."""
        return cls(
            experiment_id=data["experiment_id"],
            train_configs=tuple(
                TrainingFlowConfig(**params) for params in data["train_configs"]
            ),
            ab_pairs=tuple(tuple(p) for p in data.get("ab_pairs", [])),
            run_monitoring=bool(data.get("run_monitoring", True)),
        )


def _eval_macro_f1(run_id: str) -> float | None:
    """Read a trained model's eval macro-F1 from its local metadata, if present."""
    meta_path = MODELS_DIR / run_id / MODEL_METADATA_FILENAME
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("metrics", {}).get("eval_macro_f1")


def run_experiment(spec: ExperimentSpec) -> dict[str, Any]:
    """Run one experiment end-to-end under its own namespace; return a result dict."""
    os.environ[EXPERIMENT_ID_ENV] = spec.experiment_id

    from ab_flow import ab_flow
    from monitoring_flow import monitoring_flow
    from src.train import train

    print(f"\n########## experiment {spec.experiment_id} : start ##########")

    trained: list[dict[str, Any]] = []
    for config in spec.train_configs:
        version = config.flow_version_id()
        run_id = train(config)
        trained.append(
            {
                "flow_version_id": version,
                "run_id": run_id,
                "config": config.as_params(),
                "eval_macro_f1": _eval_macro_f1(run_id),
            }
        )
        print(f"[{spec.experiment_id}] trained {version} -> {run_id}")

    monitoring_verdict = monitoring_flow() if spec.run_monitoring else None

    ab_results: list[dict[str, Any]] = []
    for i, j in spec.ab_pairs:
        version_a = spec.train_configs[i].flow_version_id()
        version_b = spec.train_configs[j].flow_version_id()
        ab_test_id = f"{spec.experiment_id}-ab-{i}v{j}"
        results = ab_flow(version_a, version_b, ab_test_id)
        winner = max(results, key=lambda v: results[v]["macro_f1"])
        ab_results.append(
            {
                "ab_test_id": ab_test_id,
                "version_a": version_a,
                "version_b": version_b,
                "macro_f1": {v: results[v]["macro_f1"] for v in results},
                "n": {v: results[v]["n"] for v in results},
                "winner": winner,
            }
        )

    result = {
        "experiment_id": spec.experiment_id,
        "trained": trained,
        "monitoring_verdict": monitoring_verdict,
        "ab_results": ab_results,
    }
    print(f"########## experiment {spec.experiment_id} : done ##########")
    return result


def _write_result(result: dict[str, Any]) -> Path:
    """Persist an experiment result as JSON for the launcher to aggregate."""
    EXPERIMENTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPERIMENTS_OUT_DIR / f"{result['experiment_id']}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def main() -> None:
    """CLI: run a single experiment from its JSON spec (the launcher's child)."""
    parser = argparse.ArgumentParser(description="Run one experiment from a JSON spec.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--spec", help="Experiment spec as an inline JSON string.")
    group.add_argument("--spec-file", help="Path to a JSON file holding the spec.")
    args = parser.parse_args()

    raw = Path(args.spec_file).read_text(encoding="utf-8") if args.spec_file else args.spec
    spec = ExperimentSpec.from_dict(json.loads(raw))
    result = run_experiment(spec)
    out_path = _write_result(result)
    print(f"[{spec.experiment_id}] result -> {out_path}")


if __name__ == "__main__":
    main()
