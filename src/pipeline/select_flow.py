import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from sklearn.metrics import f1_score

from src.artifacts import write_json
from src.config import SELECTION_EXPERIMENT_NAME, SELECTION_WINNER_ARTIFACT
from src.flow_config import CANDIDATE_FLOW_CONFIGS
from src.flow_registry import resolve_flow_version_to_run_id
from src.mlflow_setup import configure_tracking, ensure_flow_run_id, experiment_tags
from src.registry import load_model
from src.selection import pick_winner
from src.sentiment_data import build_xy, filter_for_binary_sentiment
from src.unseen_segment import load_unseen_segment


def _resolve_candidates(names: list[str]) -> dict[str, dict[str, str]]:
    """Map each candidate name to its (flow_version_id, model_run_id) via MLflow."""
    resolved: dict[str, dict[str, str]] = {}
    for name in names:
        version = CANDIDATE_FLOW_CONFIGS[name].flow_version_id()
        run_id = resolve_flow_version_to_run_id(version)
        resolved[name] = {"flow_version_id": version, "run_id": run_id}
        print(f"candidate {name}: flow_version {version} -> model {run_id}")
    return resolved


def _score_on_shared_segment(
    resolved: dict[str, dict[str, str]],
) -> dict[str, dict[str, float]]:
    """Score every candidate's model on one shared held-out segment (apples-to-apples)."""
    segment = filter_for_binary_sentiment(load_unseen_segment())
    X, y = build_xy(segment)
    n = int(len(X))
    results: dict[str, dict[str, float]] = {}
    for name, ref in resolved.items():
        model = load_model(ref["run_id"])
        macro_f1 = float(f1_score(y, model.predict(X), average="macro"))
        results[name] = {**ref, "n": n, "macro_f1": macro_f1}
        print(f"candidate {name}: model={ref['run_id']} n={n} macro_f1={macro_f1:.6f}")
    return results


def _log_selection(
    names: list[str], results: dict[str, dict[str, float]], winner: str
) -> None:
    """Record the per-candidate macro-F1 ranking and the winner to MLflow."""
    configure_tracking(SELECTION_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="selection") as run:
        mlflow.set_tags(
            {
                "flow_name": "drug-review-selection",
                "winner_variant": winner,
                **experiment_tags(),
            }
        )
        mlflow.log_params(
            {
                "candidates": ",".join(names),
                "metric": "macro_f1",
                **{f"model_{n}": results[n]["run_id"] for n in names},
                **{f"flow_version_{n}": results[n]["flow_version_id"] for n in names},
            }
        )
        for n in names:
            mlflow.log_metrics(
                {f"macro_f1_{n}": results[n]["macro_f1"], f"n_{n}": results[n]["n"]}
            )
        print(f"selection logged to mlflow run {run.info.run_id}")


def select_step(names: list[str]) -> dict[str, object]:
    """Resolve, score on a shared eval set, and record the best-macro-F1 candidate."""
    started = time.time()
    print(f"\n=== selection flow (flow_run_id={ensure_flow_run_id()}) ===")
    resolved = _resolve_candidates(names)
    results = _score_on_shared_segment(resolved)
    winner = pick_winner(results)

    record = {
        "variant": winner,
        "flow_version_id": results[winner]["flow_version_id"],
        "run_id": results[winner]["run_id"],
        "macro_f1": results[winner]["macro_f1"],
    }
    write_json(SELECTION_WINNER_ARTIFACT, record)
    _log_selection(names, results, winner)
    print(
        f"\nselected {winner} (macro_f1={results[winner]['macro_f1']:.6f}) "
        f"in {time.time() - started:.1f}s"
    )
    return record


def main() -> None:
    """CLI entry point: compare the trained candidates and record the winner."""
    parser = argparse.ArgumentParser(
        description="Pick the best trained candidate on one shared held-out set."
    )
    parser.add_argument(
        "--candidates",
        default=",".join(CANDIDATE_FLOW_CONFIGS),
        help="Comma-separated candidate names to compare (default: all candidates).",
    )
    args = parser.parse_args()
    names = [c.strip() for c in args.candidates.split(",") if c.strip()]
    select_step(names)


if __name__ == "__main__":
    main()
