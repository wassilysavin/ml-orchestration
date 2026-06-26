import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import pandas as pd
from sklearn.metrics import f1_score

from src import promotion
from src.ab_split import assign_variant
from src.config import (
    AB_EXPERIMENT_NAME,
    AB_SPLIT_ATTRIBUTE,
    AB_VARIANT_A,
    AB_VARIANT_B,
    AB_WINNER_ARTIFACT,
    SELECTION_WINNER_ARTIFACT,
)
from src.flow_config import BASELINE_FLOW_CONFIG, CHALLENGER_FLOW_CONFIG
from src.mlflow_setup import configure_tracking, ensure_flow_run_id, experiment_tags
from src.flow_registry import resolve_flow_version_to_run_id
from src.registry import load_model
from src.sentiment_data import build_xy, filter_for_binary_sentiment
from src.unseen_segment import load_unseen_segment
from src.artifacts import read_json, write_json

_MODEL_IDS_ARTIFACT = "ab/model_ids"
_VERSIONS_ARTIFACT = "ab/versions"


def _results_artifact(variant: str) -> str:
    """Artifact name a predict branch writes its per-variant stats to."""
    return f"ab/results_{variant}"


def resolve_step(version_a: str, version_b: str) -> dict[str, str]:
    """Step 1: resolve each flow version id to its model run id via MLflow."""
    print("\n=== step: resolve-versions ===")
    run_a = resolve_flow_version_to_run_id(version_a)
    run_b = resolve_flow_version_to_run_id(version_b)
    print(f"variant {AB_VARIANT_A}: flow_version {version_a} -> model {run_a}")
    print(f"variant {AB_VARIANT_B}: flow_version {version_b} -> model {run_b}")
    return {AB_VARIANT_A: run_a, AB_VARIANT_B: run_b}


def _predict_branch(run_id: str, frame: pd.DataFrame) -> float:
    """Load one variant's model and return its macro-F1 on its routed slice."""
    X, y = build_xy(frame)
    model = load_model(run_id)
    return float(f1_score(y, model.predict(X), average="macro"))


def _route_segment(ab_test_id: str) -> pd.DataFrame:
    """Load the unseen segment and tag each row with its A/B variant."""
    segment = filter_for_binary_sentiment(load_unseen_segment())
    return segment.assign(
        variant=segment[AB_SPLIT_ATTRIBUTE].map(
            lambda rid: assign_variant(int(rid), ab_test_id)
        )
    )


def _predict_variant(
    run_id: str, variant: str, segment: pd.DataFrame
) -> dict[str, float]:
    """Score one variant's model on its routed half of an already-routed segment."""
    branch = segment[segment["variant"] == variant]
    macro_f1 = _predict_branch(run_id, branch)
    print(f"variant {variant}: model={run_id} n={len(branch)} macro_f1={macro_f1:.6f}")
    return {"model_run_id": run_id, "n": int(len(branch)), "macro_f1": macro_f1}


def predict_step(
    model_ids: dict[str, str], ab_test_id: str
) -> dict[str, dict[str, float]]:
    """Step 2: fork the unseen segment by variant and score each branch."""
    print("\n=== step: fork-and-predict ===")
    segment = _route_segment(ab_test_id)
    return {
        variant: _predict_variant(run_id, variant, segment)
        for variant, run_id in model_ids.items()
    }


def measure_step(
    results: dict[str, dict[str, float]],
    ab_test_id: str,
    versions: dict[str, str],
) -> str:
    """Step 3: log per-variant metrics under the test's namespace; return winner."""
    print("\n=== step: measure ===")
    winner = max(results, key=lambda v: results[v]["macro_f1"])

    write_json(
        AB_WINNER_ARTIFACT,
        {
            "variant": winner,
            "flow_version_id": versions[winner],
            "run_id": results[winner]["model_run_id"],
            "macro_f1": results[winner]["macro_f1"],
            "ab_test_id": ab_test_id,
        },
    )

    configure_tracking(AB_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"ab:{ab_test_id}") as run:
        mlflow.set_tags(
            {
                "flow_name": "drug-review-abtest",
                "ab_test_id": ab_test_id,
                "winner_variant": winner,
                **experiment_tags(),
            }
        )
        mlflow.log_params(
            {
                "ab_test_id": ab_test_id,
                "metric": "macro_f1",
                "split_attribute": AB_SPLIT_ATTRIBUTE,
                f"flow_version_{AB_VARIANT_A}": versions[AB_VARIANT_A],
                f"flow_version_{AB_VARIANT_B}": versions[AB_VARIANT_B],
                f"model_{AB_VARIANT_A}": results[AB_VARIANT_A]["model_run_id"],
                f"model_{AB_VARIANT_B}": results[AB_VARIANT_B]["model_run_id"],
            }
        )
        for variant, stats in results.items():
            mlflow.log_metrics(
                {
                    f"macro_f1_{variant}": stats["macro_f1"],
                    f"n_{variant}": stats["n"],
                }
            )
        print(
            f"winner={winner} "
            f"(A={results[AB_VARIANT_A]['macro_f1']:.6f} vs "
            f"B={results[AB_VARIANT_B]['macro_f1']:.6f})  "
            f"mlflow run {run.info.run_id}"
        )
    return winner


def ab_flow(version_a: str, version_b: str, ab_test_id: str) -> dict[str, dict[str, float]]:
    """Run resolve -> fork-and-predict -> measure in-process for one A/B test."""
    started = time.time()
    print(f"\n=== A/B flow (ab_test_id={ab_test_id}, flow_run_id={ensure_flow_run_id()}) ===")
    versions = {AB_VARIANT_A: version_a, AB_VARIANT_B: version_b}
    model_ids = resolve_step(version_a, version_b)
    results = predict_step(model_ids, ab_test_id)
    measure_step(results, ab_test_id, versions)
    print(f"\nA/B flow completed in {time.time() - started:.1f}s")
    return results


def resolve_command(version_a: str, version_b: str) -> None:
    """DAG step: resolve versions to model ids and persist them for the branches."""
    model_ids = resolve_step(version_a, version_b)
    write_json(_MODEL_IDS_ARTIFACT, model_ids)
    write_json(_VERSIONS_ARTIFACT, {AB_VARIANT_A: version_a, AB_VARIANT_B: version_b})


def predict_command(variant: str, ab_test_id: str) -> None:
    """DAG step (one fork branch): score `variant` and persist its stats."""
    print(f"\n=== step: fork-and-predict ({variant}) ===")
    run_id = read_json(_MODEL_IDS_ARTIFACT)[variant]
    stats = _predict_variant(run_id, variant, _route_segment(ab_test_id))
    write_json(_results_artifact(variant), stats)


def measure_command(ab_test_id: str) -> str:
    """DAG step: read both branches' stats and log the comparison to MLflow."""
    versions = read_json(_VERSIONS_ARTIFACT)
    results = {v: read_json(_results_artifact(v)) for v in (AB_VARIANT_A, AB_VARIANT_B)}
    return measure_step(results, ab_test_id, versions)


def _forward_selection_winner(ab_test_id: str) -> None:
    """No champion yet: hand the selection winner straight to Promote for bootstrap."""
    selected = read_json(SELECTION_WINNER_ARTIFACT)
    write_json(
        AB_WINNER_ARTIFACT,
        {
            "variant": AB_VARIANT_A,
            "flow_version_id": selected["flow_version_id"],
            "run_id": selected["run_id"],
            "macro_f1": selected.get("macro_f1"),
            "ab_test_id": ab_test_id,
        },
    )
    print("no champion yet -> selection winner forwarded for bootstrap promotion")


def champion_command(ab_test_id: str) -> None:
    """DAG step: A/B the selection winner (A) against the live champion (B).

    The winning candidate from the offline selection step plays the role of the
    challenger; the deployed champion is variant B. With no champion on record yet
    there is nothing to test against, so the selection winner is forwarded as-is.
    """
    print(
        f"\n=== A/B (selected vs champion, ab_test_id={ab_test_id}, "
        f"flow_run_id={ensure_flow_run_id()}) ==="
    )
    selected = read_json(SELECTION_WINNER_ARTIFACT)
    champion = promotion.get_champion()
    if champion is None:
        _forward_selection_winner(ab_test_id)
        return

    model_ids = {AB_VARIANT_A: selected["run_id"], AB_VARIANT_B: champion["run_id"]}
    versions = {
        AB_VARIANT_A: selected["flow_version_id"],
        AB_VARIANT_B: champion["flow_version_id"],
    }
    print(
        f"variant {AB_VARIANT_A} (selected {selected['variant']}): "
        f"model {selected['run_id']}; "
        f"variant {AB_VARIANT_B} (champion): model {champion['run_id']}"
    )
    results = predict_step(model_ids, ab_test_id)
    measure_step(results, ab_test_id, versions)


def main() -> None:
    """CLI entry point for running the A/B flow or its individual steps."""
    parser = argparse.ArgumentParser(description="Run an offline A/B prediction flow.")
    parser.add_argument(
        "step",
        nargs="?",
        default="run",
        choices=["run", "resolve", "predict", "measure", "champion"],
        help="Which step to run (default: run the whole flow in-process).",
    )
    parser.add_argument(
        "--version-a",
        default=BASELINE_FLOW_CONFIG.flow_version_id(),
        help="flow_version_id for variant A (default: baseline preset).",
    )
    parser.add_argument(
        "--version-b",
        default=CHALLENGER_FLOW_CONFIG.flow_version_id(),
        help="flow_version_id for variant B (default: challenger preset).",
    )
    parser.add_argument(
        "--ab-test-id",
        default="drug-sentiment-ab-001",
        help="Identifier namespacing this test's bucketing and metrics.",
    )
    parser.add_argument(
        "--variant",
        choices=[AB_VARIANT_A, AB_VARIANT_B],
        help="Which fork branch to score (required for the 'predict' step).",
    )
    args = parser.parse_args()
    if args.step == "resolve":
        resolve_command(args.version_a, args.version_b)
    elif args.step == "predict":
        if not args.variant:
            parser.error("the 'predict' step requires --variant A|B")
        predict_command(args.variant, args.ab_test_id)
    elif args.step == "measure":
        measure_command(args.ab_test_id)
    elif args.step == "champion":
        champion_command(args.ab_test_id)
    else:
        ab_flow(args.version_a, args.version_b, args.ab_test_id)


if __name__ == "__main__":
    main()
