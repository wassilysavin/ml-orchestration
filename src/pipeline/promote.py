import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import promotion
from src.config import (
    AB_EXPERIMENT_NAME,
    AB_WINNER_ARTIFACT,
    MODELS_DIR,
    PROMOTION_MIN_MACRO_F1_MARGIN,
)
from src.flow_config import CHALLENGER_FLOW_CONFIG

TRAINING_RUN_ID_ARTIFACT = "training/run_id"


def should_promote(margin: float, min_margin: float = PROMOTION_MIN_MACRO_F1_MARGIN) -> bool:
    """Pure decision: promote when the candidate beats the champion by the margin."""
    return margin >= min_margin


def _read_ab_winner() -> dict | None:
    """Return the A/B winner record if an upstream A/B step recorded one this run."""
    try:
        from src.artifacts import read_json

        return read_json(AB_WINNER_ARTIFACT)
    except FileNotFoundError:
        return None


def _resolve_candidate(candidate_version: str) -> tuple[str, str]:
    """Return (run_id, version) for the candidate: the trained run if present, else resolve."""
    try:
        from src.artifacts import read_json

        record = read_json(TRAINING_RUN_ID_ARTIFACT)
        return record["run_id"], record.get("flow_version_id", candidate_version)
    except FileNotFoundError:
        from src.flow_registry import resolve_flow_version_to_run_id

        return resolve_flow_version_to_run_id(candidate_version), candidate_version


def _eval_macro_f1(run_id: str) -> float:
    """Score one model (by run id) on the held-out current segment."""
    from sklearn.metrics import f1_score

    from src.registry import load_model
    from src.sentiment_data import build_xy, filter_for_binary_sentiment
    from src.unseen_segment import load_unseen_segment

    segment = filter_for_binary_sentiment(load_unseen_segment())
    X, y = build_xy(segment)
    model = load_model(run_id)
    return float(f1_score(y, model.predict(X), average="macro"))


def _log_promotion(outcome: dict[str, Any]) -> None:
    """Record the champion-vs-candidate comparison and decision to MLflow."""
    import mlflow

    from src.mlflow_setup import configure_tracking, experiment_tags

    configure_tracking(AB_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="promotion"):
        mlflow.set_tags(
            {
                "flow_name": "drug-review-promotion",
                "promoted": str(outcome["promoted"]),
                **experiment_tags(),
            }
        )
        mlflow.log_params(
            {
                "champion_run_id": outcome.get("champion_run_id", ""),
                "candidate_run_id": outcome["candidate_run_id"],
                "candidate_version": outcome["candidate_version"],
                "min_margin": PROMOTION_MIN_MACRO_F1_MARGIN,
            }
        )
        metrics = {"promoted": float(outcome["promoted"])}
        for key in ("champion_macro_f1", "candidate_macro_f1", "margin"):
            if outcome.get(key) is not None:
                metrics[key] = float(outcome[key])
        mlflow.log_metrics(metrics)


def promote(
    candidate_version: str, models_dir: Path = MODELS_DIR
) -> dict[str, Any]:
    """Promote the A/B winner if one was recorded; else candidate-vs-champion."""
    from src.mlflow_setup import ensure_flow_run_id

    print(f"\n=== promote step (flow_run_id={ensure_flow_run_id()}) ===")

    winner = _read_ab_winner()
    if winner is not None:
        record = promotion.set_champion(
            winner["flow_version_id"], winner["run_id"], models_dir=models_dir
        )
        outcome = {
            "candidate_run_id": winner["run_id"],
            "candidate_version": winner["flow_version_id"],
            "candidate_macro_f1": winner.get("macro_f1"),
            "promoted": True,
        }
        print(
            f"PROMOTE: A/B winner {winner['variant']} "
            f"({winner['flow_version_id']}, macro_f1={winner.get('macro_f1')}) "
            f"-> champion"
        )
        _log_promotion(outcome)
        return record

    candidate_run_id, candidate_version = _resolve_candidate(candidate_version)
    champion = promotion.get_champion(models_dir=models_dir)

    outcome: dict[str, Any] = {
        "candidate_run_id": candidate_run_id,
        "candidate_version": candidate_version,
        "promoted": False,
    }

    if champion is None:
        promotion.set_champion(candidate_version, candidate_run_id, models_dir=models_dir)
        outcome["promoted"] = True
        print(f"promote: no prior champion -> bootstrapped {candidate_version}")
        _log_promotion(outcome)
        return outcome

    champion_f1 = _eval_macro_f1(champion["run_id"])
    candidate_f1 = _eval_macro_f1(candidate_run_id)
    margin = candidate_f1 - champion_f1
    outcome.update(
        {
            "champion_run_id": champion["run_id"],
            "champion_macro_f1": champion_f1,
            "candidate_macro_f1": candidate_f1,
            "margin": margin,
            "promoted": should_promote(margin),
        }
    )

    if outcome["promoted"]:
        promotion.set_champion(candidate_version, candidate_run_id, models_dir=models_dir)
        print(
            f"PROMOTE: candidate {candidate_run_id} ({candidate_f1:.4f}) beats "
            f"champion {champion['run_id']} ({champion_f1:.4f}) by {margin:.4f}"
        )
    else:
        print(
            f"KEEP champion {champion['run_id']} ({champion_f1:.4f}): "
            f"candidate ({candidate_f1:.4f}) margin {margin:.4f} "
            f"< {PROMOTION_MIN_MACRO_F1_MARGIN}"
        )
    _log_promotion(outcome)
    return outcome


def main() -> None:
    """CLI entry point for the promotion step."""
    parser = argparse.ArgumentParser(
        description="Compare candidate vs champion on the held-out segment; promote on a win."
    )
    parser.add_argument(
        "--candidate-version",
        default=CHALLENGER_FLOW_CONFIG.flow_version_id(),
        help="Fallback flow_version_id if no trained run id artifact is present.",
    )
    args = parser.parse_args()
    promote(args.candidate_version)


if __name__ == "__main__":
    main()
