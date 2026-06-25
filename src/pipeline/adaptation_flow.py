import argparse
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow

from monitoring_flow import band_of, evaluate_step, measure_step
from src import promotion
from src.adaptation_policy import Action, decide, in_cooldown
from src.config import (
    ADAPTATION_EXPERIMENT_NAME,
    ADAPTATION_STATE_FILENAME,
    AB_VARIANT_A,
    AB_VARIANT_B,
    MODELS_DIR,
    MODEL_METADATA_FILENAME,
    PROMOTION_MIN_MACRO_F1_MARGIN,
    RETRAIN_COOLDOWN_SECONDS,
)
from src.flow_config import BASELINE_FLOW_CONFIG, CHALLENGER_FLOW_CONFIG, TrainingFlowConfig
from src.mlflow_setup import configure_tracking, ensure_flow_run_id, experiment_tags
from src.utils import read_json, write_json

VARIANTS = {"baseline": BASELINE_FLOW_CONFIG, "challenger": CHALLENGER_FLOW_CONFIG}


def _alert(message: str) -> None:
    """Surface an operator-facing alert (stdout here; a hook in a real system)."""
    print(f"ALERT: {message}")


def _state_path(models_dir: Path) -> Path:
    """Path to the adaptation cooldown/promotion state file."""
    return models_dir / ADAPTATION_STATE_FILENAME


def _load_state(models_dir: Path) -> dict[str, Any]:
    """Read the adaptation state (last-retrain timestamp), or empty if none."""
    path = _state_path(models_dir)
    return read_json(path) if path.exists() else {}


def _save_state(models_dir: Path, state: dict[str, Any]) -> None:
    """Persist the adaptation state."""
    models_dir.mkdir(parents=True, exist_ok=True)
    write_json(state, _state_path(models_dir))


def _config_from_run_id(run_id: str) -> TrainingFlowConfig:
    """Rebuild the flow config a model was trained with, from its metadata."""
    meta = read_json(MODELS_DIR / run_id / MODEL_METADATA_FILENAME)
    return TrainingFlowConfig(**meta["flow_config"])


def _ensure_champion(models_dir: Path) -> dict[str, Any]:
    """Return the live champion, bootstrapping one if none has been set yet."""
    champion = promotion.get_champion(models_dir=models_dir)
    if champion is not None:
        return champion

    from src.registry import list_models

    models = list_models()
    if models:
        run_id = models[-1]["run_id"]
        version = models[-1]["metadata"].get("flow_version_id") or _config_from_run_id(
            run_id
        ).flow_version_id()
    else:
        from training_flow import run_training_flow

        run_id = run_training_flow(BASELINE_FLOW_CONFIG)
        version = BASELINE_FLOW_CONFIG.flow_version_id()
    print(f"adaptation: bootstrapping champion -> {version} ({run_id})")
    return promotion.set_champion(version, run_id, models_dir=models_dir)


def _log_adaptation(outcome: dict[str, Any]) -> None:
    """Record the adaptation decision/outcome under its own MLflow experiment."""
    configure_tracking(ADAPTATION_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"adapt:{outcome['action']}"):
        mlflow.set_tags(
            {
                "flow_name": "drug-review-adaptation",
                "action": outcome["action"],
                "promoted": str(outcome.get("promoted", False)),
                "drift_band": outcome["band"],
                **experiment_tags(),
            }
        )
        metrics = {
            k: float(outcome[k])
            for k in ("psi", "champion_macro_f1", "candidate_macro_f1", "margin")
            if outcome.get(k) is not None
        }
        metrics["promoted"] = float(bool(outcome.get("promoted", False)))
        mlflow.log_metrics(metrics)


def _finish(outcome: dict[str, Any], started: float) -> dict[str, Any]:
    """Log the outcome, print a one-line summary, and return it."""
    _log_adaptation(outcome)
    print(
        f"\nadaptation flow completed in {time.time() - started:.1f}s "
        f"-> action={outcome['action']} promoted={outcome.get('promoted', False)}"
    )
    return outcome


def adaptation_flow(
    candidate_config: TrainingFlowConfig | None = None,
    ab_test_id: str | None = None,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    """Measure drift, decide a response, and act on it. Returns the outcome dict."""
    started = time.time()
    print(f"\n=== adaptation flow (flow_run_id={ensure_flow_run_id()}) ===")

    measurement = measure_step()
    psi = measurement["psi"]
    band = band_of(psi)
    evaluate_step(measurement)
    action = decide(band)
    print(f"\n=== adaptation: psi={psi:.6f} band={band} -> action={action.value} ===")

    outcome: dict[str, Any] = {"psi": psi, "band": band, "action": action.value, "promoted": False}

    if action is Action.NONE:
        return _finish(outcome, started)

    if action is Action.ALERT:
        _alert(f"drift band={band} (psi={psi:.6f}) — investigate; no model change")
        return _finish(outcome, started)

    now = time.time()
    state = _load_state(models_dir)
    if in_cooldown(state.get("last_retrain_ts"), now, RETRAIN_COOLDOWN_SECONDS):
        outcome["action"] = "retrain_suppressed_cooldown"
        _alert("significant drift but within retrain cooldown — skipping retrain")
        return _finish(outcome, started)

    champion = _ensure_champion(models_dir)
    candidate = candidate_config or _config_from_run_id(champion["run_id"])
    candidate_version = candidate.flow_version_id()

    from training_flow import run_training_flow

    candidate_run_id = run_training_flow(candidate)
    _save_state(models_dir, {"last_retrain_ts": now})

    from ab_flow import ab_flow

    ab_test_id = ab_test_id or f"adapt-{int(now)}"
    results = ab_flow(champion["flow_version_id"], candidate_version, ab_test_id)
    champ_f1 = results[AB_VARIANT_A]["macro_f1"]
    cand_f1 = results[AB_VARIANT_B]["macro_f1"]
    margin = cand_f1 - champ_f1

    outcome.update(
        {
            "champion_version": champion["flow_version_id"],
            "candidate_version": candidate_version,
            "champion_macro_f1": champ_f1,
            "candidate_macro_f1": cand_f1,
            "margin": margin,
            "ab_test_id": ab_test_id,
        }
    )

    if margin >= PROMOTION_MIN_MACRO_F1_MARGIN:
        promotion.set_champion(candidate_version, candidate_run_id, models_dir=models_dir)
        outcome["promoted"] = True
        print(
            f"PROMOTE: candidate {candidate_version} ({cand_f1:.4f}) beats champion "
            f"{champion['flow_version_id']} ({champ_f1:.4f}) by {margin:.4f}"
        )
    else:
        _alert(
            f"candidate {candidate_version} ({cand_f1:.4f}) did not beat champion "
            f"{champion['flow_version_id']} ({champ_f1:.4f}) by margin "
            f"{PROMOTION_MIN_MACRO_F1_MARGIN} — keeping champion (rollback target retained)"
        )

    return _finish(outcome, started)


def _candidate_from_args(args: argparse.Namespace) -> TrainingFlowConfig | None:
    """Build the retrain candidate config from CLI args, or None (= champion's)."""
    if args.variant:
        return VARIANTS[args.variant]
    if args.C is not None or args.ngram_max is not None or args.min_df is not None:
        base = TrainingFlowConfig()
        return TrainingFlowConfig(
            C=args.C if args.C is not None else base.C,
            ngram_max=args.ngram_max if args.ngram_max is not None else base.ngram_max,
            min_df=args.min_df if args.min_df is not None else base.min_df,
        )
    return None


def main() -> None:
    """CLI entry point for the closed-loop adaptation flow."""
    parser = argparse.ArgumentParser(description="Run the closed-loop drift adaptation flow.")
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        help="Named preset for the retrain candidate (default: the champion's own config).",
    )
    parser.add_argument("--C", type=float, default=None)
    parser.add_argument("--ngram-max", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--min-df", type=int, default=None)
    parser.add_argument(
        "--ab-test-id",
        default=None,
        help="Identifier for the champion-vs-candidate A/B test (default: adapt-<ts>).",
    )
    args = parser.parse_args()
    adaptation_flow(
        candidate_config=_candidate_from_args(args),
        ab_test_id=args.ab_test_id,
    )


if __name__ == "__main__":
    main()
