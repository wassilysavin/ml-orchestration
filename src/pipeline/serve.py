import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import promotion
from src.config import MODELS_DIR


class NoChampionError(RuntimeError):
    """Raised when no model has been promoted to champion yet."""


def resolve_champion(models_dir: Path = MODELS_DIR) -> dict[str, Any]:
    """Return the live champion record, or raise if the pipeline hasn't promoted one."""
    champion = promotion.get_champion(models_dir=models_dir)
    if champion is None:
        raise NoChampionError(
            "no champion model promoted yet — run the pipeline/adaptation chain first"
        )
    return champion


def format_prediction(text: str, label: int, prob: float, champion: dict[str, Any]) -> str:
    """Render one prediction line with the champion that produced it."""
    sentiment = "positive" if int(label) == 1 else "negative"
    return (
        f"label={int(label)} ({sentiment}) prob={prob:.4f} "
        f"champion={champion['run_id']} version={champion['flow_version_id']} "
        f"| {text!r}"
    )


def predict(
    texts: list[str], models_dir: Path = MODELS_DIR
) -> tuple[dict[str, Any], list[tuple[str, int, float]]]:
    """Load the champion and predict each text; return (champion, [(text,label,prob)])."""
    champion = resolve_champion(models_dir)
    from src.registry import load_model

    model = load_model(champion["run_id"])
    labels = model.predict(texts)
    probabilities = model.predict_proba(texts)

    results: list[tuple[str, int, float]] = []
    for text, label, proba in zip(texts, labels, probabilities):
        label_int = int(label)
        results.append((text, label_int, float(proba[label_int])))
    return champion, results


def main() -> None:
    """CLI entry point: `serve.py predict <text> [<text> ...]`."""
    parser = argparse.ArgumentParser(description="Predict with the champion model.")
    sub = parser.add_subparsers(dest="command", required=True)

    predict_parser = sub.add_parser("predict", help="Predict sentiment for text(s).")
    predict_parser.add_argument("text", nargs="+", help="One or more review texts.")

    sub.add_parser("champion", help="Print the current champion pointer and exit.")

    args = parser.parse_args()
    try:
        if args.command == "champion":
            champion = resolve_champion()
            print(
                f"champion run_id={champion['run_id']} "
                f"version={champion['flow_version_id']}"
            )
            return
        champion, results = predict(args.text)
        for text, label, prob in results:
            print(format_prediction(text, label, prob, champion))
    except NoChampionError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
