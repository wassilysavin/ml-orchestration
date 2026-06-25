import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import mlflow.onnx
from onnx import ModelProto
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src.config import (
    FLOW_NAME_TAG,
    FLOW_VERSION_TAG,
    MIN_TRAINING_ROWS,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_FILENAME,
    MODEL_METADATA_FILENAME,
    MODELS_DIR,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    TRAINING_FLOW_NAME,
)
from src.flow_config import TrainingFlowConfig
from src.mlflow_setup import configure_tracking, experiment_tags
from src.sentiment_data import build_xy, filter_for_binary_sentiment
from src.utils import load_processed_current, load_processed_training


class InsufficientTrainingDataError(RuntimeError):
    """Raised when the post-filter training set is below the configured minimum."""


def build_pipeline(config: TrainingFlowConfig) -> Pipeline:
    """Construct the TF-IDF + LogisticRegression pipeline from a flow config."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, config.ngram_max),
                    min_df=config.min_df,
                    max_features=100_000,
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=config.C,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=config.random_seed,
                ),
            ),
        ]
    )


def _convert_to_onnx(model: Pipeline) -> ModelProto:
    """Serialize the fitted sklearn pipeline to an ONNX ModelProto."""
    initial_types = [("review_text", StringTensorType([None, 1]))]
    return to_onnx(
        model,
        initial_types=initial_types,
        options={id(model.named_steps["clf"]): {"zipmap": False}},
    )


def _configure_mlflow() -> None:
    """Point MLflow at the local store (or remote server) and set the experiment."""
    configure_tracking(MLFLOW_EXPERIMENT_NAME)


def _persist_artifact(
    onnx_model: ModelProto, run_id: str, metadata: dict[str, Any]
) -> Path:
    """Write the ONNX bytes and metadata JSON under models/<run_id>/."""
    run_dir = MODELS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / MODEL_FILENAME
    model_path.write_bytes(onnx_model.SerializeToString())
    (run_dir / MODEL_METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return model_path


def train(config: TrainingFlowConfig | None = None) -> str:
    """Train, score, convert to ONNX, log to MLflow, persist; return the run id."""
    config = config or TrainingFlowConfig()

    training = filter_for_binary_sentiment(
        load_processed_training(),
        positive_floor=config.positive_rating_floor,
        negative_ceiling=config.negative_rating_ceiling,
    )
    current = filter_for_binary_sentiment(
        load_processed_current(),
        positive_floor=config.positive_rating_floor,
        negative_ceiling=config.negative_rating_ceiling,
    )

    X_train, y_train = build_xy(training)
    X_eval, y_eval = build_xy(current)

    if len(X_train) < MIN_TRAINING_ROWS:
        raise InsufficientTrainingDataError(
            f"Training data has {len(X_train)} rows, "
            f"below the configured minimum of {MIN_TRAINING_ROWS}."
        )

    model = build_pipeline(config)
    model.fit(X_train, y_train)

    train_macro_f1 = float(f1_score(y_train, model.predict(X_train), average="macro"))
    eval_macro_f1 = float(f1_score(y_eval, model.predict(X_eval), average="macro"))

    onnx_model = _convert_to_onnx(model)
    flow_version_id = config.flow_version_id()

    _configure_mlflow()
    with mlflow.start_run() as run:
        mlflow.set_tags(
            {
                FLOW_NAME_TAG: TRAINING_FLOW_NAME,
                FLOW_VERSION_TAG: flow_version_id,
                **experiment_tags(),
            }
        )
        mlflow.log_params(
            {
                "model_type": "tfidf-logreg-onnx",
                "min_training_rows": MIN_TRAINING_ROWS,
                "training_rows": len(X_train),
                "evaluation_rows": len(X_eval),
                **config.as_params(),
            }
        )
        mlflow.log_metrics(
            {"train_macro_f1": train_macro_f1, "eval_macro_f1": eval_macro_f1}
        )
        mlflow.onnx.log_model(onnx_model, artifact_path="onnx-model")

        metadata = {
            "run_id": run.info.run_id,
            "flow_name": TRAINING_FLOW_NAME,
            "flow_version_id": flow_version_id,
            "flow_config": config.as_params(),
            "input_schema": {"review_text": "string"},
            "output_schema": {
                "sentiment": f"int[{NEGATIVE_LABEL},{POSITIVE_LABEL}]",
                "probability": "float[0,1]",
            },
            "label_mapping": {"positive": POSITIVE_LABEL, "negative": NEGATIVE_LABEL},
            "training_rows": int(len(X_train)),
            "evaluation_rows": int(len(X_eval)),
            "metrics": {
                "train_macro_f1": train_macro_f1,
                "eval_macro_f1": eval_macro_f1,
            },
            "serialization_format": "ONNX",
            "dependencies_file": "requirements-train.txt",
        }
        _persist_artifact(onnx_model, run.info.run_id, metadata)
        return run.info.run_id


def ensure_trained_model() -> str:
    """Train and version a model only when no artifact exists yet."""
    if MODELS_DIR.exists():
        existing = sorted(p for p in MODELS_DIR.iterdir() if p.is_dir())
        if existing:
            return existing[-1].name
    return train()


def main() -> None:
    """CLI entry point: train a fresh model (with `--force`) or ensure one exists."""
    parser = argparse.ArgumentParser(
        description="Train the binary sentiment model on the reference split."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Train and version a fresh model even if one already exists.",
    )
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--ngram-max", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--min-df", type=int, default=5)
    args = parser.parse_args()
    config = TrainingFlowConfig(C=args.C, ngram_max=args.ngram_max, min_df=args.min_df)
    run_id = train(config) if args.force else ensure_trained_model()
    print(f"Model run id: {run_id}")


if __name__ == "__main__":
    main()
