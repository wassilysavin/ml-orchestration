"""Filesystem-backed model registry: list, load, and serve ONNX sentiment models."""
from typing import Iterable

import numpy as np
import onnxruntime as ort

from src.config import MODEL_FILENAME, MODEL_METADATA_FILENAME, MODELS_DIR
from src.utils import read_json


class OnnxSentimentModel:
    """Adapter exposing sklearn-style predict / predict_proba over an ONNX session."""

    def __init__(self, session: ort.InferenceSession) -> None:
        """Cache the ONNX session and resolve its input/output tensor names."""
        self._session = session
        self._input_name = session.get_inputs()[0].name
        outputs = session.get_outputs()
        self._label_output = outputs[0].name
        self._probability_output = outputs[1].name

    @staticmethod
    def _to_input(texts: Iterable[str]) -> np.ndarray:
        """Reshape an iterable of texts into the (N, 1) object array ONNX expects."""
        return np.asarray(list(texts), dtype=object).reshape(-1, 1)

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        """Return the predicted class label array for each text."""
        return self._session.run(
            [self._label_output], {self._input_name: self._to_input(texts)}
        )[0]

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        """Return the per-class probability array for each text."""
        return self._session.run(
            [self._probability_output], {self._input_name: self._to_input(texts)}
        )[0]


def list_models() -> list[dict]:
    """Return metadata for every versioned model on disk."""
    if not MODELS_DIR.exists():
        return []
    entries = []
    for run_dir in sorted(p for p in MODELS_DIR.iterdir() if p.is_dir()):
        metadata_path = run_dir / MODEL_METADATA_FILENAME
        entries.append(
            {
                "run_id": run_dir.name,
                "path": str(run_dir),
                "metadata": read_json(metadata_path) if metadata_path.exists() else {},
            }
        )
    return entries


def _onnx_bytes_from_mlflow(run_id: str) -> bytes:
    """Download the ONNX artifact for a run id from the MLflow artifact store."""
    import mlflow.onnx

    from src.mlflow_setup import configure_tracking

    configure_tracking()
    onnx_model = mlflow.onnx.load_model(f"runs:/{run_id}/onnx-model")
    return onnx_model.SerializeToString()


def load_model(run_id: str) -> OnnxSentimentModel:
    """Deserialize a model by its run id (local disk first, else MLflow artifact)."""
    model_path = MODELS_DIR / run_id / MODEL_FILENAME
    if model_path.exists():
        session = ort.InferenceSession(str(model_path))
    else:
        session = ort.InferenceSession(_onnx_bytes_from_mlflow(run_id))
    return OnnxSentimentModel(session)


def latest_run_id() -> str:
    """Return the run id of the most recently created model."""
    runs = list_models()
    if not runs:
        raise FileNotFoundError("No models found under models/.")
    return runs[-1]["run_id"]
