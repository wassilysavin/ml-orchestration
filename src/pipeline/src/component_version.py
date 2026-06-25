import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.config import (
    CURRENT_SPLIT_LABEL,
    DATE_FORMAT,
    KAGGLE_DATASET_SLUG,
    REFERENCE_SPLIT_LABEL,
    UCI_DATASET_NAME,
)

DATA_PREP_PARAMS: dict[str, Any] = {
    "dataset": UCI_DATASET_NAME,
    "date_format": DATE_FORMAT,
    "reference_split_label": REFERENCE_SPLIT_LABEL,
    "current_split_label": CURRENT_SPLIT_LABEL,
    "rating_bucket_bins": [0, 4, 7, 10],
    "rating_bucket_labels": ["low", "medium", "high"],
    "derived_columns": [
        "review_length",
        "review_word_count",
        "review_year",
        "rating_bucket",
    ],
}


def file_digest(path: Path | str) -> str:
    """Return the SHA-256 hex digest of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def named_digests(named_paths: Mapping[str, Path | str]) -> dict[str, str]:
    """Map each logical file name to the content digest of its path."""
    return {name: file_digest(path) for name, path in named_paths.items()}


def component_version_id(component: str, payload: Mapping[str, Any]) -> str:
    """Deterministic 12-char id derived from a component name + its inputs payload."""
    canonical = json.dumps(
        {"component": component, "payload": dict(payload)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def dataset_pull_payload(file_digests: Mapping[str, str]) -> dict[str, Any]:
    """Versioning payload for the dataset-pull component: slug + file content digests."""
    return {"dataset_slug": KAGGLE_DATASET_SLUG, "files": dict(file_digests)}


def data_prep_payload(dataset_version_id: str) -> dict[str, Any]:
    """Versioning payload for data-prep: the prep contract + the dataset it consumed."""
    return {"dataset_version_id": dataset_version_id, "params": DATA_PREP_PARAMS}
