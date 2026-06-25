
import hashlib
import json
from dataclasses import asdict, dataclass

from src.config import (
    NEGATIVE_RATING_CEILING,
    POSITIVE_RATING_FLOOR,
    RANDOM_SEED,
)


@dataclass(frozen=True)
class TrainingFlowConfig:
    """All flow-level knobs that affect model performance, in one versioned unit."""

    C: float = 1.0
    ngram_max: int = 2
    min_df: int = 5
    positive_rating_floor: int = POSITIVE_RATING_FLOOR
    negative_rating_ceiling: int = NEGATIVE_RATING_CEILING
    random_seed: int = RANDOM_SEED

    def as_params(self) -> dict[str, object]:
        """Return the config as a flat dict suitable for ``mlflow.log_params``."""
        return asdict(self)

    def flow_version_id(self) -> str:
        """Deterministic 12-char id derived from the configuration contents."""
        canonical = json.dumps(self.as_params(), sort_keys=True)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return digest[:12]


BASELINE_FLOW_CONFIG = TrainingFlowConfig(C=1.0, ngram_max=2, min_df=5)
CHALLENGER_FLOW_CONFIG = TrainingFlowConfig(C=0.05, ngram_max=1, min_df=50)
