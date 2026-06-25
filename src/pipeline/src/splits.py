import hashlib

from src.config import INCOMING_EVAL_FRACTION, INCOMING_SPLIT_SALT

_BUCKETS = 10_000


def is_incoming_eval(
    unique_id: int,
    eval_fraction: float = INCOMING_EVAL_FRACTION,
    salt: str = INCOMING_SPLIT_SALT,
) -> bool:
    """Return True if this record is held out as eval (vs folded into training)."""
    digest = hashlib.sha256(f"{salt}:{unique_id}".encode("utf-8")).hexdigest()
    position = (int(digest, 16) % _BUCKETS) / _BUCKETS
    return position < eval_fraction
