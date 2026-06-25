import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from full_pipeline import REQUIRED_COLUMNS, validate_csv_header

_GOOD_HEADER = ",".join(sorted(REQUIRED_COLUMNS)).encode()


def test_valid_header_passes() -> None:
    """A CSV carrying every required column validates."""
    validate_csv_header(_GOOD_HEADER + b"\n1,a,b,c,2,d,3\n")


def test_missing_column_rejected() -> None:
    """A header missing a required column raises with the offending name."""
    header = ",".join(sorted(REQUIRED_COLUMNS - {"rating"})).encode()
    with pytest.raises(ValueError, match="rating"):
        validate_csv_header(header + b"\n")


def test_empty_dataset_rejected() -> None:
    """An empty body is rejected rather than silently accepted."""
    with pytest.raises(ValueError, match="empty"):
        validate_csv_header(b"")
