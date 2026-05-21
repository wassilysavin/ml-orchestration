"""Deterministic, reproducible A/B bucketing of records."""

import hashlib

from src.config import AB_VARIANT_A, AB_VARIANT_B


def assign_variant(record_id: int, ab_test_id: str) -> str:
    """Route one record id to variant A or B, deterministically and salted by test."""
    digest = hashlib.sha256(f"{ab_test_id}:{record_id}".encode("utf-8")).hexdigest()
    return AB_VARIANT_A if int(digest, 16) % 2 == 0 else AB_VARIANT_B
