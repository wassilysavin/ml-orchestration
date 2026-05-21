"""Robustness check (disabled): predictions should survive benign text perturbations."""
import pytest

from src.config import INVARIANCE_SAMPLE_SIZE
from src.sentiment_data import filter_for_binary_sentiment
