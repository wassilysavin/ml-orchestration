import pytest

from src.flow_config import (
    BASELINE_FLOW_CONFIG,
    CHALLENGER_FLOW_CONFIG,
    TrainingFlowConfig,
)


@pytest.mark.flow_versioning
def test_flow_version_id_is_deterministic() -> None:
    """Identical configs must produce the identical flow_version_id."""
    a = TrainingFlowConfig(C=0.5, ngram_max=2, min_df=10)
    b = TrainingFlowConfig(C=0.5, ngram_max=2, min_df=10)
    assert a.flow_version_id() == b.flow_version_id()


@pytest.mark.flow_versioning
def test_changing_any_field_changes_the_version() -> None:
    """A change to any tracked field must yield a different flow_version_id."""
    base = TrainingFlowConfig()
    assert base.flow_version_id() != TrainingFlowConfig(C=0.5).flow_version_id()
    assert base.flow_version_id() != TrainingFlowConfig(ngram_max=1).flow_version_id()
    assert base.flow_version_id() != TrainingFlowConfig(min_df=50).flow_version_id()


@pytest.mark.flow_versioning
def test_named_presets_are_distinct_versions() -> None:
    """Baseline and challenger presets must be different flow versions."""
    assert (
        BASELINE_FLOW_CONFIG.flow_version_id()
        != CHALLENGER_FLOW_CONFIG.flow_version_id()
    )
