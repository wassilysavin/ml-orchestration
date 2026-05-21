"""Multi-experiment checks: namespacing isolates, and specs round-trip."""
import pytest

from src.experiment import ExperimentSpec
from src.flow_config import TrainingFlowConfig
from src.mlflow_setup import (
    EXPERIMENT_ID_ENV,
    FLOW_RUN_ID_ENV,
    current_flow_run_id,
    ensure_flow_run_id,
    experiment_name,
    experiment_tags,
)


@pytest.mark.experiments
def test_experiment_name_is_unnamespaced_without_id(monkeypatch) -> None:
    """With no EXPERIMENT_ID/FLOW_RUN_ID, names and tags are unchanged (backward compatible)."""
    monkeypatch.delenv(EXPERIMENT_ID_ENV, raising=False)
    monkeypatch.delenv(FLOW_RUN_ID_ENV, raising=False)
    assert experiment_name("drug-review-sentiment") == "drug-review-sentiment"
    assert experiment_tags() == {}


@pytest.mark.experiments
def test_experiment_id_namespaces_names_and_tags(monkeypatch) -> None:
    """An EXPERIMENT_ID prefixes every experiment name and tags every run."""
    monkeypatch.setenv(EXPERIMENT_ID_ENV, "exp-42")
    monkeypatch.delenv(FLOW_RUN_ID_ENV, raising=False)
    assert experiment_name("drug-review-sentiment") == "exp-42:drug-review-sentiment"
    assert experiment_name("drug-review-abtest") == "exp-42:drug-review-abtest"
    assert experiment_tags() == {"experiment_id": "exp-42"}


@pytest.mark.experiments
def test_flow_run_id_is_tagged_and_correlates_with_experiment_id(monkeypatch) -> None:
    """The flow run id is stamped on every run, alongside the experiment id."""
    monkeypatch.setenv(FLOW_RUN_ID_ENV, "fr-001")
    monkeypatch.delenv(EXPERIMENT_ID_ENV, raising=False)
    assert experiment_tags() == {"flow_run_id": "fr-001"}
    monkeypatch.setenv(EXPERIMENT_ID_ENV, "exp-42")
    assert experiment_tags() == {"experiment_id": "exp-42", "flow_run_id": "fr-001"}


@pytest.mark.experiments
def test_ensure_flow_run_id_reuses_then_mints(monkeypatch) -> None:
    """Reuse-or-mint: an injected id is kept; an unset one is minted and exported."""
    monkeypatch.setenv(FLOW_RUN_ID_ENV, "injected-id")
    assert ensure_flow_run_id() == "injected-id"

    monkeypatch.delenv(FLOW_RUN_ID_ENV, raising=False)
    minted = ensure_flow_run_id()
    assert len(minted) == 12
    assert current_flow_run_id() == minted
    assert ensure_flow_run_id() == minted


@pytest.mark.experiments
def test_distinct_experiments_get_distinct_namespaces(monkeypatch) -> None:
    """Two experiment ids never share a namespaced experiment name."""
    monkeypatch.setenv(EXPERIMENT_ID_ENV, "exp-a")
    name_a = experiment_name("drug-review-sentiment")
    monkeypatch.setenv(EXPERIMENT_ID_ENV, "exp-b")
    name_b = experiment_name("drug-review-sentiment")
    assert name_a != name_b


@pytest.mark.experiments
def test_experiment_spec_round_trips_through_json() -> None:
    """A spec survives to_dict/from_dict unchanged (the launcher ships it as JSON)."""
    spec = ExperimentSpec(
        experiment_id="exp-rt",
        train_configs=(
            TrainingFlowConfig(C=1.0, ngram_max=1, min_df=50),
            TrainingFlowConfig(C=0.1, ngram_max=2, min_df=20),
        ),
        ab_pairs=((0, 1),),
        run_monitoring=True,
    )
    restored = ExperimentSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert restored.train_configs[0].flow_version_id() == (
        spec.train_configs[0].flow_version_id()
    )
