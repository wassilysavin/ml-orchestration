import pytest

from src import artifacts


@pytest.fixture()
def isolated_artifacts(tmp_path, monkeypatch):
    """Point the artifact base at a tmp dir and clear any inherited FLOW_RUN_ID."""
    monkeypatch.setenv("PIPELINE_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.delenv("FLOW_RUN_ID", raising=False)
    return tmp_path


def test_distinct_flow_runs_do_not_clobber(isolated_artifacts, monkeypatch) -> None:
    """The same artifact name under two FLOW_RUN_IDs stays independent."""
    name = "training/run_id"

    monkeypatch.setenv("FLOW_RUN_ID", "run-a")
    artifacts.write_json(name, {"run_id": "model-a"})

    monkeypatch.setenv("FLOW_RUN_ID", "run-b")
    artifacts.write_json(name, {"run_id": "model-b"})

    monkeypatch.setenv("FLOW_RUN_ID", "run-a")
    assert artifacts.read_json(name) == {"run_id": "model-a"}
    monkeypatch.setenv("FLOW_RUN_ID", "run-b")
    assert artifacts.read_json(name) == {"run_id": "model-b"}


def test_same_flow_run_shares_artifacts(isolated_artifacts, monkeypatch) -> None:
    """Steps sharing a FLOW_RUN_ID (e.g. train -> robustness) see each other's writes."""
    monkeypatch.setenv("FLOW_RUN_ID", "run-shared")
    artifacts.write_json("training/run_id", {"run_id": "model-x"})
    assert artifacts.read_json("training/run_id") == {"run_id": "model-x"}


def test_run_id_namespaces_the_directory(isolated_artifacts, monkeypatch) -> None:
    """The run id appears as a path segment; without one it falls back to the base."""
    monkeypatch.setenv("FLOW_RUN_ID", "run-a")
    assert artifacts.artifacts_dir() == isolated_artifacts / "run-a"

    monkeypatch.delenv("FLOW_RUN_ID", raising=False)
    assert artifacts.artifacts_dir() == isolated_artifacts
