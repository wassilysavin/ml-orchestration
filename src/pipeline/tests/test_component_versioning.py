import pytest

from src.component_version import (
    DATA_PREP_PARAMS,
    component_version_id,
    data_prep_payload,
    dataset_pull_payload,
    file_digest,
    named_digests,
)


@pytest.mark.flow_versioning
def test_component_version_id_is_deterministic() -> None:
    """Identical (component, payload) inputs must produce the identical id."""
    payload = {"a": 1, "b": [1, 2, 3]}
    assert component_version_id("x", payload) == component_version_id("x", payload)


@pytest.mark.flow_versioning
def test_component_name_namespaces_the_id() -> None:
    """The same payload under different component names must hash differently."""
    payload = {"a": 1}
    assert component_version_id("dataset-pull", payload) != component_version_id(
        "data-prep", payload
    )


@pytest.mark.flow_versioning
def test_dataset_version_tracks_file_content() -> None:
    """Changing any pulled file's digest must change the dataset-pull version."""
    base = dataset_pull_payload({"train.csv": "aaa", "test.csv": "bbb"})
    changed = dataset_pull_payload({"train.csv": "aaa", "test.csv": "ccc"})
    assert component_version_id("dataset-pull", base) != component_version_id(
        "dataset-pull", changed
    )


@pytest.mark.flow_versioning
def test_data_prep_version_chains_from_dataset_version() -> None:
    """A different upstream dataset version must yield a different data-prep version."""
    v1 = component_version_id("data-prep", data_prep_payload("dataset-aaa"))
    v2 = component_version_id("data-prep", data_prep_payload("dataset-bbb"))
    assert v1 != v2


@pytest.mark.flow_versioning
def test_data_prep_version_tracks_prep_params() -> None:
    """A change to the prep contract must change the data-prep version."""
    baseline = component_version_id("data-prep", data_prep_payload("dataset-aaa"))
    mutated_params = {**DATA_PREP_PARAMS, "rating_bucket_bins": [0, 5, 10]}
    mutated = component_version_id(
        "data-prep",
        {"dataset_version_id": "dataset-aaa", "params": mutated_params},
    )
    assert baseline != mutated


@pytest.mark.flow_versioning
def test_file_digest_is_content_addressed(tmp_path) -> None:
    """file_digest is stable for identical content and differs when content changes."""
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"hello world")
    b.write_bytes(b"hello world")
    assert file_digest(a) == file_digest(b)
    b.write_bytes(b"different")
    assert file_digest(a) != file_digest(b)
    assert named_digests({"x": a})["x"] == file_digest(a)
