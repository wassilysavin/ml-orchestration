

import argparse
import os
import shutil
from pathlib import Path

import kagglehub
import pandas as pd

from src.component_version import (
    component_version_id,
    dataset_pull_payload,
    named_digests,
)
from src.config import (
    DATASET_PULL_COMPONENT,
    KAGGLE_DATASET_SLUG,
    KAGGLE_TEST_FILENAME,
    KAGGLE_TRAIN_FILENAME,
    TEST_RAW_FILE,
    TEST_RAW_PARQUET_FILE,
    TRAIN_RAW_FILE,
    UCI_DATASET_NAME,
)
from src.ingestion import incoming_dataset_path, plan_ingestion
from src.utils import ensure_directories


def _kaggle_download_dir() -> Path:
    """Resolve (and cache) the local Kaggle download dir for the configured dataset."""
    return Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))


def dataset_pull_version() -> tuple[str, dict[str, str]]:
    """Content-address the pulled raw files into a dataset-pull component version."""
    digests = named_digests(
        {KAGGLE_TRAIN_FILENAME: TRAIN_RAW_FILE, KAGGLE_TEST_FILENAME: TEST_RAW_FILE}
    )
    return component_version_id(DATASET_PULL_COMPONENT, dataset_pull_payload(digests)), digests


def _maybe_track_dataset_pull() -> None:
    """When tracking is active, log this dataset-pull version + file digests to MLflow."""
    if not (os.environ.get("FLOW_RUN_ID") or os.environ.get("TRACK_COMPONENTS")):
        return
    try:
        version_id, digests = dataset_pull_version()
        from src.component_tracking import log_component

        log_component(
            DATASET_PULL_COMPONENT,
            version_id,
            params={
                "dataset_slug": KAGGLE_DATASET_SLUG,
                "dataset_name": UCI_DATASET_NAME,
                **{f"sha256_{name}": digest for name, digest in digests.items()},
            },
            metrics={
                f"bytes_{name}": float(path.stat().st_size)
                for name, path in {
                    KAGGLE_TRAIN_FILENAME: TRAIN_RAW_FILE,
                    KAGGLE_TEST_FILENAME: TEST_RAW_FILE,
                }.items()
            },
        )
        print(f"dataset-pull component_version_id={version_id}")
    except Exception as exc:
        print(f"dataset-pull tracking skipped: {exc}")


def prepare_raw_dataset(overwrite: bool = False) -> list[Path]:
    """Materialize raw CSVs into data/raw/ and serialize the current split to Parquet."""
    ensure_directories()

    outputs = [TRAIN_RAW_FILE, TEST_RAW_FILE, TEST_RAW_PARQUET_FILE]
    incoming = incoming_dataset_path()
    plan = plan_ingestion(
        has_incoming=incoming is not None,
        train_exists=TRAIN_RAW_FILE.exists(),
        test_exists=TEST_RAW_FILE.exists(),
        test_parquet_exists=TEST_RAW_PARQUET_FILE.exists(),
        overwrite=overwrite,
    )

    if plan.need_kaggle_train or plan.need_kaggle_test:
        source_dir = _kaggle_download_dir()
        if plan.need_kaggle_train:
            shutil.copyfile(source_dir / KAGGLE_TRAIN_FILENAME, TRAIN_RAW_FILE)
        if plan.need_kaggle_test:
            shutil.copyfile(source_dir / KAGGLE_TEST_FILENAME, TEST_RAW_FILE)

    if plan.use_incoming:
        incoming_file = Path(incoming)
        if not incoming_file.exists():
            raise FileNotFoundError(
                f"INCOMING_DATASET points at a missing file: {incoming}"
            )
        shutil.copyfile(incoming_file, TEST_RAW_FILE)
        print(f"ingest: using incoming dataset {incoming_file.name} as the current split")

    if plan.rebuild_test_parquet:
        pd.read_csv(TEST_RAW_FILE).to_parquet(TEST_RAW_PARQUET_FILE, index=False)

    _maybe_track_dataset_pull()
    return outputs


def main() -> None:
    """CLI entry point: parse `--overwrite` and run the raw-download routine."""
    parser = argparse.ArgumentParser(
        description=(
            "Download the UCI Drug Review raw CSVs from the Kaggle dataset "
            f"'{KAGGLE_DATASET_SLUG}' into data/raw/, and serialize the test split "
            "to a columnar Parquet copy for later use."
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download and rewrite the raw files even if they already exist.",
    )
    args = parser.parse_args()

    prepared_paths = prepare_raw_dataset(overwrite=args.overwrite)
    for path in prepared_paths:
        print(f"Raw file ready: {path}")


if __name__ == "__main__":
    main()
