"""Download the UCI Drug Review dataset from Kaggle and serialize to deterministic paths."""


import argparse
import shutil
from pathlib import Path

import kagglehub
import pandas as pd

from src.config import (
    KAGGLE_DATASET_SLUG,
    KAGGLE_TEST_FILENAME,
    KAGGLE_TRAIN_FILENAME,
    TEST_RAW_FILE,
    TEST_RAW_PARQUET_FILE,
    TRAIN_RAW_FILE,
)
from src.utils import ensure_directories


def _kaggle_download_dir() -> Path:
    """Resolve (and cache) the local Kaggle download dir for the configured dataset."""
    return Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))


def prepare_raw_dataset(overwrite: bool = False) -> list[Path]:
    """Fetch raw CSVs from Kaggle into data/raw/ and serialize test split to Parquet."""
    ensure_directories()

    outputs = [TRAIN_RAW_FILE, TEST_RAW_FILE, TEST_RAW_PARQUET_FILE]
    if not overwrite and all(path.exists() for path in outputs):
        return outputs

    source_dir = _kaggle_download_dir()

    if overwrite or not TRAIN_RAW_FILE.exists():
        shutil.copyfile(source_dir / KAGGLE_TRAIN_FILENAME, TRAIN_RAW_FILE)

    if overwrite or not TEST_RAW_FILE.exists():
        shutil.copyfile(source_dir / KAGGLE_TEST_FILENAME, TEST_RAW_FILE)

    if overwrite or not TEST_RAW_PARQUET_FILE.exists():
        pd.read_csv(TEST_RAW_FILE).to_parquet(TEST_RAW_PARQUET_FILE, index=False)

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
