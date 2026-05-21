"""Transform raw Kaggle CSVs into typed Parquet splits + a quality summary."""
import argparse
import html
from typing import Any

import pandas as pd

from src.config import (
    COMBINED_PROCESSED_FILE,
    CURRENT_PROCESSED_FILE,
    CURRENT_SPLIT_LABEL,
    DATE_FORMAT,
    REFERENCE_PROCESSED_FILE,
    REFERENCE_SPLIT_LABEL,
    SUMMARY_FILE,
    TEST_RAW_FILE,
    TRAIN_RAW_FILE,
    UCI_DATASET_NAME,
)
from src.download_data import prepare_raw_dataset
from src.quality_checks import (
    categorical_total_variation_distance,
    condition_missing_rate,
    rating_ks_statistic,
)
from src.utils import (
    ensure_directories,
    processed_outputs_exist,
    read_json,
    read_csv,
    write_json,
    write_parquet,
)


def _clean_review_text(review: object) -> str:
    """Decode HTML entities and collapse whitespace; return empty string for NaN."""
    text = "" if pd.isna(review) else html.unescape(str(review))
    return " ".join(text.split())


def _prepare_split(dataframe: pd.DataFrame, split_label: str) -> pd.DataFrame:
    """Rename columns, normalize dtypes, and add derived feature columns for one split."""
    prepared = dataframe.rename(
        columns={
            "uniqueID": "unique_id",
            "drugName": "drug_name",
            "review": "review_text",
            "date": "review_date",
            "usefulCount": "useful_count",
        }
    ).copy()

    prepared["split"] = split_label
    prepared["review_text"] = prepared["review_text"].map(_clean_review_text).astype("string")
    prepared["drug_name"] = prepared["drug_name"].astype("string")
    prepared["condition"] = prepared["condition"].astype("string")
    prepared["review_date"] = pd.to_datetime(prepared["review_date"], format=DATE_FORMAT)

    prepared["review_length"] = prepared["review_text"].str.len().astype("int32")
    prepared["review_word_count"] = (
        prepared["review_text"].str.split().str.len().fillna(0).astype("int32")
    )
    prepared["review_year"] = prepared["review_date"].dt.year.astype("int16")
    prepared["rating_bucket"] = pd.cut(
        prepared["rating"],
        bins=[0, 4, 7, 10],
        labels=["low", "medium", "high"],
        include_lowest=True,
    ).astype("category")

    prepared["unique_id"] = prepared["unique_id"].astype("int32")
    prepared["rating"] = prepared["rating"].astype("int16")
    prepared["useful_count"] = prepared["useful_count"].astype("int32")
    prepared["split"] = prepared["split"].astype("category")
    prepared["drug_name"] = prepared["drug_name"].astype("category")
    prepared["condition"] = prepared["condition"].astype("category")

    ordered_columns = [
        "unique_id",
        "drug_name",
        "condition",
        "review_text",
        "rating",
        "rating_bucket",
        "review_date",
        "review_year",
        "useful_count",
        "review_length",
        "review_word_count",
        "split",
    ]
    return prepared[ordered_columns]


def _build_summary(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    """Compute row counts plus drift/missingness stats between the two splits."""
    return {
        "dataset": UCI_DATASET_NAME,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "condition_missing_rate_reference": round(condition_missing_rate(reference), 6),
        "condition_missing_rate_current": round(condition_missing_rate(current), 6),
        "rating_ks_reference_vs_current": round(
            rating_ks_statistic(reference["rating"], current["rating"]), 6
        ),
        "condition_tvd_reference_vs_current": round(
            categorical_total_variation_distance(
                reference["condition"], current["condition"]
            ),
            6,
        ),
    }


def build_processed_datasets(overwrite: bool = False) -> dict[str, Any]:
    """Create processed Parquet files for the reference and current splits."""
    ensure_directories()
    prepare_raw_dataset(overwrite=overwrite)

    if processed_outputs_exist() and not overwrite:
        return read_json(SUMMARY_FILE)

    reference_raw = read_csv(TRAIN_RAW_FILE)
    current_raw = read_csv(TEST_RAW_FILE)

    reference = _prepare_split(reference_raw, REFERENCE_SPLIT_LABEL)
    current = _prepare_split(current_raw, CURRENT_SPLIT_LABEL)

    combined = pd.concat([reference, current], ignore_index=True)
    combined = combined.sort_values(["split", "review_date", "unique_id"]).reset_index(drop=True)

    write_parquet(reference, REFERENCE_PROCESSED_FILE)
    write_parquet(current, CURRENT_PROCESSED_FILE)
    write_parquet(combined, COMBINED_PROCESSED_FILE)

    summary = _build_summary(reference, current)
    write_json(summary, SUMMARY_FILE)
    return summary


def ensure_processed_datasets() -> None:
    """Build processed data only when the artifacts are missing."""
    if not processed_outputs_exist():
        build_processed_datasets(overwrite=False)


def main() -> None:
    """CLI entry point: parse `--overwrite` and print the resulting summary."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a processed Parquet version of the UCI Drug Review dataset and "
            "write baseline/current quality summary."
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite the processed files even if they already exist.",
    )
    args = parser.parse_args()

    summary = build_processed_datasets(overwrite=args.overwrite)
    print("Processed data written to data/processed/")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
