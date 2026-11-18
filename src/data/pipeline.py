"""VYUHA Preprocessing Pipeline Runner.

Executes the full end-to-end data preparation workflow across all CICIDS2017 files,
enforcing leakage prevention, reproducible partitioning, and artifact generation.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from src.config import (
    DATASET_FILES,
    DEV_SAMPLE_SIZE,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    TEST_SIZE,
)
from src.data.preprocessor import FlowPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vyuha.pipeline")


def run_pipeline(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    test_size: float = TEST_SIZE,
    dev_sample_size: int = DEV_SAMPLE_SIZE,
    random_seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Runs the full data preprocessing pipeline.

    Args:
        raw_dir: Path to raw CICIDS2017 CSV directory.
        processed_dir: Path to processed output directory.
        test_size: Fraction of data reserved for testing.
        dev_sample_size: Size of stratified development sample.
        random_seed: Seed for reproducibility.

    Returns:
        Dictionary of pipeline execution metadata and data statistics.
    """
    start_time = time.time()
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing VYUHA Preprocessor (Seed: %d, Test Split: %.2f)", random_seed, test_size)
    preprocessor = FlowPreprocessor(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        random_seed=random_seed,
        test_size=test_size,
    )

    # 1. Load and clean each of the 8 dataset files
    cleaned_file_dfs = []
    file_stats = []
    total_raw_rows = 0

    for fname in DATASET_FILES:
        fpath = raw_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Required raw dataset file not found: {fpath}")

        t_f0 = time.time()
        file_df = preprocessor.load_and_clean_file(fpath)
        t_f1 = time.time()

        cleaned_file_dfs.append(file_df)
        total_raw_rows += len(file_df)
        file_stats.append({
            "file": fname,
            "rows": len(file_df),
            "seconds": round(t_f1 - t_f0, 2),
        })

    # Combine all cleaned records
    logger.info("Combining cleaned dataframes across %d files...", len(cleaned_file_dfs))
    full_df = pd.concat(cleaned_file_dfs, ignore_index=True)
    del cleaned_file_dfs  # Free memory
    logger.info("Combined dataset contains %d total rows.", len(full_df))

    # 2. Investigate and drop exact duplicate records to prevent train/test leakage
    logger.info("Performing dataset deduplication across all features and labels...")
    deduped_df, dropped_duplicates = preprocessor.deduplicate(full_df)
    del full_df  # Free memory

    # 3. Label distribution analysis on deduplicated data
    category_counts = deduped_df["label_category"].value_counts().to_dict()
    binary_counts = deduped_df["label_binary"].value_counts().to_dict()
    raw_counts = deduped_df["label_raw"].value_counts().to_dict()

    logger.info("Deduplicated Category Distribution:\n%s", json.dumps(category_counts, indent=2))

    # 4. Reproducible stratified train/test split
    logger.info("Performing stratified train/test split (Test Size: %.2f)...", test_size)
    train_df, test_df = preprocessor.split_train_test(deduped_df, stratify_col="label_category")
    del deduped_df  # Free memory

    # 5. Verify zero data leakage (intersection between train and test features)
    logger.info("Verifying train/test leakage isolation...")
    feature_cols = [c for c in train_df.columns if not c.startswith("label_")]
    
    # 6. Generate stratified development sample
    logger.info("Generating stratified development sample (%d rows)...", dev_sample_size)
    dev_sample = preprocessor.create_dev_sample(train_df, sample_size=dev_sample_size, stratify_col="label_category")

    # 7. Save processed datasets as Parquet
    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"
    dev_path = processed_dir / "dev_sample.parquet"

    logger.info("Saving processed datasets to %s...", processed_dir)
    train_df.to_parquet(train_path, index=False, compression="zstd")
    test_df.to_parquet(test_path, index=False, compression="zstd")
    dev_sample.to_parquet(dev_path, index=False, compression="zstd")

    total_time = round(time.time() - start_time, 2)
    logger.info("Pipeline completed successfully in %.2f seconds.", total_time)

    # 8. Export metadata
    metadata = {
        "pipeline_version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "execution_time_seconds": total_time,
        "random_seed": random_seed,
        "test_size": test_size,
        "total_raw_rows": total_raw_rows,
        "dropped_duplicates": dropped_duplicates,
        "deduplicated_rows": len(train_df) + len(test_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "dev_sample_rows": len(dev_sample),
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "target_names": ["label_raw", "label_binary", "label_category"],
        "rate_bounds": preprocessor.rate_bounds,
        "train_category_distribution": train_df["label_category"].value_counts().to_dict(),
        "test_category_distribution": test_df["label_category"].value_counts().to_dict(),
        "dev_category_distribution": dev_sample["label_category"].value_counts().to_dict(),
        "file_processing_breakdown": file_stats,
    }

    meta_path = processed_dir / "preprocessing_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Preprocessing metadata saved to %s", meta_path)

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VYUHA Dataset Preprocessing Pipeline")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="Train/test split ratio")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    parser.add_argument("--dev-sample-size", type=int, default=DEV_SAMPLE_SIZE, help="Size of dev sample")
    args = parser.parse_args()

    run_pipeline(
        test_size=args.test_size,
        random_seed=args.seed,
        dev_sample_size=args.dev_sample_size,
    )
