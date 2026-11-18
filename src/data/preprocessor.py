"""VYUHA Flow Preprocessor Module.

Orchestrates chunked loading, cleaning, deduplication, train/test partitioning,
and sample extraction with strict zero-leakage guarantees.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CHUNK_SIZE,
    DEV_SAMPLE_SIZE,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    TEST_SIZE,
)
from src.data.cleaner import (
    clean_numerical_anomalies,
    normalize_labels,
    standardize_columns,
)

logger = logging.getLogger(__name__)


class FlowPreprocessor:
    """End-to-end network flow preprocessing engine for CICIDS2017."""

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        chunk_size: int = CHUNK_SIZE,
        random_seed: int = RANDOM_SEED,
        test_size: float = TEST_SIZE,
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else RAW_DATA_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DATA_DIR
        self.chunk_size = chunk_size
        self.random_seed = random_seed
        self.test_size = test_size
        self.rate_bounds: Dict[str, float] = {}

    def clean_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Cleans a single chunk of network flows."""
        # 1. Standardize column names
        chunk = standardize_columns(chunk)

        # 2. Normalize and encode labels
        chunk = normalize_labels(chunk)

        # 3. Clean numerical anomalies and bounds
        chunk, bounds = clean_numerical_anomalies(chunk, rate_bounds=self.rate_bounds if self.rate_bounds else None)
        if not self.rate_bounds:
            self.rate_bounds = bounds

        return chunk

    def load_and_clean_file(self, file_path: Path) -> pd.DataFrame:
        """Loads a single CSV file using chunked streaming and cleans each chunk."""
        logger.info("Processing file: %s", file_path.name)
        chunks = []
        total_rows = 0

        # Read in chunks with fallback encodings
        for chunk in pd.read_csv(file_path, chunksize=self.chunk_size, encoding="cp1252", low_memory=False):
            cleaned = self.clean_chunk(chunk)
            chunks.append(cleaned)
            total_rows += len(cleaned)

        combined = pd.concat(chunks, ignore_index=True)
        logger.info("Loaded %d rows from %s", len(combined), file_path.name)
        return combined

    def deduplicate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Removes exact duplicate rows across all features and target labels to prevent train/test leakage.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (deduplicated_df, count_of_dropped_duplicates)
        """
        initial_count = len(df)
        df_deduped = df.drop_duplicates(keep="first").reset_index(drop=True)
        dropped_count = initial_count - len(df_deduped)
        logger.info("Deduplication: Dropped %d duplicate rows (%.2f%%)", dropped_count, (dropped_count / initial_count) * 100 if initial_count > 0 else 0)
        return df_deduped, dropped_count

    def separate_features_and_targets(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Separates network flow features (X) from target labels (y).

        Args:
            df: Cleaned DataFrame.

        Returns:
            Tuple of (X, y_df)
        """
        target_columns = ["label_raw", "label_binary", "label_category"]
        existing_targets = [c for c in target_columns if c in df.columns]

        X = df.drop(columns=existing_targets)
        y = df[existing_targets].copy()
        return X, y

    def split_train_test(
        self,
        df: pd.DataFrame,
        stratify_col: str = "label_category",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Performs reproducible, leakage-free stratified train/test split.

        Args:
            df: Deduplicated DataFrame.
            stratify_col: Target column name to preserve class proportions.

        Returns:
            Tuple of (train_df, test_df)
        """
        stratify_target = df[stratify_col] if stratify_col in df.columns else None

        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=stratify_target,
            shuffle=True,
        )

        logger.info(
            "Split complete: Train=%d (%.1f%%), Test=%d (%.1f%%)",
            len(train_df),
            (1 - self.test_size) * 100,
            len(test_df),
            self.test_size * 100,
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def create_dev_sample(
        self,
        df: pd.DataFrame,
        sample_size: int = DEV_SAMPLE_SIZE,
        stratify_col: str = "label_category",
    ) -> pd.DataFrame:
        """Creates a representative stratified development subset for rapid model iteration on local machines.

        Args:
            df: Input DataFrame (e.g., training split).
            sample_size: Desired number of samples.
            stratify_col: Column for stratified sampling.

        Returns:
            Sampled DataFrame.
        """
        if len(df) <= sample_size:
            return df.copy()

        # Compute sampling fraction
        fraction = sample_size / len(df)
        _, sample_df = train_test_split(
            df,
            test_size=fraction,
            random_state=self.random_seed,
            stratify=df[stratify_col] if stratify_col in df.columns else None,
            shuffle=True,
        )
        logger.info("Created stratified dev sample with %d records.", len(sample_df))
        return sample_df.reset_index(drop=True)
