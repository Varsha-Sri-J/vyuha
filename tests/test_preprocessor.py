"""Unit and Integration Tests for VYUHA Preprocessing Pipeline.

Validates column cleaning, label mapping, sentinel preservation,
infinite/null imputation, deduplication, and stratified train/test splitting.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import (
    clean_numerical_anomalies,
    normalize_labels,
    standardize_columns,
)
from src.data.preprocessor import FlowPreprocessor


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """Creates a synthetic DataFrame mimicking raw CICIDS2017 flow records."""
    data = {
        " Destination Port": [80, 22, 443, 80, 80, 21, 80],
        " Flow Duration": [1000, 500, -13, 0, 0, 2000, 1000],  # Includes negative and zero
        " Total Fwd Packets": [2, 5, 1, 10, 10, 3, 2],
        " Total Backward Packets": [2, 4, 0, 0, 0, 2, 2],
        "Total Length of Fwd Packets": [100, 250, 0, 500, 500, 150, 100],
        " Total Length of Bwd Packets": [200, 300, 0, 0, 0, 100, 200],
        "Flow Bytes/s": ["150000.0", "1100000.0", "0.0", "Infinity", np.nan, "125000.0", "150000.0"],
        " Flow Packets/s": [4000.0, 18000.0, 0.0, np.inf, 20000.0, 2500.0, 4000.0],
        "Fwd Header Length": [40, 100, 20, 200, 200, 60, 40],
        "Fwd Header Length.1": [40, 100, 20, 200, 200, 60, 40],  # Duplicate column
        "Init_Win_bytes_forward": [29200, -1, 65535, 14600, 14600, -1, 29200],  # Sentinel -1
        " Init_Win_bytes_backward": [-1, 28960, 65535, -1, -1, 5840, -1],  # Sentinel -1
        " Label": [
            "BENIGN",
            "SSH-Patator",
            "BENIGN",
            "DoS Hulk",
            "DoS Hulk",
            "FTP-Patator",
            "BENIGN",  # Row 6 is an exact duplicate of Row 0
        ],
    }
    return pd.DataFrame(data)


def test_standardize_columns(sample_raw_dataframe):
    """Verifies that headers are stripped of whitespace and duplicate column is removed."""
    cleaned = standardize_columns(sample_raw_dataframe)
    
    assert "Destination Port" in cleaned.columns
    assert "Flow Duration" in cleaned.columns
    assert "Label" in cleaned.columns
    assert "Fwd Header Length.1" not in cleaned.columns
    # Check no column name has leading/trailing whitespace
    for col in cleaned.columns:
        assert col == col.strip()


def test_normalize_labels():
    """Verifies binary and multi-class categorization across attack scenarios."""
    df = pd.DataFrame({
        "Label": [
            "BENIGN",
            "DoS Hulk",
            "DDoS",
            "PortScan",
            "FTP-Patator",
            "SSH-Patator",
            "Web Attack  Brute Force",
            "Web Attack – XSS",
            "Bot",
            "Infiltration",
            "Heartbleed",
        ]
    })
    
    normalized = normalize_labels(df)
    
    assert "Label" not in normalized.columns
    assert "label_raw" in normalized.columns
    assert "label_binary" in normalized.columns
    assert "label_category" in normalized.columns
    
    # Binary checks
    assert normalized.loc[0, "label_binary"] == 0
    assert (normalized.loc[1:, "label_binary"] == 1).all()
    
    # Category checks
    assert normalized.loc[0, "label_category"] == "BENIGN"
    assert normalized.loc[1, "label_category"] == "DoS_DDoS"
    assert normalized.loc[2, "label_category"] == "DoS_DDoS"
    assert normalized.loc[3, "label_category"] == "PortScan"
    assert normalized.loc[4, "label_category"] == "BruteForce"
    assert normalized.loc[5, "label_category"] == "BruteForce"
    assert normalized.loc[6, "label_category"] == "WebAttack"
    assert normalized.loc[7, "label_category"] == "WebAttack"
    assert normalized.loc[8, "label_category"] == "Botnet"
    assert normalized.loc[9, "label_category"] == "Infiltration"
    assert normalized.loc[10, "label_category"] == "Heartbleed"


def test_clean_numerical_anomalies_negative_duration(sample_raw_dataframe):
    """Verifies that negative Flow Duration rows are filtered out."""
    std_df = standardize_columns(sample_raw_dataframe)
    cleaned_df, _ = clean_numerical_anomalies(std_df)
    
    # Row with Flow Duration -13 should be removed
    assert (cleaned_df["Flow Duration"] >= 0).all()
    assert len(cleaned_df) == len(sample_raw_dataframe) - 1


def test_sentinel_preservation(sample_raw_dataframe):
    """Verifies that -1 in Init_Win_bytes_forward and backward is preserved as discrete sentinel."""
    std_df = standardize_columns(sample_raw_dataframe)
    cleaned_df, _ = clean_numerical_anomalies(std_df)
    
    assert -1 in cleaned_df["Init_Win_bytes_forward"].values
    assert -1 in cleaned_df["Init_Win_bytes_backward"].values


def test_infinities_and_nulls_imputed(sample_raw_dataframe):
    """Verifies that infinity and NaN values in rate columns are cleanly handled."""
    std_df = standardize_columns(sample_raw_dataframe)
    cleaned_df, _ = clean_numerical_anomalies(std_df)
    
    for col in ["Flow Bytes/s", "Flow Packets/s"]:
        assert not cleaned_df[col].isna().any(), f"{col} has NaNs"
        assert not np.isinf(cleaned_df[col]).any(), f"{col} has infinities"
        assert (cleaned_df[col] >= 0).all(), f"{col} has negative values"


def test_deduplication(sample_raw_dataframe):
    """Verifies that exact duplicate records are identified and removed."""
    preprocessor = FlowPreprocessor()
    std_df = standardize_columns(sample_raw_dataframe)
    labeled_df = normalize_labels(std_df)
    cleaned_df, _ = clean_numerical_anomalies(labeled_df)
    
    # In sample_raw_dataframe, row 0 and row 6 are identical
    deduped_df, dropped = preprocessor.deduplicate(cleaned_df)
    
    assert dropped == 1
    assert len(deduped_df) == len(cleaned_df) - 1


def test_separate_features_and_targets(sample_raw_dataframe):
    """Verifies clean separation of X (features) and y (targets)."""
    preprocessor = FlowPreprocessor()
    std_df = standardize_columns(sample_raw_dataframe)
    labeled_df = normalize_labels(std_df)
    cleaned_df, _ = clean_numerical_anomalies(labeled_df)
    
    X, y = preprocessor.separate_features_and_targets(cleaned_df)
    
    assert "label_raw" not in X.columns
    assert "label_binary" not in X.columns
    assert "label_category" not in X.columns
    assert "label_binary" in y.columns
    assert "label_category" in y.columns
    assert len(X) == len(y)


def test_stratified_train_test_split():
    """Verifies reproducible stratified split preserving category ratios."""
    # Create synthetic dataset with balanced classes
    np.random.seed(42)
    categories = ["BENIGN"] * 80 + ["DoS_DDoS"] * 40 + ["PortScan"] * 30 + ["BruteForce"] * 20
    df = pd.DataFrame({
        "feature_1": np.random.randn(len(categories)),
        "feature_2": np.random.randint(0, 100, size=len(categories)),
        "label_category": categories,
        "label_binary": [0 if c == "BENIGN" else 1 for c in categories],
    })
    
    preprocessor = FlowPreprocessor(random_seed=42, test_size=0.2)
    train_df, test_df = preprocessor.split_train_test(df, stratify_col="label_category")
    
    assert len(train_df) == int(len(df) * 0.8)
    assert len(test_df) == int(len(df) * 0.2)
    
    # Check category distribution proportions match within 2%
    for cat in ["BENIGN", "DoS_DDoS", "PortScan", "BruteForce"]:
        train_prop = (train_df["label_category"] == cat).mean()
        test_prop = (test_df["label_category"] == cat).mean()
        assert abs(train_prop - test_prop) < 0.05
