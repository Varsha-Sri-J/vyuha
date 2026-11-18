"""Unit Tests for VYUHA Feature Selection, Pipeline, and Leakage Prevention.

Verifies:
1. Strict target column exclusion.
2. Constant and zero-variance feature pruning.
3. Exact duplicate and collinear twin feature identification.
4. Leakage-free fit/transform execution (trained strictly on training fold).
5. Full vs Port-Agnostic feature sets.
6. Classification and anomaly pipeline construction.
7. Processed dataset immutability.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.models.feature_selector import (
    FeatureSelector,
    extract_features_and_targets,
    get_port_agnostic_features,
)
from src.models.pipeline import (
    build_classification_pipeline,
    build_anomaly_pipeline,
)
from src.models.evaluate import LeakageSafeCrossValidator


@pytest.fixture
def synthetic_flow_dataframe() -> pd.DataFrame:
    """Generates synthetic flow data with known properties."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "Destination Port": [80] * 50 + [443] * 50,
        "Flow Duration": np.random.exponential(1000, size=n),
        "Total Fwd Packets": np.random.randint(1, 20, size=n),
        "Total Backward Packets": np.random.randint(0, 15, size=n),
        "Total Length of Fwd Packets": np.random.randint(50, 2000, size=n),
        # Exact duplicate of Total Length of Fwd Packets:
        "Subflow Fwd Bytes": 0,  # Will be set below
        # Zero-variance constant feature:
        "Bwd PSH Flags": [0.0] * n,
        # Another constant feature:
        "Fwd Avg Bulk Rate": [0.0] * n,
        # Informative features:
        "Flow Bytes/s": np.random.uniform(100, 50000, size=n),
        "PSH Flag Count": np.random.choice([0.0, 1.0], size=n),
        # Target columns:
        "label_raw": ["BENIGN"] * 80 + ["DoS Hulk"] * 20,
        "label_binary": [0] * 80 + [1] * 20,
        "label_category": ["BENIGN"] * 80 + ["DoS_DDoS"] * 20,
    })
    df["Subflow Fwd Bytes"] = df["Total Length of Fwd Packets"]  # Exact duplicate
    return df


def test_extract_features_and_targets_strict_isolation(synthetic_flow_dataframe):
    """Verifies that all target columns are strictly separated and cannot enter X."""
    X, y_dict = extract_features_and_targets(synthetic_flow_dataframe)

    assert "label_raw" not in X.columns
    assert "label_binary" not in X.columns
    assert "label_category" not in X.columns
    assert "binary" in y_dict
    assert "category" in y_dict
    assert "raw" in y_dict

    # Check that no column in X starts with 'label_'
    for col in X.columns:
        assert not col.startswith("label_")


def test_zero_variance_removal(synthetic_flow_dataframe):
    """Verifies that constant features are identified and removed during fit()."""
    X, _ = extract_features_and_targets(synthetic_flow_dataframe)
    selector = FeatureSelector(variance_threshold=0.0, drop_collinear_twins=False)
    selector.fit(X)

    # Bwd PSH Flags and Fwd Avg Bulk Rate must be dropped
    assert "Bwd PSH Flags" in selector.dropped_features_
    assert "Fwd Avg Bulk Rate" in selector.dropped_features_
    assert "Bwd PSH Flags" not in selector.selected_features_
    assert "Fwd Avg Bulk Rate" not in selector.selected_features_

    X_transformed = selector.transform(X)
    assert "Bwd PSH Flags" not in X_transformed.columns


def test_collinear_twin_removal(synthetic_flow_dataframe):
    """Verifies that exact subflow duplicates are identified and pruned."""
    X, _ = extract_features_and_targets(synthetic_flow_dataframe)
    selector = FeatureSelector(drop_collinear_twins=True)
    selector.fit(X)

    # Subflow Fwd Bytes is a known exact twin of Total Length of Fwd Packets
    assert "Subflow Fwd Bytes" in selector.dropped_features_
    assert "Total Length of Fwd Packets" in selector.selected_features_

    X_transformed = selector.transform(X)
    assert "Subflow Fwd Bytes" not in X_transformed.columns
    assert "Total Length of Fwd Packets" in X_transformed.columns


def test_port_agnostic_feature_set(synthetic_flow_dataframe):
    """Verifies that exclude_ports=True prunes Destination Port."""
    X, _ = extract_features_and_targets(synthetic_flow_dataframe)

    # Full feature set
    selector_full = FeatureSelector(exclude_ports=False)
    selector_full.fit(X)
    assert "Destination Port" in selector_full.selected_features_

    # Port-agnostic feature set
    selector_port_agnostic = FeatureSelector(exclude_ports=True)
    selector_port_agnostic.fit(X)
    assert "Destination Port" in selector_port_agnostic.dropped_features_
    assert "Destination Port" not in selector_port_agnostic.selected_features_

    X_agnostic = selector_port_agnostic.transform(X)
    assert "Destination Port" not in X_agnostic.columns


def test_feature_selection_fitted_only_on_training_fold():
    """Verifies that feature selection learns only from the training fold without leakage.

    Constructs a feature that is CONSTANT in training fold, but HAS VARIANCE in validation fold.
    The selector must drop it because it only learns from the training slice.
    """
    X_train = pd.DataFrame({
        "feature_stable": [1.0, 2.0, 3.0, 4.0],
        "feature_split_dependent": [0.0, 0.0, 0.0, 0.0],  # Constant in train!
    })
    X_val = pd.DataFrame({
        "feature_stable": [5.0, 6.0],
        "feature_split_dependent": [10.0, 20.0],  # Varies in val!
    })

    selector = FeatureSelector(variance_threshold=0.0)
    selector.fit(X_train)

    # Must be dropped based solely on X_train
    assert "feature_split_dependent" in selector.dropped_features_

    # Transform on X_val must respect the train-learned mask
    X_val_trans = selector.transform(X_val)
    assert list(X_val_trans.columns) == ["feature_stable"]


def test_pipeline_integration_and_cross_validation(synthetic_flow_dataframe):
    """Verifies end-to-end integration with LeakageSafeCrossValidator."""
    X, y_dict = extract_features_and_targets(synthetic_flow_dataframe)
    y = y_dict["binary"]

    # Construct pipeline with dummy classifier
    pipeline = build_classification_pipeline(
        classifier=DummyClassifier(strategy="most_frequent"),
        exclude_ports=False,
    )

    validator = LeakageSafeCrossValidator(n_splits=3, random_state=42)
    summary = validator.evaluate_pipeline(pipeline, X, y)

    assert "mean_f1_macro" in summary
    assert "feature_selection_stability" in summary
    assert summary["n_splits"] == 3
    # Check that feature stability is recorded
    assert summary["feature_selection_stability"] == 1.0


def test_unsupervised_anomaly_pipeline():
    """Verifies anomaly detection pipeline construction for unscaled baseline and scaled variants."""
    pipe_unscaled = build_anomaly_pipeline(scaler=None)
    assert "selector" in pipe_unscaled.named_steps
    assert "scaler" not in pipe_unscaled.named_steps

    from sklearn.preprocessing import RobustScaler
    pipe_scaled = build_anomaly_pipeline(scaler=RobustScaler())
    assert "selector" in pipe_scaled.named_steps
    assert "scaler" in pipe_scaled.named_steps


def test_processed_dataset_immutability():
    """Verifies that the underlying processed parquet files have not been modified or corrupted."""
    processed_dir = Path("datasets/processed")
    if not processed_dir.exists():
        processed_dir = Path("../datasets/processed")

    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"
    dev_path = processed_dir / "dev_sample.parquet"

    assert train_path.exists(), "train.parquet missing!"
    assert test_path.exists(), "test.parquet missing!"
    assert dev_path.exists(), "dev_sample.parquet missing!"

    # Check record counts match baseline
    dev_df = pd.read_parquet(dev_path)
    assert len(dev_df) == 200000, f"dev_sample altered! Expected 200,000, got {len(dev_df)}"
    assert len(dev_df.columns) == 80, f"dev_sample columns altered! Expected 80, got {len(dev_df.columns)}"
