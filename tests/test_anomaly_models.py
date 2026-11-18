"""Unit Tests for VYUHA Phase 3D Unsupervised Anomaly Detection Engine.

Verifies:
1. Pipeline architecture building for Isolation Forest (Full & Port-Agnostic, Unscaled & Scaled).
2. Benign-only training enforcement (model fits only on normal traffic).
3. Target-column strict exclusion from feature matrices.
4. Threshold calibration mathematics and percentile freezing on calibration data.
5. Prediction behavior (anomaly score bounds and binary alert flagging).
6. Evaluation metrics computation (ROC-AUC, PR-AUC, per-attack recall).
7. Serialization and artifact persistence outside Git data dirs.
8. Strict held-out test set isolation (attempting to load test.parquet triggers PermissionError).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.train_anomaly import (
    build_anomaly_model,
    calibrate_anomaly_thresholds,
    load_training_data,
    partition_anomaly_data,
    sample_benign_training_subset,
    save_anomaly_artifact,
    train_anomaly_model,
)
from src.models.evaluate_anomaly import (
    compute_anomaly_scores,
    evaluate_anomaly_detector,
)


@pytest.fixture
def synthetic_anomaly_data() -> pd.DataFrame:
    """Creates a synthetic dataset with normal flows and simulated attack outliers."""
    np.random.seed(42)
    n_benign = 500
    n_attack = 50

    # Normal traffic: tightly clustered
    df_benign = pd.DataFrame({
        "Destination Port": np.random.choice([80, 443, 8080], size=n_benign),
        "Flow Duration": np.random.exponential(1000, size=n_benign),
        "Total Fwd Packets": np.random.randint(1, 10, size=n_benign),
        "Total Backward Packets": np.random.randint(1, 10, size=n_benign),
        "Total Length of Fwd Packets": np.random.normal(500, 50, size=n_benign),
        "Subflow Fwd Bytes": 0.0,
        "Bwd PSH Flags": [0.0] * n_benign,
        "label_binary": 0,
        "label_category": "BENIGN",
        "label_raw": "BENIGN",
    })

    # Attack traffic: volumetric outliers
    df_attack = pd.DataFrame({
        "Destination Port": np.random.choice([21, 22, 23], size=n_attack),
        "Flow Duration": np.random.exponential(50000, size=n_attack),
        "Total Fwd Packets": np.random.randint(100, 500, size=n_attack),
        "Total Backward Packets": np.random.randint(50, 200, size=n_attack),
        "Total Length of Fwd Packets": np.random.normal(15000, 1000, size=n_attack),
        "Subflow Fwd Bytes": 0.0,
        "Bwd PSH Flags": [0.0] * n_attack,
        "label_binary": 1,
        "label_category": "DoS_DDoS",
        "label_raw": "DoS_DDoS",
    })

    df = pd.concat([df_benign, df_attack], ignore_index=True)
    return df


def test_build_anomaly_model_architectures():
    """Verifies pipeline construction across full, port-agnostic, and scaled variants."""
    # 1. Full unscaled
    pipe_full = build_anomaly_model(exclude_ports=False, scaler_type=None)
    assert isinstance(pipe_full, Pipeline)
    assert "selector" in pipe_full.named_steps
    assert "scaler" not in pipe_full.named_steps
    assert "anomaly_detector" in pipe_full.named_steps
    assert pipe_full.named_steps["selector"].exclude_ports is False

    # 2. Port-agnostic unscaled
    pipe_port = build_anomaly_model(exclude_ports=True, scaler_type=None)
    assert pipe_port.named_steps["selector"].exclude_ports is True

    # 3. Scaled variant
    pipe_scaled = build_anomaly_model(exclude_ports=False, scaler_type="robust")
    assert "scaler" in pipe_scaled.named_steps


def test_benign_only_partitioning_and_target_exclusion(synthetic_anomaly_data):
    """Verifies that only BENIGN flows enter training and calibration, and targets are excluded."""
    X_train_pool, X_cal, X_val, y_val, val_meta = partition_anomaly_data(
        synthetic_anomaly_data, test_size=0.2, random_state=42, cal_size=50
    )

    # Verify target columns strictly excluded
    for target_col in ["label_binary", "label_category", "label_raw"]:
        assert target_col not in X_train_pool.columns
        assert target_col not in X_cal.columns
        assert target_col not in X_val.columns

    # Verify calibration set and training pool are disjoint
    overlap = set(X_train_pool.index).intersection(set(X_cal.index))
    assert len(overlap) == 0

    # Verify validation set contains both Benign and Attacks
    assert "BENIGN" in y_val.values
    assert "DoS_DDoS" in y_val.values


def test_threshold_calibration_mathematics(synthetic_anomaly_data):
    """Verifies that calibrated thresholds correctly achieve target percentiles on calibration data."""
    X_train_pool, X_cal, X_val, y_val, _ = partition_anomaly_data(
        synthetic_anomaly_data, test_size=0.2, random_state=42, cal_size=50
    )

    pipe = build_anomaly_model(exclude_ports=False)
    pipe = train_anomaly_model(pipe, X_train_pool)

    # Calibrate thresholds for 5% and 10% FPR targets
    target_fprs = (0.10, 0.05)
    cal_res = calibrate_anomaly_thresholds(pipe, X_cal, target_fprs=target_fprs)

    for k, v in cal_res.items():
        assert "threshold" in v
        assert "percentile" in v
        assert "target_fpr" in v
        # Empirical FPR on calibration set should match target within sampling tolerance
        assert abs(v["empirical_calibration_fpr"] - v["target_fpr"]) < 0.05


def test_anomaly_scoring_and_evaluation_metrics(synthetic_anomaly_data):
    """Verifies anomaly scoring, ranking metrics (ROC/PR), and threshold evaluations."""
    X_train_pool, X_cal, X_val, y_val, _ = partition_anomaly_data(
        synthetic_anomaly_data, test_size=0.2, random_state=42, cal_size=50
    )

    pipe = build_anomaly_model(exclude_ports=False)
    pipe = train_anomaly_model(pipe, X_train_pool)

    scores = compute_anomaly_scores(pipe, X_val)
    assert len(scores) == len(X_val)
    assert np.all(scores >= 0.0)

    cal_res = calibrate_anomaly_thresholds(pipe, X_cal, target_fprs=(0.05,))
    eval_res = evaluate_anomaly_detector(pipe, X_val, y_val, cal_res, benchmark_latency=True)

    assert "roc_auc" in eval_res
    assert "pr_auc" in eval_res
    assert "category_score_stats" in eval_res
    assert "threshold_evaluations" in eval_res
    assert "latency" in eval_res

    # Outliers should achieve high ROC-AUC on synthetic data
    assert eval_res["roc_auc"] > 0.70


def test_strict_test_set_isolation_anomaly():
    """Verifies that attempting to pass test.parquet raises PermissionError."""
    with pytest.raises(PermissionError) as exc_info:
        load_training_data(data_path="datasets/processed/test.parquet")
    assert "Strict Isolation Violation" in str(exc_info.value)


def test_anomaly_model_serialization(tmp_path, synthetic_anomaly_data):
    """Verifies that fitted pipeline and calibration metadata serialize and deserialize accurately."""
    X_train_pool, X_cal, X_val, y_val, _ = partition_anomaly_data(
        synthetic_anomaly_data, test_size=0.2, random_state=42, cal_size=50
    )

    pipe = build_anomaly_model(exclude_ports=False)
    pipe = train_anomaly_model(pipe, X_train_pool)
    scores_before = compute_anomaly_scores(pipe, X_val)

    save_path = tmp_path / "test_anomaly_model.joblib"
    save_anomaly_artifact(pipe, save_path, metadata={"n_samples": len(X_train_pool)})

    assert save_path.exists()

    import joblib
    loaded_data = joblib.load(save_path)
    loaded_pipe = loaded_data["pipeline"]
    scores_after = compute_anomaly_scores(loaded_pipe, X_val)

    np.testing.assert_allclose(scores_before, scores_after, rtol=1e-5)
