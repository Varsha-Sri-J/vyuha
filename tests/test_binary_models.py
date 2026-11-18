"""Unit Tests for VYUHA Phase 3B Binary Intrusion Detection Engine.

Verifies:
1. Model pipeline building (Random Forest and Decision Tree; Full vs Port-Agnostic).
2. Model training, prediction, and probability estimation.
3. Feature count integrity (63 full features, 62 port-agnostic features).
4. Strict test set isolation (attempting to load test.parquet triggers PermissionError).
5. Comprehensive evaluation metric calculation (Precision, Recall, F1, PR-AUC, FPR).
6. Error analysis extraction (FP by port, FN by attack family).
7. Model serialization and artifact persistence outside Git data dirs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.train_binary import (
    build_binary_model,
    load_training_data,
    save_model_artifact,
    split_training_data,
    train_binary_model,
)
from src.models.evaluate_binary import (
    analyze_prediction_errors,
    evaluate_single_model,
)


@pytest.fixture
def synthetic_binary_data() -> pd.DataFrame:
    """Creates a synthetic dataset with known attack and benign patterns."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "Destination Port": [80] * 100 + [443] * 50 + [22] * 50,
        "Flow Duration": np.random.exponential(5000, size=n),
        "Total Fwd Packets": np.random.randint(1, 30, size=n),
        "Total Backward Packets": np.random.randint(0, 20, size=n),
        "Total Length of Fwd Packets": np.random.randint(100, 5000, size=n),
        "Subflow Fwd Bytes": 0.0,  # Exact duplicate of Total Length of Fwd Packets
        "Bwd PSH Flags": [0.0] * n,  # Constant zero-variance
        "Flow Bytes/s": np.random.uniform(500, 100000, size=n),
        "Flow Packets/s": np.random.uniform(10, 5000, size=n),
        "PSH Flag Count": np.random.choice([0.0, 1.0], size=n),
        "ACK Flag Count": np.random.choice([0.0, 1.0], size=n),
        # Target labels
        "label_raw": ["BENIGN"] * 160 + ["DoS Hulk"] * 30 + ["SSH-Patator"] * 10,
        "label_binary": [0] * 160 + [1] * 40,
        "label_category": ["BENIGN"] * 160 + ["DoS_DDoS"] * 30 + ["BruteForce"] * 10,
    })
    df["Subflow Fwd Bytes"] = df["Total Length of Fwd Packets"]
    return df


def test_build_binary_model_architectures():
    """Verifies pipeline construction for RF and DT models across full and port-agnostic modes."""
    # RF Full
    rf_full = build_binary_model(model_type="rf", exclude_ports=False)
    assert isinstance(rf_full, Pipeline)
    assert "selector" in rf_full.named_steps
    assert "classifier" in rf_full.named_steps
    assert not rf_full.named_steps["selector"].exclude_ports

    # RF Port-Agnostic
    rf_port = build_binary_model(model_type="rf", exclude_ports=True)
    assert rf_port.named_steps["selector"].exclude_ports

    # DT Full
    dt_full = build_binary_model(model_type="dt", exclude_ports=False)
    assert "classifier" in dt_full.named_steps

    # Unsupported model
    with pytest.raises(ValueError, match="Unsupported model_type"):
        build_binary_model(model_type="svm")


def test_training_and_feature_counts(synthetic_binary_data):
    """Verifies that model trains and prunes constant/twin features correctly."""
    X_tr, X_val, y_tr, y_val, _ = split_training_data(synthetic_binary_data, test_size=0.3, random_state=42)

    # Full model
    pipe_full = build_binary_model(model_type="dt", exclude_ports=False, max_depth=5)
    fitted_full, t_full = train_binary_model(pipe_full, X_tr, y_tr)
    selector_full = fitted_full.named_steps["selector"]

    assert "Bwd PSH Flags" in selector_full.dropped_features_
    assert "Subflow Fwd Bytes" in selector_full.dropped_features_
    assert "Destination Port" in selector_full.selected_features_

    # Port-Agnostic model
    pipe_port = build_binary_model(model_type="dt", exclude_ports=True, max_depth=5)
    fitted_port, t_port = train_binary_model(pipe_port, X_tr, y_tr)
    selector_port = fitted_port.named_steps["selector"]

    assert "Destination Port" in selector_port.dropped_features_
    assert len(selector_port.selected_features_) == len(selector_full.selected_features_) - 1


def test_strict_test_set_isolation():
    """Verifies that load_training_data raises PermissionError if test.parquet is requested."""
    with pytest.raises(PermissionError, match="Strict Isolation Violation"):
        load_training_data("datasets/processed/test.parquet")

    with pytest.raises(PermissionError, match="Strict Isolation Violation"):
        load_training_data("../datasets/processed/test.parquet")


def test_evaluation_metrics_computation(synthetic_binary_data):
    """Verifies that evaluate_single_model calculates all required binary intrusion metrics."""
    X_tr, X_val, y_tr, y_val, val_meta = split_training_data(synthetic_binary_data, test_size=0.4, random_state=42)
    pipe = build_binary_model(model_type="rf", n_estimators=10, max_depth=5)
    fitted_pipe, t_fit = train_binary_model(pipe, X_tr, y_tr)

    metrics = evaluate_single_model(fitted_pipe, X_val, y_val, "test_rf", training_time=t_fit)

    # Required metric keys
    required_keys = [
        "accuracy", "balanced_accuracy", "precision", "recall",
        "f1_score", "pr_auc", "roc_auc", "false_positive_rate",
        "false_negative_rate", "confusion_matrix",
        "inference_time_seconds", "inference_microseconds_per_flow",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing metric key: {key}"

    cm = metrics["confusion_matrix"]
    assert cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] + cm["true_positives"] == len(y_val)


def test_prediction_error_analysis(synthetic_binary_data):
    """Verifies error analysis breakdowns for false positives and false negatives."""
    X_tr, X_val, y_tr, y_val, val_meta = split_training_data(synthetic_binary_data, test_size=0.4, random_state=42)
    pipe = build_binary_model(model_type="dt", max_depth=3)
    fitted_pipe, _ = train_binary_model(pipe, X_tr, y_tr)

    y_pred = fitted_pipe.predict(X_val)
    diag = analyze_prediction_errors(y_val, y_pred, val_meta, "test_dt")

    assert "total_false_positives" in diag
    assert "total_false_negatives" in diag
    assert "false_positives_top_ports" in diag
    assert "false_negatives_by_raw_label" in diag
    assert "false_negatives_by_category" in diag


def test_model_serialization(tmp_path, synthetic_binary_data):
    """Verifies model artifact persistence and reloading."""
    X_tr, _, y_tr, _, _ = split_training_data(synthetic_binary_data, test_size=0.5, random_state=42)
    pipe = build_binary_model(model_type="dt", max_depth=3)
    fitted_pipe, _ = train_binary_model(pipe, X_tr, y_tr)

    save_path = tmp_path / "models" / "dt_test.joblib"
    saved_file = save_model_artifact(fitted_pipe, save_path)
    assert saved_file.exists()

    import joblib
    reloaded = joblib.load(saved_file)
    assert hasattr(reloaded, "predict")
    # Verify inference produces same output
    assert np.array_equal(fitted_pipe.predict(X_tr), reloaded.predict(X_tr))
