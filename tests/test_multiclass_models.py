"""Unit Tests for VYUHA Phase 3C Multiclass Attack Family Detection Engine.

Verifies:
1. Pipeline architecture building for multiclass models (RF & DT; Full & Port-Agnostic).
2. Multiclass training across all 8 canonical categories.
3. Strict test set isolation (attempting to load test.parquet triggers PermissionError).
4. Full metric calculations (Macro F1, Balanced Accuracy, Per-Class Precision/Recall/F1/Support).
5. Error analysis extraction (attacks misclassified as BENIGN, inter-attack confusion).
6. Model serialization and artifact persistence outside Git data dirs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.train_multiclass import (
    CANONICAL_CATEGORIES,
    build_multiclass_model,
    load_training_data,
    save_model_artifact,
    split_multiclass_data,
    train_multiclass_model,
)
from src.models.evaluate_multiclass import (
    analyze_multiclass_errors,
    evaluate_single_multiclass_model,
)


@pytest.fixture
def synthetic_multiclass_data() -> pd.DataFrame:
    """Creates a synthetic dataset with records for all 8 canonical categories."""
    np.random.seed(42)
    # Ensure all 8 categories have at least 2 samples for stratified splitting
    cat_counts = {
        "BENIGN": 100,
        "DoS_DDoS": 30,
        "PortScan": 20,
        "BruteForce": 10,
        "WebAttack": 10,
        "Botnet": 10,
        "Infiltration": 4,
        "Heartbleed": 4,
    }
    total_n = sum(cat_counts.values())

    categories_list = []
    for cat, count in cat_counts.items():
        categories_list.extend([cat] * count)

    df = pd.DataFrame({
        "Destination Port": np.random.choice([80, 443, 21, 22, 8080], size=total_n),
        "Flow Duration": np.random.exponential(5000, size=total_n),
        "Total Fwd Packets": np.random.randint(1, 30, size=total_n),
        "Total Backward Packets": np.random.randint(0, 20, size=total_n),
        "Total Length of Fwd Packets": np.random.randint(100, 5000, size=total_n),
        "Subflow Fwd Bytes": 0.0,
        "Bwd PSH Flags": [0.0] * total_n,
        "Flow Bytes/s": np.random.uniform(500, 100000, size=total_n),
        "Flow Packets/s": np.random.uniform(10, 5000, size=total_n),
        "PSH Flag Count": np.random.choice([0.0, 1.0], size=total_n),
        "ACK Flag Count": np.random.choice([0.0, 1.0], size=total_n),
        # Target labels
        "label_raw": categories_list,
        "label_binary": [0 if c == "BENIGN" else 1 for c in categories_list],
        "label_category": categories_list,
    })
    df["Subflow Fwd Bytes"] = df["Total Length of Fwd Packets"]
    return df


def test_build_multiclass_model_architectures():
    """Verifies pipeline construction for multiclass RF and DT models."""
    rf_full = build_multiclass_model(model_type="rf", exclude_ports=False)
    assert isinstance(rf_full, Pipeline)
    assert "selector" in rf_full.named_steps
    assert "classifier" in rf_full.named_steps
    assert not rf_full.named_steps["selector"].exclude_ports

    rf_port = build_multiclass_model(model_type="rf", exclude_ports=True)
    assert rf_port.named_steps["selector"].exclude_ports

    dt_full = build_multiclass_model(model_type="dt", exclude_ports=False)
    assert "classifier" in dt_full.named_steps

    with pytest.raises(ValueError, match="Unsupported model_type"):
        build_multiclass_model(model_type="svm")


def test_multiclass_training_and_feature_counts(synthetic_multiclass_data):
    """Verifies multiclass training, label representation, and feature counts."""
    X_tr, X_val, y_tr, y_val, _ = split_multiclass_data(synthetic_multiclass_data, test_size=0.3, random_state=42)

    # All 8 categories represented
    assert len(np.unique(y_tr)) == len(CANONICAL_CATEGORIES)

    # Full model
    pipe_full = build_multiclass_model(model_type="dt", exclude_ports=False, max_depth=5)
    fitted_full, t_full = train_multiclass_model(pipe_full, X_tr, y_tr)
    selector_full = fitted_full.named_steps["selector"]

    assert "Bwd PSH Flags" in selector_full.dropped_features_
    assert "Subflow Fwd Bytes" in selector_full.dropped_features_
    assert "Destination Port" in selector_full.selected_features_

    # Port-Agnostic model
    pipe_port = build_multiclass_model(model_type="dt", exclude_ports=True, max_depth=5)
    fitted_port, t_port = train_multiclass_model(pipe_port, X_tr, y_tr)
    selector_port = fitted_port.named_steps["selector"]

    assert "Destination Port" in selector_port.dropped_features_
    assert len(selector_port.selected_features_) == len(selector_full.selected_features_) - 1


def test_strict_test_set_isolation_multiclass():
    """Verifies that load_training_data raises PermissionError if test.parquet is requested."""
    with pytest.raises(PermissionError, match="Strict Isolation Violation"):
        load_training_data("datasets/processed/test.parquet")

    with pytest.raises(PermissionError, match="Strict Isolation Violation"):
        load_training_data("../datasets/processed/test.parquet")


def test_multiclass_metrics_computation(synthetic_multiclass_data):
    """Verifies that evaluate_single_multiclass_model computes all multiclass metrics."""
    X_tr, X_val, y_tr, y_val, _ = split_multiclass_data(synthetic_multiclass_data, test_size=0.4, random_state=42)
    pipe = build_multiclass_model(model_type="rf", n_estimators=10, max_depth=5)
    fitted_pipe, t_fit = train_multiclass_model(pipe, X_tr, y_tr)

    metrics = evaluate_single_multiclass_model(
        fitted_pipe, X_val, y_val, "test_multiclass_rf", training_time=t_fit
    )

    required_keys = [
        "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
        "macro_f1", "weighted_f1", "per_class", "confusion_matrix_raw",
        "confusion_matrix_normalized", "inference_microseconds_per_flow",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing metric key: {key}"

    # Verify per-class structure contains all 8 categories
    for cat in CANONICAL_CATEGORIES:
        assert cat in metrics["per_class"]
        assert "precision" in metrics["per_class"][cat]
        assert "recall" in metrics["per_class"][cat]
        assert "f1_score" in metrics["per_class"][cat]
        assert "support" in metrics["per_class"][cat]

    # Verify confusion matrix shape is 8x8
    cm = np.array(metrics["confusion_matrix_raw"])
    assert cm.shape == (8, 8)
    assert cm.sum() == len(y_val)


def test_multiclass_error_analysis(synthetic_multiclass_data):
    """Verifies multiclass error diagnostics."""
    X_tr, X_val, y_tr, y_val, val_meta = split_multiclass_data(synthetic_multiclass_data, test_size=0.4, random_state=42)
    pipe = build_multiclass_model(model_type="dt", max_depth=3)
    fitted_pipe, _ = train_multiclass_model(pipe, X_tr, y_tr)

    y_pred = fitted_pipe.predict(X_val)
    diag = analyze_multiclass_errors(y_val, y_pred, val_meta, "test_multiclass_dt")

    assert "total_validation_flows" in diag
    assert "total_attacks_misclassified_as_benign" in diag
    assert "attacks_misclassified_as_benign_counts" in diag
    assert "attacks_misclassified_as_benign_rates" in diag
    assert "total_false_alarms" in diag
    assert "total_inter_attack_confusions" in diag


def test_multiclass_model_serialization(tmp_path, synthetic_multiclass_data):
    """Verifies multiclass model artifact serialization and reloading."""
    X_tr, _, y_tr, _, _ = split_multiclass_data(synthetic_multiclass_data, test_size=0.5, random_state=42)
    pipe = build_multiclass_model(model_type="dt", max_depth=3)
    fitted_pipe, _ = train_multiclass_model(pipe, X_tr, y_tr)

    save_path = tmp_path / "models" / "multiclass_dt_test.joblib"
    saved_file = save_model_artifact(fitted_pipe, save_path)
    assert saved_file.exists()

    import joblib
    reloaded = joblib.load(saved_file)
    assert hasattr(reloaded, "predict")
    assert np.array_equal(fitted_pipe.predict(X_tr), reloaded.predict(X_tr))
