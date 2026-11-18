"""VYUHA Unsupervised Anomaly Detection Training Engine (Phase 3D).

Implements novelty-based anomaly detection using Isolation Forest:
1. Strict test.parquet isolation (never accessed).
2. Benign-only model fitting: learns normal flow patterns without attack labels.
3. Strict target column exclusion via FeatureSelector.
4. Separate benign calibration subset for target false-positive rate threshold calibration.
5. Support for controlled training-size benchmarks (50k, 100k, 200k).
6. Port-agnostic and scaling comparative variants.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.models.feature_selector import (
    FeatureSelector,
    TARGET_COLUMNS,
    extract_features_and_targets,
)
from src.models.pipeline import build_anomaly_pipeline

logger = logging.getLogger(__name__)


def load_training_data(
    data_path: Union[str, Path] = "datasets/processed/train.parquet",
    dev_mode: bool = False,
) -> pd.DataFrame:
    """Loads processed training dataset, ensuring test.parquet is NEVER accessed.

    Args:
        data_path: Path to train.parquet or dev_sample.parquet.
        dev_mode: If True, uses dev_sample.parquet.

    Returns:
        Loaded DataFrame.

    Raises:
        PermissionError: If data_path references test.parquet.
    """
    path = Path(data_path)
    if not path.is_absolute() and not path.exists():
        alt_path = Path("..") / data_path
        if alt_path.exists():
            path = alt_path

    # Strict isolation check: NEVER allow loading test.parquet
    if "test.parquet" in str(path).lower():
        raise PermissionError(
            f"Strict Isolation Violation: Attempted to load {path}. "
            "test.parquet is strictly held-out and must never be accessed during training/validation!"
        )

    if dev_mode and "dev_sample.parquet" not in str(path):
        dev_path = path.parent / "dev_sample.parquet"
        if dev_path.exists():
            path = dev_path
            logger.info("Dev mode active: Using %s (200k records).", path)

    logger.info("Loading anomaly training dataset from: %s", path)
    df = pd.read_parquet(path)
    logger.info("Loaded %d records with %d columns.", len(df), len(df.columns))
    return df


def partition_anomaly_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    cal_size: int = 50000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Partitions dataset into benign training pool, benign calibration set, and mixed validation set.

    Guarantees:
    1. Validation split is stratified on label_category across all 8 classes (identical to Phase 3B/3C).
    2. Training pool and calibration set contain ONLY benign flows.
    3. Calibration set is disjoint from training pool.
    4. Target columns never enter feature matrices.

    Args:
        df: Input DataFrame containing flow features and labels.
        test_size: Proportion of dataset reserved for validation (default: 0.2).
        random_state: Random seed for reproducibility.
        cal_size: Number of benign flows reserved strictly for threshold calibration.

    Returns:
        Tuple of (X_train_benign_pool, X_cal_benign, X_val, y_val, val_meta).
    """
    X, y_dict = extract_features_and_targets(df)
    y_category = y_dict["category"]

    # Preserve metadata for validation diagnostics
    metadata_cols = ["label_raw", "label_category", "Destination Port"]
    meta_df = df[[c for c in metadata_cols if c in df.columns]].copy()

    # Stratified split identical to Phase 3B and Phase 3C
    X_train, X_val, y_train, y_val, _, val_meta = train_test_split(
        X,
        y_category,
        meta_df,
        test_size=test_size,
        random_state=random_state,
        stratify=y_category,
    )

    logger.info(
        "Initial split complete: Train=%d, Val=%d",
        len(X_train),
        len(X_val),
    )

    # Filter strictly BENIGN flows for unsupervised modeling
    benign_mask = (y_train == "BENIGN")
    X_benign_all = X_train[benign_mask].copy()
    total_benign = len(X_benign_all)
    logger.info("Total benign training flows available: %d", total_benign)

    if total_benign < cal_size + 10000:
        # Scale cal_size down if testing on tiny synthetic subsets
        cal_size = int(total_benign * 0.2)

    # Deterministic partition of benign training pool into calibration set & training pool
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(X_benign_all.index)

    cal_indices = shuffled_indices[:cal_size]
    train_pool_indices = shuffled_indices[cal_size:]

    X_cal_benign = X_benign_all.loc[cal_indices].copy()
    X_train_benign_pool = X_benign_all.loc[train_pool_indices].copy()

    logger.info(
        "Benign partitioning complete: Calibration Set=%d, Training Pool=%d",
        len(X_cal_benign),
        len(X_train_benign_pool),
    )

    # Strict assertion: Ensure calibration set and training pool are disjoint
    overlap = set(X_cal_benign.index).intersection(set(X_train_benign_pool.index))
    assert len(overlap) == 0, f"Integrity Failure: {len(overlap)} overlapping indices between train and cal!"

    # Strict assertion: Ensure no target column exists in any feature matrix
    for col in TARGET_COLUMNS:
        assert col not in X_train_benign_pool.columns, f"Target column {col} in X_train_benign_pool!"
        assert col not in X_cal_benign.columns, f"Target column {col} in X_cal_benign!"
        assert col not in X_val.columns, f"Target column {col} in X_val!"

    return X_train_benign_pool, X_cal_benign, X_val, y_val, val_meta


def sample_benign_training_subset(
    X_train_benign_pool: pd.DataFrame,
    n_samples: int = 100000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Extracts a reproducible sample from the benign training pool.

    Args:
        X_train_benign_pool: Disjoint benign training pool.
        n_samples: Number of samples to draw.
        random_state: Random seed for reproducibility.

    Returns:
        Sampled DataFrame.
    """
    if n_samples >= len(X_train_benign_pool):
        logger.warning(
            "Requested %d samples but pool only has %d. Using full pool.",
            n_samples,
            len(X_train_benign_pool),
        )
        return X_train_benign_pool.copy()

    sample = X_train_benign_pool.sample(n=n_samples, random_state=random_state).copy()
    logger.info("Sampled %d benign training flows (seed=%d).", len(sample), random_state)
    return sample


def build_anomaly_model(
    exclude_ports: bool = False,
    scaler_type: Optional[str] = None,
    n_estimators: int = 100,
    max_samples: Union[int, str] = 256,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """Constructs an unsupervised anomaly detection pipeline with encapsulated feature selection.

    Args:
        exclude_ports: If True, drops 'Destination Port' for port-agnostic evaluation.
        scaler_type: Optional scaler ("robust", "standard", or None for tree invariance).
        n_estimators: Number of trees in Isolation Forest (default: 100).
        max_samples: Number of samples to draw per tree (default: 256).
        random_state: Random seed for reproducibility.
        n_jobs: Parallel worker count (default: -1).

    Returns:
        Scikit-Learn Pipeline.
    """
    scaler = None
    if scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()

    detector = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    pipeline = build_anomaly_pipeline(
        exclude_ports=exclude_ports,
        scaler=scaler,
        anomaly_detector=detector,
    )

    return pipeline


def train_anomaly_model(
    pipeline: Pipeline,
    X_train_benign: pd.DataFrame,
) -> Pipeline:
    """Fits the anomaly detection pipeline strictly on benign training flows.

    Args:
        pipeline: Constructed pipeline.
        X_train_benign: Benign training flow feature matrix.

    Returns:
        Fitted pipeline.
    """
    logger.info("Fitting anomaly detection pipeline on %d benign records...", len(X_train_benign))
    pipeline.fit(X_train_benign)

    selector = pipeline.named_steps["selector"]
    retained = len(selector.selected_features_)
    dropped = len(selector.dropped_features_)
    logger.info("Pipeline fit complete. Features retained: %d, dropped: %d.", retained, dropped)
    return pipeline


def calibrate_anomaly_thresholds(
    pipeline: Pipeline,
    X_cal_benign: pd.DataFrame,
    target_fprs: Tuple[float, ...] = (0.05, 0.01, 0.005, 0.001),
) -> Dict[str, Dict[str, float]]:
    """Calibrates frozen decision thresholds on a held-out benign calibration subset.

    Threshold Convention:
    scikit-learn IsolationForest score_samples(X) returns negative values,
    where lower/more negative means more anomalous, and higher/closer to 0 means normal.
    We define the standard normalized Anomaly Score:
        A(x) = - score_samples(x)
    where higher A(x) means more anomalous (A(x) in [0, 1]).

    For target benign false positive rate alpha:
        threshold = Percentile_{100 * (1 - alpha)}(A(X_cal))
    A flow is flagged as an anomaly if A(x) >= threshold.

    Args:
        pipeline: Fitted anomaly detection pipeline.
        X_cal_benign: Held-out benign calibration flows.
        target_fprs: Tuple of target benign false positive rates (e.g. 0.01 for 1.0%).

    Returns:
        Dictionary mapping target FPR name to threshold metadata.
    """
    logger.info("Calibrating thresholds on %d benign calibration records...", len(X_cal_benign))

    # IsolationForest is the last step
    detector = pipeline.named_steps["anomaly_detector"]

    # Transform through preceding steps (selector, scaler)
    if len(pipeline.steps) > 1:
        prep_pipeline = Pipeline(pipeline.steps[:-1])
        X_cal_trans = prep_pipeline.transform(X_cal_benign)
    else:
        X_cal_trans = X_cal_benign

    # Compute raw scikit-learn scores and inverted anomaly scores
    raw_scores = detector.score_samples(X_cal_trans)
    anomaly_scores = -raw_scores

    calibration_results = {}
    for fpr in target_fprs:
        percentile = 100.0 * (1.0 - fpr)
        threshold = float(np.percentile(anomaly_scores, percentile))
        empirical_cal_fpr = float(np.mean(anomaly_scores >= threshold))

        key = f"fpr_{fpr * 100:.1f}pct" if fpr >= 0.01 else f"fpr_{fpr * 100:.2f}pct"
        calibration_results[key] = {
            "target_fpr": fpr,
            "percentile": percentile,
            "threshold": threshold,
            "empirical_calibration_fpr": empirical_cal_fpr,
        }
        logger.info(
            "Calibrated %s: Target FPR=%.3f, Percentile=%.1f, Threshold=%.4f (Empirical Cal FPR=%.4f)",
            key,
            fpr,
            percentile,
            threshold,
            empirical_cal_fpr,
        )

    return calibration_results


def save_anomaly_artifact(
    pipeline: Pipeline,
    filepath: Union[str, Path],
    metadata: Optional[dict] = None,
) -> Path:
    """Serializes the fitted anomaly detection pipeline and metadata to disk.

    Args:
        pipeline: Fitted pipeline.
        filepath: Target destination path.
        metadata: Optional experiment metadata.

    Returns:
        Path of saved artifact.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": pipeline,
        "metadata": metadata or {},
    }

    joblib.dump(artifact, path)
    logger.info("Saved anomaly model artifact to: %s", path)
    return path
