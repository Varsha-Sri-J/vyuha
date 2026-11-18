"""VYUHA Binary Intrusion Detection Training Engine.

Trains and serializes supervised machine-learning models (Random Forest and Decision Tree)
for binary classification: BENIGN (0) vs ATTACK (1).
Supports both Full (63 features) and Port-Agnostic (62 features) configurations.
Enforces strict inside-pipeline feature selection to prevent cross-validation leakage.
"""

import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.models.feature_selector import (
    FeatureSelector,
    extract_features_and_targets,
)
from src.models.pipeline import build_classification_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_training_data(
    data_path: Union[str, Path] = "datasets/processed/train.parquet",
    dev_mode: bool = False,
) -> pd.DataFrame:
    """Loads the processed training dataset, ensuring test.parquet is NEVER accessed.

    Args:
        data_path: Path to train.parquet or dev_sample.parquet.
        dev_mode: If True and data_path is default, falls back to dev_sample.parquet.

    Returns:
        Loaded DataFrame.
    """
    path = Path(data_path)
    if not path.is_absolute() and not path.exists():
        # Check parent workspace
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

    logger.info("Loading training dataset from: %s", path)
    df = pd.read_parquet(path)
    logger.info("Loaded %d records with %d columns.", len(df), len(df.columns))
    return df


def split_training_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Splits processed training data into an in-distribution Train and Validation set.

    Preserves target metadata (label_raw, label_category) for validation error analysis.

    Args:
        df: Input DataFrame containing flow features and target labels.
        test_size: Proportion of training data reserved for validation (default: 0.2).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_val, y_train, y_val, df_val_metadata)
    """
    X, y_dict = extract_features_and_targets(df)
    y_binary = y_dict["binary"]

    # Preserve metadata columns for downstream validation error analysis
    metadata_cols = ["label_raw", "label_category", "Destination Port"]
    meta_df = df[[c for c in metadata_cols if c in df.columns]].copy()

    X_train, X_val, y_train, y_val, _, val_meta = train_test_split(
        X,
        y_binary,
        meta_df,
        test_size=test_size,
        random_state=random_state,
        stratify=y_binary,
    )

    logger.info(
        "Stratified split complete: Train=%d (Attack=%.2f%%), Val=%d (Attack=%.2f%%)",
        len(X_train),
        (y_train.mean() * 100),
        len(X_val),
        (y_val.mean() * 100),
    )
    return X_train, X_val, y_train, y_val, val_meta


def build_binary_model(
    model_type: str = "rf",
    exclude_ports: bool = False,
    random_state: int = 42,
    class_weight: str = "balanced",
    max_depth: int = 20,
    n_estimators: int = 100,
) -> Pipeline:
    """Constructs a binary classification pipeline with encapsulated feature selection.

    Args:
        model_type: 'rf' for Random Forest, 'dt' for Decision Tree.
        exclude_ports: If True, excludes Destination Port (port-agnostic model).
        random_state: Seed for reproducible initialization.
        class_weight: Cost-sensitive weight strategy ('balanced' handles ~83/17 skew).
        max_depth: Maximum tree depth to control model complexity and prevent memorization.
        n_estimators: Number of trees for Random Forest (default: 100).

    Returns:
        Unfitted Scikit-Learn Pipeline.
    """
    selector = FeatureSelector(
        exclude_ports=exclude_ports,
        variance_threshold=0.0,
        drop_collinear_twins=True,
    )

    if model_type.lower() in ("rf", "random_forest"):
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type.lower() in ("dt", "decision_tree"):
        classifier = DecisionTreeClassifier(
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Expected 'rf' or 'dt'.")

    return build_classification_pipeline(
        feature_selector=selector,
        classifier=classifier,
    )


def train_binary_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Pipeline, float]:
    """Fits an end-to-end binary classification pipeline on the training partition.

    Learns feature selection filters strictly from X_train.

    Args:
        pipeline: Unfitted classification pipeline.
        X_train: Training feature matrix.
        y_train: Training binary labels.

    Returns:
        Tuple of (fitted_pipeline, training_time_seconds).
    """
    t_start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_time = time.perf_counter() - t_start

    selector = pipeline.named_steps["selector"]
    logger.info(
        "Model training completed in %.2fs. Features selected: %d (Dropped: %d)",
        training_time,
        len(selector.selected_features_),
        len(selector.dropped_features_),
    )
    return pipeline, training_time


def save_model_artifact(
    pipeline: Pipeline,
    output_path: Union[str, Path],
) -> Path:
    """Serializes a trained pipeline to disk outside Git-tracked dataset directories.

    Args:
        pipeline: Fitted pipeline.
        output_path: Target path for the joblib file.

    Returns:
        Absolute Path of saved file.
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, target, compress=3)
    logger.info("Model saved to: %s (%.2f MB)", target, target.stat().st_size / (1024 * 1024))
    return target.resolve()


def run_training_suite(
    data_path: Union[str, Path] = "datasets/processed/train.parquet",
    output_dir: Union[str, Path] = "models",
    dev_mode: bool = False,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Orchestrates training for all 4 Phase 3B experimental configurations.

    Configurations:
    1. rf_full: Random Forest — Full feature set (63 features)
    2. rf_port_agnostic: Random Forest — Port-agnostic feature set (62 features)
    3. dt_full: Decision Tree — Full feature set (63 features)
    4. dt_port_agnostic: Decision Tree — Port-agnostic feature set (62 features)

    Returns:
        Dictionary mapping config_name to {pipeline, training_time, X_val, y_val, val_meta}.
    """
    df = load_training_data(data_path, dev_mode=dev_mode)
    X_train, X_val, y_train, y_val, val_meta = split_training_data(
        df,
        test_size=0.2,
        random_state=random_state,
    )

    configs = {
        "rf_full": {"model_type": "rf", "exclude_ports": False, "desc": "Random Forest — Full (63 features)"},
        "rf_port_agnostic": {"model_type": "rf", "exclude_ports": True, "desc": "Random Forest — Port-Agnostic (62 features)"},
        "dt_full": {"model_type": "dt", "exclude_ports": False, "desc": "Decision Tree — Full (63 features)"},
        "dt_port_agnostic": {"model_type": "dt", "exclude_ports": True, "desc": "Decision Tree — Port-Agnostic (62 features)"},
    }

    out_path = Path(output_dir)
    results = {}

    for name, cfg in configs.items():
        logger.info("=== Starting Training: %s (%s) ===", name, cfg["desc"])
        pipe = build_binary_model(
            model_type=cfg["model_type"],
            exclude_ports=cfg["exclude_ports"],
            random_state=random_state,
            n_estimators=n_estimators,
        )
        fitted_pipe, train_time = train_binary_model(pipe, X_train, y_train)

        # Save model artifact
        model_file = out_path / f"{name}.joblib"
        save_model_artifact(fitted_pipe, model_file)

        results[name] = {
            "pipeline": fitted_pipe,
            "training_time": train_time,
            "config": cfg,
            "model_path": str(model_file),
            "selected_features": fitted_pipe.named_steps["selector"].selected_features_,
            "feature_count": len(fitted_pipe.named_steps["selector"].selected_features_),
        }

    results["data"] = {
        "X_val": X_val,
        "y_val": y_val,
        "val_meta": val_meta,
        "n_train": len(X_train),
        "n_val": len(X_val),
    }

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VYUHA Phase 3B Binary Model Trainer")
    parser.add_argument("--data", default="datasets/processed/train.parquet", help="Path to training parquet")
    parser.add_argument("--out-dir", default="models", help="Output directory for model binaries")
    parser.add_argument("--dev", action="store_true", help="Run in fast dev mode (200k dev_sample)")
    parser.add_argument("--n-trees", type=int, default=100, help="Number of trees in Random Forest")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_training_suite(
        data_path=args.data,
        output_dir=args.out_dir,
        dev_mode=args.dev,
        n_estimators=args.n_trees,
        random_state=args.seed,
    )
