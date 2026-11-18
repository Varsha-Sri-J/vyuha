"""VYUHA Leakage-Safe Cross-Validation & Evaluation Engine.

Executes rigorous K-Fold cross-validation ensuring all feature selection,
scaling, and model fitting happen strictly inside training folds.
"""

import logging
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class LeakageSafeCrossValidator:
    """Evaluates ML pipelines across stratified K-folds ensuring zero data leakage.

    Guarantees:
    - Feature selection is fitted strictly on X_train_fold.
    - Transformation is applied to X_val_fold using fold-fitted parameters.
    - Test set (test.parquet) remains completely untouched and isolated.
    """

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        """Initializes the cross-validator.

        Args:
            n_splits: Number of stratified folds (default: 5).
            random_state: Seed for fold shuffling reproducibility.
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def evaluate_pipeline(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        scoring_average: str = "macro",
    ) -> Dict[str, Any]:
        """Runs stratified K-fold cross-validation on an end-to-end pipeline.

        Args:
            pipeline: Scikit-learn Pipeline containing FeatureSelector + optional Scaler + Model.
            X: Training feature matrix (never test data).
            y: Training target Series (binary or category).
            scoring_average: Averaging method for multi-class metrics ('macro' or 'weighted').

        Returns:
            Dictionary containing per-fold metrics, aggregate means, and feature stability stats.
        """
        fold_metrics = []
        selected_feature_sets = []

        logger.info(
            "Starting %d-Fold Stratified Cross-Validation on %d samples...",
            self.n_splits,
            len(X),
        )

        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Fit the ENTIRE pipeline strictly on the training fold
            # Feature selection and scaling parameters are learned ONLY from X_tr
            pipeline.fit(X_tr, y_tr)

            # Record features selected in this fold
            if "selector" in pipeline.named_steps:
                selector = pipeline.named_steps["selector"]
                selected_features = selector.get_feature_names_out()
                selected_feature_sets.append(set(selected_features))
            else:
                selected_feature_sets.append(set(X.columns))

            # Predict on the held-out validation fold
            y_pred = pipeline.predict(X_val)

            # Calculate metrics
            acc = accuracy_score(y_val, y_pred)
            bal_acc = balanced_accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average=scoring_average, zero_division=0)
            rec = recall_score(y_val, y_pred, average=scoring_average, zero_division=0)
            f1 = f1_score(y_val, y_pred, average=scoring_average, zero_division=0)

            fold_result = {
                "fold": fold_idx + 1,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                f"precision_{scoring_average}": prec,
                f"recall_{scoring_average}": rec,
                f"f1_{scoring_average}": f1,
                "selected_feature_count": len(selected_feature_sets[-1]),
            }
            fold_metrics.append(fold_result)
            logger.info(
                "Fold %d/%d - F1 (%s): %.4f, BalAcc: %.4f, Features: %d",
                fold_idx + 1,
                self.n_splits,
                scoring_average,
                f1,
                bal_acc,
                len(selected_feature_sets[-1]),
            )

        # Compute feature selection stability across folds (Jaccard similarity of selected sets)
        intersection_features = set.intersection(*selected_feature_sets) if selected_feature_sets else set()
        union_features = set.union(*selected_feature_sets) if selected_feature_sets else set()
        stability_ratio = len(intersection_features) / len(union_features) if union_features else 1.0

        metrics_df = pd.DataFrame(fold_metrics)
        summary = {
            "n_splits": self.n_splits,
            "random_state": self.random_state,
            "fold_results": fold_metrics,
            "mean_accuracy": float(metrics_df["accuracy"].mean()),
            "std_accuracy": float(metrics_df["accuracy"].std()),
            "mean_balanced_accuracy": float(metrics_df["balanced_accuracy"].mean()),
            "std_balanced_accuracy": float(metrics_df["balanced_accuracy"].std()),
            f"mean_f1_{scoring_average}": float(metrics_df[f"f1_{scoring_average}"].mean()),
            f"std_f1_{scoring_average}": float(metrics_df[f"f1_{scoring_average}"].std()),
            f"mean_precision_{scoring_average}": float(metrics_df[f"precision_{scoring_average}"].mean()),
            f"mean_recall_{scoring_average}": float(metrics_df[f"recall_{scoring_average}"].mean()),
            "mean_selected_feature_count": float(metrics_df["selected_feature_count"].mean()),
            "feature_selection_stability": float(stability_ratio),
            "common_features_count": len(intersection_features),
            "common_features": sorted(list(intersection_features)),
        }

        return summary
