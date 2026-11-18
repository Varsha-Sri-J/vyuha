"""VYUHA Feature Selector & Engineering Module.

Provides leakage-free, Scikit-Learn compatible transformers for:
- Strict target column exclusion.
- In-pipeline zero-variance and constant feature removal.
- Exact duplicate and collinear twin feature pruning.
- Port-agnostic feature set partitioning.
- Validation-based permutation and tree-based importance evaluation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)

TARGET_COLUMNS = ["label_raw", "label_binary", "label_category"]

# Known exact subflow and segment duplicates from CICIDS2017 CICFlowMeter export
EXACT_COLLINEAR_TWINS = [
    ("Subflow Fwd Bytes", "Total Length of Fwd Packets"),
    ("Subflow Bwd Bytes", "Total Length of Bwd Packets"),
    ("Subflow Fwd Packets", "Total Fwd Packets"),
    ("Subflow Bwd Packets", "Total Backward Packets"),
    ("Avg Fwd Segment Size", "Fwd Packet Length Mean"),
    ("Avg Bwd Segment Size", "Bwd Packet Length Mean"),
]


def extract_features_and_targets(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """Separates network flow features (X) from all target columns (y) with strict isolation.

    Ensures target columns cannot accidentally enter the feature matrix.

    Args:
        df: Input DataFrame containing flow records and target columns.

    Returns:
        Tuple of (X, y_dict) where y_dict contains 'binary', 'category', and 'raw' Series.
    """
    targets_to_drop = [col for col in df.columns if col.startswith("label_") or col in TARGET_COLUMNS]

    y_dict = {}
    if "label_binary" in df.columns:
        y_dict["binary"] = df["label_binary"].copy()
    if "label_category" in df.columns:
        y_dict["category"] = df["label_category"].copy()
    if "label_raw" in df.columns:
        y_dict["raw"] = df["label_raw"].copy()

    X = df.drop(columns=targets_to_drop).copy()

    # Integrity verification
    for target_col in TARGET_COLUMNS:
        assert target_col not in X.columns, f"Leakage Error: {target_col} found in feature matrix X!"

    return X, y_dict


def get_port_agnostic_features(feature_names: List[str]) -> List[str]:
    """Returns the subset of features excluding 'Destination Port' to prevent topology shortcutting.

    Args:
        feature_names: List of all feature names.

    Returns:
        List of feature names with 'Destination Port' removed.
    """
    return [col for col in feature_names if col != "Destination Port"]


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Leakage-free feature selector designed for strict inside-CV training execution.

    Learns constant features, exact duplicates, and collinear twins solely from
    the training slice passed to `fit()`.
    """

    def __init__(
        self,
        exclude_ports: bool = False,
        variance_threshold: float = 0.0,
        drop_collinear_twins: bool = True,
        correlation_threshold: Optional[float] = None,
    ):
        """Initializes the FeatureSelector.

        Args:
            exclude_ports: If True, drops 'Destination Port' to construct a port-agnostic model.
            variance_threshold: Drops features with variance <= threshold (default: 0.0 for constants).
            drop_collinear_twins: If True, prunes identical duplicate features (e.g. Subflow duplicates).
            correlation_threshold: If set (e.g. 0.98), prunes pairwise features exceeding this correlation.
        """
        self.exclude_ports = exclude_ports
        self.variance_threshold = variance_threshold
        self.drop_collinear_twins = drop_collinear_twins
        self.correlation_threshold = correlation_threshold

        self.selected_features_: List[str] = []
        self.dropped_features_: List[str] = []
        self.drop_reasons_: Dict[str, str] = {}

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Fits feature selection criteria strictly on the training partition.

        Args:
            X: Training feature matrix (DataFrame or 2D array).
            y: Optional training targets.

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            # Convert to DataFrame with synthetic column names if array passed
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)

        cols = list(X.columns)
        dropped: Dict[str, str] = {}

        # 1. Port-Agnostic Exclusion (if enabled)
        if self.exclude_ports and "Destination Port" in cols:
            dropped["Destination Port"] = "Port-agnostic feature set exclusion"

        # 2. Zero-Variance & Constant Features
        variances = X.var(numeric_only=True)
        for col, var in variances.items():
            if col in dropped:
                continue
            if var <= self.variance_threshold or X[col].nunique() <= 1:
                dropped[col] = f"Zero variance (var={var:.6f}, nunique={X[col].nunique()})"

        # 3. Collinear Twins Pruning (Exact Subflow duplicates)
        if self.drop_collinear_twins:
            for twin_to_drop, primary in EXACT_COLLINEAR_TWINS:
                if twin_to_drop in cols and twin_to_drop not in dropped:
                    if primary in cols and primary not in dropped:
                        dropped[twin_to_drop] = f"Exact collinear twin of '{primary}' (r=1.000)"

        # 4. Pairwise Correlation Pruning (Optional)
        remaining = [c for c in cols if c not in dropped]
        if self.correlation_threshold is not None and len(remaining) > 1:
            corr_matrix = X[remaining].corr(method="spearman").abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            for col in upper_tri.columns:
                if col in dropped:
                    continue
                high_corr_partners = upper_tri.index[upper_tri[col] > self.correlation_threshold].tolist()
                if high_corr_partners:
                    partner = high_corr_partners[0]
                    dropped[col] = f"High correlation (|r| > {self.correlation_threshold}) with '{partner}'"

        self.dropped_features_ = list(dropped.keys())
        self.drop_reasons_ = dropped
        self.selected_features_ = [c for c in cols if c not in dropped]

        logger.info(
            "FeatureSelector fit complete: %d features retained, %d dropped (exclude_ports=%s).",
            len(self.selected_features_),
            len(self.dropped_features_),
            self.exclude_ports,
        )
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Transforms X by selecting the retained feature subset.

        Args:
            X: Input feature matrix.

        Returns:
            Filtered feature matrix.
        """
        if not self.selected_features_:
            raise RuntimeError("FeatureSelector must be fitted before calling transform().")

        if isinstance(X, pd.DataFrame):
            # Ensure all selected features exist
            missing = set(self.selected_features_) - set(X.columns)
            if missing:
                raise ValueError(f"Input DataFrame is missing fitted features: {missing}")
            return X[self.selected_features_].copy()
        else:
            # If numpy array, slice by column indices
            return X[:, [i for i, c in enumerate(X.columns) if c in self.selected_features_]]

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Returns the list of selected feature column names."""
        return list(self.selected_features_)


def compute_permutation_feature_importance(
    estimator,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scoring: str = "f1_macro",
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Computes Permutation Feature Importance strictly on held-out validation data.

    Why Permutation Importance over Impurity (Gini):
    - Impurity-based importance evaluates training set splits and is biased toward
      high-cardinality continuous features.
    - Permutation importance measures true generalization drop on unseen validation data
      when a feature's values are randomly shuffled.

    Args:
        estimator: Trained model pipeline.
        X_val: Held-out validation features.
        y_val: Held-out validation labels.
        scoring: Metric to evaluate performance degradation.
        n_repeats: Number of permutation iterations per feature.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame ranking features by mean importance and standard deviation.
    """
    perm_result = permutation_importance(
        estimator,
        X_val,
        y_val,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    feature_names = list(X_val.columns)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std,
    }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

    return importance_df


def compute_tree_feature_importance(
    tree_estimator,
    feature_names: List[str],
) -> pd.DataFrame:
    """Extracts MDI (Mean Decrease in Impurity) feature importance from a trained tree model.

    Note on Limitations:
    - Impurity-based importances can over-represent continuous features and only evaluate
      in-sample split contributions.

    Args:
        tree_estimator: Trained Random Forest or Decision Tree model.
        feature_names: List of input feature names.

    Returns:
        DataFrame of features ranked by Gini/impurity importance.
    """
    if not hasattr(tree_estimator, "feature_importances_"):
        raise AttributeError("Estimator does not possess 'feature_importances_' attribute.")

    importances = tree_estimator.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return df
