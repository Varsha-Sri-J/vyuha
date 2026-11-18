"""VYUHA ML Pipeline Factory.

Constructs modular Scikit-Learn pipelines for supervised classification
and unsupervised anomaly detection, enforcing strict leakage boundaries.
"""

from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.models.feature_selector import FeatureSelector


def build_classification_pipeline(
    feature_selector: Optional[FeatureSelector] = None,
    scaler=None,
    classifier=None,
    exclude_ports: bool = False,
    variance_threshold: float = 0.0,
    drop_collinear_twins: bool = True,
    correlation_threshold: Optional[float] = None,
) -> Pipeline:
    """Builds an end-to-end supervised classification pipeline.

    Feature selection, scaling, and classification happen strictly within
    the pipeline during fit(), guaranteeing zero cross-validation leakage.

    Args:
        feature_selector: Custom FeatureSelector instance or None (uses defaults).
        scaler: Optional feature scaler (e.g., StandardScaler or None for tree models).
        classifier: Estimator instance (Random Forest, Decision Tree, etc.).
        exclude_ports: If True, builds a port-agnostic classification pipeline.
        variance_threshold: Threshold for constant feature pruning.
        drop_collinear_twins: If True, drops exact duplicate subflow features.
        correlation_threshold: Optional correlation pruning ceiling.

    Returns:
        Configured Scikit-Learn Pipeline.
    """
    steps = []

    # 1. Feature Selection Step
    if feature_selector is None:
        feature_selector = FeatureSelector(
            exclude_ports=exclude_ports,
            variance_threshold=variance_threshold,
            drop_collinear_twins=drop_collinear_twins,
            correlation_threshold=correlation_threshold,
        )
    steps.append(("selector", feature_selector))

    # 2. Optional Feature Scaling Step (Tree models are scale-invariant)
    if scaler is not None:
        steps.append(("scaler", scaler))

    # 3. Classifier Step
    if classifier is not None:
        steps.append(("classifier", classifier))

    return Pipeline(steps)


def build_anomaly_pipeline(
    feature_selector: Optional[FeatureSelector] = None,
    scaler=None,
    anomaly_detector=None,
    exclude_ports: bool = False,
    variance_threshold: float = 0.0,
    drop_collinear_twins: bool = True,
) -> Pipeline:
    """Builds an unsupervised anomaly detection pipeline (e.g., Isolation Forest).

    Keeps anomaly detection feature preparation and scaling strictly separated
    from supervised classification. Supports baseline unscaled models as well
    as experimental scaled variants.

    Args:
        feature_selector: FeatureSelector instance.
        scaler: Optional scaler (None for unscaled baseline; RobustScaler for experimental comparison).
        anomaly_detector: Anomaly detection model (e.g., IsolationForest).
        exclude_ports: Whether to exclude port numbers.
        variance_threshold: Variance cutoff.
        drop_collinear_twins: Subflow duplicate cutoff.

    Returns:
        Configured Scikit-Learn Pipeline.
    """
    steps = []

    if feature_selector is None:
        feature_selector = FeatureSelector(
            exclude_ports=exclude_ports,
            variance_threshold=variance_threshold,
            drop_collinear_twins=drop_collinear_twins,
        )
    steps.append(("selector", feature_selector))

    if scaler is not None:
        steps.append(("scaler", scaler))

    if anomaly_detector is not None:
        steps.append(("anomaly_detector", anomaly_detector))

    return Pipeline(steps)
