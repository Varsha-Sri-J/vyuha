"""VYUHA Detection Engine Alert Adapters (Phase 4.1).

Implements dedicated normalization adapters for:
1. Binary Intrusion Detection (Phase 3B): Distinguishes benign vs attack, confidence probabilities.
2. Multiclass Intrusion Detection (Phase 3C): Preserves attack categories, class-level confidence.
3. Unsupervised Anomaly Detection (Phase 3D): Preserves anomaly scores, calibrated thresholds,
   and avoids treating statistical anomalies as confirmed attacks.

Each adapter encapsulates engine-specific semantics and translates raw model outputs into the
canonical VYUHA Alert representation.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd

from src.intelligence.extractor import EntityExtractor, ExtractedEntities
from src.intelligence.schema import Alert, AlertSeverity, DetectionEngineType, ThreatStatus

logger = logging.getLogger(__name__)


# Standard category to severity mapping for supervised multiclass predictions
CATEGORY_SEVERITY_MAP: Dict[str, str] = {
    "Heartbleed": AlertSeverity.CRITICAL.value,
    "Infiltration": AlertSeverity.CRITICAL.value,
    "DoS_DDoS": AlertSeverity.HIGH.value,
    "Botnet": AlertSeverity.HIGH.value,
    "WebAttack": AlertSeverity.MEDIUM.value,
    "BruteForce": AlertSeverity.MEDIUM.value,
    "PortScan": AlertSeverity.LOW.value,
    "BENIGN": AlertSeverity.INFO.value,
}


def _resolve_flow_records(
    flow_records: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
    n_expected: int,
) -> List[ExtractedEntities]:
    """Helper to extract entities from either DataFrame or list of dicts."""
    if isinstance(flow_records, pd.DataFrame):
        if len(flow_records) != n_expected:
            raise ValueError(
                f"Length mismatch: {n_expected} model outputs but {len(flow_records)} flow records"
            )
        return EntityExtractor.extract_batch(flow_records)
    elif isinstance(flow_records, (list, tuple)):
        if len(flow_records) != n_expected:
            raise ValueError(
                f"Length mismatch: {n_expected} model outputs but {len(flow_records)} flow records"
            )
        return [
            EntityExtractor.extract(rec, index=i)
            for i, rec in enumerate(flow_records)
        ]
    else:
        raise TypeError(f"Unsupported flow_records type: {type(flow_records)}")


class BinaryAlertAdapter:
    """Adapts outputs from binary classification models (Phase 3B) into normalized Alerts."""

    def __init__(
        self,
        model_name: str = "rf_full",
        alert_on_attack_only: bool = True,
        high_confidence_threshold: float = 0.90,
        med_confidence_threshold: float = 0.70,
    ):
        """Initializes the BinaryAlertAdapter.

        Args:
            model_name: Identifier of the binary model (e.g. 'rf_full', 'dt_port_agnostic').
            alert_on_attack_only: If True, only emits alerts for flows predicted as ATTACK (1).
            high_confidence_threshold: Confidence threshold for HIGH severity assignment.
            med_confidence_threshold: Confidence threshold for MEDIUM severity assignment.
        """
        self.model_name = model_name
        self.alert_on_attack_only = alert_on_attack_only
        self.high_conf = high_confidence_threshold
        self.med_conf = med_confidence_threshold

    def adapt(
        self,
        predictions: Union[np.ndarray, Sequence[Union[int, str]]],
        flow_records: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
        probabilities: Optional[np.ndarray] = None,
        metadata_extra: Optional[Dict[str, Any]] = None,
    ) -> List[Alert]:
        """Converts binary predictions and probabilities into canonical Alerts.

        Args:
            predictions: Array of binary predictions (0=BENIGN, 1=ATTACK).
            flow_records: Originating flow records (DataFrame or list of dicts).
            probabilities: Optional array of shape (N, 2) or (N,) containing class probabilities.
            metadata_extra: Optional dictionary of metadata to attach to generated alerts.

        Returns:
            List of normalized Alert objects.
        """
        n_samples = len(predictions)
        entities_list = _resolve_flow_records(flow_records, n_samples)
        meta = metadata_extra or {}

        alerts: List[Alert] = []

        for i in range(n_samples):
            pred_raw = predictions[i]
            # Normalize prediction to integer 0 or 1
            if isinstance(pred_raw, str):
                pred_int = 0 if pred_raw.strip().upper() == "BENIGN" else 1
            else:
                pred_int = int(pred_raw)

            is_attack = bool(pred_int == 1)

            # Filter if alert_on_attack_only is active
            if self.alert_on_attack_only and not is_attack:
                continue

            # Extract confidence if probabilities are provided (do NOT fabricate)
            confidence: Optional[float] = None
            attack_prob: Optional[float] = None

            if probabilities is not None:
                probs = np.asarray(probabilities)
                if probs.ndim == 2 and probs.shape[1] >= 2:
                    confidence = float(probs[i, pred_int])
                    attack_prob = float(probs[i, 1])
                elif probs.ndim == 1:
                    p1 = float(probs[i])
                    attack_prob = p1
                    confidence = p1 if is_attack else (1.0 - p1)

            # Determine severity
            if not is_attack:
                severity = AlertSeverity.INFO.value
            else:
                if confidence is not None:
                    if confidence >= self.high_conf:
                        severity = AlertSeverity.HIGH.value
                    elif confidence >= self.med_conf:
                        severity = AlertSeverity.MEDIUM.value
                    else:
                        severity = AlertSeverity.LOW.value
                else:
                    severity = AlertSeverity.MEDIUM.value

            threat_status = ThreatStatus.ATTACK.value if is_attack else ThreatStatus.BENIGN.value

            ent = entities_list[i]
            evidence = {
                "binary_prediction": "ATTACK" if is_attack else "BENIGN",
                "raw_prediction": pred_int,
                "threat_status": threat_status,
                "confidence": confidence,
                "attack_probability": attack_prob,
                "validation_errors": ent.validation_errors,
            }

            alert = Alert(
                alert_id=f"alt_bin_{uuid.uuid4().hex[:12]}",
                flow_id=ent.flow_id,
                detection_engine=DetectionEngineType.BINARY.value,
                model_name=self.model_name,
                prediction="ATTACK" if is_attack else "BENIGN",
                threat_status=threat_status,
                is_attack=is_attack,
                attack_category="ATTACK" if is_attack else None,
                source_ip=ent.source_ip,
                destination_ip=ent.destination_ip,
                source_port=ent.source_port,
                destination_port=ent.destination_port,
                protocol=ent.protocol,
                timestamp=ent.timestamp,
                flow_duration=ent.flow_duration,
                confidence=confidence,
                severity=severity,
                evidence=evidence,
                metadata={
                    "engine_type": "supervised_binary",
                    **meta,
                },
            )
            alerts.append(alert)

        return alerts


class MulticlassAlertAdapter:
    """Adapts outputs from multiclass attack-family classifiers (Phase 3C) into normalized Alerts."""

    def __init__(
        self,
        model_name: str = "multiclass_rf_full",
        alert_on_attack_only: bool = True,
        classes: Optional[Sequence[str]] = None,
    ):
        """Initializes the MulticlassAlertAdapter.

        Args:
            model_name: Identifier of the multiclass model.
            alert_on_attack_only: If True, only emits alerts for non-BENIGN predictions.
            classes: Sequence of class names corresponding to probability columns.
        """
        self.model_name = model_name
        self.alert_on_attack_only = alert_on_attack_only
        self.classes = list(classes) if classes else None

    def adapt(
        self,
        predictions: Union[np.ndarray, Sequence[str]],
        flow_records: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
        probabilities: Optional[np.ndarray] = None,
        classes: Optional[Sequence[str]] = None,
        metadata_extra: Optional[Dict[str, Any]] = None,
    ) -> List[Alert]:
        """Converts multiclass predictions and probabilities into canonical Alerts.

        Preserves predicted attack category across canonical categories:
        BENIGN, DoS_DDoS, PortScan, BruteForce, WebAttack, Botnet, Infiltration, Heartbleed.

        Args:
            predictions: Array of category predictions.
            flow_records: Originating flow records (DataFrame or list of dicts).
            probabilities: Optional array of shape (N, n_classes).
            classes: Class labels corresponding to probability columns.
            metadata_extra: Optional metadata dictionary.

        Returns:
            List of normalized Alert objects.
        """
        n_samples = len(predictions)
        entities_list = _resolve_flow_records(flow_records, n_samples)
        meta = metadata_extra or {}
        active_classes = list(classes) if classes else self.classes

        alerts: List[Alert] = []

        for i in range(n_samples):
            pred_cat = str(predictions[i]).strip()
            is_attack = bool(pred_cat != "BENIGN")

            if self.alert_on_attack_only and not is_attack:
                continue

            # Confidence extraction (do NOT fabricate)
            confidence: Optional[float] = None
            class_probs_dict: Optional[Dict[str, float]] = None

            if probabilities is not None:
                probs = np.asarray(probabilities)
                if probs.ndim == 2:
                    if active_classes and pred_cat in active_classes:
                        class_idx = active_classes.index(pred_cat)
                        confidence = float(probs[i, class_idx])
                        class_probs_dict = {
                            c: float(probs[i, idx])
                            for idx, c in enumerate(active_classes)
                        }
                    else:
                        # Max probability if exact class index unknown
                        confidence = float(np.max(probs[i]))

            # Determine severity based on specific attack family
            severity = CATEGORY_SEVERITY_MAP.get(pred_cat, AlertSeverity.MEDIUM.value)

            threat_status = ThreatStatus.ATTACK.value if is_attack else ThreatStatus.BENIGN.value

            ent = entities_list[i]
            evidence = {
                "predicted_category": pred_cat,
                "threat_status": threat_status,
                "is_attack": is_attack,
                "confidence": confidence,
                "class_probabilities": class_probs_dict,
                "validation_errors": ent.validation_errors,
            }

            alert = Alert(
                alert_id=f"alt_mc_{uuid.uuid4().hex[:12]}",
                flow_id=ent.flow_id,
                detection_engine=DetectionEngineType.MULTICLASS.value,
                model_name=self.model_name,
                prediction=pred_cat,
                threat_status=threat_status,
                is_attack=is_attack,
                attack_category=pred_cat if is_attack else None,
                source_ip=ent.source_ip,
                destination_ip=ent.destination_ip,
                source_port=ent.source_port,
                destination_port=ent.destination_port,
                protocol=ent.protocol,
                timestamp=ent.timestamp,
                flow_duration=ent.flow_duration,
                confidence=confidence,
                severity=severity,
                evidence=evidence,
                metadata={
                    "engine_type": "supervised_multiclass",
                    **meta,
                },
            )
            alerts.append(alert)

        return alerts


class AnomalyAlertAdapter:
    """Adapts outputs from unsupervised anomaly detection models (Phase 3D) into normalized Alerts.

    Preserves anomaly scores and calibrated decision thresholds.
    Ensures statistical anomalies are NOT automatically branded as confirmed attacks.
    """

    def __init__(
        self,
        model_name: str = "iforest_port_100k",
        calibrated_threshold: float = 0.6430,
        threshold_name: str = "fpr_1.0pct",
        alert_on_anomaly_only: bool = True,
    ):
        """Initializes the AnomalyAlertAdapter.

        Args:
            model_name: Identifier of the anomaly detection model.
            calibrated_threshold: Frozen decision threshold tau_alpha calibrated on benign data.
            threshold_name: Target FPR calibration label (e.g. 'fpr_1.0pct').
            alert_on_anomaly_only: If True, only emits alerts for flows where A(x) >= threshold.
        """
        self.model_name = model_name
        self.calibrated_threshold = float(calibrated_threshold)
        self.threshold_name = threshold_name
        self.alert_on_anomaly_only = alert_on_anomaly_only

    def adapt(
        self,
        anomaly_scores: Union[np.ndarray, Sequence[float]],
        flow_records: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
        calibrated_threshold: Optional[float] = None,
        threshold_name: Optional[str] = None,
        metadata_extra: Optional[Dict[str, Any]] = None,
    ) -> List[Alert]:
        """Converts normalized anomaly scores A(x) into canonical Alerts.

        Args:
            anomaly_scores: Array of normalized anomaly scores in [0, 1].
            flow_records: Originating flow records (DataFrame or list of dicts).
            calibrated_threshold: Optional override for decision threshold.
            threshold_name: Optional calibration name override.
            metadata_extra: Optional metadata dictionary.

        Returns:
            List of normalized Alert objects.
        """
        n_samples = len(anomaly_scores)
        entities_list = _resolve_flow_records(flow_records, n_samples)
        meta = metadata_extra or {}

        threshold = (
            float(calibrated_threshold)
            if calibrated_threshold is not None
            else self.calibrated_threshold
        )
        t_name = threshold_name or self.threshold_name

        alerts: List[Alert] = []

        for i in range(n_samples):
            score = float(anomaly_scores[i])
            is_anomaly = bool(score >= threshold)

            if self.alert_on_anomaly_only and not is_anomaly:
                continue

            # Determine severity based on empirical anomaly score tiers
            if not is_anomaly:
                severity = AlertSeverity.INFO.value
            elif score >= 0.70:
                severity = AlertSeverity.HIGH.value
            elif score >= 0.64:
                severity = AlertSeverity.MEDIUM.value
            else:
                severity = AlertSeverity.LOW.value

            threat_status = ThreatStatus.ANOMALY.value if is_anomaly else ThreatStatus.BENIGN.value
            margin = score - threshold
            ent = entities_list[i]

            evidence = {
                "anomaly_score": round(score, 6),
                "calibrated_threshold": round(threshold, 6),
                "threshold_name": t_name,
                "threshold_margin": round(margin, 6),
                "threat_status": threat_status,
                "is_anomaly": is_anomaly,
                "assessment": (
                    "Statistical novelty detected; unconfirmed as malicious attack"
                    if is_anomaly
                    else "In-distribution normal flow"
                ),
                "validation_errors": ent.validation_errors,
            }

            alert = Alert(
                alert_id=f"alt_ano_{uuid.uuid4().hex[:12]}",
                flow_id=ent.flow_id,
                detection_engine=DetectionEngineType.ANOMALY.value,
                model_name=self.model_name,
                prediction="ANOMALOUS" if is_anomaly else "NORMAL",
                threat_status=threat_status,
                # An anomaly is NOT automatically a confirmed attack
                is_attack=False,
                # Unsupervised models have no attack labels
                attack_category=None,
                source_ip=ent.source_ip,
                destination_ip=ent.destination_ip,
                source_port=ent.source_port,
                destination_port=ent.destination_port,
                protocol=ent.protocol,
                timestamp=ent.timestamp,
                flow_duration=ent.flow_duration,
                anomaly_score=round(score, 6),
                anomaly_threshold=round(threshold, 6),
                # Non-probabilistic tree isolation score -> no fabricated confidence
                confidence=None,
                severity=severity,
                evidence=evidence,
                metadata={
                    "engine_type": "unsupervised_anomaly",
                    "calibration_subset_size": 50000,
                    **meta,
                },
            )
            alerts.append(alert)

        return alerts
