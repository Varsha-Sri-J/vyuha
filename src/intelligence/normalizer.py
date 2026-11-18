"""VYUHA Multi-Engine Alert Normalizer Module (Phase 4.1).

Coordinates alerts generated across multiple detection engines evaluating the same
network traffic flows.
Enforces:
1. Deduplication: Eliminates duplicate alert generation for identical flows.
2. Evidence Preservation: Preserves engine-specific predictions, confidences, thresholds,
   and features within a unified multi-engine evidence dictionary.
3. Negative Assessment Tracking: Engines that evaluate a flow but do not flag it are
   explicitly recorded with `flagged_as_threat=False`, preventing them from appearing as missing.
4. Consensus & Threat Synthesis: Unifies `threat_status` (ATTACK, ANOMALY, BENIGN, UNKNOWN)
   and computes cross-engine agreement.
"""

from collections import defaultdict
import logging
from typing import Any, Dict, List, Optional, Sequence, Union
import uuid
import numpy as np
import pandas as pd

from src.intelligence.adapters import (
    AnomalyAlertAdapter,
    BinaryAlertAdapter,
    MulticlassAlertAdapter,
)
from src.intelligence.schema import (
    Alert,
    AlertSeverity,
    DetectionEngineType,
    SEVERITY_ORDER,
    ThreatStatus,
)

logger = logging.getLogger(__name__)


class MultiEngineAlertNormalizer:
    """Consolidates alerts across binary, multiclass, and anomaly detection engines."""

    @staticmethod
    def consolidate_alerts(
        alerts: Sequence[Alert],
        alert_on_threat_only: bool = False,
    ) -> List[Alert]:
        """Consolidates a heterogeneous sequence of alerts by flow_id without duplicate generation.

        If a flow was evaluated by multiple engines (e.g. Binary, Multiclass, Anomaly):
        - A single consolidated Alert is created for that flow.
        - Engine-specific predictions, scores, and evidence are preserved under `evidence['engines']`.
        - Engines that evaluated the flow but did not flag it are recorded with `flagged_as_threat=False`.
        - Severity is reconciled to the highest provisional severity among participating engines.
        - Supervised attack categories take precedence for primary classification.

        Args:
            alerts: Sequence of raw Alert objects from any combination of engines.
            alert_on_threat_only: If True, only emits consolidated alerts where at least one engine
                                  flagged a threat (ATTACK or ANOMALY).

        Returns:
            List of consolidated Alert objects with guaranteed unique flow_ids.
        """
        if not alerts:
            return []

        # Group alerts by flow_id
        flows_map: Dict[str, List[Alert]] = defaultdict(list)
        for alt in alerts:
            flows_map[alt.flow_id].append(alt)

        consolidated_results: List[Alert] = []

        for flow_id, group in flows_map.items():
            if len(group) == 1:
                alt = group[0]
                if alert_on_threat_only and alt.threat_status not in (
                    ThreatStatus.ATTACK.value,
                    ThreatStatus.ANOMALY.value,
                ):
                    continue
                consolidated_results.append(alt)
                continue

            # Multiple engines evaluated the same flow -> Consolidate
            engines_evidence: Dict[str, Any] = {}
            participating_engines: List[str] = []
            model_names: List[str] = []

            binary_alt: Optional[Alert] = None
            multiclass_alt: Optional[Alert] = None
            anomaly_alt: Optional[Alert] = None

            for alt in group:
                participating_engines.append(alt.detection_engine)
                model_names.append(alt.model_name)

                engine_key = alt.detection_engine
                # Handle possible multiple alerts of same engine type gracefully
                if engine_key in engines_evidence:
                    engine_key = f"{alt.detection_engine}_{alt.model_name}"

                flagged = alt.threat_status in (
                    ThreatStatus.ATTACK.value,
                    ThreatStatus.ANOMALY.value,
                )

                engines_evidence[engine_key] = {
                    "alert_id": alt.alert_id,
                    "model_name": alt.model_name,
                    "prediction": alt.prediction,
                    "threat_status": alt.threat_status,
                    "flagged_as_threat": flagged,
                    "is_attack": alt.is_attack,
                    "attack_category": alt.attack_category,
                    "confidence": alt.confidence,
                    "anomaly_score": alt.anomaly_score,
                    "anomaly_threshold": alt.anomaly_threshold,
                    "severity": alt.severity,
                    "evidence": alt.evidence,
                }

                if alt.detection_engine == DetectionEngineType.BINARY.value:
                    binary_alt = alt
                elif alt.detection_engine == DetectionEngineType.MULTICLASS.value:
                    multiclass_alt = alt
                elif alt.detection_engine == DetectionEngineType.ANOMALY.value:
                    anomaly_alt = alt

            # 2. Reconcile explicit threat_status and is_attack
            if any(a.threat_status == ThreatStatus.ATTACK.value for a in group):
                threat_status = ThreatStatus.ATTACK.value
                is_attack = True
            elif any(a.threat_status == ThreatStatus.ANOMALY.value for a in group):
                threat_status = ThreatStatus.ANOMALY.value
                is_attack = False  # Anomaly is NOT a confirmed attack
            elif all(a.threat_status == ThreatStatus.BENIGN.value for a in group):
                threat_status = ThreatStatus.BENIGN.value
                is_attack = False
            else:
                threat_status = ThreatStatus.UNKNOWN.value
                is_attack = False

            # Filter if alert_on_threat_only is active
            if alert_on_threat_only and threat_status not in (
                ThreatStatus.ATTACK.value,
                ThreatStatus.ANOMALY.value,
            ):
                continue

            # 3. Reconcile primary attack category
            attack_category: Optional[str] = None
            if multiclass_alt and multiclass_alt.attack_category:
                attack_category = multiclass_alt.attack_category
            elif binary_alt and binary_alt.is_attack:
                attack_category = "ATTACK"

            # Primary prediction
            if multiclass_alt and multiclass_alt.is_attack:
                primary_pred = multiclass_alt.prediction
            elif binary_alt and binary_alt.is_attack:
                primary_pred = binary_alt.prediction
            elif anomaly_alt and anomaly_alt.threat_status == ThreatStatus.ANOMALY.value:
                primary_pred = anomaly_alt.prediction
            elif multiclass_alt:
                primary_pred = multiclass_alt.prediction
            elif binary_alt:
                primary_pred = binary_alt.prediction
            elif anomaly_alt:
                primary_pred = anomaly_alt.prediction
            else:
                primary_pred = group[0].prediction

            # 4. Reconcile confidence and anomaly metrics without cross-engine contamination
            confidence: Optional[float] = None
            if multiclass_alt and multiclass_alt.confidence is not None:
                confidence = multiclass_alt.confidence
            elif binary_alt and binary_alt.confidence is not None:
                confidence = binary_alt.confidence

            anomaly_score = anomaly_alt.anomaly_score if anomaly_alt else None
            anomaly_threshold = anomaly_alt.anomaly_threshold if anomaly_alt else None

            # 5. Reconcile provisional maximum severity
            max_severity = max(
                (a.severity for a in group),
                key=lambda s: SEVERITY_ORDER.get(s, 0),
            )

            # 6. Extract shared entities from first alert
            base_alt = group[0]

            # 7. Build consolidated evidence and negative assessment tracking
            engines_flagging = [
                eng for eng, info in engines_evidence.items() if info["flagged_as_threat"]
            ]
            engines_not_flagging = [
                eng for eng, info in engines_evidence.items() if not info["flagged_as_threat"]
            ]

            consolidated_evidence = {
                "participating_engines": participating_engines,
                "engine_count": len(group),
                "engines": engines_evidence,
                "consensus_summary": {
                    "threat_status": threat_status,
                    "engines_flagging_threat": engines_flagging,
                    "engines_not_flagging": engines_not_flagging,
                    "supervised_attack_detected": is_attack,
                    "anomaly_detected": (
                        anomaly_alt is not None
                        and anomaly_alt.threat_status == ThreatStatus.ANOMALY.value
                    ),
                    "all_engines_agreed_attack": (
                        bool(binary_alt and binary_alt.is_attack)
                        and bool(multiclass_alt and multiclass_alt.is_attack)
                    ),
                },
            }

            consolidated_alert = Alert(
                alert_id=f"alt_multi_{uuid.uuid4().hex[:12]}",
                flow_id=flow_id,
                detection_engine=DetectionEngineType.MULTI_ENGINE.value,
                model_name="+".join(sorted(set(model_names))),
                prediction=primary_pred,
                threat_status=threat_status,
                is_attack=is_attack,
                attack_category=attack_category,
                source_ip=base_alt.source_ip,
                destination_ip=base_alt.destination_ip,
                source_port=base_alt.source_port,
                destination_port=base_alt.destination_port,
                protocol=base_alt.protocol,
                timestamp=base_alt.timestamp,
                flow_duration=base_alt.flow_duration,
                anomaly_score=anomaly_score,
                anomaly_threshold=anomaly_threshold,
                confidence=confidence,
                severity=max_severity,
                evidence=consolidated_evidence,
                metadata={
                    "consolidated_from_alerts": [a.alert_id for a in group],
                    "engine_count": len(group),
                    "severity_semantics": "provisional_flow_heuristic",
                },
            )
            consolidated_results.append(consolidated_alert)

        return consolidated_results

    @classmethod
    def process_flow_batch(
        cls,
        flow_records: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
        binary_predictions: Optional[Union[np.ndarray, Sequence]] = None,
        binary_probabilities: Optional[np.ndarray] = None,
        binary_model_name: str = "rf_full",
        multiclass_predictions: Optional[Union[np.ndarray, Sequence]] = None,
        multiclass_probabilities: Optional[np.ndarray] = None,
        multiclass_classes: Optional[Sequence[str]] = None,
        multiclass_model_name: str = "multiclass_rf_full",
        anomaly_scores: Optional[Union[np.ndarray, Sequence]] = None,
        anomaly_threshold: float = 0.6430,
        anomaly_model_name: str = "iforest_port_100k",
        alert_on_threat_only: bool = True,
    ) -> List[Alert]:
        """Processes a batch of flows across all active engines and returns deduplicated alerts.

        Preserves negative assessments: engines that do not flag a flow are fully recorded in
        evidence['engines'] rather than appearing as missing predictions.

        Args:
            flow_records: Originating flow records (DataFrame or list of dicts).
            binary_predictions: Optional binary model outputs.
            binary_probabilities: Optional binary model probabilities.
            binary_model_name: Identifier for binary model.
            multiclass_predictions: Optional multiclass model outputs.
            multiclass_probabilities: Optional multiclass probabilities.
            multiclass_classes: Canonical class list.
            multiclass_model_name: Identifier for multiclass model.
            anomaly_scores: Optional anomaly scores A(x).
            anomaly_threshold: Calibrated decision threshold.
            anomaly_model_name: Identifier for anomaly model.
            alert_on_threat_only: If True, only emits alerts for flows where at least one engine
                                  flagged a threat.

        Returns:
            List of consolidated Alert objects with guaranteed unique flow_ids.
        """
        raw_alerts: List[Alert] = []

        # 1. Run Binary Adapter across all flows (capture both attack and benign)
        if binary_predictions is not None:
            bin_adapter = BinaryAlertAdapter(
                model_name=binary_model_name,
                alert_on_attack_only=False,
            )
            bin_alerts = bin_adapter.adapt(
                predictions=binary_predictions,
                flow_records=flow_records,
                probabilities=binary_probabilities,
            )
            raw_alerts.extend(bin_alerts)

        # 2. Run Multiclass Adapter across all flows (capture both attack and benign)
        if multiclass_predictions is not None:
            mc_adapter = MulticlassAlertAdapter(
                model_name=multiclass_model_name,
                alert_on_attack_only=False,
                classes=multiclass_classes,
            )
            mc_alerts = mc_adapter.adapt(
                predictions=multiclass_predictions,
                flow_records=flow_records,
                probabilities=multiclass_probabilities,
                classes=multiclass_classes,
            )
            raw_alerts.extend(mc_alerts)

        # 3. Run Anomaly Adapter across all flows (capture both anomalous and normal)
        if anomaly_scores is not None:
            ano_adapter = AnomalyAlertAdapter(
                model_name=anomaly_model_name,
                calibrated_threshold=anomaly_threshold,
                alert_on_anomaly_only=False,
            )
            ano_alerts = ano_adapter.adapt(
                anomaly_scores=anomaly_scores,
                flow_records=flow_records,
                calibrated_threshold=anomaly_threshold,
            )
            raw_alerts.extend(ano_alerts)

        # 4. Consolidate and filter threats with complete evidence preservation
        return cls.consolidate_alerts(
            raw_alerts,
            alert_on_threat_only=alert_on_threat_only,
        )
