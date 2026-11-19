"""VYUHA Machine Learning Inference & Fusion Engine (Phase 6).

Executes genuine on-demand inference across the three primary trained models:
1. Binary Random Forest (rf_full.joblib)
2. Multiclass Random Forest (multiclass_rf_full.joblib)
3. Isolation Forest Anomaly Detection (iforest_full_100k.joblib)

Enforces:
- Models are loaded once via ModelRegistry.
- Exact 63-feature input validation and sanitization via FlowInferencePreprocessor.
- Probability mapping using estimator.classes_.
- Anomaly score inversion: A(x) = -score_samples(x).
- Multi-engine consensus fusion without conflating anomalies with confirmed attacks.
- Preserves individual engine evidence and negative evaluations.
- Generates transparent, explainable triage summaries.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.models.preprocessor import CANONICAL_63_FEATURES, FlowInferencePreprocessor
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class BinaryInferenceResult:
    """Inference output from the Supervised Binary Random Forest model."""
    prediction: int  # 0 = BENIGN, 1 = ATTACK
    label: str       # "BENIGN" or "ATTACK"
    is_attack: bool
    confidence: float
    benign_probability: float
    attack_probability: float
    model_name: str = "rf_full"


@dataclass
class MulticlassInferenceResult:
    """Inference output from the Supervised Multiclass Random Forest model."""
    predicted_category: str
    is_attack: bool
    confidence: float
    class_probabilities: Dict[str, float]
    model_name: str = "multiclass_rf_full"


@dataclass
class AnomalyInferenceResult:
    """Inference output from the Unsupervised Isolation Forest model."""
    anomaly_score: float
    calibrated_threshold: float
    threshold_name: str
    is_anomaly: bool
    threat_status: str  # "ANOMALY" if is_anomaly else "BENIGN"
    is_attack: bool = False  # Strictly False: unsupervised anomaly is not a confirmed attack
    model_name: str = "iforest_full_100k"


@dataclass
class FusedFlowEvaluation:
    """Consolidated multi-engine evaluation result for a single network flow."""
    flow_index: int
    flow_id: str
    binary: BinaryInferenceResult
    multiclass: MulticlassInferenceResult
    anomaly: AnomalyInferenceResult
    consensus_status: str
    consensus_threat: str  # "ATTACK", "ANOMALY", or "BENIGN"
    is_attack: bool
    flagged_as_threat: bool
    primary_category: Optional[str]
    provisional_severity: str
    explanation: str
    raw_flow_features: Dict[str, Any] = field(default_factory=dict)
    endpoint_metadata: Dict[str, Any] = field(default_factory=dict)


class InferenceEngine:
    """Executes multi-model inference and synthesizes cross-engine consensus."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        preprocessor: Optional[FlowInferencePreprocessor] = None,
        default_anomaly_tier: str = "fpr_1.0pct",
    ) -> None:
        """Initializes the inference engine.

        Args:
            registry: ModelRegistry instance (defaults to singleton registry).
            preprocessor: FlowInferencePreprocessor instance.
            default_anomaly_tier: Target calibration tier ('fpr_1.0pct', 'fpr_5.0pct', etc.).
        """
        self.registry = registry or ModelRegistry()
        self.preprocessor = preprocessor or FlowInferencePreprocessor()
        self.default_anomaly_tier = default_anomaly_tier

    def evaluate_flows(
        self,
        flow_records: Sequence[Dict[str, Any]],
        anomaly_tier: Optional[str] = None,
    ) -> List[FusedFlowEvaluation]:
        """Evaluates a batch of flow records through all three trained models.

        Args:
            flow_records: List of flow dictionaries.
            anomaly_tier: Optional anomaly threshold tier override.

        Returns:
            List of FusedFlowEvaluation objects, one for each input flow.
        """
        if not flow_records:
            return []

        # 1. Validate and prepare input DataFrame with strict 63 features
        X_df = self.preprocessor.prepare_flow_dataframe(flow_records)
        n_samples = len(X_df)

        # 2. Execute Binary Intrusion Detection (rf_full)
        binary_results = self._predict_binary(X_df)

        # 3. Execute Multiclass Attack Classification (multiclass_rf_full)
        multiclass_results = self._predict_multiclass(X_df)

        # 4. Execute Isolation Forest Anomaly Detection (iforest_full_100k)
        active_tier = anomaly_tier or self.default_anomaly_tier
        anomaly_results = self._predict_anomaly(X_df, tier=active_tier)

        # 5. Extract unrecorded or provided endpoint metadata (without fabrication)
        endpoint_metas = [self._extract_endpoint_metadata(r) for r in flow_records]

        # 6. Fuse results into transparent evaluations
        evaluations: List[FusedFlowEvaluation] = []
        for i in range(n_samples):
            bin_res = binary_results[i]
            mc_res = multiclass_results[i]
            ano_res = anomaly_results[i]
            raw_rec = flow_records[i]
            meta = endpoint_metas[i]

            flow_id = raw_rec.get("Flow ID") or raw_rec.get("flow_id") or f"flow_{i:04d}"

            fused = self._fuse_predictions(
                index=i,
                flow_id=str(flow_id),
                bin_res=bin_res,
                mc_res=mc_res,
                ano_res=ano_res,
                raw_rec=raw_rec,
                metadata=meta,
            )
            evaluations.append(fused)

        return evaluations

    def evaluate_flow(
        self,
        flow_record: Dict[str, Any],
        anomaly_tier: Optional[str] = None,
    ) -> FusedFlowEvaluation:
        """Evaluates a single network flow record."""
        results = self.evaluate_flows([flow_record], anomaly_tier=anomaly_tier)
        return results[0]

    # -------------------------------------------------------------------------
    # Internal Engine Predictions
    # -------------------------------------------------------------------------

    def _predict_binary(self, X_df: pd.DataFrame) -> List[BinaryInferenceResult]:
        """Executes binary pipeline inference."""
        pipeline = self.registry.get_binary_model()

        # Run pipeline predict and predict_proba
        preds = pipeline.predict(X_df)
        probs = pipeline.predict_proba(X_df)

        clf = pipeline.named_steps["classifier"]
        classes = list(clf.classes_)  # [0, 1]
        idx_0 = classes.index(0) if 0 in classes else 0
        idx_1 = classes.index(1) if 1 in classes else 1

        results = []
        for i in range(len(preds)):
            pred_val = int(preds[i])
            is_atk = bool(pred_val == 1)
            b_prob = float(probs[i, idx_0])
            a_prob = float(probs[i, idx_1])
            conf = a_prob if is_atk else b_prob

            results.append(
                BinaryInferenceResult(
                    prediction=pred_val,
                    label="ATTACK" if is_atk else "BENIGN",
                    is_attack=is_atk,
                    confidence=round(conf, 4),
                    benign_probability=round(b_prob, 4),
                    attack_probability=round(a_prob, 4),
                )
            )
        return results

    def _predict_multiclass(self, X_df: pd.DataFrame) -> List[MulticlassInferenceResult]:
        """Executes multiclass pipeline inference."""
        pipeline = self.registry.get_multiclass_model()

        preds = pipeline.predict(X_df)
        probs = pipeline.predict_proba(X_df)

        clf = pipeline.named_steps["classifier"]
        classes = list(clf.classes_)

        results = []
        for i in range(len(preds)):
            cat = str(preds[i]).strip()
            is_atk = bool(cat != "BENIGN")

            class_probs = {c: round(float(probs[i, idx]), 4) for idx, c in enumerate(classes)}
            top_conf = round(float(np.max(probs[i])), 4)

            results.append(
                MulticlassInferenceResult(
                    predicted_category=cat,
                    is_attack=is_atk,
                    confidence=top_conf,
                    class_probabilities=class_probs,
                )
            )
        return results

    def _predict_anomaly(
        self,
        X_df: pd.DataFrame,
        tier: str = "fpr_1.0pct",
    ) -> List[AnomalyInferenceResult]:
        """Executes unsupervised Isolation Forest scoring and threshold comparison."""
        pipeline, metadata = self.registry.get_anomaly_model()

        # Step through pipeline selector
        selector = pipeline.named_steps["selector"]
        detector = pipeline.named_steps["anomaly_detector"]

        X_trans = selector.transform(X_df)

        # Scikit-Learn score_samples is negative; invert for A(x) in [0, 1]
        raw_scores = detector.score_samples(X_trans)
        anomaly_scores = -raw_scores

        # Retrieve calibrated threshold
        thresholds_dict = metadata.get("calibrated_thresholds", {})
        if tier not in thresholds_dict:
            available = list(thresholds_dict.keys())
            logger.warning(
                "Requested threshold tier '%s' not found. Available: %s. Using default 'fpr_1.0pct'.",
                tier,
                available,
            )
            tier = "fpr_1.0pct" if "fpr_1.0pct" in thresholds_dict else available[0]

        threshold_val = float(thresholds_dict[tier]["threshold"])

        results = []
        for i in range(len(anomaly_scores)):
            score = float(anomaly_scores[i])
            is_anomaly = bool(score >= threshold_val)

            results.append(
                AnomalyInferenceResult(
                    anomaly_score=round(score, 4),
                    calibrated_threshold=round(threshold_val, 4),
                    threshold_name=tier,
                    is_anomaly=is_anomaly,
                    threat_status="ANOMALY" if is_anomaly else "BENIGN",
                    is_attack=False,  # Unsupervised anomaly is not an attack
                )
            )
        return results

    # -------------------------------------------------------------------------
    # Multi-Engine Consensus Fusion
    # -------------------------------------------------------------------------

    def _fuse_predictions(
        self,
        index: int,
        flow_id: str,
        bin_res: BinaryInferenceResult,
        mc_res: MulticlassInferenceResult,
        ano_res: AnomalyInferenceResult,
        raw_rec: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> FusedFlowEvaluation:
        """Synthesizes consensus assessment across binary, multiclass, and anomaly detection."""
        # Severity calculation based on threat category
        severity_map = {
            "Heartbleed": "Critical",
            "Infiltration": "Critical",
            "DoS_DDoS": "High",
            "Botnet": "High",
            "WebAttack": "Medium",
            "BruteForce": "Medium",
            "PortScan": "Low",
            "BENIGN": "Info",
        }

        # Case 1: Corroborated Supervised Detection (Binary ATTACK + Multiclass Attack)
        if bin_res.is_attack and mc_res.is_attack:
            status = "CORROBORATED_ATTACK"
            threat = "ATTACK"
            is_attack = True
            flagged = True
            cat = mc_res.predicted_category
            sev = severity_map.get(cat, "High")
            ano_note = (
                f" Concurrently scored {ano_res.anomaly_score:.3f} by Isolation Forest (threshold {ano_res.calibrated_threshold:.3f})."
                if ano_res.is_anomaly
                else " Isolation Forest scored flow within nominal baseline."
            )
            explanation = (
                f"Corroborated supervised detection: Binary Random Forest flagged ATTACK ({bin_res.attack_probability * 100:.1f}%) "
                f"and Multiclass Random Forest classified threat family as '{cat}' ({mc_res.confidence * 100:.1f}% confidence).{ano_note}"
            )

        # Case 2: Conflicting Supervised Assessment (Binary ATTACK, Multiclass BENIGN)
        elif bin_res.is_attack and not mc_res.is_attack:
            status = "SUPERVISED_CONFLICT"
            threat = "ATTACK"
            is_attack = True
            flagged = True
            cat = "Unspecified Attack"
            sev = "Medium"
            ano_note = (
                f" Isolation Forest concurrently flagged novelty (score {ano_res.anomaly_score:.4f} >= {ano_res.calibrated_threshold:.4f} threshold)."
                if ano_res.is_anomaly
                else " Isolation Forest scored flow within nominal baseline."
            )
            explanation = (
                f"Conflicting supervised assessment: Binary model flagged ATTACK ({bin_res.attack_probability * 100:.1f}%), "
                f"but Multiclass model classified flow as BENIGN ({mc_res.confidence * 100:.1f}%).{ano_note} "
                f"Preserving conflicting evidence for analyst review rather than misleading three-engine agreement."
            )

        # Case 3: Conflicting Supervised Assessment (Binary BENIGN, Multiclass Attack)
        elif not bin_res.is_attack and mc_res.is_attack:
            status = "SUPERVISED_CONFLICT"
            threat = "ATTACK"
            is_attack = True
            flagged = True
            cat = mc_res.predicted_category
            sev = "Medium"
            ano_note = (
                f" Isolation Forest concurrently flagged novelty (score {ano_res.anomaly_score:.4f} >= {ano_res.calibrated_threshold:.4f} threshold)."
                if ano_res.is_anomaly
                else " Isolation Forest scored flow within nominal baseline."
            )
            explanation = (
                f"Conflicting supervised assessment: Multiclass model predicted '{cat}' ({mc_res.confidence * 100:.1f}%), "
                f"but Binary model evaluated flow as BENIGN ({bin_res.benign_probability * 100:.1f}%).{ano_note} "
                f"Preserving multiclass signature indicator for triage inspection."
            )

        # Case 4: Statistical Anomaly Only (Supervised models BENIGN, Isolation Forest flagged)
        elif ano_res.is_anomaly:
            status = "NOVEL_ANOMALY"
            threat = "ANOMALY"
            is_attack = False  # Anomaly is NOT an attack
            flagged = True
            cat = None
            sev = "Medium" if ano_res.anomaly_score >= 0.65 else "Low"
            explanation = (
                f"Unsupervised statistical novelty: Isolation Forest scored flow at {ano_res.anomaly_score:.4f}, "
                f"exceeding the calibrated {ano_res.threshold_name} threshold ({ano_res.calibrated_threshold:.4f}). "
                f"Supervised models evaluated flow as BENIGN (Binary {bin_res.benign_probability * 100:.1f}%, Multiclass {mc_res.confidence * 100:.1f}%). "
                f"This flow represents out-of-distribution traffic, not a confirmed attack signature."
            )

        # Case 5: Corroborated Benign (All engines normal)
        else:
            status = "CORROBORATED_BENIGN"
            threat = "BENIGN"
            is_attack = False
            flagged = False
            cat = None
            sev = "Info"
            explanation = (
                f"All detection engines evaluated flow as nominal: Binary BENIGN ({bin_res.benign_probability * 100:.1f}%), "
                f"Multiclass BENIGN ({mc_res.confidence * 100:.1f}%), Isolation Forest score {ano_res.anomaly_score:.4f} "
                f"below threshold {ano_res.calibrated_threshold:.4f}."
            )

        return FusedFlowEvaluation(
            flow_index=index,
            flow_id=flow_id,
            binary=bin_res,
            multiclass=mc_res,
            anomaly=ano_res,
            consensus_status=status,
            consensus_threat=threat,
            is_attack=is_attack,
            flagged_as_threat=flagged,
            primary_category=cat,
            provisional_severity=sev,
            explanation=explanation,
            raw_flow_features=raw_rec,
            endpoint_metadata=metadata,
        )

    def _extract_endpoint_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts available endpoint metadata without fabricating missing fields."""
        return {
            "source_ip": record.get("source_ip") or record.get("Source IP"),
            "destination_ip": record.get("destination_ip") or record.get("Destination IP"),
            "source_port": record.get("source_port") or record.get("Source Port"),
            "destination_port": record.get("destination_port") or record.get("Destination Port"),
            "protocol": record.get("protocol") or record.get("Protocol"),
            "timestamp": record.get("timestamp") or record.get("Timestamp"),
        }
