"""VYUHA Temporal Correlation Engine Module (Phase 4.2).

Provides explainable, sliding-window temporal correlation across normalized Alert objects.
Key Capabilities:
1. Timezone-aware timestamp standardization (normalized to UTC).
2. Chronological event sorting and delta-time calculations (Delta t = t_B - t_A >= 0).
3. Configurable sliding-window observation boundaries.
4. Transparent handling of missing/unparseable timestamps (marked as ineligible with reasons).
5. Modular, independently testable correlation rules:
   - Temporal Proximity Rule (co-occurrence within window W)
   - Repeated Alert Rule (repeated model/prediction within window W)
   - Repeated Attack Category Rule (bursts of identical threat family within window W)
   - Sequential Ordering Rule (candidate attack-pattern sequences: Reconnaissance -> Exploitation -> C2)
6. Transitive single-linkage temporal clustering and explainable relationship graph evidence.

Strict Cybersecurity Grounding:
- Candidate attack-pattern sequences represent structural hypotheses; temporal ordering does NOT
  establish causal connection, coordinated multi-stage execution, or common attacker identity.
- Missing IP or port values NEVER count as shared entities (None == None is never treated as a match).
- Temporal proximity alone is circumstantial co-occurrence and does NOT establish an attack relationship.
- Anomalies and attacks are never automatically merged or classified as the same campaign
  solely based on time co-occurrence.
- CICIDS2017 MachineLearningCVE lacks timestamps; the engine gracefully rejects timestamp-free alerts
  without fabricating timestamps or using row indices as timestamps.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field
import datetime
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import uuid
import pandas as pd

from src.intelligence.schema import Alert, AlertSeverity, ThreatStatus

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Models for Temporal Correlation
# -----------------------------------------------------------------------------

@dataclass
class IneligibleAlertRecord:
    """Documents alerts that could not participate in temporal correlation."""
    alert_id: str
    flow_id: str
    reason: str
    raw_timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemporalRelationship:
    """Encapsulates an explainable directed temporal link between two alerts (t_A <= t_B)."""
    relationship_id: str
    source_alert_id: str
    target_alert_id: str
    rule_name: str
    source_timestamp: str
    target_timestamp: str
    time_delta_seconds: float
    window_seconds: float
    is_sequential: bool  # True if t_A < t_B; False if simultaneous (t_A == t_B)
    evidence: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemporalCluster:
    """Represents a coherent cluster of alerts formed via transitive single-linkage temporal chaining."""
    cluster_id: str
    first_timestamp: str  # ISO-8601 UTC of earliest event in cluster
    last_timestamp: str  # ISO-8601 UTC of latest event in cluster
    total_duration_seconds: float  # Elapsed seconds between first and last event
    max_cluster_gap_seconds: float  # Maximum allowed inter-event gap
    alerts_count: int  # Number of alerts in cluster
    alert_ids: List[str]  # IDs of all alerts in cluster
    relationships: List[TemporalRelationship] = field(default_factory=list)
    relationship_count: int = 0
    dominant_categories: List[str] = field(default_factory=list)
    clustering_method: str = "transitive_single_linkage"

    @property
    def start_time(self) -> str:
        """Backward-compatible alias for first_timestamp."""
        return self.first_timestamp

    @property
    def end_time(self) -> str:
        """Backward-compatible alias for last_timestamp."""
        return self.last_timestamp

    @property
    def duration_seconds(self) -> float:
        """Backward-compatible alias for total_duration_seconds."""
        return self.total_duration_seconds

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["relationships"] = [r.to_dict() for r in self.relationships]
        return d


@dataclass
class TemporalCorrelationResult:
    """Comprehensive output of the temporal correlation engine."""
    total_alerts_ingested: int
    eligible_alerts_count: int
    ineligible_alerts_count: int
    ineligible_records: List[IneligibleAlertRecord]
    ordered_alerts: List[Alert]
    relationships: List[TemporalRelationship]
    clusters: List[TemporalCluster]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_alerts_ingested": self.total_alerts_ingested,
            "eligible_alerts_count": self.eligible_alerts_count,
            "ineligible_alerts_count": self.ineligible_alerts_count,
            "ineligible_records": [r.to_dict() for r in self.ineligible_records],
            "ordered_alerts": [a.to_dict() for a in self.ordered_alerts],
            "relationships": [r.to_dict() for r in self.relationships],
            "clusters": [c.to_dict() for c in self.clusters],
            "metadata": self.metadata,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def get_relationships_by_rule(self, rule_name: str) -> List[TemporalRelationship]:
        """Filters relationships by matched rule name."""
        return [r for r in self.relationships if r.rule_name == rule_name]

    def get_relationships_for_alert(self, alert_id: str) -> List[TemporalRelationship]:
        """Finds all relationships involving a specific alert ID (as source or target)."""
        return [
            r for r in self.relationships
            if r.source_alert_id == alert_id or r.target_alert_id == alert_id
        ]


# -----------------------------------------------------------------------------
# Timestamp Normalization Utilities
# -----------------------------------------------------------------------------

def parse_and_normalize_timestamp(ts_val: Any) -> Optional[datetime.datetime]:
    """Parses a timestamp value and normalizes it to a UTC timezone-aware datetime.

    Handles:
    - Timezone-aware datetimes/timestamps (e.g. UTC, +05:30, -05:00) -> Converted to UTC.
    - Naive datetimes/timestamps -> Assumed UTC and localized.
    - Standard ISO-8601 strings, RFC strings, and standard date strings.
    - None / NaN / Empty strings -> Returns None.
    - Malformed / Unparseable strings -> Returns None.

    Args:
        ts_val: Input timestamp (string, datetime, pd.Timestamp, or None).

    Returns:
        UTC datetime object or None if unparseable/missing.
    """
    if ts_val is None or (isinstance(ts_val, float) and pd.isna(ts_val)):
        return None

    if isinstance(ts_val, (datetime.datetime, pd.Timestamp)):
        if ts_val.tzinfo is None:
            return ts_val.replace(tzinfo=datetime.timezone.utc)
        return ts_val.astimezone(datetime.timezone.utc)

    ts_str = str(ts_val).strip()
    if not ts_str or ts_str.lower() in ("nan", "none", "null"):
        return None

    try:
        dt = pd.to_datetime(ts_str)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        else:
            dt = dt.tz_convert("UTC")
        return dt.to_pydatetime()
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Correlation Rules Interface & Implementations
# -----------------------------------------------------------------------------

class BaseTemporalRule(ABC):
    """Abstract base class for all temporal correlation rules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique rule name."""
        pass

    @abstractmethod
    def evaluate(
        self,
        alert_a: Alert,
        alert_b: Alert,
        dt_seconds: float,
        window_seconds: float,
        ts_a: datetime.datetime,
        ts_b: datetime.datetime,
    ) -> Optional[TemporalRelationship]:
        """Evaluates whether a directed temporal relationship exists between alert_a and alert_b (t_A <= t_B).

        Args:
            alert_a: Chronologically earlier (or simultaneous) alert.
            alert_b: Chronologically later (or simultaneous) alert.
            dt_seconds: Time difference in seconds (ts_b - ts_a >= 0).
            window_seconds: Configured observation window boundary.
            ts_a: Normalized UTC datetime for alert_a.
            ts_b: Normalized UTC datetime for alert_b.

        Returns:
            TemporalRelationship if rule matches; None otherwise.
        """
        pass

    def _extract_entity_evidence(self, alert_a: Alert, alert_b: Alert) -> Dict[str, Any]:
        """Extracts shared entities, missing entities, and classifies proximity type.

        CRITICAL INTEGRITY GUARANTEE:
        Missing values (None in alert_a and None in alert_b) NEVER count as shared entities.
        Only genuine, matching non-None values count as shared entities.
        """
        shared_entities: Dict[str, Any] = {}
        missing_entities: Dict[str, List[str]] = {
            "alert_a": [],
            "alert_b": [],
        }

        # Check source_ip
        if not alert_a.source_ip:
            missing_entities["alert_a"].append("source_ip")
        if not alert_b.source_ip:
            missing_entities["alert_b"].append("source_ip")
        if alert_a.source_ip and alert_b.source_ip and alert_a.source_ip == alert_b.source_ip:
            shared_entities["source_ip"] = alert_a.source_ip

        # Check destination_ip
        if not alert_a.destination_ip:
            missing_entities["alert_a"].append("destination_ip")
        if not alert_b.destination_ip:
            missing_entities["alert_b"].append("destination_ip")
        if alert_a.destination_ip and alert_b.destination_ip and alert_a.destination_ip == alert_b.destination_ip:
            shared_entities["destination_ip"] = alert_a.destination_ip

        # Check destination_port
        if alert_a.destination_port is None:
            missing_entities["alert_a"].append("destination_port")
        if alert_b.destination_port is None:
            missing_entities["alert_b"].append("destination_port")
        if (alert_a.destination_port is not None and alert_b.destination_port is not None and
                alert_a.destination_port == alert_b.destination_port):
            shared_entities["destination_port"] = alert_a.destination_port

        # Check source_port
        if alert_a.source_port is None:
            missing_entities["alert_a"].append("source_port")
        if alert_b.source_port is None:
            missing_entities["alert_b"].append("source_port")
        if (alert_a.source_port is not None and alert_b.source_port is not None and
                alert_a.source_port == alert_b.source_port):
            shared_entities["source_port"] = alert_a.source_port

        # Differentiate proximity classification
        has_shared = bool(shared_entities)
        if not has_shared:
            proximity_classification = "time_only_proximity"
        elif len(shared_entities) == 1:
            if "source_ip" in shared_entities:
                proximity_classification = "shared_source_ip"
            elif "destination_ip" in shared_entities:
                proximity_classification = "shared_destination_ip"
            elif "destination_port" in shared_entities:
                proximity_classification = "shared_destination_port"
            else:
                proximity_classification = "shared_network_entity"
        else:
            proximity_classification = "shared_network_entity"

        return {
            "shared_entities": shared_entities,
            "has_shared_entities": has_shared,
            "proximity_classification": proximity_classification,
            "missing_entities": missing_entities,
        }

    def _extract_shared_entities(self, alert_a: Alert, alert_b: Alert) -> Dict[str, Any]:
        """Backward-compatible helper returning dictionary of shared endpoints."""
        return self._extract_entity_evidence(alert_a, alert_b)["shared_entities"]


class TemporalProximityRule(BaseTemporalRule):
    """Identifies alerts occurring within observation window W.

    Strict Cybersecurity Grounding:
    Circumstantial temporal co-occurrence without claiming attack causation.
    Differentiates time-only proximity from shared entity proximity.
    """

    def __init__(self, name: str = "temporal_proximity"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def evaluate(
        self,
        alert_a: Alert,
        alert_b: Alert,
        dt_seconds: float,
        window_seconds: float,
        ts_a: datetime.datetime,
        ts_b: datetime.datetime,
    ) -> Optional[TemporalRelationship]:
        if not (0.0 <= dt_seconds <= window_seconds):
            return None

        entity_info = self._extract_entity_evidence(alert_a, alert_b)
        is_seq = dt_seconds > 0.0

        evidence = {
            "time_delta_seconds": round(dt_seconds, 4),
            "window_seconds": window_seconds,
            "proximity_classification": entity_info["proximity_classification"],
            "has_shared_entities": entity_info["has_shared_entities"],
            "shared_entities": entity_info["shared_entities"],
            "missing_entities": entity_info["missing_entities"],
            "threat_statuses": [alert_a.threat_status, alert_b.threat_status],
            "predictions": [alert_a.prediction, alert_b.prediction],
            "categories": [alert_a.attack_category, alert_b.attack_category],
            "provisional_severities": [alert_a.severity, alert_b.severity],
            "causation_caveat": (
                "Temporal proximity is circumstantial co-occurrence within the observation window; "
                "time-only proximity alone does NOT establish an attack relationship, causal connection, "
                "or coordinated campaign."
            ),
            "interpretation": (
                f"Circumstantial temporal proximity within observation window. "
                f"Classification: {entity_info['proximity_classification']}."
            ),
        }

        return TemporalRelationship(
            relationship_id=f"trel_prox_{uuid.uuid4().hex[:12]}",
            source_alert_id=alert_a.alert_id,
            target_alert_id=alert_b.alert_id,
            rule_name=self.name,
            source_timestamp=ts_a.isoformat(),
            target_timestamp=ts_b.isoformat(),
            time_delta_seconds=round(dt_seconds, 4),
            window_seconds=window_seconds,
            is_sequential=is_seq,
            evidence=evidence,
        )


class RepeatedAlertRule(BaseTemporalRule):
    """Identifies repeated identical alert predictions or repeated engine detections within window W."""

    def __init__(self, name: str = "repeated_alert"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def evaluate(
        self,
        alert_a: Alert,
        alert_b: Alert,
        dt_seconds: float,
        window_seconds: float,
        ts_a: datetime.datetime,
        ts_b: datetime.datetime,
    ) -> Optional[TemporalRelationship]:
        if not (0.0 <= dt_seconds <= window_seconds):
            return None

        is_same_model_and_pred = (
            alert_a.model_name == alert_b.model_name and
            alert_a.prediction == alert_b.prediction and
            alert_a.threat_status == alert_b.threat_status
        )
        is_same_category = (
            alert_a.attack_category is not None and
            alert_a.attack_category == alert_b.attack_category
        )

        if not (is_same_model_and_pred or is_same_category):
            return None

        entity_info = self._extract_entity_evidence(alert_a, alert_b)
        is_seq = dt_seconds > 0.0

        evidence = {
            "time_delta_seconds": round(dt_seconds, 4),
            "window_seconds": window_seconds,
            "repetition_type": (
                "identical_model_prediction" if is_same_model_and_pred else "identical_category"
            ),
            "repeated_prediction": alert_a.prediction,
            "repeated_threat_status": alert_a.threat_status,
            "proximity_classification": entity_info["proximity_classification"],
            "shared_entities": entity_info["shared_entities"],
            "has_shared_entities": entity_info["has_shared_entities"],
            "missing_entities": entity_info["missing_entities"],
            "causation_caveat": (
                "Recurrent detection pattern indicates repeated or automated activity; "
                "temporal recurrence alone does not establish causal campaign linkage."
            ),
            "interpretation": (
                "Recurrent detection pattern observed within time window. "
                "Suggests sustained, repeated, or automated activity."
            ),
        }

        return TemporalRelationship(
            relationship_id=f"trel_rep_alt_{uuid.uuid4().hex[:12]}",
            source_alert_id=alert_a.alert_id,
            target_alert_id=alert_b.alert_id,
            rule_name=self.name,
            source_timestamp=ts_a.isoformat(),
            target_timestamp=ts_b.isoformat(),
            time_delta_seconds=round(dt_seconds, 4),
            window_seconds=window_seconds,
            is_sequential=is_seq,
            evidence=evidence,
        )


class RepeatedCategoryRule(BaseTemporalRule):
    """Identifies bursts of alerts sharing the same non-benign attack category within window W."""

    def __init__(self, name: str = "repeated_category"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def evaluate(
        self,
        alert_a: Alert,
        alert_b: Alert,
        dt_seconds: float,
        window_seconds: float,
        ts_a: datetime.datetime,
        ts_b: datetime.datetime,
    ) -> Optional[TemporalRelationship]:
        if not (0.0 <= dt_seconds <= window_seconds):
            return None

        cat_a = alert_a.attack_category
        cat_b = alert_b.attack_category

        if not cat_a or not cat_b or cat_a == "BENIGN" or cat_b == "BENIGN":
            return None

        if cat_a != cat_b:
            return None

        entity_info = self._extract_entity_evidence(alert_a, alert_b)
        is_seq = dt_seconds > 0.0

        evidence = {
            "time_delta_seconds": round(dt_seconds, 4),
            "window_seconds": window_seconds,
            "attack_category": cat_a,
            "burst_interval_seconds": round(dt_seconds, 4),
            "proximity_classification": entity_info["proximity_classification"],
            "shared_entities": entity_info["shared_entities"],
            "has_shared_entities": entity_info["has_shared_entities"],
            "missing_entities": entity_info["missing_entities"],
            "target_ports": [alert_a.destination_port, alert_b.destination_port],
            "causation_caveat": (
                f"Repeated '{cat_a}' occurrences reflect temporal category clustering; "
                "does not establish identical actor attribution without endpoint corroboration."
            ),
            "interpretation": (
                f"Wave or burst of repeated '{cat_a}' threat category within observation window."
            ),
        }

        return TemporalRelationship(
            relationship_id=f"trel_rep_cat_{uuid.uuid4().hex[:12]}",
            source_alert_id=alert_a.alert_id,
            target_alert_id=alert_b.alert_id,
            rule_name=self.name,
            source_timestamp=ts_a.isoformat(),
            target_timestamp=ts_b.isoformat(),
            time_delta_seconds=round(dt_seconds, 4),
            window_seconds=window_seconds,
            is_sequential=is_seq,
            evidence=evidence,
        )


class SequentialOrderingRule(BaseTemporalRule):
    """Identifies strict directed chronological progression (t_A < t_B) representing candidate attack-pattern sequences.

    Models plausible cyber attack progression patterns:
    1. Reconnaissance -> Targeted Exploitation / DoS:
       - PortScan -> BruteForce
       - PortScan -> WebAttack
       - PortScan -> DoS_DDoS
       - PortScan -> Infiltration
       - PortScan -> Heartbleed
    2. Exploitation -> Botnet / Persistence / C2:
       - WebAttack -> Botnet
       - Infiltration -> Botnet
       - BruteForce -> Botnet
    3. Exploitation Sequencing:
       - Heartbleed -> Infiltration

    Strict Cybersecurity Grounding:
    Sequential patterns represent candidate attack-pattern hypotheses based on timing;
    temporal ordering does NOT establish causal connection, coordinated multi-stage execution,
    or common attacker identity.
    """

    CANDIDATE_ATTACK_SEQUENCES: Set[Tuple[str, str]] = {
        ("PortScan", "BruteForce"),
        ("PortScan", "WebAttack"),
        ("PortScan", "DoS_DDoS"),
        ("PortScan", "Infiltration"),
        ("PortScan", "Heartbleed"),
        ("WebAttack", "Botnet"),
        ("Infiltration", "Botnet"),
        ("BruteForce", "Botnet"),
        ("Heartbleed", "Infiltration"),
    }

    def __init__(self, name: str = "sequential_ordering"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def evaluate(
        self,
        alert_a: Alert,
        alert_b: Alert,
        dt_seconds: float,
        window_seconds: float,
        ts_a: datetime.datetime,
        ts_b: datetime.datetime,
    ) -> Optional[TemporalRelationship]:
        # Strict chronological precedence required: t_A < t_B
        if not (0.0 < dt_seconds <= window_seconds):
            return None

        cat_a = alert_a.attack_category
        cat_b = alert_b.attack_category

        if not cat_a or not cat_b or cat_a == "BENIGN" or cat_b == "BENIGN":
            return None

        pair = (cat_a, cat_b)
        is_candidate_sequence = pair in self.CANDIDATE_ATTACK_SEQUENCES

        # Match strictly if pair reflects a recognized candidate attack pattern
        if not is_candidate_sequence:
            return None

        entity_info = self._extract_entity_evidence(alert_a, alert_b)
        has_shared_target = bool(
            entity_info["shared_entities"].get("destination_ip") or
            entity_info["shared_entities"].get("destination_port")
        )

        if pair[0] == "PortScan":
            candidate_stage = "candidate_reconnaissance_to_exploitation"
            progression_stage = "reconnaissance_to_exploitation"
        elif pair[1] == "Botnet":
            candidate_stage = "candidate_exploitation_to_c2"
            progression_stage = "exploitation_to_c2"
        else:
            candidate_stage = "candidate_sequential_attack_pattern"
            progression_stage = "sequential_threat_progression"

        evidence = {
            "earlier_alert_id": alert_a.alert_id,
            "later_alert_id": alert_b.alert_id,
            "earlier_category": cat_a,
            "later_category": cat_b,
            "time_delta_seconds": round(dt_seconds, 4),
            "window_seconds": window_seconds,
            "candidate_stage": candidate_stage,
            "is_candidate_pattern_sequence": True,
            "shared_target_detected": has_shared_target,
            "shared_entity_evidence": entity_info["shared_entities"],
            "has_shared_entities": entity_info["has_shared_entities"],
            "proximity_classification": entity_info["proximity_classification"],
            "missing_entity_information": entity_info["missing_entities"],
            "causation_caveat": (
                f"Observed sequential pattern is a candidate attack-pattern hypothesis "
                f"('{cat_a}' -> '{cat_b}' within {round(dt_seconds, 2)}s); temporal ordering does NOT "
                "establish causal connection, coordinated multi-stage execution, or common attacker identity."
            ),
            # Backward-compatible fields
            "preceding_category": cat_a,
            "subsequent_category": cat_b,
            "progression_stage": progression_stage,
            "shared_entities": entity_info["shared_entities"],
        }

        return TemporalRelationship(
            relationship_id=f"trel_seq_{uuid.uuid4().hex[:12]}",
            source_alert_id=alert_a.alert_id,
            target_alert_id=alert_b.alert_id,
            rule_name=self.name,
            source_timestamp=ts_a.isoformat(),
            target_timestamp=ts_b.isoformat(),
            time_delta_seconds=round(dt_seconds, 4),
            window_seconds=window_seconds,
            is_sequential=True,
            evidence=evidence,
        )


# Default suite of rules
DEFAULT_TEMPORAL_RULES: List[BaseTemporalRule] = [
    TemporalProximityRule(),
    RepeatedAlertRule(),
    RepeatedCategoryRule(),
    SequentialOrderingRule(),
]


# -----------------------------------------------------------------------------
# Temporal Correlation Engine
# -----------------------------------------------------------------------------

class TemporalCorrelationEngine:
    """Configurable temporal correlation engine operating across canonical Alert streams."""

    def __init__(
        self,
        window_seconds: float = 300.0,
        rules: Optional[Sequence[BaseTemporalRule]] = None,
        max_cluster_gap_seconds: Optional[float] = None,
    ):
        """Initializes the TemporalCorrelationEngine.

        Args:
            window_seconds: Observation window boundary in seconds (default: 300.0s / 5 minutes).
            rules: Sequence of temporal correlation rules (defaults to standard 4 rules).
            max_cluster_gap_seconds: Maximum time gap between consecutive alerts before starting
                                     a new temporal cluster (defaults to window_seconds).
        """
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")

        self.window_seconds = float(window_seconds)
        self.rules: List[BaseTemporalRule] = list(rules) if rules is not None else DEFAULT_TEMPORAL_RULES
        self.max_cluster_gap = (
            float(max_cluster_gap_seconds)
            if max_cluster_gap_seconds is not None
            else self.window_seconds
        )

    def normalize_alert_timestamps(
        self, alerts: Sequence[Alert]
    ) -> Tuple[List[Tuple[datetime.datetime, Alert]], List[IneligibleAlertRecord]]:
        """Parses timestamps, normalizes to UTC, and partitions into eligible and ineligible sets.

        Args:
            alerts: Input sequence of Alert objects.

        Returns:
            Tuple of:
            - Sorted list of (normalized_utc_datetime, alert) tuples in ascending chronological order.
            - List of IneligibleAlertRecord objects with recorded exclusion reasons.
        """
        eligible: List[Tuple[datetime.datetime, Alert]] = []
        ineligible: List[IneligibleAlertRecord] = []

        for alt in alerts:
            raw_ts = alt.timestamp
            if raw_ts is None or not str(raw_ts).strip():
                ineligible.append(
                    IneligibleAlertRecord(
                        alert_id=alt.alert_id,
                        flow_id=alt.flow_id,
                        reason="missing_timestamp",
                        raw_timestamp=None,
                    )
                )
                continue

            parsed_dt = parse_and_normalize_timestamp(raw_ts)
            if parsed_dt is None:
                ineligible.append(
                    IneligibleAlertRecord(
                        alert_id=alt.alert_id,
                        flow_id=alt.flow_id,
                        reason=f"unparseable_timestamp: '{raw_ts}'",
                        raw_timestamp=str(raw_ts),
                    )
                )
                continue

            eligible.append((parsed_dt, alt))

        # Sort strictly chronologically in ascending order
        eligible.sort(key=lambda item: item[0])

        return eligible, ineligible

    def form_clusters(
        self,
        ordered_timed_alerts: List[Tuple[datetime.datetime, Alert]],
        relationships: List[TemporalRelationship],
    ) -> List[TemporalCluster]:
        """Groups chronologically ordered alerts into temporal clusters via transitive single-linkage clustering.

        Transitive Single-Linkage Behavior:
        If event A connects to B (gap <= max_cluster_gap), and B connects to C (gap <= max_cluster_gap),
        all three events [A, B, C] are chained into a single cluster, even if the total elapsed duration
        between A and C exceeds the pairwise window or gap threshold.

        Args:
            ordered_timed_alerts: Ascending sorted list of (datetime, alert) tuples.
            relationships: List of generated relationships across the alert set.

        Returns:
            List of TemporalCluster objects.
        """
        if not ordered_timed_alerts:
            return []

        clusters: List[TemporalCluster] = []
        current_cluster_alerts: List[Tuple[datetime.datetime, Alert]] = [ordered_timed_alerts[0]]

        for i in range(1, len(ordered_timed_alerts)):
            prev_dt, _ = ordered_timed_alerts[i - 1]
            curr_dt, _ = ordered_timed_alerts[i]
            gap = (curr_dt - prev_dt).total_seconds()

            if gap <= self.max_cluster_gap:
                current_cluster_alerts.append(ordered_timed_alerts[i])
            else:
                # Close current cluster
                cluster = self._build_cluster(current_cluster_alerts, relationships)
                clusters.append(cluster)
                current_cluster_alerts = [ordered_timed_alerts[i]]

        # Close final cluster
        if current_cluster_alerts:
            cluster = self._build_cluster(current_cluster_alerts, relationships)
            clusters.append(cluster)

        return clusters

    def _build_cluster(
        self,
        cluster_items: List[Tuple[datetime.datetime, Alert]],
        relationships: List[TemporalRelationship],
    ) -> TemporalCluster:
        """Helper to construct a TemporalCluster from a list of grouped alert tuples."""
        start_dt = cluster_items[0][0]
        end_dt = cluster_items[-1][0]
        duration = (end_dt - start_dt).total_seconds()

        alert_ids = [item[1].alert_id for item in cluster_items]
        alert_ids_set = set(alert_ids)

        # Extract pairwise relationships where both source and target belong to this cluster
        cluster_relationships = [
            r for r in relationships
            if r.source_alert_id in alert_ids_set and r.target_alert_id in alert_ids_set
        ]

        # Find dominant categories
        categories = [
            item[1].attack_category for item in cluster_items
            if item[1].attack_category and item[1].attack_category != "BENIGN"
        ]
        counts: Dict[str, int] = defaultdict(int)
        for cat in categories:
            counts[cat] += 1
        sorted_categories = sorted(counts.keys(), key=lambda c: counts[c], reverse=True)

        return TemporalCluster(
            cluster_id=f"tclust_{uuid.uuid4().hex[:12]}",
            first_timestamp=start_dt.isoformat(),
            last_timestamp=end_dt.isoformat(),
            total_duration_seconds=round(duration, 4),
            max_cluster_gap_seconds=self.max_cluster_gap,
            alerts_count=len(alert_ids),
            alert_ids=alert_ids,
            relationships=cluster_relationships,
            relationship_count=len(cluster_relationships),
            dominant_categories=sorted_categories,
            clustering_method="transitive_single_linkage",
        )

    def correlate(self, alerts: Sequence[Alert]) -> TemporalCorrelationResult:
        """Executes sliding-window temporal correlation across the provided alert stream.

        Args:
            alerts: Sequence of canonical Alert objects.

        Returns:
            TemporalCorrelationResult with chronological alerts, relationships, clusters,
            and ineligible alert records.
        """
        n_ingested = len(alerts)

        # 1. Normalize timestamps, validate timezones, and partition eligibility
        ordered_timed, ineligible = self.normalize_alert_timestamps(alerts)
        n_eligible = len(ordered_timed)

        # If no alerts have valid timestamps, return graceful result with limitations documented
        if n_eligible == 0:
            return TemporalCorrelationResult(
                total_alerts_ingested=n_ingested,
                eligible_alerts_count=0,
                ineligible_alerts_count=len(ineligible),
                ineligible_records=ineligible,
                ordered_alerts=[],
                relationships=[],
                clusters=[],
                metadata={
                    "status": "all_alerts_ineligible_missing_timestamps",
                    "window_seconds": self.window_seconds,
                    "dataset_limitation_warning": (
                        "None of the provided alerts contained a valid timestamp. "
                        "CICIDS2017 MachineLearningCVE lacks timestamp columns. "
                        "Temporal correlation requires genuine timestamped alert telemetry."
                    ),
                },
            )

        # 2. Sliding-Window Pairwise Rule Evaluation
        relationships: List[TemporalRelationship] = []

        # Compare pairs (i, j) where i < j and t_j - t_i <= window_seconds
        for i in range(n_eligible):
            ts_a, alert_a = ordered_timed[i]

            for j in range(i + 1, n_eligible):
                ts_b, alert_b = ordered_timed[j]
                dt = (ts_b - ts_a).total_seconds()

                # Because alerts are sorted chronologically, break once boundary is exceeded
                if dt > self.window_seconds:
                    break

                # Evaluate active rules
                for rule in self.rules:
                    rel = rule.evaluate(
                        alert_a=alert_a,
                        alert_b=alert_b,
                        dt_seconds=dt,
                        window_seconds=self.window_seconds,
                        ts_a=ts_a,
                        ts_b=ts_b,
                    )
                    if rel is not None:
                        relationships.append(rel)

        # 3. Form Temporal Clusters via Transitive Single-Linkage
        clusters = self.form_clusters(ordered_timed, relationships)

        ordered_alerts = [item[1] for item in ordered_timed]

        return TemporalCorrelationResult(
            total_alerts_ingested=n_ingested,
            eligible_alerts_count=n_eligible,
            ineligible_alerts_count=len(ineligible),
            ineligible_records=ineligible,
            ordered_alerts=ordered_alerts,
            relationships=relationships,
            clusters=clusters,
            metadata={
                "status": "success",
                "window_seconds": self.window_seconds,
                "max_cluster_gap_seconds": self.max_cluster_gap,
                "clustering_method": "transitive_single_linkage",
                "rules_active": [r.name for r in self.rules],
                "clusters_formed": len(clusters),
                "relationships_discovered": len(relationships),
            },
        )
