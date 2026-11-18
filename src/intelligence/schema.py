"""VYUHA Common Alert Schema Module (Phase 4.1).

Defines the unified canonical data representation for security alerts generated
across all three detection engines:
1. Binary Intrusion Detection (Phase 3B)
2. Multiclass Intrusion Detection (Phase 3C)
3. Unsupervised Anomaly Detection (Phase 3D)
4. Multi-Engine Consolidated Alerts

Provides strict validation, nullable field handling, serialization, explicit threat status semantics,
and provisional severity ranking.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from typing import Any, Dict, List, Optional, Union


class ThreatStatus(str, Enum):
    """Explicit operational threat status of an alert.

    - BENIGN: Verified or predicted normal traffic.
    - ATTACK: Confirmed malicious threat identified by supervised detection.
    - ANOMALY: Statistical distribution novelty identified by unsupervised isolation.
               Importantly: An anomaly is NOT a confirmed attack.
    - UNKNOWN: Missing, uninitialized, or unclassifiable traffic.
    """
    BENIGN = "BENIGN"
    ATTACK = "ATTACK"
    ANOMALY = "ANOMALY"
    UNKNOWN = "UNKNOWN"


class AlertSeverity(str, Enum):
    """Provisional alert severity levels for triage prioritization.

    NOTE: Alert-level severity is provisional based on model outputs.
    True incident severity and operational prioritization require asset criticality,
    temporal frequency, and relationship graph context (developed in later milestones).
    """
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DetectionEngineType(str, Enum):
    """Categorization of alert-generating engines."""
    BINARY = "binary"
    MULTICLASS = "multiclass"
    ANOMALY = "anomaly"
    MULTI_ENGINE = "multi_engine"


SEVERITY_ORDER: Dict[str, int] = {
    AlertSeverity.INFO.value: 0,
    AlertSeverity.LOW.value: 1,
    AlertSeverity.MEDIUM.value: 2,
    AlertSeverity.HIGH.value: 3,
    AlertSeverity.CRITICAL.value: 4,
}


@dataclass
class Alert:
    """Unified security alert representation across all VYUHA detection engines.

    Accommodates supervised classifications, unsupervised statistical anomalies,
    and multi-engine consolidated assessments without loss of engine-specific evidence.
    """

    alert_id: str
    flow_id: str
    detection_engine: str
    model_name: str
    prediction: Union[str, int, bool]
    threat_status: str = ThreatStatus.UNKNOWN.value
    is_attack: bool = False
    attack_category: Optional[str] = None
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    source_port: Optional[int] = None
    destination_port: Optional[int] = None
    protocol: Optional[str] = None
    timestamp: Optional[str] = None
    flow_duration: Optional[float] = None
    anomaly_score: Optional[float] = None
    anomaly_threshold: Optional[float] = None
    confidence: Optional[float] = None
    severity: str = AlertSeverity.LOW.value
    evidence: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validates threat status, bounds, and severity constraints."""
        # 1. Validate threat status
        valid_statuses = [s.value for s in ThreatStatus]
        if self.threat_status not in valid_statuses:
            raise ValueError(
                f"Invalid threat_status '{self.threat_status}'. Must be one of {valid_statuses}"
            )

        # 2. Enforce unambiguous threat_status and is_attack semantics
        if self.threat_status == ThreatStatus.ANOMALY.value:
            if self.is_attack:
                raise ValueError(
                    "Semantic violation: An anomaly alert cannot have is_attack=True. "
                    "Anomalies represent statistical novelty, not confirmed attacks."
                )
        elif self.threat_status == ThreatStatus.BENIGN.value:
            if self.is_attack:
                raise ValueError(
                    "Semantic violation: A benign alert cannot have is_attack=True."
                )
        elif self.threat_status == ThreatStatus.ATTACK.value:
            if not self.is_attack:
                raise ValueError(
                    "Semantic violation: An attack alert must have is_attack=True."
                )

        # 3. Ensure severity is valid
        if self.severity not in SEVERITY_ORDER:
            raise ValueError(
                f"Invalid severity '{self.severity}'. Must be one of {list(SEVERITY_ORDER.keys())}"
            )

        # 4. Validate confidence score bounds if present
        if self.confidence is not None:
            if not (0.0 <= self.confidence <= 1.0):
                raise ValueError(
                    f"Confidence score must be in [0.0, 1.0], got {self.confidence}"
                )

        # 5. Validate anomaly score bounds if present
        if self.anomaly_score is not None:
            if not (0.0 <= self.anomaly_score <= 1.0):
                raise ValueError(
                    f"Anomaly score must be in [0.0, 1.0], got {self.anomaly_score}"
                )

        # 6. Validate port ranges if present
        for port_field, port_val in [
            ("source_port", self.source_port),
            ("destination_port", self.destination_port),
        ]:
            if port_val is not None:
                if not (0 <= port_val <= 65535):
                    raise ValueError(
                        f"{port_field} must be between 0 and 65535, got {port_val}"
                    )

    @property
    def severity_rank(self) -> int:
        """Returns integer rank (0-4) corresponding to provisional severity level."""
        return SEVERITY_ORDER.get(self.severity, 0)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes alert to a pure Python dictionary."""
        return asdict(self)

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serializes alert to a JSON-formatted string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Alert":
        """Instantiates an Alert from a dictionary, handling missing/nullable fields."""
        d = data.copy()
        if "evidence" not in d or d["evidence"] is None:
            d["evidence"] = {}
        if "metadata" not in d or d["metadata"] is None:
            d["metadata"] = {}
        return cls(**d)

    @classmethod
    def from_json(cls, json_str: str) -> "Alert":
        """Instantiates an Alert from a JSON string."""
        return cls.from_dict(json.loads(json_str))
