"""VYUHA API Schema Models.

Defines Pydantic request and response models bridging the Phase 1-4 ML detection
and incident intelligence backend with the React frontend.
Enforces explicit data provenance, indicating whether records represent genuine
offline benchmarks, validated demonstration telemetry, or illustrative sample flows.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Primitive / UI Value Types
# -----------------------------------------------------------------------------

FrontendSeverity = Literal["High", "Medium", "Low"]
FrontendIncidentStatus = Literal["Review", "Monitoring", "Closed candidate"]
FrontendTone = Literal["sage", "terracotta", "yellow", "lavender"]
FrontendConfidence = Literal["Low", "Moderate", "Strong"]


class TrafficWindow(BaseModel):
    """Traffic volume aggregated across an observation window."""
    name: str = Field(description="Window label, e.g. '08:00'")
    incoming: int = Field(default=0, description="Inbound flow count")
    outgoing: int = Field(default=0, description="Outbound flow count")
    unusual: int = Field(default=0, description="Flagged or unusual flow count")


class WeekDayData(BaseModel):
    """Weekly traffic distribution across days."""
    name: str = Field(description="Day abbreviation, e.g. 'Mon'")
    normal: int = Field(default=0, description="Observed normal flow count")
    unusual: int = Field(default=0, description="Observed unusual/flagged flow count")


class CategoryDistribution(BaseModel):
    """Attack or anomaly category breakdown."""
    name: str = Field(description="Category name, e.g. 'PortScan'")
    value: int = Field(description="Count or representation")
    color: str = Field(description="Hex color code for visual charts")


class EngineMetric(BaseModel):
    """Detection engine participation and evaluation metrics."""
    name: str = Field(description="Detection engine identifier")
    matched: int = Field(default=0, description="Alerts flagged or matched")
    reviewed: int = Field(default=0, description="Evaluated or reviewed baseline count")


class MomentItem(BaseModel):
    """Chronological traffic observation moment."""
    model_config = {"protected_namespaces": ()}
    time: str = Field(description="Timestamp or window label")
    title: str = Field(description="Observation summary title")
    copy: str = Field(description="Descriptive explanation")


class TimelineEvent(BaseModel):
    """Timestamped event within an incident timeline."""
    model_config = {"protected_namespaces": ()}
    time: str = Field(description="Time string, e.g. '21:28'")
    title: str = Field(description="Headline of the timeline step")
    copy: str = Field(description="Descriptive context")
    tone: FrontendTone = Field(default="sage", description="Color tone token")


class IncidentEvidenceItem(BaseModel):
    """Corroborating line of evidence supporting an incident grouping."""
    model_config = {"protected_namespaces": ()}
    icon: str = Field(description="Icon identifier: activity | network | clock | trending_up | eye | zap | check | info")
    title: str = Field(description="Evidence title")
    copy: str = Field(description="Detailed explanation of the evidence")
    tone: FrontendTone = Field(default="sage", description="Visual indicator tone")


# -----------------------------------------------------------------------------
# Alert & Incident Response Models
# -----------------------------------------------------------------------------

class AlertItem(BaseModel):
    """Canonical alert presentation object matching frontend AlertItem schema."""
    id: str = Field(description="Unique alert identifier")
    title: str = Field(description="Human-readable headline")
    category: str = Field(description="Attack or anomaly category")
    source: str = Field(description="Detection engine name")
    severity: FrontendSeverity = Field(description="Provisional severity: High | Medium | Low")
    summary: str = Field(description="One-sentence plain-language summary")
    time: str = Field(description="Relative or ISO timestamp")
    detail: str = Field(description="What the flow evaluation revealed")
    evidence: str = Field(description="Specific technical or comparative evidence")
    whyFlagged: str = Field(description="Explanation of why this was marked for review")
    technical: str = Field(description="Underlying technical parameters (ports, duration, engine)")
    threat_status: Optional[str] = Field(default=None, description="ATTACK | ANOMALY | BENIGN | UNKNOWN")
    flow_id: Optional[str] = Field(default=None, description="Deterministic flow identifier")
    destination_port: Optional[int] = Field(default=None, description="Destination port if available")
    confidence: Optional[float] = Field(default=None, description="Model probability or confidence if available")
    anomaly_score: Optional[float] = Field(default=None, description="Unsupervised anomaly score if available")
    is_attack: Optional[bool] = Field(default=None, description="Supervised attack flag")


class IncidentItem(BaseModel):
    """Candidate incident presentation object matching frontend Incident schema."""
    id: str = Field(description="Incident ID, e.g. 'INC-001'")
    title: str = Field(description="Human-readable incident title")
    summary: str = Field(description="Plain-language incident summary")
    severity: FrontendSeverity = Field(description="Provisional severity: High | Medium | Low")
    alerts: int = Field(description="Number of correlated alerts in this incident")
    categories: List[str] = Field(default_factory=list, description="Attack or anomaly categories involved")
    status: FrontendIncidentStatus = Field(default="Review", description="Review | Monitoring | Closed candidate")
    updated: str = Field(description="Human-readable update time string")
    engines: List[str] = Field(default_factory=list, description="Names of contributing detection engines")
    relationship: str = Field(description="Plain-language explanation of why events were grouped")
    timeline: List[TimelineEvent] = Field(default_factory=list, description="Chronological timeline of events")
    nextSteps: List[str] = Field(default_factory=list, description="Evidence-based investigation recommendations")
    confidence: FrontendConfidence = Field(default="Moderate", description="Connection strength: Low | Moderate | Strong")
    evidence: List[IncidentEvidenceItem] = Field(default_factory=list, description="Corroborating lines of evidence")
    missing_evidence: List[str] = Field(default_factory=list, description="Explicitly missing metadata (e.g. IPs, hostnames)")
    alert_ids: List[str] = Field(default_factory=list, description="List of alert IDs clustered in this incident")


# -----------------------------------------------------------------------------
# Page-Level Response DTOs
# -----------------------------------------------------------------------------

class FlowTotals(BaseModel):
    """Aggregated traffic flow counts."""
    incoming: int = 0
    outgoing: int = 0
    suspicious: int = 0
    total_observed: int = 0


class OverviewResponse(BaseModel):
    """Payload for the Overview dashboard page."""
    posture: str = Field(default="Steady", description="High-level posture label")
    posture_detail: str = Field(default="Baseline view", description="Posture context subtitle")
    posture_summary: str = Field(description="One-sentence posture summary")
    flow_totals: FlowTotals
    categories_count: int
    open_incidents_count: int
    traffic_data: List[TrafficWindow]
    recent_alerts: List[AlertItem]
    attention_items: List[Dict[str, str]]
    data_source: str = Field(description="Explains whether telemetry is benchmark or demonstration")
    is_live_telemetry: bool = Field(default=False, description="Explicit flag: never claims live when offline")


class NetworkActivityResponse(BaseModel):
    """Payload for the Network Activity page."""
    flow_totals: FlowTotals
    traffic_data: List[TrafficWindow]
    week_data: List[WeekDayData]
    moments: List[MomentItem]
    data_source: str
    is_live_telemetry: bool = False
    disclaimer: str = Field(default="Capacity, latency, packet loss, and congestion are not available in this dataset.")


class SecurityAlertsResponse(BaseModel):
    """Payload for the Security Alerts page."""
    alerts: List[AlertItem]
    total_count: int
    suspicious_flow_count: int
    data_source: str
    is_live_telemetry: bool = False


class IncidentsResponse(BaseModel):
    """Payload for the Incidents page."""
    incidents: List[IncidentItem]
    total_count: int
    open_count: int
    data_source: str
    is_live_telemetry: bool = False


class InsightsResponse(BaseModel):
    """Payload for the Insights page."""
    category_data: List[CategoryDistribution]
    week_data: List[WeekDayData]
    engine_data: List[EngineMetric]
    benchmark_metrics: Dict[str, Any] = Field(default_factory=dict, description="Verified Phase 1-3 model benchmarks")
    data_source: str
    is_live_telemetry: bool = False


class BinaryEvaluationDetail(BaseModel):
    """Detailed output from the Supervised Binary Random Forest model."""
    prediction: int
    label: str
    is_attack: bool
    confidence: float
    attack_probability: float
    benign_probability: float
    model_name: str = "rf_full"


class MulticlassEvaluationDetail(BaseModel):
    """Detailed output from the Supervised Multiclass Random Forest model."""
    predicted_category: str
    is_attack: bool
    confidence: float
    class_probabilities: Dict[str, float]
    model_name: str = "multiclass_rf_full"


class AnomalyEvaluationDetail(BaseModel):
    """Detailed output from the Unsupervised Isolation Forest model."""
    anomaly_score: float
    calibrated_threshold: float
    threshold_name: str
    is_anomaly: bool
    threat_status: str
    is_attack: bool = False
    model_name: str = "iforest_full_100k"


class FlowEvaluationItem(BaseModel):
    """Multi-engine consensus evaluation for a single network flow."""
    flow_index: int
    flow_id: str
    consensus_status: str
    consensus_threat: str
    is_attack: bool
    flagged_as_threat: bool
    primary_category: Optional[str] = None
    provisional_severity: str
    explanation: str
    binary: BinaryEvaluationDetail
    multiclass: MulticlassEvaluationDetail
    anomaly: AnomalyEvaluationDetail


class AnalyzeFlowRequest(BaseModel):
    """Input payload to evaluate one or more network flows dynamically."""
    flows: List[Dict[str, Any]] = Field(description="List of flow records with 63 feature values")
    alert_on_threat_only: bool = Field(default=False, description="Filter output to threats only")
    anomaly_threshold_tier: Optional[str] = Field(default="fpr_1.0pct", description="Isolation Forest FPR tier: fpr_5.0pct | fpr_1.0pct | fpr_0.50pct | fpr_0.10pct")


class AnalyzeFlowResponse(BaseModel):
    """Output payload from dynamic flow evaluation."""
    alerts: List[AlertItem]
    incidents: List[IncidentItem]
    evaluated_flows_count: int
    threats_count: int
    evaluations: List[FlowEvaluationItem] = Field(default_factory=list, description="Per-flow multi-engine predictions")
    models_used: Dict[str, str] = Field(default_factory=dict, description="Active model artifacts used for inference")
    anomaly_threshold_tier: str = Field(default="fpr_1.0pct", description="Active operating anomaly threshold tier")


# -----------------------------------------------------------------------------
# Live Network Monitoring Models (Phase 7)
# -----------------------------------------------------------------------------

class NetworkInterfaceInfo(BaseModel):
    """Network interface details discovered on the host."""
    name: str
    ip: str
    all_ips: List[str] = Field(default_factory=list)
    is_active: bool
    is_loopback: bool


class LiveStatusResponse(BaseModel):
    """Operational state of Zeek live network monitoring."""
    status: str
    collector_status: str
    mode: str
    interface: str
    available_interfaces: List[NetworkInterfaceInfo] = Field(default_factory=list)
    zeek_installed: bool
    zeek_path: Optional[str] = None
    zeek_version: Optional[str] = None
    bpf_permission: bool
    permission_instruction: Optional[str] = None
    total_connections: int = 0
    latest_event_timestamp: Optional[float] = None
    error_message: Optional[str] = None
    log_path: Optional[str] = None
    is_live_telemetry: bool = True


class LiveConnectionItem(BaseModel):
    """Normalized connection record observed from Zeek."""
    uid: str
    timestamp: str
    iso_time: str
    ts: float
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    proto: str
    service: Optional[str] = None
    duration: float = 0.0
    orig_bytes: int = 0
    resp_bytes: int = 0
    total_bytes: int = 0
    orig_pkts: int = 0
    resp_pkts: int = 0
    conn_state: str
    history: Optional[str] = None
    detection_status: str
    detection_severity: str
    detection_reason: str
    ml_evaluated: bool = False
    ml_status: str = "INSUFFICIENT_FEATURES"
    missing_features_count: int = 63
    explanation: str


class LiveConnectionsResponse(BaseModel):
    """Response containing observed live network connections."""
    connections: List[LiveConnectionItem]
    total_count: int
    returned_count: int
    interface: str
    is_monitoring: bool


class PortCountItem(BaseModel):
    """Destination port observation count."""
    port: int
    count: int


class LiveSummaryResponse(BaseModel):
    """Aggregated live traffic observation statistics."""
    total_connections: int
    protocol_distribution: Dict[str, int]
    status_distribution: Dict[str, int]
    top_destination_ports: List[PortCountItem]
    latest_timestamp: Optional[float] = None
    malformed_records_count: int = 0
    buffered_records_count: int = 0
    interface: str
    is_monitoring: bool


class LiveStartRequest(BaseModel):
    """Request payload to initiate live network observation."""
    interface: Optional[str] = Field(default=None, description="Network interface, e.g. en0 or lo0")
    allow_demo_fallback: bool = Field(default=True, description="Fallback to safe local telemetry if macOS BPF permissions denied")


class LiveStartResponse(BaseModel):
    """Response confirming live monitoring launch."""
    status: str
    interface: str
    mode: str = "live_pcap"
    bpf_permission: bool = True
    permission_instruction: Optional[str] = None
    message: str


class LiveStopResponse(BaseModel):
    """Response confirming live monitoring shutdown."""
    status: str
    interface: str
    was_running: bool
    message: str
    total_connections: int
