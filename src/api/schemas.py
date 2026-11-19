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


class AnalyzeFlowRequest(BaseModel):
    """Input payload to evaluate one or more network flows dynamically."""
    flows: List[Dict[str, Any]] = Field(description="List of flow records with feature values")
    alert_on_threat_only: bool = Field(default=False, description="Filter output to threats only")


class AnalyzeFlowResponse(BaseModel):
    """Output payload from dynamic flow evaluation."""
    alerts: List[AlertItem]
    incidents: List[IncidentItem]
    evaluated_flows_count: int
    threats_count: int
