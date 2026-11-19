"""VYUHA API Route Handlers.

Defines REST endpoints serving the React frontend views with verified telemetry,
benchmark statistics, normalized alerts, and candidate security incidents.
"""

from typing import Optional
from fastapi import APIRouter, Body, Depends, HTTPException, Query, status

from src.models.preprocessor import (
    IncompleteFlowError,
    InvalidFeatureValueError,
    get_sample_flow_preset,
)
from src.models.registry import (
    ModelCorruptError,
    ModelIncompatibleError,
    ModelNotFoundError,
)
from src.api.schemas import (
    AnalyzeFlowRequest,
    AnalyzeFlowResponse,
    IncidentItem,
    IncidentsResponse,
    InsightsResponse,
    LiveConnectionsResponse,
    LiveStartRequest,
    LiveStartResponse,
    LiveStatusResponse,
    LiveStopResponse,
    LiveSummaryResponse,
    NetworkActivityResponse,
    OverviewResponse,
    SecurityAlertsResponse,
)
from src.api.service import IntelligenceService

router = APIRouter(prefix="/api", tags=["Intelligence API"])

# Singleton service instance
_service_instance: Optional[IntelligenceService] = None


def get_service() -> IntelligenceService:
    """Dependency provider for IntelligenceService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = IntelligenceService()
    return _service_instance


@router.get("/health", summary="Health Check")
def health_check(
    service: IntelligenceService = Depends(get_service),
) -> dict:
    """Returns service health, model readiness, and operational status."""
    status_info = service.get_model_status()
    return {
        "status": "healthy",
        "service": "VYUHA Incident Intelligence API",
        "version": "0.1.0",
        "data_provenance": "CICIDS2017 Verified Model Benchmarks & On-Demand Inference",
        "is_live_telemetry": False,
        "models_loaded": status_info["is_loaded"],
        "models_status": status_info["models"],
    }


@router.get("/analyze-flow/presets", summary="Sample Flow Presets")
def get_flow_presets() -> dict:
    """Returns verified synthetic 63-feature flow presets for on-demand inference testing."""
    return {
        "presets": {
            "benign": {
                "name": "Benign Web Browsing",
                "category": "BENIGN",
                "description": "Standard HTTPS TLS flow on port 443 with balanced payload and legitimate TCP window parameters.",
                "flow": get_sample_flow_preset("benign"),
            },
            "portscan": {
                "name": "Stealth SYN PortScan",
                "category": "PortScan",
                "description": "Single-packet SYN probe targeting SSH (port 22) with zero duration and inactive backward window.",
                "flow": get_sample_flow_preset("portscan"),
            },
            "dos": {
                "name": "Volumetric DoS Flood",
                "category": "DoS_DDoS",
                "description": "High-packet-volume HTTP flood targeting port 80 with compressed inter-arrival times.",
                "flow": get_sample_flow_preset("dos"),
            },
            "anomaly": {
                "name": "Statistical Novelty / Distribution Shift",
                "category": "ANOMALY",
                "description": "Out-of-distribution flow on port 8088 exhibiting rare TCP flags (URG, CWE, ECE) and abnormal header ratios.",
                "flow": get_sample_flow_preset("anomaly"),
            },
        }
    }


@router.get("/overview", response_model=OverviewResponse, summary="Security Overview")
def get_overview(
    service: IntelligenceService = Depends(get_service),
) -> OverviewResponse:
    """Returns high-level posture, diurnal traffic summary, and latest alerts for the Overview view."""
    return service.get_overview()


@router.get("/network-activity", response_model=NetworkActivityResponse, summary="Network Activity")
def get_network_activity(
    service: IntelligenceService = Depends(get_service),
) -> NetworkActivityResponse:
    """Returns flow volume by window, diurnal and weekly breakdowns, and highlighted traffic moments."""
    return service.get_network_activity()


@router.get("/alerts", response_model=SecurityAlertsResponse, summary="Security Alerts")
def get_alerts(
    query: Optional[str] = Query(default=None, description="Filter alerts by keyword in title/summary/category"),
    severity: Optional[str] = Query(default=None, description="Filter alerts by severity: High | Medium | Low | All"),
    service: IntelligenceService = Depends(get_service),
) -> SecurityAlertsResponse:
    """Returns normalized security alerts from all detection engines with optional filtering."""
    return service.get_alerts(query=query, severity=severity)


@router.get("/incidents", response_model=IncidentsResponse, summary="Candidate Incidents")
def get_incidents(
    status_filter: Optional[str] = Query(default=None, alias="status", description="Filter by status: Review | Monitoring | Closed candidate | All"),
    service: IntelligenceService = Depends(get_service),
) -> IncidentsResponse:
    """Returns candidate security incidents generated by the Phase 4 incident engine."""
    return service.get_incidents(status=status_filter)


@router.get("/incidents/{incident_id}", response_model=IncidentItem, summary="Incident Details")
def get_incident_details(
    incident_id: str,
    service: IntelligenceService = Depends(get_service),
) -> IncidentItem:
    """Returns an in-depth candidate incident dossier by incident ID."""
    incident = service.get_incident_by_id(incident_id)
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate incident '{incident_id}' not found.",
        )
    return incident


@router.get("/insights", response_model=InsightsResponse, summary="Security Insights")
def get_insights(
    service: IntelligenceService = Depends(get_service),
) -> InsightsResponse:
    """Returns verified category distributions, weekly activity rhythm, and engine benchmark metrics."""
    return service.get_insights()


@router.post("/analyze-flow", response_model=AnalyzeFlowResponse, summary="Dynamic Flow Analysis")
def analyze_flows(
    request: AnalyzeFlowRequest,
    service: IntelligenceService = Depends(get_service),
) -> AnalyzeFlowResponse:
    """Dynamically evaluates supplied flow records through genuine Phase 6 ML inference and Phase 4 incident clustering."""
    try:
        alerts, incidents, evaluations, models_used, active_tier = service.analyze_flows(
            flow_records=request.flows,
            alert_on_threat_only=request.alert_on_threat_only,
            anomaly_threshold_tier=request.anomaly_threshold_tier,
        )
    except IncompleteFlowError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "IncompleteFlowError",
                "message": str(exc),
                "record_index": exc.record_index,
                "missing_features": exc.missing_features,
            },
        ) from exc
    except InvalidFeatureValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "InvalidFeatureValueError",
                "message": str(exc),
                "record_index": exc.record_index,
                "feature": exc.feature_name,
            },
        ) from exc
    except (ModelNotFoundError, ModelCorruptError, ModelIncompatibleError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "ModelLoadError",
                "message": f"Trained ML models could not be loaded: {exc}",
            },
        ) from exc

    threats_cnt = len([a for a in alerts if a.threat_status in ("ATTACK", "ANOMALY")])
    return AnalyzeFlowResponse(
        alerts=alerts,
        incidents=incidents,
        evaluated_flows_count=len(request.flows),
        threats_count=threats_cnt,
        evaluations=evaluations,
        models_used=models_used,
        anomaly_threshold_tier=active_tier,
    )


# -----------------------------------------------------------------------------
# Live Network Monitoring Endpoints (Phase 7)
# -----------------------------------------------------------------------------

@router.get("/live/status", response_model=LiveStatusResponse, summary="Live Monitoring Status")
def get_live_status(
    service: IntelligenceService = Depends(get_service),
) -> LiveStatusResponse:
    """Returns current Zeek live monitoring operational status and interface diagnostics."""
    status_dict = service.get_live_status()
    return LiveStatusResponse(**status_dict)


@router.get("/live/connections", response_model=LiveConnectionsResponse, summary="Live Observed Connections")
def get_live_connections(
    limit: int = Query(default=50, ge=1, le=500, description="Maximum connection records to return"),
    status_filter: Optional[str] = Query(default=None, alias="status", description="Filter by status: NORMAL | SUSPICIOUS | POTENTIAL THREAT | ALL"),
    service: IntelligenceService = Depends(get_service),
) -> LiveConnectionsResponse:
    """Returns recent network connection flows observed live by Zeek."""
    connections, total, returned, interface, is_monitoring = service.get_live_connections(
        limit=limit, status_filter=status_filter
    )
    return LiveConnectionsResponse(
        connections=connections,
        total_count=total,
        returned_count=returned,
        interface=interface,
        is_monitoring=is_monitoring,
    )


@router.get("/live/summary", response_model=LiveSummaryResponse, summary="Live Monitoring Summary")
def get_live_summary(
    service: IntelligenceService = Depends(get_service),
) -> LiveSummaryResponse:
    """Returns aggregated live traffic metrics, protocol distribution, and top target ports."""
    summary = service.get_live_summary()
    return LiveSummaryResponse(**summary)


@router.post("/live/start", response_model=LiveStartResponse, summary="Start Live Monitoring")
def start_live_monitoring(
    request: Optional[LiveStartRequest] = None,
    service: IntelligenceService = Depends(get_service),
) -> LiveStartResponse:
    """Starts live packet capture via Zeek on the host."""
    req = request or LiveStartRequest()
    res = service.start_live_monitoring(
        interface=req.interface,
        allow_demo_fallback=req.allow_demo_fallback,
    )
    if res.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=res.get("message", "Failed to start live network monitoring"),
        )
    return LiveStartResponse(
        status=res.get("status", "started"),
        interface=res.get("interface", "en0"),
        mode=res.get("mode", "live_pcap"),
        bpf_permission=res.get("bpf_permission", True),
        permission_instruction=res.get("permission_instruction"),
        message=res.get("message", "Live network monitoring started"),
    )


@router.post("/live/stop", response_model=LiveStopResponse, summary="Stop Live Monitoring")
def stop_live_monitoring(
    service: IntelligenceService = Depends(get_service),
) -> LiveStopResponse:
    """Stops live Zeek network capture safely."""
    res = service.stop_live_monitoring()
    return LiveStopResponse(
        status=res.get("status", "stopped"),
        interface=res.get("interface", "en0"),
        was_running=res.get("was_running", False),
        message=res.get("message", "Live network monitoring stopped"),
        total_connections=res.get("total_connections", 0),
    )
