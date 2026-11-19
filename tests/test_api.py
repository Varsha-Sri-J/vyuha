"""VYUHA API Integration Test Suite.

Verifies:
1. Health check and root endpoints.
2. Overview payload integrity.
3. Network activity metrics and moment feeds.
4. Security alerts retrieval and query/severity filtering.
5. Incidents retrieval and status filtering.
6. Incident details lookup (200 for valid, 404 for missing).
7. Insights category distributions and benchmark metrics.
8. Dynamic flow analysis endpoint.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_health_check():
    """Verifies the health check endpoint returns 200 and declares offline provenance."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["is_live_telemetry"] is False
    assert "data_provenance" in data


def test_root_endpoint():
    """Verifies the root endpoint lists registered API routes."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "/api/overview" in data["endpoints"]


def test_get_overview():
    """Verifies the overview endpoint returns complete posture and traffic feeds."""
    response = client.get("/api/overview")
    assert response.status_code == 200
    data = response.json()
    assert "posture" in data
    assert "flow_totals" in data
    assert data["flow_totals"]["total_observed"] > 0
    assert len(data["traffic_data"]) > 0
    assert len(data["recent_alerts"]) > 0
    assert len(data["attention_items"]) > 0
    assert data["is_live_telemetry"] is False
    # Verify attention items have tone, target, and icon for dynamic card presentation
    for item in data["attention_items"]:
        assert "title" in item
        assert "copy" in item
        assert "tone" in item
        assert "target" in item
        assert "icon" in item


def test_get_network_activity():
    """Verifies network activity returns diurnal windows, weekly bar data, and moments."""
    response = client.get("/api/network-activity")
    assert response.status_code == 200
    data = response.json()
    assert "flow_totals" in data
    assert len(data["traffic_data"]) == 8
    assert len(data["week_data"]) == 7
    assert len(data["moments"]) > 0
    assert "disclaimer" in data


def test_get_alerts_unfiltered():
    """Verifies security alerts endpoint returns all canonical demonstration alerts."""
    response = client.get("/api/alerts")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] > 0
    assert len(data["alerts"]) == data["total_count"]
    # Check alert structure
    first_alt = data["alerts"][0]
    assert "id" in first_alt
    assert "severity" in first_alt
    assert first_alt["severity"] in ("High", "Medium", "Low")
    assert "whyFlagged" in first_alt
    assert "evidence" in first_alt


def test_get_alerts_filtered_by_severity():
    """Verifies security alerts filtering by severity."""
    response = client.get("/api/alerts?severity=High")
    assert response.status_code == 200
    data = response.json()
    for alert in data["alerts"]:
        assert alert["severity"] == "High"


def test_get_alerts_filtered_by_query():
    """Verifies security alerts keyword search."""
    response = client.get("/api/alerts?query=PortScan")
    assert response.status_code == 200
    data = response.json()
    for alert in data["alerts"]:
        text = f"{alert['title']} {alert['category']} {alert['summary']}".lower()
        assert "portscan" in text


def test_get_incidents():
    """Verifies candidate incidents endpoint returns valid incidents grouped by Phase 4."""
    response = client.get("/api/incidents")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] > 0
    assert len(data["incidents"]) == data["total_count"]
    first_inc = data["incidents"][0]
    assert "id" in first_inc
    assert "severity" in first_inc
    assert "status" in first_inc
    assert first_inc["status"] in ("Review", "Monitoring", "Closed candidate")
    assert len(first_inc["nextSteps"]) > 0
    assert "relationship" in first_inc


def test_get_incidents_filtered_by_status():
    """Verifies incident filtering by status."""
    response = client.get("/api/incidents?status=Review")
    assert response.status_code == 200
    data = response.json()
    for incident in data["incidents"]:
        assert incident["status"] == "Review"


def test_get_incident_details_found():
    """Verifies retrieval of specific candidate incident details."""
    # First get an incident ID from list
    list_res = client.get("/api/incidents")
    incidents = list_res.json()["incidents"]
    inc_id = incidents[0]["id"]

    res = client.get(f"/api/incidents/{inc_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == inc_id
    assert "timeline" in data
    assert "evidence" in data
    assert "confidence" in data
    assert data["confidence"] in ("Low", "Moderate", "Strong")
    # Verify relationship string is non-empty (falls back to summary if grouping text is empty)
    for inc in incidents:
        assert inc["relationship"] is not None
        assert len(inc["relationship"].strip()) > 0


def test_get_incident_details_not_found():
    """Verifies 404 response for nonexistent incident."""
    res = client.get("/api/incidents/INC-NONEXISTENT-999")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_get_insights():
    """Verifies security insights payload with category breakdown and benchmark data."""
    response = client.get("/api/insights")
    assert response.status_code == 200
    data = response.json()
    assert len(data["category_data"]) > 0
    assert len(data["engine_data"]) > 0
    assert "benchmark_metrics" in data
    assert data["is_live_telemetry"] is False
    assert "binary_rf_precision" in data["benchmark_metrics"]
    assert "anomaly_iforest_roc_auc" in data["benchmark_metrics"]


def test_analyze_flow_dynamic():
    """Verifies the on-demand dynamic flow evaluation endpoint."""
    sample_flows = [
        {"Destination Port": 80, "attack_category": "PortScan"},
        {"Destination Port": 80, "attack_category": "PortScan"},
        {"Destination Port": 443, "attack_category": "BENIGN"},
    ]
    payload = {
        "flows": sample_flows,
        "alert_on_threat_only": False,
    }
    response = client.post("/api/analyze-flow", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["evaluated_flows_count"] == 3
    assert len(data["alerts"]) == 3
    assert data["threats_count"] == 2
    # Should group the two PortScan flows on port 80 into a candidate incident
    assert len(data["incidents"]) >= 1
