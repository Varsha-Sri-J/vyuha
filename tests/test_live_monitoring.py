"""Comprehensive test suite for Phase 7: Live Network Monitoring with Zeek.

Tests:
1. Zeek installation detection
2. BPF permission detection and interface discovery
3. Feature validation and strict 63-feature ML eligibility check
4. Rule-based heuristic observation (NORMAL, SUSPICIOUS, POTENTIAL THREAT)
5. JSON and TSV log parsing
6. Incremental log tailing and file offset tracking
7. Duplicate record prevention using Zeek uid
8. Malformed record safety and error resilience
9. Live monitoring process manager lifecycle (start / stop)
10. FastAPI REST live monitoring endpoints
"""

import os
import json
import tempfile
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.monitoring.validator import LiveFeatureValidator
from src.monitoring.collector import ZeekLogCollector
from src.monitoring.manager import LiveMonitoringManager, get_monitoring_manager


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


# -----------------------------------------------------------------------------
# 1. Zeek Installation & System Discovery Tests
# -----------------------------------------------------------------------------

def test_zeek_binary_discovery():
    """Verifies Zeek binary locator finds homebrew or system zeek."""
    zeek_bin = LiveMonitoringManager.find_zeek_binary()
    # On this mac environment, Zeek was installed at /opt/homebrew/bin/zeek
    assert zeek_bin is not None
    assert os.path.exists(zeek_bin)
    version = LiveMonitoringManager.get_zeek_version(zeek_bin)
    assert version is not None
    assert "9." in version or "version" in version.lower()


def test_bpf_permission_check():
    """Verifies BPF permission check reports boolean and helpful instruction."""
    has_bpf, instruction = LiveMonitoringManager.check_bpf_permission()
    assert isinstance(has_bpf, bool)
    if not has_bpf:
        assert instruction is not None
        assert "sudo chmod 666 /dev/bpf" in instruction


def test_interface_discovery():
    """Verifies that network interface discovery returns active interfaces."""
    interfaces = LiveMonitoringManager.detect_network_interfaces()
    assert len(interfaces) > 0
    names = [i["name"] for i in interfaces]
    assert "lo0" in names or "en0" in names


# -----------------------------------------------------------------------------
# 2. ML Feature Completeness & Security Observation Tests
# -----------------------------------------------------------------------------

def test_live_feature_validator_insufficient_features():
    """Verifies that raw Zeek records lacking 63 features are flagged as INSUFFICIENT_FEATURES."""
    raw_zeek_flow = {
        "id.orig_h": "192.168.1.50",
        "id.orig_p": 54321,
        "id.resp_h": "142.250.190.46",
        "id.resp_p": 443,
        "proto": "tcp",
        "duration": 0.5,
    }
    is_eligible, missing, explanation = LiveFeatureValidator.validate_ml_eligibility(raw_zeek_flow)
    assert is_eligible is False
    assert len(missing) > 50
    assert "required statistical dispersion features" in explanation


def test_live_feature_validator_complete_features():
    """Verifies that complete 63-feature records are recognized as eligible."""
    from src.models.preprocessor import get_sample_flow_preset
    preset = get_sample_flow_preset("benign")
    is_eligible, missing, explanation = LiveFeatureValidator.validate_ml_eligibility(preset)
    assert is_eligible is True
    assert len(missing) == 0


def test_heuristic_classification():
    """Tests rule-based security classification for standard, suspicious, and threat scenarios."""
    # Normal HTTPS
    status, sev, desc = LiveFeatureValidator.classify_connection(
        src_ip="172.51.128.120", dst_ip="142.250.190.46", src_port=52341, dst_port=443,
        proto="tcp", conn_state="SF"
    )
    assert status == "NORMAL"
    assert sev == "INFO"

    # Suspicious SSH rejected probe
    status, sev, desc = LiveFeatureValidator.classify_connection(
        src_ip="203.0.113.5", dst_ip="172.51.128.120", src_port=41234, dst_port=22,
        proto="tcp", conn_state="REJ"
    )
    assert status == "SUSPICIOUS"
    assert sev == "MEDIUM"
    assert "SSH" in desc

    # Potential threat: Volumetric flood
    status, sev, desc = LiveFeatureValidator.classify_connection(
        src_ip="198.51.100.99", dst_ip="172.51.128.120", src_port=55555, dst_port=80,
        proto="tcp", conn_state="SF", duration=1.2, orig_pkts=12000
    )
    assert status == "POTENTIAL THREAT"
    assert sev == "HIGH"


# -----------------------------------------------------------------------------
# 3. Log Collector Parsing & Incremental Polling Tests
# -----------------------------------------------------------------------------

def test_json_log_parsing():
    """Verifies parsing of standard Zeek JSON conn.log entries."""
    collector = ZeekLogCollector()
    raw_json = json.dumps({
        "ts": 1731998400.123,
        "uid": "CHhAvV3ug9PqY1u4",
        "id.orig_h": "192.168.1.10",
        "id.orig_p": 49152,
        "id.resp_h": "1.1.1.1",
        "id.resp_p": 53,
        "proto": "udp",
        "service": "dns",
        "duration": 0.024,
        "orig_bytes": 48,
        "resp_bytes": 128,
        "conn_state": "SF",
    })

    rec = collector.parse_log_line(raw_json)
    assert rec is not None
    assert rec["uid"] == "CHhAvV3ug9PqY1u4"
    assert rec["src_ip"] == "192.168.1.10"
    assert rec["dst_port"] == 53
    assert rec["proto"] == "UDP"
    assert rec["ml_status"] == "INSUFFICIENT_FEATURES"
    assert rec["ml_evaluated"] is False


def test_tsv_log_parsing():
    """Verifies fallback parsing of standard Zeek TSV log lines."""
    collector = ZeekLogCollector()
    # ts uid id.orig_h id.orig_p id.resp_h id.resp_p proto service duration orig_bytes resp_bytes conn_state
    tsv_line = "1731998400.000\tCTSV001\t10.0.0.5\t12345\t8.8.8.8\t53\tudp\tdns\t0.01\t60\t120\tSF"
    rec = collector.parse_log_line(tsv_line)
    assert rec is not None
    assert rec["uid"] == "CTSV001"
    assert rec["dst_ip"] == "8.8.8.8"
    assert rec["dst_port"] == 53


def test_malformed_record_handling():
    """Verifies malformed lines are skipped without throwing exceptions."""
    collector = ZeekLogCollector()
    assert collector.parse_log_line("") is None
    assert collector.parse_log_line("#fields ts uid id.orig_h") is None
    assert collector.parse_log_line("{invalid-json-data") is None
    assert collector.parse_log_line("not-enough-tsv-columns") is None
    assert collector._malformed_records_count == 2


def test_duplicate_prevention():
    """Verifies that records with identical uids are not duplicated."""
    collector = ZeekLogCollector()
    rec1 = {
        "uid": "DUP_UID_001",
        "ts": 1731998400.0,
        "src_ip": "127.0.0.1",
        "dst_ip": "127.0.0.1",
        "dst_port": 8000,
        "proto": "TCP",
    }
    rec2 = dict(rec1)  # Exact duplicate

    added = collector.ingest_records([rec1, rec2])
    assert added == 1
    summary = collector.get_summary()
    assert summary["total_connections"] == 1


def test_incremental_file_polling():
    """Verifies that poll_file reads newly appended lines incrementally."""
    with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
        log_path = tf.name

    try:
        collector = ZeekLogCollector(log_path=log_path)

        # 1. Write first record
        with open(log_path, "a") as f:
            f.write(json.dumps({"ts": 1731998400.0, "uid": "POLL_01", "id.orig_h": "127.0.0.1", "id.resp_p": 80}) + "\n")

        added_1 = collector.poll_file()
        assert added_1 == 1
        assert collector.get_summary()["total_connections"] == 1

        # 2. Polling again without new writes adds 0
        added_empty = collector.poll_file()
        assert added_empty == 0

        # 3. Append second record
        with open(log_path, "a") as f:
            f.write(json.dumps({"ts": 1731998401.0, "uid": "POLL_02", "id.orig_h": "127.0.0.1", "id.resp_p": 443}) + "\n")

        added_2 = collector.poll_file()
        assert added_2 == 1
        assert collector.get_summary()["total_connections"] == 2

        conns = collector.get_connections(limit=10)
        assert len(conns) == 2
        assert conns[0]["uid"] == "POLL_02"  # Newest first

    finally:
        if os.path.exists(log_path):
            os.remove(log_path)


# -----------------------------------------------------------------------------
# 4. Live Process Manager Lifecycle Tests
# -----------------------------------------------------------------------------

def test_monitoring_manager_singleton():
    """Verifies that get_monitoring_manager returns the identical instance."""
    m1 = get_monitoring_manager()
    m2 = get_monitoring_manager()
    assert m1 is m2


def test_monitoring_manager_status():
    """Verifies manager returns full status dictionary with diagnostics."""
    manager = get_monitoring_manager()
    status = manager.get_status()
    assert "status" in status
    assert "interface" in status
    assert "available_interfaces" in status
    assert "zeek_installed" in status
    assert "bpf_permission" in status
    assert "total_connections" in status


def test_monitoring_start_stop_demonstration():
    """Verifies start and stop lifecycle in demonstration stream mode."""
    manager = get_monitoring_manager()
    # Start monitoring with allow_demo_fallback=True
    res_start = manager.start_monitoring(interface="lo0", allow_demo_fallback=True)
    assert res_start["status"] in ("started", "already_running")

    # Verify is_running
    assert manager.is_running is True

    # Stop monitoring
    res_stop = manager.stop_monitoring()
    assert res_stop["status"] == "stopped"
    assert manager.is_running is False


# -----------------------------------------------------------------------------
# 5. FastAPI Live Endpoints Integration Tests
# -----------------------------------------------------------------------------

def test_api_live_status(client):
    """Verifies GET /api/live/status response contract."""
    res = client.get("/api/live/status")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "collector_status" in data
    assert "interface" in data
    assert "zeek_installed" in data
    assert "available_interfaces" in data
    assert data["is_live_telemetry"] is True


def test_api_live_connections(client):
    """Verifies GET /api/live/connections response structure."""
    res = client.get("/api/live/connections?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert "connections" in data
    assert "total_count" in data
    assert "returned_count" in data
    assert "interface" in data


def test_api_live_summary(client):
    """Verifies GET /api/live/summary response structure."""
    res = client.get("/api/live/summary")
    assert res.status_code == 200
    data = res.json()
    assert "total_connections" in data
    assert "protocol_distribution" in data
    assert "status_distribution" in data
    assert "top_destination_ports" in data


def test_api_live_start_and_stop(client):
    """Verifies POST /api/live/start and POST /api/live/stop."""
    # Start
    start_res = client.post("/api/live/start", json={"interface": "lo0", "allow_demo_fallback": True})
    assert start_res.status_code == 200
    start_data = start_res.json()
    assert start_data["status"] in ("started", "already_running")
    assert "interface" in start_data

    # Stop
    stop_res = client.post("/api/live/stop")
    assert stop_res.status_code == 200
    stop_data = stop_res.json()
    assert stop_data["status"] == "stopped"
    assert "total_connections" in stop_data
