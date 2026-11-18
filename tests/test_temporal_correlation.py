"""VYUHA Test Suite: Temporal Correlation Engine (Phase 4.2).

Covers:
1. Chronological ordering.
2. Time-window boundaries.
3. Timezone handling (UTC, positive offset, negative offset, naive).
4. Missing timestamps (ineligible partitioning and dataset limitation).
5. Invalid timestamps.
6. Repeated alerts rule.
7. Repeated categories rule.
8. Sequential event ordering rule.
9. Evidence preservation and JSON serialization.
10. Multiple independent time windows (sliding-window clustering).
11. Alerts close in time with no shared entities.
"""

import datetime
import json
from typing import Any, Dict, List, Optional
import pytest

from src.intelligence.schema import (
    Alert,
    AlertSeverity,
    DetectionEngineType,
    ThreatStatus,
)
from src.intelligence.temporal import (
    IneligibleAlertRecord,
    RepeatedAlertRule,
    RepeatedCategoryRule,
    SequentialOrderingRule,
    TemporalCluster,
    TemporalCorrelationEngine,
    TemporalCorrelationResult,
    TemporalProximityRule,
    TemporalRelationship,
    parse_and_normalize_timestamp,
)


# -----------------------------------------------------------------------------
# Test Fixture Helpers (Explicitly identified as synthetic test fixtures)
# -----------------------------------------------------------------------------

def make_test_alert(
    alert_id: str,
    timestamp: Optional[str],
    category: Optional[str] = None,
    prediction: str = "ATTACK",
    threat_status: str = ThreatStatus.ATTACK.value,
    is_attack: bool = True,
    src_ip: Optional[str] = "192.168.1.50",
    dst_ip: Optional[str] = "172.16.0.10",
    src_port: Optional[int] = 50000,
    dst_port: Optional[int] = 80,
    model_name: str = "rf_full",
) -> Alert:
    """Helper factory for creating synthetic test alert fixtures."""
    return Alert(
        alert_id=alert_id,
        flow_id=f"flow_{alert_id}",
        detection_engine=DetectionEngineType.BINARY.value,
        model_name=model_name,
        prediction=prediction,
        threat_status=threat_status,
        is_attack=is_attack,
        attack_category=category,
        source_ip=src_ip,
        destination_ip=dst_ip,
        source_port=src_port,
        destination_port=dst_port,
        protocol="TCP",
        timestamp=timestamp,
        flow_duration=1000.0,
        confidence=0.95 if is_attack else None,
        severity=AlertSeverity.HIGH.value if is_attack else AlertSeverity.INFO.value,
        metadata={"fixture_type": "synthetic_test_alert"},
    )


# -----------------------------------------------------------------------------
# 1. Chronological Ordering
# -----------------------------------------------------------------------------
def test_chronological_ordering():
    """Verifies that alerts arriving in scrambled order are sorted in strict ascending time."""
    engine = TemporalCorrelationEngine(window_seconds=300.0)

    # Scrambled timestamps: t3, t1, t4, t2
    alerts = [
        make_test_alert("alt_3", "2026-11-19T00:03:00Z"),
        make_test_alert("alt_1", "2026-11-19T00:01:00Z"),
        make_test_alert("alt_4", "2026-11-19T00:04:00Z"),
        make_test_alert("alt_2", "2026-11-19T00:02:00Z"),
    ]

    result = engine.correlate(alerts)

    assert result.eligible_alerts_count == 4
    ordered_ids = [a.alert_id for a in result.ordered_alerts]
    assert ordered_ids == ["alt_1", "alt_2", "alt_3", "alt_4"]


# -----------------------------------------------------------------------------
# 2. Time-Window Boundaries
# -----------------------------------------------------------------------------
def test_time_window_boundaries():
    """Verifies strict adherence to observation window boundary W."""
    engine = TemporalCorrelationEngine(
        window_seconds=60.0,
        rules=[TemporalProximityRule()],
    )

    # Alert A at t=0
    # Alert B at t=60s (exact boundary)
    # Alert C at t=61s (outside boundary)
    alerts = [
        make_test_alert("alt_a", "2026-11-19T00:00:00Z"),
        make_test_alert("alt_b", "2026-11-19T00:01:00Z"),  # dt = 60s
        make_test_alert("alt_c", "2026-11-19T00:01:01Z"),  # dt = 61s from A, 1s from B
    ]

    result = engine.correlate(alerts)

    # Expected relationships:
    # (alt_a, alt_b) with dt=60s: within boundary
    # (alt_b, alt_c) with dt=1s: within boundary
    # (alt_a, alt_c) with dt=61s: outside boundary (must NOT exist)
    rel_pairs = [(r.source_alert_id, r.target_alert_id) for r in result.relationships]

    assert ("alt_a", "alt_b") in rel_pairs
    assert ("alt_b", "alt_c") in rel_pairs
    assert ("alt_a", "alt_c") not in rel_pairs


# -----------------------------------------------------------------------------
# 3. Timezone Handling
# -----------------------------------------------------------------------------
def test_timezone_handling():
    """Verifies that timezones (UTC, positive offsets, negative offsets, and naive) normalize correctly."""
    # Standard UTC
    ts_utc = parse_and_normalize_timestamp("2026-11-19T00:00:00Z")
    assert ts_utc.tzinfo == datetime.timezone.utc
    assert ts_utc.hour == 0 and ts_utc.minute == 0

    # +05:30 offset: 05:31:00 IST is exactly 00:01:00 UTC (dt = 60s from ts_utc)
    ts_ist = parse_and_normalize_timestamp("2026-11-19T05:31:00+05:30")
    assert ts_ist.tzinfo == datetime.timezone.utc
    assert (ts_ist - ts_utc).total_seconds() == 60.0

    # -05:00 offset: 19:02:00 EST on Nov 18 is exactly 00:02:00 UTC on Nov 19 (dt = 120s from ts_utc)
    ts_est = parse_and_normalize_timestamp("2026-11-18T19:02:00-05:00")
    assert ts_est.tzinfo == datetime.timezone.utc
    assert (ts_est - ts_utc).total_seconds() == 120.0

    # Naive timestamp string: assumed UTC
    ts_naive = parse_and_normalize_timestamp("2026-11-19 00:03:00")
    assert ts_naive.tzinfo == datetime.timezone.utc
    assert (ts_naive - ts_utc).total_seconds() == 180.0

    # Run correlation across all 4 mixed-timezone alerts
    engine = TemporalCorrelationEngine(window_seconds=300.0)
    alerts = [
        make_test_alert("alt_est", "2026-11-18T19:02:00-05:00"),  # 00:02:00 UTC
        make_test_alert("alt_utc", "2026-11-19T00:00:00Z"),        # 00:00:00 UTC
        make_test_alert("alt_naive", "2026-11-19 00:03:00"),      # 00:03:00 UTC
        make_test_alert("alt_ist", "2026-11-19T05:31:00+05:30"),  # 00:01:00 UTC
    ]

    result = engine.correlate(alerts)
    ordered_ids = [a.alert_id for a in result.ordered_alerts]
    assert ordered_ids == ["alt_utc", "alt_ist", "alt_est", "alt_naive"]


# -----------------------------------------------------------------------------
# 4. Missing Timestamps Handling
# -----------------------------------------------------------------------------
def test_missing_timestamps_handling():
    """Verifies that alerts without timestamps are partitioned as ineligible with recorded reasons."""
    engine = TemporalCorrelationEngine(window_seconds=300.0)

    alerts = [
        make_test_alert("alt_valid_1", "2026-11-19T00:01:00Z"),
        make_test_alert("alt_missing_1", None),
        make_test_alert("alt_valid_2", "2026-11-19T00:02:00Z"),
        make_test_alert("alt_missing_2", ""),
    ]

    result = engine.correlate(alerts)

    assert result.total_alerts_ingested == 4
    assert result.eligible_alerts_count == 2
    assert result.ineligible_alerts_count == 2

    ineligible_reasons = {r.alert_id: r.reason for r in result.ineligible_records}
    assert ineligible_reasons["alt_missing_1"] == "missing_timestamp"
    assert ineligible_reasons["alt_missing_2"] == "missing_timestamp"

    # Verify that a pure CICIDS2017 batch (where all timestamps are None) is handled cleanly
    cicids_alerts = [
        make_test_alert("alt_c1", None),
        make_test_alert("alt_c2", None),
    ]
    empty_result = engine.correlate(cicids_alerts)
    assert empty_result.eligible_alerts_count == 0
    assert empty_result.ineligible_alerts_count == 2
    assert empty_result.relationships == []
    assert empty_result.metadata["status"] == "all_alerts_ineligible_missing_timestamps"
    assert "dataset_limitation_warning" in empty_result.metadata


# -----------------------------------------------------------------------------
# 5. Invalid Timestamps Handling
# -----------------------------------------------------------------------------
def test_invalid_timestamps_handling():
    """Verifies that malformed or unparseable timestamps are rejected and documented."""
    engine = TemporalCorrelationEngine(window_seconds=300.0)

    alerts = [
        make_test_alert("alt_ok", "2026-11-19T00:01:00Z"),
        make_test_alert("alt_bad_1", "not_a_valid_timestamp_string"),
        make_test_alert("alt_bad_2", "9999-99-99 99:99:99"),
    ]

    result = engine.correlate(alerts)

    assert result.eligible_alerts_count == 1
    assert result.ineligible_alerts_count == 2

    bad_reasons = {r.alert_id: r.reason for r in result.ineligible_records}
    assert "unparseable_timestamp" in bad_reasons["alt_bad_1"]
    assert "unparseable_timestamp" in bad_reasons["alt_bad_2"]


# -----------------------------------------------------------------------------
# 6. Repeated Alerts Rule
# -----------------------------------------------------------------------------
def test_repeated_alerts_rule():
    """Verifies that RepeatedAlertRule identifies repeated identical model predictions."""
    rule = RepeatedAlertRule()
    engine = TemporalCorrelationEngine(window_seconds=120.0, rules=[rule])

    # Two alerts from the same model with identical prediction within window
    alt1 = make_test_alert("alt_1", "2026-11-19T00:00:00Z", model_name="rf_full", prediction="ATTACK")
    alt2 = make_test_alert("alt_2", "2026-11-19T00:01:30Z", model_name="rf_full", prediction="ATTACK")

    # A third alert from a different model with a different prediction
    alt3 = make_test_alert(
        "alt_3",
        "2026-11-19T00:01:45Z",
        model_name="dt_port_agnostic",
        prediction="BENIGN",
        threat_status=ThreatStatus.BENIGN.value,
        is_attack=False,
    )

    result = engine.correlate([alt1, alt2, alt3])

    repeated_rels = result.get_relationships_by_rule("repeated_alert")
    assert len(repeated_rels) == 1

    rel = repeated_rels[0]
    assert rel.source_alert_id == "alt_1"
    assert rel.target_alert_id == "alt_2"
    assert rel.time_delta_seconds == pytest.approx(90.0)
    assert rel.evidence["repeated_prediction"] == "ATTACK"


# -----------------------------------------------------------------------------
# 7. Repeated Categories Rule
# -----------------------------------------------------------------------------
def test_repeated_categories_rule():
    """Verifies that RepeatedCategoryRule identifies bursts of identical attack families."""
    rule = RepeatedCategoryRule()
    engine = TemporalCorrelationEngine(window_seconds=180.0, rules=[rule])

    # Wave of DoS_DDoS attacks
    dos1 = make_test_alert("dos_1", "2026-11-19T00:00:00Z", category="DoS_DDoS")
    dos2 = make_test_alert("dos_2", "2026-11-19T00:01:00Z", category="DoS_DDoS")
    # Different category (PortScan)
    scan = make_test_alert("scan_1", "2026-11-19T00:01:30Z", category="PortScan")

    result = engine.correlate([dos1, dos2, scan])

    rep_cat_rels = result.get_relationships_by_rule("repeated_category")
    assert len(rep_cat_rels) == 1

    rel = rep_cat_rels[0]
    assert rel.source_alert_id == "dos_1"
    assert rel.target_alert_id == "dos_2"
    assert rel.evidence["attack_category"] == "DoS_DDoS"
    assert rel.time_delta_seconds == pytest.approx(60.0)


# -----------------------------------------------------------------------------
# 8. Sequential Event Ordering Rule
# -----------------------------------------------------------------------------
def test_sequential_event_ordering():
    """Verifies that SequentialOrderingRule identifies reconnaissance followed by exploitation."""
    rule = SequentialOrderingRule()
    engine = TemporalCorrelationEngine(window_seconds=300.0, rules=[rule])

    # 1. Valid sequence: PortScan at t=0, BruteForce at t=45s
    scan = make_test_alert("scan_1", "2026-11-19T00:00:00Z", category="PortScan")
    brute = make_test_alert("brute_1", "2026-11-19T00:00:45Z", category="BruteForce")

    result = engine.correlate([scan, brute])
    seq_rels = result.get_relationships_by_rule("sequential_ordering")
    assert len(seq_rels) == 1

    rel = seq_rels[0]
    assert rel.source_alert_id == "scan_1"
    assert rel.target_alert_id == "brute_1"
    assert rel.is_sequential is True
    assert rel.evidence["preceding_category"] == "PortScan"
    assert rel.evidence["subsequent_category"] == "BruteForce"
    assert rel.evidence["progression_stage"] == "reconnaissance_to_exploitation"

    # 2. Inverted sequence: BruteForce followed by PortScan (not a recognized progression)
    brute_first = make_test_alert("brute_a", "2026-11-19T00:00:00Z", category="BruteForce", dst_port=22)
    scan_later = make_test_alert("scan_b", "2026-11-19T00:00:30Z", category="PortScan", dst_port=80)
    inv_result = engine.correlate([brute_first, scan_later])
    assert len(inv_result.get_relationships_by_rule("sequential_ordering")) == 0

    # 3. Simultaneous events (dt = 0): Must NOT match sequential rule
    scan_sim = make_test_alert("scan_s", "2026-11-19T00:00:00Z", category="PortScan")
    brute_sim = make_test_alert("brute_s", "2026-11-19T00:00:00Z", category="BruteForce")
    sim_result = engine.correlate([scan_sim, brute_sim])
    assert len(sim_result.get_relationships_by_rule("sequential_ordering")) == 0


# -----------------------------------------------------------------------------
# 9. Evidence Preservation and Serialization
# -----------------------------------------------------------------------------
def test_evidence_preservation():
    """Verifies that all required relationship fields and evidence details are preserved and serializable."""
    engine = TemporalCorrelationEngine(window_seconds=120.0)

    alt_a = make_test_alert("alt_a", "2026-11-19T00:00:00Z", category="PortScan")
    alt_b = make_test_alert("alt_b", "2026-11-19T00:01:00Z", category="WebAttack")

    result = engine.correlate([alt_a, alt_b])
    assert len(result.relationships) > 0

    rel = result.relationships[0]
    # Required core fields
    assert rel.relationship_id.startswith("trel_")
    assert rel.source_alert_id == "alt_a"
    assert rel.target_alert_id == "alt_b"
    assert rel.source_timestamp == "2026-11-19T00:00:00+00:00"
    assert rel.target_timestamp == "2026-11-19T00:01:00+00:00"
    assert rel.time_delta_seconds == pytest.approx(60.0)
    assert rel.window_seconds == pytest.approx(120.0)
    assert rel.is_sequential is True
    assert isinstance(rel.evidence, dict)

    # JSON serialization
    json_str = result.to_json()
    assert "trel_" in json_str
    assert "alt_a" in json_str
    assert "alt_b" in json_str

    parsed_json = json.loads(json_str)
    assert parsed_json["total_alerts_ingested"] == 2
    assert parsed_json["eligible_alerts_count"] == 2
    assert len(parsed_json["relationships"]) == len(result.relationships)


# -----------------------------------------------------------------------------
# 10. Multiple Independent Time Windows (Clustering)
# -----------------------------------------------------------------------------
def test_multiple_independent_time_windows():
    """Verifies that alert groups separated by large time gaps form independent temporal clusters."""
    engine = TemporalCorrelationEngine(window_seconds=180.0, max_cluster_gap_seconds=180.0)

    # Cluster 1: 3 alerts spaced 30 seconds apart (total 60s)
    c1_alerts = [
        make_test_alert("c1_1", "2026-11-19T00:00:00Z", category="PortScan"),
        make_test_alert("c1_2", "2026-11-19T00:00:30Z", category="PortScan"),
        make_test_alert("c1_3", "2026-11-19T00:01:00Z", category="PortScan"),
    ]

    # Large gap of 1800s (30 minutes)
    # Cluster 2: 2 alerts spaced 20 seconds apart
    c2_alerts = [
        make_test_alert("c2_1", "2026-11-19T00:31:00Z", category="DoS_DDoS"),
        make_test_alert("c2_2", "2026-11-19T00:31:20Z", category="DoS_DDoS"),
    ]

    result = engine.correlate(c1_alerts + c2_alerts)

    # Must form exactly TWO clusters
    assert len(result.clusters) == 2

    # Cluster 1 assertions
    clust1 = result.clusters[0]
    assert clust1.alerts_count == 3
    assert clust1.duration_seconds == pytest.approx(60.0)
    assert clust1.dominant_categories == ["PortScan"]

    # Cluster 2 assertions
    clust2 = result.clusters[1]
    assert clust2.alerts_count == 2
    assert clust2.duration_seconds == pytest.approx(20.0)
    assert clust2.dominant_categories == ["DoS_DDoS"]

    # Verify no relationships bridge across the 30-minute gap (window is 180s)
    cross_cluster_rels = [
        r for r in result.relationships
        if r.source_alert_id.startswith("c1_") and r.target_alert_id.startswith("c2_")
    ]
    assert len(cross_cluster_rels) == 0


# -----------------------------------------------------------------------------
# 11. Alerts Close in Time with No Shared Entities
# -----------------------------------------------------------------------------
def test_close_in_time_no_shared_entities():
    """Verifies that temporal proximity is preserved without erroneously inventing shared entities."""
    engine = TemporalCorrelationEngine(window_seconds=60.0, rules=[TemporalProximityRule()])

    # Two completely distinct hosts and targets, separated by 5 seconds
    alt1 = make_test_alert(
        "alt_alpha",
        "2026-11-19T00:00:00Z",
        src_ip="10.0.0.5",
        dst_ip="172.16.100.1",
        src_port=40001,
        dst_port=80,
    )
    alt2 = make_test_alert(
        "alt_beta",
        "2026-11-19T00:00:05Z",
        src_ip="192.168.1.99",
        dst_ip="10.200.0.50",
        src_port=40002,
        dst_port=22,
    )

    result = engine.correlate([alt1, alt2])

    prox_rels = result.get_relationships_by_rule("temporal_proximity")
    assert len(prox_rels) == 1

    rel = prox_rels[0]
    assert rel.time_delta_seconds == pytest.approx(5.0)

    # CRITICAL: Verify that evidence explicitly shows NO shared entities
    assert rel.evidence["has_shared_entities"] is False
    assert rel.evidence["shared_entities"] == {}
    assert "Circumstantial temporal proximity" in rel.evidence["interpretation"]


# -----------------------------------------------------------------------------
# 12. Transitive Single-Linkage Clustering (A -> B -> C where Total > Window)
# -----------------------------------------------------------------------------
def test_transitive_clustering_and_duration_exceeding_window():
    """Verifies that transitive single-linkage clustering correctly groups chained events.

    Test scenario:
    - Event A at 10:00:00
    - Event B at 10:00:09 (dt_AB = 9s <= 10s max_gap)
    - Event C at 10:00:18 (dt_BC = 9s <= 10s max_gap)
    Total duration = 18s (> 10s window).
    Expected:
    - All three belong to ONE cluster via transitive chaining.
    - Pairwise relationships exist for (A, B) and (B, C), but NOT (A, C) because dt_AC = 18s > 10s.
    """
    engine = TemporalCorrelationEngine(
        window_seconds=10.0,
        max_cluster_gap_seconds=10.0,
        rules=[TemporalProximityRule()],
    )

    alert_a = make_test_alert("alt_a", "2026-11-19T10:00:00Z")
    alert_b = make_test_alert("alt_b", "2026-11-19T10:00:09Z")
    alert_c = make_test_alert("alt_c", "2026-11-19T10:00:18Z")

    result = engine.correlate([alert_a, alert_b, alert_c])

    # Exactly one single cluster formed
    assert len(result.clusters) == 1
    cluster = result.clusters[0]

    assert cluster.alerts_count == 3
    assert cluster.alert_ids == ["alt_a", "alt_b", "alt_c"]
    assert cluster.first_timestamp == "2026-11-19T10:00:00+00:00"
    assert cluster.last_timestamp == "2026-11-19T10:00:18+00:00"
    assert cluster.total_duration_seconds == pytest.approx(18.0)
    assert cluster.total_duration_seconds > engine.window_seconds  # 18.0 > 10.0
    assert cluster.max_cluster_gap_seconds == pytest.approx(10.0)
    assert cluster.clustering_method == "transitive_single_linkage"

    # Pairwise relationship inspection
    rel_pairs = [(r.source_alert_id, r.target_alert_id) for r in result.relationships]
    assert ("alt_a", "alt_b") in rel_pairs
    assert ("alt_b", "alt_c") in rel_pairs
    assert ("alt_a", "alt_c") not in rel_pairs  # 18s exceeds pairwise window of 10s

    assert cluster.relationship_count == 2
    assert len(cluster.relationships) == 2


# -----------------------------------------------------------------------------
# 13. Cluster Metrics Verification
# -----------------------------------------------------------------------------
def test_cluster_metrics_verification():
    """Verifies all required cluster metrics and serialization attributes."""
    engine = TemporalCorrelationEngine(
        window_seconds=120.0,
        max_cluster_gap_seconds=120.0,
        rules=[TemporalProximityRule(), RepeatedCategoryRule()],
    )

    alerts = [
        make_test_alert("c_1", "2026-11-19T10:00:00Z", category="PortScan"),
        make_test_alert("c_2", "2026-11-19T10:01:00Z", category="PortScan"),
        make_test_alert("c_3", "2026-11-19T10:01:30Z", category="DoS_DDoS"),
    ]

    result = engine.correlate(alerts)
    assert len(result.clusters) == 1
    cluster = result.clusters[0]

    # Metrics verification
    assert cluster.first_timestamp == "2026-11-19T10:00:00+00:00"
    assert cluster.last_timestamp == "2026-11-19T10:01:30+00:00"
    assert cluster.total_duration_seconds == pytest.approx(90.0)
    assert cluster.max_cluster_gap_seconds == pytest.approx(120.0)
    assert cluster.alerts_count == 3
    assert cluster.alert_ids == ["c_1", "c_2", "c_3"]
    assert cluster.clustering_method == "transitive_single_linkage"

    # Dominant categories: PortScan (count 2) precedes DoS_DDoS (count 1)
    assert cluster.dominant_categories == ["PortScan", "DoS_DDoS"]
    assert cluster.relationship_count > 0

    # Serialization dictionary schema check
    c_dict = cluster.to_dict()
    assert "cluster_id" in c_dict
    assert "first_timestamp" in c_dict
    assert "last_timestamp" in c_dict
    assert "total_duration_seconds" in c_dict
    assert "max_cluster_gap_seconds" in c_dict
    assert "alerts_count" in c_dict
    assert "alert_ids" in c_dict
    assert "relationships" in c_dict
    assert "relationship_count" in c_dict
    assert "dominant_categories" in c_dict
    assert "clustering_method" in c_dict


# -----------------------------------------------------------------------------
# 14. Proximity Evidence Classification Differentiation
# -----------------------------------------------------------------------------
def test_proximity_evidence_classification_differentiation():
    """Verifies explicit differentiation across:
    - time_only_proximity
    - shared_source_ip
    - shared_destination_ip
    - shared_network_entity
    """
    rule = TemporalProximityRule()
    engine = TemporalCorrelationEngine(window_seconds=60.0, rules=[rule])

    # Case 1: time_only_proximity (distinct sources, destinations, and ports)
    a1 = make_test_alert("a1", "2026-11-19T10:00:00Z", src_ip="10.0.0.1", dst_ip="172.16.1.1", src_port=40001, dst_port=80)
    a2 = make_test_alert("a2", "2026-11-19T10:00:05Z", src_ip="10.0.0.2", dst_ip="172.16.2.2", src_port=40002, dst_port=443)
    r1 = engine.correlate([a1, a2])
    rel1 = r1.relationships[0]
    assert rel1.evidence["proximity_classification"] == "time_only_proximity"
    assert rel1.evidence["has_shared_entities"] is False
    assert rel1.evidence["shared_entities"] == {}

    # Case 2: shared_source_ip
    b1 = make_test_alert("b1", "2026-11-19T10:00:00Z", src_ip="10.0.0.100", dst_ip="172.16.1.1", src_port=40001, dst_port=80)
    b2 = make_test_alert("b2", "2026-11-19T10:00:05Z", src_ip="10.0.0.100", dst_ip="172.16.2.2", src_port=40002, dst_port=443)
    r2 = engine.correlate([b1, b2])
    rel2 = r2.relationships[0]
    assert rel2.evidence["proximity_classification"] == "shared_source_ip"
    assert rel2.evidence["has_shared_entities"] is True
    assert rel2.evidence["shared_entities"] == {"source_ip": "10.0.0.100"}

    # Case 3: shared_destination_ip
    c1 = make_test_alert("c1", "2026-11-19T10:00:00Z", src_ip="10.0.0.1", dst_ip="172.16.0.50", src_port=40001, dst_port=80)
    c2 = make_test_alert("c2", "2026-11-19T10:00:05Z", src_ip="10.0.0.2", dst_ip="172.16.0.50", src_port=40002, dst_port=443)
    r3 = engine.correlate([c1, c2])
    rel3 = r3.relationships[0]
    assert rel3.evidence["proximity_classification"] == "shared_destination_ip"
    assert rel3.evidence["has_shared_entities"] is True
    assert rel3.evidence["shared_entities"] == {"destination_ip": "172.16.0.50"}

    # Case 4: shared_network_entity (shared destination IP and destination port)
    d1 = make_test_alert("d1", "2026-11-19T10:00:00Z", src_ip="10.0.0.1", dst_ip="172.16.0.50", src_port=40001, dst_port=443)
    d2 = make_test_alert("d2", "2026-11-19T10:00:05Z", src_ip="10.0.0.2", dst_ip="172.16.0.50", src_port=40002, dst_port=443)
    r4 = engine.correlate([d1, d2])
    rel4 = r4.relationships[0]
    assert rel4.evidence["proximity_classification"] == "shared_network_entity"
    assert rel4.evidence["has_shared_entities"] is True
    assert rel4.evidence["shared_entities"] == {"destination_ip": "172.16.0.50", "destination_port": 443}


# -----------------------------------------------------------------------------
# 15. Missing Entity Values Never Treated as Shared
# -----------------------------------------------------------------------------
def test_missing_entity_values_never_treated_as_shared():
    """Verifies that missing entity values (None in both alerts) are never treated as shared."""
    engine = TemporalCorrelationEngine(window_seconds=60.0, rules=[TemporalProximityRule()])

    # Both alerts have completely missing endpoint metadata
    alt_none_1 = make_test_alert(
        "alt_n1",
        "2026-11-19T10:00:00Z",
        src_ip=None,
        dst_ip=None,
        src_port=None,
        dst_port=None,
    )
    alt_none_2 = make_test_alert(
        "alt_n2",
        "2026-11-19T10:00:05Z",
        src_ip=None,
        dst_ip=None,
        src_port=None,
        dst_port=None,
    )

    result = engine.correlate([alt_none_1, alt_none_2])
    assert len(result.relationships) == 1

    rel = result.relationships[0]
    # Critical verification: None == None must NOT be shared!
    assert rel.evidence["has_shared_entities"] is False
    assert rel.evidence["shared_entities"] == {}
    assert rel.evidence["proximity_classification"] == "time_only_proximity"

    # Missing entity information must be explicitly tracked
    assert set(rel.evidence["missing_entities"]["alert_a"]) == {
        "source_ip", "destination_ip", "destination_port", "source_port"
    }
    assert set(rel.evidence["missing_entities"]["alert_b"]) == {
        "source_ip", "destination_ip", "destination_port", "source_port"
    }


# -----------------------------------------------------------------------------
# 16. Candidate Attack-Pattern Sequence Evidence Contents
# -----------------------------------------------------------------------------
def test_candidate_attack_pattern_sequence_evidence_contents():
    """Verifies that candidate attack-pattern sequences preserve all required audit evidence:
    - Earlier and later alerts
    - Attack categories
    - Time difference
    - Observation window
    - Shared entity evidence
    - Missing entity information
    - Clear statement that temporal ordering does not establish causation
    """
    rule = SequentialOrderingRule()
    engine = TemporalCorrelationEngine(window_seconds=300.0, rules=[rule])

    scan_alert = make_test_alert(
        "scan_01",
        "2026-11-19T10:00:00Z",
        category="PortScan",
        src_ip="192.168.10.50",
        dst_ip="172.16.0.1",
        src_port=45000,
        dst_port=22,
    )
    brute_alert = make_test_alert(
        "brute_02",
        "2026-11-19T10:00:30Z",
        category="BruteForce",
        src_ip="192.168.10.50",
        dst_ip="172.16.0.1",
        src_port=45002,
        dst_port=22,
    )

    result = engine.correlate([scan_alert, brute_alert])
    assert len(result.relationships) == 1

    rel = result.relationships[0]
    assert rel.rule_name == "sequential_ordering"
    assert rel.is_sequential is True

    ev = rel.evidence
    # 1. Earlier and later alerts
    assert ev["earlier_alert_id"] == "scan_01"
    assert ev["later_alert_id"] == "brute_02"

    # 2. Attack categories
    assert ev["earlier_category"] == "PortScan"
    assert ev["later_category"] == "BruteForce"

    # 3. Time difference
    assert ev["time_delta_seconds"] == pytest.approx(30.0)

    # 4. Observation window
    assert ev["window_seconds"] == pytest.approx(300.0)

    # 5. Shared entity evidence
    assert ev["has_shared_entities"] is True
    assert ev["shared_entity_evidence"] == {
        "source_ip": "192.168.10.50",
        "destination_ip": "172.16.0.1",
        "destination_port": 22,
    }
    assert ev["shared_target_detected"] is True

    # 6. Missing entity information
    assert "missing_entity_information" in ev
    assert "alert_a" in ev["missing_entity_information"]
    assert "alert_b" in ev["missing_entity_information"]

    # 7. Non-causation caveat and candidate pattern status
    assert ev["is_candidate_pattern_sequence"] is True
    assert ev["candidate_stage"] == "candidate_reconnaissance_to_exploitation"
    assert "candidate attack-pattern hypothesis" in ev["causation_caveat"]
    assert "temporal ordering does NOT establish causal connection" in ev["causation_caveat"]

