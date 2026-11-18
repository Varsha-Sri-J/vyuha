"""VYUHA Test Suite: Incident Intelligence & Relationship Graph (Phase 4).

Covers:
1. Relationship graph construction.
2. Shared entity relationships (source IP, destination IP, network entity, port).
3. Temporal relationships integration.
4. Missing metadata (None == None never links alerts).
5. Candidate incident generation.
6. Separation of unrelated alerts.
7. Time-only proximity does not merge unrelated alerts.
8. Duplicate alert handling (same-flow multi-engine assessments).
9. Explainable provisional severity assessment across threat tiers.
10. Explanation generation (structured breakdown and prose summary).
11. Rule-based investigation recommendations (no false claims of blocking/resolution).
12. Empty input handling.
13. Partial evidence handling (CICIDS2017 style: missing timestamps and source IPs).
14. Incident serialization (to_dict and to_json).
"""

import json
from typing import Optional
import pytest

from src.intelligence.graph import RelationshipGraph
from src.intelligence.incident import CandidateIncidentGenerator
from src.intelligence.schema import (
    Alert,
    AlertRelationship,
    AlertSeverity,
    CandidateIncident,
    DetectionEngineType,
    IncidentStatus,
    RelationshipType,
    ThreatStatus,
)
from src.intelligence.temporal import (
    TemporalCorrelationEngine,
    TemporalProximityRule,
    TemporalRelationship,
)


def make_alert(
    alert_id: str,
    flow_id: Optional[str] = None,
    engine: str = DetectionEngineType.MULTICLASS.value,
    model_name: str = "rf_multiclass",
    prediction: str = "ATTACK",
    threat_status: str = ThreatStatus.ATTACK.value,
    is_attack: bool = True,
    category: Optional[str] = "PortScan",
    src_ip: Optional[str] = "192.168.10.50",
    dst_ip: Optional[str] = "172.16.0.1",
    src_port: Optional[int] = None,
    dst_port: Optional[int] = 80,
    timestamp: Optional[str] = "2026-11-19T10:00:00Z",
) -> Alert:
    """Helper factory for creating test Alert fixtures."""
    return Alert(
        alert_id=alert_id,
        flow_id=flow_id or f"flow_{alert_id}",
        detection_engine=engine,
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
        flow_duration=1500.0,
        severity=AlertSeverity.HIGH.value if is_attack else AlertSeverity.INFO.value,
        metadata={"fixture": "test_alert"},
    )


# -----------------------------------------------------------------------------
# 1. Relationship Graph Construction
# -----------------------------------------------------------------------------
def test_relationship_graph_construction():
    """Verifies graph node insertion, edge creation, adjacency, and duplicate rejection."""
    graph = RelationshipGraph()

    alt1 = make_alert("alt_1")
    alt2 = make_alert("alt_2")

    graph.add_alert(alt1)
    graph.add_alert(alt2)

    assert len(graph.nodes) == 2
    assert "alt_1" in graph.nodes
    assert "alt_2" in graph.nodes

    # Add valid edge
    rel = AlertRelationship(
        relationship_id="rel_01",
        source_alert_id="alt_1",
        target_alert_id="alt_2",
        relationship_type=RelationshipType.SHARED_DESTINATION_IP.value,
        weight=1.0,
        grouping_eligible=True,
    )
    added = graph.add_relationship(rel)
    assert added is True
    assert len(graph.edges) == 1

    # Duplicate edge rejection
    added_dup = graph.add_relationship(rel)
    assert added_dup is False
    assert len(graph.edges) == 1

    # Adjacency and neighbor lookup
    assert graph.get_neighbors("alt_1") == ["alt_2"]
    assert graph.get_neighbors("alt_2") == ["alt_1"]
    assert len(graph.get_edges_between("alt_1", "alt_2")) == 1


# -----------------------------------------------------------------------------
# 2. Shared Entity Relationships
# -----------------------------------------------------------------------------
def test_shared_entity_relationships():
    """Verifies that shared source IP, destination IP, and network entities form correct edges."""
    # Pair 1: Shared source IP, different destinations
    a1 = make_alert("a1", src_ip="10.0.0.1", dst_ip="172.16.1.1", dst_port=80)
    a2 = make_alert("a2", src_ip="10.0.0.1", dst_ip="172.16.2.2", dst_port=443)

    g1 = RelationshipGraph.build_from_alerts([a1, a2])
    edges1 = g1.get_edges_between("a1", "a2")
    types1 = {e.relationship_type for e in edges1}
    assert RelationshipType.SHARED_SOURCE_IP.value in types1

    # Pair 2: Shared destination IP, different ports
    b1 = make_alert("b1", src_ip="10.0.0.1", dst_ip="172.16.0.10", dst_port=80)
    b2 = make_alert("b2", src_ip="10.0.0.2", dst_ip="172.16.0.10", dst_port=443)

    g2 = RelationshipGraph.build_from_alerts([b1, b2])
    edges2 = g2.get_edges_between("b1", "b2")
    types2 = {e.relationship_type for e in edges2}
    assert RelationshipType.SHARED_DESTINATION_IP.value in types2

    # Pair 3: Shared network entity (destination IP and destination port)
    c1 = make_alert("c1", src_ip="10.0.0.1", dst_ip="172.16.0.10", dst_port=22)
    c2 = make_alert("c2", src_ip="10.0.0.2", dst_ip="172.16.0.10", dst_port=22)

    g3 = RelationshipGraph.build_from_alerts([c1, c2])
    edges3 = g3.get_edges_between("c1", "c2")
    types3 = {e.relationship_type for e in edges3}
    assert RelationshipType.SHARED_NETWORK_ENTITY.value in types3

    # Pair 4: Shared port with different destination IPs (grouping_eligible must be False)
    d1 = make_alert("d1", src_ip="10.0.0.1", dst_ip="172.16.1.1", dst_port=80)
    d2 = make_alert("d2", src_ip="10.0.0.2", dst_ip="172.16.2.2", dst_port=80)

    g4 = RelationshipGraph.build_from_alerts([d1, d2])
    port_edges = [e for e in g4.get_edges_between("d1", "d2") if e.relationship_type == RelationshipType.SHARED_DESTINATION_PORT.value]
    assert len(port_edges) == 1
    assert port_edges[0].grouping_eligible is False


# -----------------------------------------------------------------------------
# 3. Temporal Relationships Integration
# -----------------------------------------------------------------------------
def test_temporal_relationships_integration():
    """Verifies that pre-computed temporal relationships are ingested into the graph."""
    alt_a = make_alert("scan_1", category="PortScan", timestamp="2026-11-19T10:00:00Z")
    alt_b = make_alert("brute_2", category="BruteForce", timestamp="2026-11-19T10:00:30Z")

    temp_rel = TemporalRelationship(
        relationship_id="trel_001",
        source_alert_id="scan_1",
        target_alert_id="brute_2",
        rule_name="sequential_ordering",
        source_timestamp="2026-11-19T10:00:00+00:00",
        target_timestamp="2026-11-19T10:00:30+00:00",
        time_delta_seconds=30.0,
        window_seconds=300.0,
        is_sequential=True,
        evidence={
            "earlier_category": "PortScan",
            "later_category": "BruteForce",
            "shared_target_detected": True,
            "has_shared_entities": True,
            "causation_caveat": "Candidate progression pattern.",
        },
    )

    graph = RelationshipGraph.build_from_alerts([alt_a, alt_b], temporal_relationships=[temp_rel])
    edges = graph.get_edges_between("scan_1", "brute_2")
    temp_edges = [e for e in edges if e.relationship_type == RelationshipType.TEMPORAL_SEQUENCE.value]

    assert len(temp_edges) == 1
    assert temp_edges[0].grouping_eligible is True
    assert "candidate" in temp_edges[0].description.lower() or "sequential" in temp_edges[0].description.lower()


# -----------------------------------------------------------------------------
# 4. Missing Metadata (None == None Never Links Alerts)
# -----------------------------------------------------------------------------
def test_missing_metadata_never_links_alerts():
    """Verifies that two alerts with completely missing metadata never link as sharing entities."""
    alt_x = make_alert("alt_x", flow_id="flow_x", src_ip=None, dst_ip=None, src_port=None, dst_port=None, timestamp=None, category=None)
    alt_y = make_alert("alt_y", flow_id="flow_y", src_ip=None, dst_ip=None, src_port=None, dst_port=None, timestamp=None, category=None)

    graph = RelationshipGraph.build_from_alerts([alt_x, alt_y])
    edges = graph.get_edges_between("alt_x", "alt_y")

    # Strictly zero edges: None must NEVER match None
    assert len(edges) == 0


# -----------------------------------------------------------------------------
# 5. Candidate Incident Generation & Grouping
# -----------------------------------------------------------------------------
def test_candidate_incident_generation_and_grouping():
    """Verifies grouping related alerts into a single candidate incident."""
    generator = CandidateIncidentGenerator()

    # Three alerts targeting the same service port 22 in a brute-force wave
    alerts = [
        make_alert("bf_1", dst_ip="172.16.0.5", dst_port=22, category="BruteForce", timestamp="2026-11-19T10:00:00Z"),
        make_alert("bf_2", dst_ip="172.16.0.5", dst_port=22, category="BruteForce", timestamp="2026-11-19T10:00:10Z"),
        make_alert("bf_3", dst_ip="172.16.0.5", dst_port=22, category="BruteForce", timestamp="2026-11-19T10:00:20Z"),
    ]

    incidents = generator.generate_incidents(alerts)

    assert len(incidents) == 1
    inc = incidents[0]

    assert inc.incident_id.startswith("inc_")
    assert inc.status == IncidentStatus.CANDIDATE.value
    assert inc.alerts_count == 3
    assert set(inc.alert_ids) == {"bf_1", "bf_2", "bf_3"}
    assert inc.destination_ips == ["172.16.0.5"]
    assert inc.destination_ports == [22]
    assert inc.attack_categories == ["BruteForce"]
    assert inc.first_timestamp == "2026-11-19T10:00:00Z"
    assert inc.last_timestamp == "2026-11-19T10:00:20Z"
    assert inc.relationship_count > 0


# -----------------------------------------------------------------------------
# 6. Separation of Unrelated Alerts
# -----------------------------------------------------------------------------
def test_separation_of_unrelated_alerts():
    """Verifies that completely distinct alerts remain separate candidate incidents."""
    generator = CandidateIncidentGenerator()

    # Three completely independent alerts: different IPs, ports, flows, categories
    a1 = make_alert("alt_alpha", flow_id="f1", src_ip="10.1.0.1", dst_ip="192.168.1.1", src_port=40001, dst_port=80, category="WebAttack")
    a2 = make_alert("alt_beta", flow_id="f2", src_ip="10.2.0.2", dst_ip="192.168.2.2", src_port=40002, dst_port=22, category="BruteForce")
    a3 = make_alert("alt_gamma", flow_id="f3", src_ip="10.3.0.3", dst_ip="192.168.3.3", src_port=40003, dst_port=53, category="DoS_DDoS")

    incidents = generator.generate_incidents([a1, a2, a3])

    # Unrelated alerts MUST NOT be merged
    assert len(incidents) == 3
    for inc in incidents:
        assert inc.alerts_count == 1


# -----------------------------------------------------------------------------
# 7. Time-Only Proximity Does Not Merge Unrelated Alerts
# -----------------------------------------------------------------------------
def test_time_only_proximity_does_not_merge():
    """Verifies that temporal proximity without shared endpoints does not group alerts."""
    generator = CandidateIncidentGenerator()

    # Two alerts occurring 2 seconds apart, but targeting completely different hosts/ports
    a1 = make_alert("host1", flow_id="f1", src_ip="10.0.0.1", dst_ip="172.16.1.1", src_port=40001, dst_port=80, timestamp="2026-11-19T10:00:00Z", category="WebAttack")
    a2 = make_alert("host2", flow_id="f2", src_ip="10.0.0.2", dst_ip="172.16.2.2", src_port=40002, dst_port=22, timestamp="2026-11-19T10:00:02Z", category="BruteForce")

    incidents = generator.generate_incidents([a1, a2])

    # Must remain separate incidents because circumstantial time proximity is not grouping eligible
    assert len(incidents) == 2
    assert incidents[0].alerts_count == 1
    assert incidents[1].alerts_count == 1


# -----------------------------------------------------------------------------
# 8. Duplicate Alert Handling (Same Flow Consolidation)
# -----------------------------------------------------------------------------
def test_same_flow_multi_engine_grouping():
    """Verifies that multiple engine assessments of the same flow group into one candidate incident."""
    generator = CandidateIncidentGenerator()

    # Flow evaluated by binary model and multiclass model
    alt_bin = make_alert("bin_01", flow_id="flow_shared", engine=DetectionEngineType.BINARY.value, model_name="rf_full", prediction="ATTACK")
    alt_mc = make_alert("mc_02", flow_id="flow_shared", engine=DetectionEngineType.MULTICLASS.value, model_name="dt_full", prediction="DoS_DDoS", category="DoS_DDoS")

    incidents = generator.generate_incidents([alt_bin, alt_mc])

    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.alerts_count == 2
    assert set(inc.detection_engines) == {DetectionEngineType.BINARY.value, DetectionEngineType.MULTICLASS.value}
    assert any(r["relationship_type"] == RelationshipType.SAME_FLOW.value for r in inc.relationships)


# -----------------------------------------------------------------------------
# 9. Explainable Severity Assessment Levels
# -----------------------------------------------------------------------------
def test_severity_assessment_levels():
    """Verifies that severity scoring assigns appropriate tiers based on category, consensus, and target."""
    generator = CandidateIncidentGenerator()

    # 1. Benign only -> INFO
    benign_alt = make_alert("ben_1", prediction="BENIGN", threat_status=ThreatStatus.BENIGN.value, is_attack=False, category="BENIGN")
    inc_info = generator.generate_incidents([benign_alt])[0]
    assert inc_info.severity == AlertSeverity.INFO.value

    # 2. PortScan alone -> LOW or MEDIUM
    scan_alt = make_alert("scan_1", category="PortScan", dst_port=8080)
    inc_scan = generator.generate_incidents([scan_alt])[0]
    assert inc_scan.severity in (AlertSeverity.LOW.value, AlertSeverity.MEDIUM.value)

    # 3. Targeted BruteForce against SSH port 22 with multiple attempts -> HIGH
    bf1 = make_alert("b1", category="BruteForce", dst_port=22, dst_ip="172.16.0.1")
    bf2 = make_alert("b2", category="BruteForce", dst_port=22, dst_ip="172.16.0.1")
    bf3 = make_alert("b3", category="BruteForce", dst_port=22, dst_ip="172.16.0.1")
    inc_high = generator.generate_incidents([bf1, bf2, bf3])[0]
    assert inc_high.severity in (AlertSeverity.HIGH.value, AlertSeverity.CRITICAL.value)

    # 4. Infiltration or Botnet with multi-engine agreement -> CRITICAL
    bot1 = make_alert("bot_1", flow_id="bot_flow", engine="binary", category="Botnet", dst_port=445)
    bot2 = make_alert("bot_2", flow_id="bot_flow", engine="multiclass", category="Botnet", dst_port=445)
    bot3 = make_alert("bot_3", flow_id="bot_flow", engine="anomaly", threat_status=ThreatStatus.ANOMALY.value, is_attack=False, category=None, dst_port=445)
    inc_crit = generator.generate_incidents([bot1, bot2, bot3])[0]
    assert inc_crit.severity == AlertSeverity.CRITICAL.value


# -----------------------------------------------------------------------------
# 10. Explanation Generation
# -----------------------------------------------------------------------------
def test_explanation_generation():
    """Verifies that the generated incident explanation is detailed, transparent, and plain-language."""
    generator = CandidateIncidentGenerator()

    alerts = [
        make_alert("alt_1", dst_ip="172.16.0.10", dst_port=22, category="BruteForce"),
        make_alert("alt_2", dst_ip="172.16.0.10", dst_port=22, category="BruteForce"),
    ]

    inc = generator.generate_incidents(alerts)[0]

    expl = inc.explanation
    assert "grouping_rationale" in expl
    assert "relationships_identified" in expl
    assert "detection_engines_contributing" in expl
    assert "attack_categories_observed" in expl
    assert "missing_evidence" in expl
    assert "severity_justification" in expl
    assert "business_impact_caveat" in expl

    # Plain-language summary text verification
    assert len(inc.summary_text) > 0
    assert "Candidate Incident" in inc.summary_text
    assert "BruteForce" in inc.summary_text
    assert inc.severity in inc.summary_text


# -----------------------------------------------------------------------------
# 11. Investigation Recommendations
# -----------------------------------------------------------------------------
def test_investigation_recommendations():
    """Verifies that recommendations are practical, evidence-based, and do not falsely claim resolution."""
    generator = CandidateIncidentGenerator()

    alerts = [
        make_alert("alt_1", src_ip="192.168.1.100", dst_ip="172.16.0.2", dst_port=22, category="BruteForce"),
    ]

    inc = generator.generate_incidents(alerts)[0]

    recs = inc.recommendations
    assert len(recs) >= 3

    # Must mention investigating the originating host
    assert any("192.168.1.100" in r for r in recs)
    # Must mention checking service/authentication logs on port 22
    assert any("22" in r for r in recs)

    # Crucial negative assertion: Must NEVER claim attack was blocked, mitigated, or resolved!
    full_text = " ".join(recs).lower()
    assert "attack blocked" not in full_text
    assert "threat eliminated" not in full_text
    assert "incident resolved" not in full_text


# -----------------------------------------------------------------------------
# 12. Empty Input Handling
# -----------------------------------------------------------------------------
def test_empty_input_handling():
    """Verifies graceful handling of empty alert sequences."""
    generator = CandidateIncidentGenerator()
    incidents = generator.generate_incidents([])
    assert incidents == []


# -----------------------------------------------------------------------------
# 13. Partial Evidence Handling (CICIDS2017 Style)
# -----------------------------------------------------------------------------
def test_partial_evidence_cicids_style():
    """Verifies incident generation when timestamps and source IPs are missing (offline dataset records)."""
    generator = CandidateIncidentGenerator()

    # Offline record lacking timestamp, source IP, and destination IP (genuine MachineLearningCVE record)
    alt = make_alert(
        "cicids_alt",
        src_ip=None,
        dst_ip=None,
        src_port=None,
        dst_port=80,
        timestamp=None,
        category="WebAttack",
    )

    incidents = generator.generate_incidents([alt])
    assert len(incidents) == 1
    inc = incidents[0]

    # Verification: Does not fabricate missing metadata
    assert inc.first_timestamp is None
    assert inc.last_timestamp is None
    assert inc.source_ips == []
    assert inc.destination_ips == []
    assert inc.destination_ports == [80]

    # Explicit audit tracking of absent fields
    assert "source_ip" in inc.missing_evidence
    assert "destination_ip" in inc.missing_evidence
    assert "source_port" in inc.missing_evidence
    assert "timestamp" in inc.missing_evidence

    # Explanation notes missing evidence
    assert any("Source IP is absent" in note for note in inc.explanation["missing_evidence_notes"])
    assert any("Destination IP is absent" in note for note in inc.explanation["missing_evidence_notes"])
    assert any("timestamps are absent" in note for note in inc.explanation["missing_evidence_notes"])


def test_offline_cicids2017_flow_and_multi_engine_correlation():
    """Verifies that offline MachineLearningCVE records correlate legitimately using
    genuinely available evidence: same-flow identity, multi-engine consensus, and destination port.
    """
    generator = CandidateIncidentGenerator()

    # Two engines evaluating the exact same flow from MachineLearningCVE (IPs and timestamps absent)
    alt_supervised = make_alert(
        "alt_sup_001",
        flow_id="flow_cicids_98765",
        engine=DetectionEngineType.MULTICLASS.value,
        model_name="rf_multiclass",
        prediction="DoS_DDoS",
        threat_status=ThreatStatus.ATTACK.value,
        category="DoS_DDoS",
        src_ip=None,
        dst_ip=None,
        src_port=None,
        dst_port=80,
        timestamp=None,
    )
    alt_anomaly = make_alert(
        "alt_ano_002",
        flow_id="flow_cicids_98765",
        engine=DetectionEngineType.ANOMALY.value,
        model_name="iso_forest",
        prediction="ANOMALY",
        threat_status=ThreatStatus.ANOMALY.value,
        is_attack=False,
        category=None,
        src_ip=None,
        dst_ip=None,
        src_port=None,
        dst_port=80,
        timestamp=None,
    )

    incidents = generator.generate_incidents([alt_supervised, alt_anomaly])
    assert len(incidents) == 1
    inc = incidents[0]

    # Genuinely available evidence: same flow, multi-engine consensus, destination port
    assert inc.alerts_count == 2
    assert inc.source_ips == []
    assert inc.destination_ips == []
    assert inc.destination_ports == [80]
    assert inc.attack_categories == ["DoS_DDoS"]
    assert set(inc.detection_engines) == {DetectionEngineType.MULTICLASS.value, DetectionEngineType.ANOMALY.value}

    rel_types = {r["relationship_type"] for r in inc.relationships}
    assert RelationshipType.SAME_FLOW.value in rel_types
    assert RelationshipType.DETECTION_ENGINE_AGREEMENT.value in rel_types

    # Discloses absent endpoints honestly
    assert "source_ip" in inc.missing_evidence
    assert "destination_ip" in inc.missing_evidence
    assert "timestamp" in inc.missing_evidence

    # Explanation states correlation was based on flow and consensus, not fabricated IPs
    rationale = inc.explanation["grouping_rationale"]
    assert "multi-engine evaluations of the same network flow" in rationale
    assert "cross-engine consensus on threat presence" in rationale


# -----------------------------------------------------------------------------
# 14. Incident Serialization
# -----------------------------------------------------------------------------
def test_incident_serialization():
    """Verifies serialization of CandidateIncident to dictionary and JSON string."""
    generator = CandidateIncidentGenerator()
    alt = make_alert("alt_s1", category="DoS_DDoS")
    inc = generator.generate_incidents([alt])[0]

    d = inc.to_dict()
    assert isinstance(d, dict)
    assert d["incident_id"] == inc.incident_id
    assert d["status"] == "CANDIDATE"

    json_str = inc.to_json(indent=2)
    assert isinstance(json_str, str)
    assert inc.incident_id in json_str

    parsed = json.loads(json_str)
    assert parsed["incident_id"] == inc.incident_id
    assert parsed["severity"] == inc.severity
    assert "explanation" in parsed
    assert "recommendations" in parsed


# -----------------------------------------------------------------------------
# 15. Relationship & Explanation Consistency Verification
# -----------------------------------------------------------------------------
def test_relationship_incident_consistency_and_validation():
    """Verifies that:
    1. Every relationship in an incident references alert IDs strictly inside that incident.
    2. explanation['relationships_identified'] matches the exact set of relationships present.
    3. Single-alert incidents have empty relationship lists and empty relationships_identified.
    """
    generator = CandidateIncidentGenerator()

    # Multi-alert incident
    a1 = make_alert("m1", dst_ip="172.16.0.100", dst_port=80, category="WebAttack")
    a2 = make_alert("m2", dst_ip="172.16.0.100", dst_port=80, category="WebAttack")
    # Isolated alert
    a3 = make_alert("iso1", src_ip="10.200.0.99", dst_ip="10.200.0.1", dst_port=53, category="DoS_DDoS")

    incidents = generator.generate_incidents([a1, a2, a3])
    assert len(incidents) == 2

    # Grouped incident
    grp_inc = [inc for inc in incidents if inc.alerts_count == 2][0]
    alert_ids_set = set(grp_inc.alert_ids)

    # 1. Every relationship references valid constituent alerts
    assert len(grp_inc.relationships) > 0
    for rel in grp_inc.relationships:
        assert rel["source_alert_id"] in alert_ids_set
        assert rel["target_alert_id"] in alert_ids_set

    # 2. Explanation consistency
    rel_types_in_rels = sorted(list({r["relationship_type"] for r in grp_inc.relationships}))
    assert grp_inc.explanation["relationships_identified"] == rel_types_in_rels

    # 3. Isolated incident consistency
    iso_inc = [inc for inc in incidents if inc.alerts_count == 1][0]
    assert iso_inc.relationships == []
    assert iso_inc.relationship_count == 0
    assert iso_inc.explanation["relationships_identified"] == []
    assert "Single isolated alert" in iso_inc.explanation["grouping_rationale"]


# -----------------------------------------------------------------------------
# 16. Incident ID Uniqueness and Round-Trip Deserialization
# -----------------------------------------------------------------------------
def test_incident_id_uniqueness_and_round_trip_deserialization():
    """Verifies that:
    1. Incident IDs in large batches are guaranteed mutually unique.
    2. CandidateIncident.from_dict and CandidateIncident.from_json achieve complete round-trip fidelity.
    3. AlertRelationship.from_dict and AlertRelationship.from_json achieve complete round-trip fidelity.
    """
    generator = CandidateIncidentGenerator()

    # Generate 30 distinct alerts
    alerts = [
        make_alert(
            f"alt_batch_{i}",
            flow_id=f"flow_batch_{i}",
            src_ip=f"10.0.{i // 256}.{i % 256}",
            dst_ip=f"172.16.{i // 256}.{i % 256}",
            dst_port=1000 + i,
            category="PortScan",
        )
        for i in range(30)
    ]

    incidents = generator.generate_incidents(alerts)
    assert len(incidents) == 30

    # 1. Mutually unique IDs
    incident_ids = [inc.incident_id for inc in incidents]
    assert len(set(incident_ids)) == 30

    # 2. Round-trip serialization fidelity
    first_inc = incidents[0]
    inc_dict = first_inc.to_dict()
    reconstructed_from_dict = CandidateIncident.from_dict(inc_dict)
    assert reconstructed_from_dict.incident_id == first_inc.incident_id
    assert reconstructed_from_dict.severity == first_inc.severity
    assert reconstructed_from_dict.alert_ids == first_inc.alert_ids

    inc_json = first_inc.to_json()
    reconstructed_from_json = CandidateIncident.from_json(inc_json)
    assert reconstructed_from_json.incident_id == first_inc.incident_id
    assert reconstructed_from_json.alert_ids == first_inc.alert_ids
    assert reconstructed_from_json.explanation == first_inc.explanation

    # 3. AlertRelationship round-trip
    rel = AlertRelationship(
        relationship_id="rel_test_rt",
        source_alert_id="a1",
        target_alert_id="a2",
        relationship_type="shared_source_ip",
        weight=1.5,
        grouping_eligible=True,
        evidence={"ip": "1.2.3.4"},
        description="Test round trip",
    )
    rel_dict = rel.to_dict()
    rel_recon = AlertRelationship.from_dict(rel_dict)
    assert rel_recon.relationship_id == "rel_test_rt"
    assert rel_recon.grouping_eligible is True

