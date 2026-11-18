"""VYUHA Candidate Incident Generation Engine (Phase 4).

Transforms normalized Alert objects and RelationshipGraph topologies into explainable
CandidateIncident dossiers.

Core Capabilities:
1. Incident Grouping:
   - Partitions alerts into coherent candidate security incidents using grouping-eligible
     edges in the RelationshipGraph.
   - Strictly keeps unrelated alerts separate (avoids over-clustering).
   - Never groups alerts based solely on circumstantial temporal proximity without endpoint anchors.
2. Provisional Severity Assessment:
   - Evaluates multi-factor, transparent severity (INFO, LOW, MEDIUM, HIGH, CRITICAL).
   - Accounts for threat categories, alert volumes, multi-engine consensus, and critical ports.
   - Disclaims business impact (notes that true impact requires asset criticality absent in CICIDS2017).
3. Human-Centric Explainability:
   - Produces clear, plain-language explanations answering:
     * Why alerts were grouped.
     * Which relationships were identified.
     * Which detection engines contributed.
     * Which attack categories were observed.
     * What evidence is missing.
     * Why the provisional severity was assigned.
4. Actionable Investigation Recommendations:
   - Generates deterministic, evidence-based triage steps.
   - Never claims attacks were blocked, prevented, or resolved.
5. Missing Telemetry Resilience:
   - Gracefully handles partial telemetry (e.g. offline CICIDS2017 lacking timestamps/source IPs).
   - Never fabricates missing entities or relationships.
"""

from collections import defaultdict
import datetime
import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import uuid

from src.intelligence.graph import RelationshipGraph
from src.intelligence.schema import (
    Alert,
    AlertSeverity,
    CandidateIncident,
    IncidentStatus,
    RelationshipType,
    SEVERITY_ORDER,
    ThreatStatus,
)
from src.intelligence.temporal import (
    TemporalCorrelationEngine,
    TemporalCorrelationResult,
    TemporalRelationship,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Severity Scoring Weights & Configuration
# -----------------------------------------------------------------------------

# Threat category base weights
CATEGORY_WEIGHTS: Dict[str, int] = {
    "Infiltration": 3,
    "Botnet": 3,
    "Heartbleed": 3,
    "WebAttack": 2,
    "DoS_DDoS": 2,
    "BruteForce": 2,
    "PortScan": 1,
    "ATTACK": 1,
    "ANOMALY": 1,
    "BENIGN": 0,
}

# Well-known critical / sensitive service ports
CRITICAL_PORTS: Set[int] = {
    21,    # FTP
    22,    # SSH
    23,    # Telnet
    25,    # SMTP
    53,    # DNS
    80,    # HTTP
    88,    # Kerberos
    110,   # POP3
    135,   # MS RPC
    139,   # NetBIOS
    443,   # HTTPS
    445,   # SMB
    1433,  # MS SQL
    3306,  # MySQL
    3389,  # RDP
    5432,  # PostgreSQL
    8080,  # HTTP Alt / Proxy
    8443,  # HTTPS Alt
}


class CandidateIncidentGenerator:
    """Orchestrates relationship graph evaluation, incident grouping, and explainable reporting."""

    def __init__(
        self,
        temporal_window_seconds: float = 300.0,
        enable_temporal_correlation: bool = True,
    ):
        """Initializes CandidateIncidentGenerator.

        Args:
            temporal_window_seconds: Observation window boundary in seconds for temporal correlation.
            enable_temporal_correlation: Whether to automatically run TemporalCorrelationEngine
                                         if alerts contain valid timestamps.
        """
        self.temporal_window_seconds = temporal_window_seconds
        self.enable_temporal_correlation = enable_temporal_correlation
        self.temporal_engine = TemporalCorrelationEngine(window_seconds=temporal_window_seconds)
        self._used_ids: Set[str] = set()

    def generate_incidents(
        self,
        alerts: Sequence[Alert],
        temporal_relationships: Optional[Sequence[TemporalRelationship]] = None,
    ) -> List[CandidateIncident]:
        """Transforms a sequence of alerts into prioritized, explainable candidate incidents.

        Args:
            alerts: Sequence of canonical Alert objects.
            temporal_relationships: Optional pre-computed temporal relationships. If None and
                                     alerts have timestamps, TemporalCorrelationEngine is executed.

        Returns:
            List of CandidateIncident objects sorted by severity descending.
        """
        if not alerts:
            return []

        # 1. Obtain temporal relationships if timestamps exist and none were provided
        t_rels: List[TemporalRelationship] = []
        if temporal_relationships is not None:
            t_rels = list(temporal_relationships)
        elif self.enable_temporal_correlation:
            has_timestamps = any(a.timestamp and str(a.timestamp).strip() for a in alerts)
            if has_timestamps:
                t_result = self.temporal_engine.correlate(alerts)
                t_rels = t_result.relationships

        # 2. Construct Relationship Graph
        graph = RelationshipGraph.build_from_alerts(alerts, temporal_relationships=t_rels)

        # 3. Partition Graph into Connected Components over Grouping-Eligible Edges
        # Unrelated alerts that share no grouping-eligible edges form singleton components
        components = graph.get_connected_components(grouping_only=True)

        incidents: List[CandidateIncident] = []

        # 4. Synthesize CandidateIncident for each Component
        for comp_ids in components:
            comp_alerts = [graph.nodes[aid] for aid in comp_ids if aid in graph.nodes]
            if not comp_alerts:
                continue

            incident = self._build_candidate_incident(comp_alerts, graph)
            incidents.append(incident)

        # 5. Prioritize Incidents: Severity Descending, then Alert Count Descending
        incidents.sort(
            key=lambda inc: (SEVERITY_ORDER.get(inc.severity, 0), inc.alerts_count),
            reverse=True,
        )

        return incidents

    def _build_candidate_incident(
        self,
        alerts: List[Alert],
        graph: RelationshipGraph,
    ) -> CandidateIncident:
        """Constructs an individual CandidateIncident from a grouped cluster of alerts."""
        alert_ids = [a.alert_id for a in alerts]
        alert_ids_set = set(alert_ids)

        # 1. Extract Edges Connecting Alerts in this Incident
        incident_rels: List[Dict[str, Any]] = []
        for rel in graph.edges:
            if rel.source_alert_id in alert_ids_set and rel.target_alert_id in alert_ids_set:
                incident_rels.append(rel.to_dict())

        # 2. Extract Timestamps
        valid_timestamps = [a.timestamp for a in alerts if a.timestamp and str(a.timestamp).strip()]
        first_ts: Optional[str] = None
        last_ts: Optional[str] = None
        if valid_timestamps:
            first_ts = min(valid_timestamps)
            last_ts = max(valid_timestamps)

        # 3. Extract Distinct Network Endpoints (STRICT INTEGRITY: Excludes None)
        source_ips = sorted(list({a.source_ip for a in alerts if a.source_ip}))
        destination_ips = sorted(list({a.destination_ip for a in alerts if a.destination_ip}))
        source_ports = sorted(list({a.source_port for a in alerts if a.source_port is not None}))
        destination_ports = sorted(list({a.destination_port for a in alerts if a.destination_port is not None}))

        # 4. Extract Threat Categories & Engines
        categories = sorted(list({
            a.attack_category for a in alerts
            if a.attack_category and a.attack_category != "BENIGN"
        }))
        threat_statuses = sorted(list({a.threat_status for a in alerts if a.threat_status}))
        detection_engines = sorted(list({a.detection_engine for a in alerts if a.detection_engine}))

        # 5. Missing Telemetry Audit
        missing_evidence: List[str] = []
        if not source_ips:
            missing_evidence.append("source_ip")
        if not destination_ips:
            missing_evidence.append("destination_ip")
        if not source_ports:
            missing_evidence.append("source_port")
        if not destination_ports:
            missing_evidence.append("destination_port")
        if not valid_timestamps:
            missing_evidence.append("timestamp")

        # 6. Assess Provisional Severity
        severity, severity_reasons, score = self._assess_provisional_severity(
            alerts=alerts,
            categories=categories,
            destination_ports=destination_ports,
            incident_rels=incident_rels,
        )

        # 7. Generate Explainability Dossier
        explanation, summary_text = self._generate_explanation(
            alerts=alerts,
            categories=categories,
            source_ips=source_ips,
            destination_ips=destination_ips,
            destination_ports=destination_ports,
            detection_engines=detection_engines,
            incident_rels=incident_rels,
            missing_evidence=missing_evidence,
            severity=severity,
            severity_reasons=severity_reasons,
            score=score,
            first_ts=first_ts,
            last_ts=last_ts,
        )

        # 8. Generate Rule-Based Investigation Recommendations
        recommendations = self._generate_recommendations(
            source_ips=source_ips,
            destination_ips=destination_ips,
            destination_ports=destination_ports,
            categories=categories,
            threat_statuses=threat_statuses,
            alerts_count=len(alerts),
            has_sequence=any(
                r.get("relationship_type") == RelationshipType.TEMPORAL_SEQUENCE.value
                for r in incident_rels
            ),
        )

        inc_id = f"inc_{uuid.uuid4().hex[:12]}"
        while inc_id in self._used_ids:
            inc_id = f"inc_{uuid.uuid4().hex[:12]}"
        self._used_ids.add(inc_id)

        return CandidateIncident(
            incident_id=inc_id,
            alert_ids=alert_ids,
            status=IncidentStatus.CANDIDATE.value,
            severity=severity,
            first_timestamp=first_ts,
            last_timestamp=last_ts,
            source_ips=source_ips,
            destination_ips=destination_ips,
            source_ports=source_ports,
            destination_ports=destination_ports,
            attack_categories=categories,
            threat_statuses=threat_statuses,
            detection_engines=detection_engines,
            alerts_count=len(alerts),
            relationships=incident_rels,
            relationship_count=len(incident_rels),
            explanation=explanation,
            summary_text=summary_text,
            recommendations=recommendations,
            missing_evidence=missing_evidence,
            metadata={
                "scoring_points": score,
                "dataset_source": "CICIDS2017" if "timestamp" in missing_evidence else "live_telemetry",
            },
        )

    def _assess_provisional_severity(
        self,
        alerts: List[Alert],
        categories: List[str],
        destination_ports: List[int],
        incident_rels: List[Dict[str, Any]],
    ) -> Tuple[str, List[str], int]:
        """Evaluates explainable multi-factor severity scoring.

        Factors:
        - Critical categories (Infiltration, Botnet, Heartbleed): +3
        - High-impact categories (WebAttack, DoS_DDoS, BruteForce): +2
        - Reconnaissance / Generic ATTACK: +1
        - Unsupervised Anomaly presence: +1
        - Multi-engine consensus: +1
        - Alert burst / high volume (count >= 3): +1
        - Sensitive destination port targeted: +1
        - Candidate attack pattern sequence: +1

        Returns:
            Tuple of (severity_string, list_of_reason_strings, total_score_int).
        """
        # If all constituent alerts are BENIGN, assign INFO
        if all(a.threat_status == ThreatStatus.BENIGN.value for a in alerts):
            return AlertSeverity.INFO.value, ["All constituent alerts classified as BENIGN traffic."], 0

        score = 0
        reasons: List[str] = []

        # (a) Category Weight
        cat_scores = [CATEGORY_WEIGHTS.get(c, 1) for c in categories]
        max_cat_score = max(cat_scores) if cat_scores else 0

        # If no named categories, check threat status
        if max_cat_score == 0:
            if any(a.threat_status == ThreatStatus.ATTACK.value for a in alerts):
                max_cat_score = 1
            elif any(a.threat_status == ThreatStatus.ANOMALY.value for a in alerts):
                max_cat_score = 1

        score += max_cat_score
        if max_cat_score == 3:
            severe_cats = [c for c in categories if CATEGORY_WEIGHTS.get(c, 0) == 3]
            reasons.append(f"High-impact exploitation threat category detected: {', '.join(severe_cats)} (+3)")
        elif max_cat_score == 2:
            mod_cats = [c for c in categories if CATEGORY_WEIGHTS.get(c, 0) == 2]
            reasons.append(f"Active exploitation category detected: {', '.join(mod_cats)} (+2)")
        elif max_cat_score == 1:
            reasons.append("Reconnaissance, generic attack, or statistical anomaly detected (+1)")

        # (b) Detection Engine Consensus
        engines = {a.detection_engine for a in alerts}
        has_supervised = any(a.threat_status == ThreatStatus.ATTACK.value for a in alerts)
        has_anomaly = any(a.threat_status == ThreatStatus.ANOMALY.value for a in alerts)
        has_multi_engine_alert = any(a.detection_engine == "multi_engine" for a in alerts)

        if (has_supervised and has_anomaly) or len(engines) >= 2 or has_multi_engine_alert:
            score += 1
            reasons.append("Cross-engine detection consensus corroborated suspicious activity (+1)")

        # (c) High Alert Volume / Repeated Burst
        if len(alerts) >= 3:
            score += 1
            reasons.append(f"Repeated activity observed across {len(alerts)} alerts (+1)")

        # (d) Critical Service Port Targeted
        targeted_critical = set(destination_ports).intersection(CRITICAL_PORTS)
        if targeted_critical:
            score += 1
            reasons.append(f"Targeted well-known service port(s): {sorted(list(targeted_critical))} (+1)")

        # (e) Confirmed Targeted Network Entity
        has_network_entity = any(
            r.get("relationship_type") == RelationshipType.SHARED_NETWORK_ENTITY.value
            for r in incident_rels
        )
        if has_network_entity and len(alerts) >= 2:
            score += 1
            reasons.append("Multiple alerts confirmed targeting identical host and service port (+1)")

        # (f) Candidate Attack Pattern Sequence
        has_seq = any(
            r.get("relationship_type") == RelationshipType.TEMPORAL_SEQUENCE.value
            for r in incident_rels
        )
        if has_seq:
            score += 1
            reasons.append("Observed chronological progression matching candidate attack-pattern sequence (+1)")

        # Mapping score to discrete severity levels
        if score >= 5:
            severity = AlertSeverity.CRITICAL.value
        elif score >= 3:
            severity = AlertSeverity.HIGH.value
        elif score == 2:
            severity = AlertSeverity.MEDIUM.value
        else:
            severity = AlertSeverity.LOW.value

        return severity, reasons, score

    def _generate_explanation(
        self,
        alerts: List[Alert],
        categories: List[str],
        source_ips: List[str],
        destination_ips: List[str],
        destination_ports: List[int],
        detection_engines: List[str],
        incident_rels: List[Dict[str, Any]],
        missing_evidence: List[str],
        severity: str,
        severity_reasons: List[str],
        score: int,
        first_ts: Optional[str],
        last_ts: Optional[str],
    ) -> Tuple[Dict[str, Any], str]:
        """Generates plain-language explainability dictionary and human-readable narrative."""
        # 1. Grouping Rationale
        rel_types = {r.get("relationship_type") for r in incident_rels}
        if len(alerts) == 1:
            grouping_rationale = (
                "Single isolated alert; no corroborating endpoint or flow links connected this "
                "event to other observed traffic."
            )
        else:
            reasons_parts = []
            if RelationshipType.SAME_FLOW.value in rel_types:
                reasons_parts.append("multi-engine evaluations of the same network flow")

            # Extract exact matched service endpoints from relationships
            matched_services = sorted(list({
                f"{r['evidence']['destination_ip']}:{r['evidence']['destination_port']}"
                for r in incident_rels
                if r.get("relationship_type") == RelationshipType.SHARED_NETWORK_ENTITY.value
                and "destination_ip" in r.get("evidence", {}) and "destination_port" in r.get("evidence", {})
            }))
            if matched_services:
                reasons_parts.append(f"targeting the same destination service ({', '.join(matched_services)})")
            elif RelationshipType.SHARED_DESTINATION_IP.value in rel_types:
                matched_dst_ips = sorted(list({
                    r['evidence']['destination_ip']
                    for r in incident_rels
                    if r.get("relationship_type") == RelationshipType.SHARED_DESTINATION_IP.value
                    and "destination_ip" in r.get("evidence", {})
                }))
                if matched_dst_ips:
                    reasons_parts.append(f"targeting shared victim host(s) ({', '.join(matched_dst_ips)})")

            if RelationshipType.SHARED_SOURCE_IP.value in rel_types:
                matched_src_ips = sorted(list({
                    r['evidence']['source_ip']
                    for r in incident_rels
                    if r.get("relationship_type") == RelationshipType.SHARED_SOURCE_IP.value
                    and "source_ip" in r.get("evidence", {})
                }))
                if matched_src_ips:
                    reasons_parts.append(f"originating from shared source host ({', '.join(matched_src_ips)})")

            if RelationshipType.TEMPORAL_SEQUENCE.value in rel_types:
                reasons_parts.append("demonstrating candidate multi-stage attack sequence")

            if RelationshipType.DETECTION_ENGINE_AGREEMENT.value in rel_types:
                reasons_parts.append("cross-engine consensus on threat presence")

            if RelationshipType.REPEATED_CATEGORY.value in rel_types:
                matched_cats = sorted(list({
                    r['evidence']['attack_category']
                    for r in incident_rels
                    if r.get("relationship_type") == RelationshipType.REPEATED_CATEGORY.value
                    and "attack_category" in r.get("evidence", {})
                }))
                if matched_cats:
                    reasons_parts.append(f"exhibiting repeated '{', '.join(matched_cats)}' threat category")

            grouping_rationale = (
                f"Grouped {len(alerts)} alerts based on " + "; ".join(reasons_parts) + "."
                if reasons_parts else f"Grouped {len(alerts)} alerts based on corroborated relationship evidence."
            )

        # 2. Missing Evidence Details
        missing_notes: List[str] = []
        if "source_ip" in missing_evidence:
            missing_notes.append("Source IP is absent in dataset; attacker attribution cannot be established from flow records.")
        if "destination_ip" in missing_evidence:
            missing_notes.append("Destination IP is absent in dataset; target host attribution cannot be established from flow records.")
        if "timestamp" in missing_evidence:
            missing_notes.append("Packet timestamps are absent in offline CICIDS2017 dataset; temporal duration was not evaluated.")
        if not missing_notes:
            missing_notes.append("All primary network telemetry fields (IPs, ports, timestamps) were available.")

        # 3. Structured Explanation
        explanation = {
            "grouping_rationale": grouping_rationale,
            "relationships_identified": sorted(list(rel_types)),
            "detection_engines_contributing": detection_engines,
            "attack_categories_observed": categories if categories else ["None / Unclassified"],
            "missing_evidence": missing_evidence,
            "missing_evidence_notes": missing_notes,
            "severity_assigned": severity,
            "severity_score": score,
            "severity_justification": "; ".join(severity_reasons),
            "business_impact_caveat": (
                "Provisional severity reflects technical anomaly and attack metrics. "
                "True business impact requires asset criticality and environment context, "
                "which are absent in offline network benchmark datasets."
            ),
        }

        # 4. Human-Readable Prose Summary (for investigator and hackathon judge)
        cats_str = ", ".join(categories) if categories else "unclassified/anomalous"
        eng_str = ", ".join(detection_engines)
        time_str = f"between {first_ts} and {last_ts}" if (first_ts and last_ts and first_ts != last_ts) else (f"at {first_ts}" if first_ts else "within offline dataset records")

        summary_text = (
            f"Candidate Incident groups {len(alerts)} alert(s) observed {time_str}. "
            f"Observed threat activity involves '{cats_str}' identified by {eng_str}. "
            f"{grouping_rationale} "
            f"Provisional severity is assessed as {severity} ({'; '.join(severity_reasons)}). "
            f"Note: {' '.join(missing_notes)}"
        )

        return explanation, summary_text

    def _generate_recommendations(
        self,
        source_ips: List[str],
        destination_ips: List[str],
        destination_ports: List[int],
        categories: List[str],
        threat_statuses: List[str],
        alerts_count: int,
        has_sequence: bool,
    ) -> List[str]:
        """Generates deterministic, evidence-grounded investigation recommendations."""
        recs: List[str] = []

        # 1. Source Host Investigation
        if source_ips:
            recs.append(
                f"Investigate originating host(s) {', '.join(source_ips)} for unauthorized software, "
                "active scanning tools, or indicators of external compromise."
            )
        else:
            recs.append(
                "Identify originating host IP address via perimeter firewall, NAT session tables, "
                "or NetFlow logs (source IP absent in flow record)."
            )

        # 2. Destination Host & Port Inspection
        if destination_ips and destination_ports:
            recs.append(
                f"Inspect service and authentication logs on destination host(s) {', '.join(destination_ips)} "
                f"on targeted port(s) {', '.join(map(str, destination_ports))} for repeated failed authentications "
                "or unexpected service restarts."
            )
        elif destination_ports:
            recs.append(
                f"Review service logs for port(s) {', '.join(map(str, destination_ports))} across internal hosts "
                "to verify whether services accepted suspicious connections."
            )

        # 3. Threat-Specific Investigation Playbooks
        if "BruteForce" in categories:
            recs.append(
                "Audit authentication logs for account lockout events and inspect successful logins "
                "immediately following the alert timeframe to verify if credentials were compromised."
            )
        if "PortScan" in categories:
            recs.append(
                "Check network firewall logs for horizontal port sweeps or host discovery probes "
                "originating from the same network segment."
            )
        if "WebAttack" in categories:
            recs.append(
                "Examine web application firewall (WAF) and HTTP access logs for SQL injection, "
                "cross-site scripting, or path traversal URI patterns."
            )
        if "DoS_DDoS" in categories:
            recs.append(
                "Review perimeter bandwidth utilization, SYN flood protection counters, and connection "
                "state tables during the incident window."
            )
        if "Botnet" in categories:
            recs.append(
                "Isolate suspicious internal hosts immediately from the local VLAN and inspect DNS query logs "
                "for domain generation algorithms (DGA) or known command-and-control (C2) beaconing."
            )
        if "Infiltration" in categories:
            recs.append(
                "Initiate host-level forensic analysis on target systems: examine active user sessions, "
                "recently created scheduled tasks/cron jobs, and modified binary executables."
            )
        if "Heartbleed" in categories:
            recs.append(
                "Confirm TLS/SSL library versions on target servers to ensure OpenSSL patches "
                "mitigating CVE-2014-0160 are active."
            )
        if ThreatStatus.ANOMALY.value in threat_statuses and not categories:
            recs.append(
                "Cross-reference anomalous traffic metrics (unusual packet sizes or flow rates) against "
                "scheduled administrative transfers or automated backup operations."
            )

        # 4. Sequential Pattern Progression
        if has_sequence:
            recs.append(
                "Candidate reconnaissance-to-exploitation progression observed: correlate endpoint logs "
                "to determine if subsequent lateral movement or persistence attempts occurred."
            )

        # 5. Activity Continuity Check
        recs.append(
            "Review network flow telemetry immediately following the incident window to verify "
            "whether suspicious connection attempts ceased or persisted."
        )

        return recs
