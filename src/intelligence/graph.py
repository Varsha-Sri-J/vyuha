"""VYUHA Alert Relationship Graph Module (Phase 4).

Constructs an explainable, lightweight relationship graph connecting normalized Alert
objects using multiple corroborating lines of evidence:
1. Shared Source IP (attacker host attribution / scanning).
2. Shared Destination IP (victim host correlation).
3. Shared Network Entity (destination IP + service port).
4. Shared Destination Port (targeted service).
5. Same Flow ID (multi-engine evaluations of the same network event).
6. Repeated Attack Category (coordinated bursts).
7. Detection-Engine Agreement (cross-engine consensus on threats).
8. Temporal Correlation Integration (sequential ordering, recurrence, proximity).

Strict Cybersecurity Semantics & Integrity Guarantees:
- Missing values (None in alert_a and None in alert_b) NEVER form entities or relationships.
- Time proximity alone is circumstantial; it is preserved as an informational edge
  with grouping_eligible=False so it never merges unrelated hosts into one incident.
- Every edge preserves an explicit explanation, edge type, weight, and evidence dictionary.
- No heavy external graph libraries required; uses lightweight adjacency representations.
"""

from collections import defaultdict
from dataclasses import asdict
import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import uuid

from src.intelligence.schema import (
    Alert,
    AlertRelationship,
    RelationshipType,
    ThreatStatus,
)
from src.intelligence.temporal import TemporalRelationship

logger = logging.getLogger(__name__)


class RelationshipGraph:
    """Lightweight explainable graph structure connecting alerts via corroborating evidence."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Alert] = {}
        self.edges: List[AlertRelationship] = []
        self._adj: Dict[str, List[AlertRelationship]] = defaultdict(list)
        self._edge_lookup: Set[Tuple[str, str, str]] = set()

    def add_alert(self, alert: Alert) -> None:
        """Adds an alert node to the graph if not already present."""
        if alert.alert_id not in self.nodes:
            self.nodes[alert.alert_id] = alert

    def add_relationship(self, relationship: AlertRelationship) -> bool:
        """Adds an explainable relationship edge between two existing alert nodes.

        Prevents duplicate edges between the same alert pair with the same relationship type.

        Args:
            relationship: AlertRelationship instance.

        Returns:
            True if edge was added; False if duplicate or endpoints missing.
        """
        src = relationship.source_alert_id
        dst = relationship.target_alert_id
        rtype = relationship.relationship_type

        # Ensure both endpoints exist in the graph
        if src not in self.nodes or dst not in self.nodes:
            logger.warning(
                "Cannot add relationship %s: endpoints (%s, %s) not both in graph.",
                relationship.relationship_id,
                src,
                dst,
            )
            return False

        # Prevent duplicate bidirectional edge insertion
        key_fwd = (src, dst, rtype)
        key_rev = (dst, src, rtype)
        if key_fwd in self._edge_lookup or key_rev in self._edge_lookup:
            return False

        self.edges.append(relationship)
        self._adj[src].append(relationship)
        self._adj[dst].append(relationship)
        self._edge_lookup.add(key_fwd)
        return True

    def get_neighbors(self, alert_id: str, grouping_only: bool = False) -> List[str]:
        """Returns adjacent alert IDs connected to the specified alert.

        Args:
            alert_id: Identifier of the alert node.
            grouping_only: If True, only follows edges where grouping_eligible is True.
        """
        neighbors: Set[str] = set()
        for rel in self._adj.get(alert_id, []):
            if grouping_only and not rel.grouping_eligible:
                continue
            nbr = rel.target_alert_id if rel.source_alert_id == alert_id else rel.source_alert_id
            neighbors.add(nbr)
        return list(neighbors)

    def get_edges_for_alert(self, alert_id: str) -> List[AlertRelationship]:
        """Returns all relationship edges incident to the specified alert."""
        return list(self._adj.get(alert_id, []))

    def get_edges_between(self, alert_id_a: str, alert_id_b: str) -> List[AlertRelationship]:
        """Returns all relationship edges connecting two specific alerts."""
        return [
            rel for rel in self._adj.get(alert_id_a, [])
            if (rel.source_alert_id == alert_id_b or rel.target_alert_id == alert_id_b)
        ]

    def get_connected_components(self, grouping_only: bool = True) -> List[List[str]]:
        """Computes connected components across the graph.

        Args:
            grouping_only: If True (default), traverses only grouping-eligible edges.
                           This guarantees that circumstantial temporal proximity alone
                           never chains unrelated alerts together.

        Returns:
            List of components, each being a list of alert IDs.
        """
        visited: Set[str] = set()
        components: List[List[str]] = []

        # Process nodes in deterministic insertion order
        for alert_id in self.nodes:
            if alert_id in visited:
                continue

            component: List[str] = []
            queue: List[str] = [alert_id]
            visited.add(alert_id)

            while queue:
                curr = queue.pop(0)
                component.append(curr)

                for nbr in self.get_neighbors(curr, grouping_only=grouping_only):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)

            components.append(component)

        return components

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the graph to a dictionary structure."""
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "nodes": [a.to_dict() for a in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def build_from_alerts(
        cls,
        alerts: Sequence[Alert],
        temporal_relationships: Optional[Sequence[TemporalRelationship]] = None,
    ) -> "RelationshipGraph":
        """Constructs and populates a RelationshipGraph across a sequence of alerts.

        Evaluates pairwise relationships:
        1. Same flow: Multiple engine evaluations of the identical network flow.
        2. Shared source IP: Identical originating host.
        3. Shared destination IP: Identical targeted host.
        4. Shared network entity: Identical destination host and service port.
        5. Shared destination port: Targeted service port.
        6. Repeated attack category: Recurrent threat family.
        7. Detection-engine agreement: Cross-engine consensus on threat presence.
        8. Ingests pre-computed temporal relationships.

        Args:
            alerts: Sequence of normalized Alert objects.
            temporal_relationships: Optional sequence of pre-computed TemporalRelationship objects.

        Returns:
            Populated RelationshipGraph instance.
        """
        graph = cls()
        if not alerts:
            return graph

        for a in alerts:
            graph.add_alert(a)

        n = len(alerts)

        # 1. Pairwise Non-Temporal Relationship Evaluation
        for i in range(n):
            alert_a = alerts[i]

            for j in range(i + 1, n):
                alert_b = alerts[j]

                # (a) Same Flow Relationship
                if alert_a.flow_id and alert_b.flow_id and alert_a.flow_id == alert_b.flow_id:
                    graph.add_relationship(
                        AlertRelationship(
                            relationship_id=f"rel_flow_{uuid.uuid4().hex[:12]}",
                            source_alert_id=alert_a.alert_id,
                            target_alert_id=alert_b.alert_id,
                            relationship_type=RelationshipType.SAME_FLOW.value,
                            weight=2.0,
                            grouping_eligible=True,
                            evidence={
                                "flow_id": alert_a.flow_id,
                                "models": [alert_a.model_name, alert_b.model_name],
                                "engines": [alert_a.detection_engine, alert_b.detection_engine],
                            },
                            description=f"Both alerts evaluate the identical network flow '{alert_a.flow_id}'.",
                        )
                    )

                    # Check detection engine agreement on same flow
                    if alert_a.detection_engine != alert_b.detection_engine:
                        both_threat = (
                            alert_a.threat_status in (ThreatStatus.ATTACK.value, ThreatStatus.ANOMALY.value)
                            and alert_b.threat_status in (ThreatStatus.ATTACK.value, ThreatStatus.ANOMALY.value)
                        )
                        if both_threat:
                            graph.add_relationship(
                                AlertRelationship(
                                    relationship_id=f"rel_agr_{uuid.uuid4().hex[:12]}",
                                    source_alert_id=alert_a.alert_id,
                                    target_alert_id=alert_b.alert_id,
                                    relationship_type=RelationshipType.DETECTION_ENGINE_AGREEMENT.value,
                                    weight=1.5,
                                    grouping_eligible=True,
                                    evidence={
                                        "flow_id": alert_a.flow_id,
                                        "engine_a": alert_a.detection_engine,
                                        "engine_b": alert_b.detection_engine,
                                        "threat_status_a": alert_a.threat_status,
                                        "threat_status_b": alert_b.threat_status,
                                    },
                                    description=(
                                        f"Detection engines '{alert_a.detection_engine}' and '{alert_b.detection_engine}' "
                                        "both identified threat activity on this flow."
                                    ),
                                )
                            )

                # (b) Shared Source IP Relationship
                # STRICT INTEGRITY: Missing source_ip (None) never matches None!
                if alert_a.source_ip and alert_b.source_ip and alert_a.source_ip == alert_b.source_ip:
                    graph.add_relationship(
                        AlertRelationship(
                            relationship_id=f"rel_src_{uuid.uuid4().hex[:12]}",
                            source_alert_id=alert_a.alert_id,
                            target_alert_id=alert_b.alert_id,
                            relationship_type=RelationshipType.SHARED_SOURCE_IP.value,
                            weight=1.2,
                            grouping_eligible=True,
                            evidence={"source_ip": alert_a.source_ip},
                            description=f"Both alerts originate from shared source host {alert_a.source_ip}.",
                        )
                    )

                # (c) Shared Destination IP / Shared Network Entity
                # STRICT INTEGRITY: Missing destination_ip (None) never matches None!
                if alert_a.destination_ip and alert_b.destination_ip and alert_a.destination_ip == alert_b.destination_ip:
                    has_shared_port = (
                        alert_a.destination_port is not None
                        and alert_b.destination_port is not None
                        and alert_a.destination_port == alert_b.destination_port
                    )

                    if has_shared_port:
                        graph.add_relationship(
                            AlertRelationship(
                                relationship_id=f"rel_ent_{uuid.uuid4().hex[:12]}",
                                source_alert_id=alert_a.alert_id,
                                target_alert_id=alert_b.alert_id,
                                relationship_type=RelationshipType.SHARED_NETWORK_ENTITY.value,
                                weight=1.8,
                                grouping_eligible=True,
                                evidence={
                                    "destination_ip": alert_a.destination_ip,
                                    "destination_port": alert_a.destination_port,
                                },
                                description=(
                                    f"Both alerts target the exact same service endpoint "
                                    f"{alert_a.destination_ip}:{alert_a.destination_port}."
                                ),
                            )
                        )
                    else:
                        graph.add_relationship(
                            AlertRelationship(
                                relationship_id=f"rel_dst_{uuid.uuid4().hex[:12]}",
                                source_alert_id=alert_a.alert_id,
                                target_alert_id=alert_b.alert_id,
                                relationship_type=RelationshipType.SHARED_DESTINATION_IP.value,
                                weight=1.0,
                                grouping_eligible=True,
                                evidence={"destination_ip": alert_a.destination_ip},
                                description=f"Both alerts target shared destination host {alert_a.destination_ip}.",
                            )
                        )

                # (d) Shared Destination Port (when destination IPs differ or are missing)
                # If destination IPs differ, port alone is informational and NOT grouping eligible.
                if (
                    alert_a.destination_port is not None
                    and alert_b.destination_port is not None
                    and alert_a.destination_port == alert_b.destination_port
                ):
                    ips_match = (
                        alert_a.destination_ip and alert_b.destination_ip
                        and alert_a.destination_ip == alert_b.destination_ip
                    )
                    if not ips_match:
                        graph.add_relationship(
                            AlertRelationship(
                                relationship_id=f"rel_port_{uuid.uuid4().hex[:12]}",
                                source_alert_id=alert_a.alert_id,
                                target_alert_id=alert_b.alert_id,
                                relationship_type=RelationshipType.SHARED_DESTINATION_PORT.value,
                                weight=0.4,
                                grouping_eligible=False,  # Port alone across distinct/missing IPs does NOT group
                                evidence={"destination_port": alert_a.destination_port},
                                description=(
                                    f"Both alerts involve destination port {alert_a.destination_port} "
                                    "on distinct or unverified target hosts."
                                ),
                            )
                        )

                # (e) Repeated Threat Category
                cat_a = alert_a.attack_category
                cat_b = alert_b.attack_category
                if cat_a and cat_b and cat_a != "BENIGN" and cat_a == cat_b:
                    # Category alone is grouping eligible only if there is also an entity anchor
                    has_anchor = bool(
                        (alert_a.source_ip and alert_a.source_ip == alert_b.source_ip)
                        or (alert_a.destination_ip and alert_a.destination_ip == alert_b.destination_ip)
                        or (alert_a.flow_id and alert_a.flow_id == alert_b.flow_id)
                    )
                    graph.add_relationship(
                        AlertRelationship(
                            relationship_id=f"rel_cat_{uuid.uuid4().hex[:12]}",
                            source_alert_id=alert_a.alert_id,
                            target_alert_id=alert_b.alert_id,
                            relationship_type=RelationshipType.REPEATED_CATEGORY.value,
                            weight=0.8,
                            grouping_eligible=has_anchor,
                            evidence={
                                "attack_category": cat_a,
                                "has_endpoint_anchor": has_anchor,
                            },
                            description=f"Both alerts report identical threat category '{cat_a}'.",
                        )
                    )

        # 2. Ingest Temporal Relationships (if provided)
        if temporal_relationships:
            for trel in temporal_relationships:
                rname = trel.rule_name
                ev = trel.evidence

                if rname == "sequential_ordering":
                    rtype = RelationshipType.TEMPORAL_SEQUENCE.value
                    # Sequential ordering with shared target is strongly grouping eligible
                    has_endpoint_anchor = bool(
                        ev.get("shared_target_detected")
                        or ev.get("shared_entities", {}).get("destination_ip")
                        or ev.get("shared_entities", {}).get("source_ip")
                    )
                    grouping = has_endpoint_anchor
                    weight = 1.6
                    desc = (
                        f"Sequential candidate attack progression ('{ev.get('earlier_category')}' -> "
                        f"'{ev.get('later_category')}' within {ev.get('time_delta_seconds')}s)."
                    )
                elif rname in ("repeated_alert", "repeated_category"):
                    rtype = RelationshipType.TEMPORAL_RECURRENCE.value
                    has_endpoint_anchor = bool(
                        ev.get("shared_entities", {}).get("source_ip")
                        or ev.get("shared_entities", {}).get("destination_ip")
                    )
                    grouping = has_endpoint_anchor
                    weight = 1.0
                    desc = f"Temporal recurrence within observation window ({ev.get('time_delta_seconds')}s)."
                else:  # temporal_proximity
                    rtype = RelationshipType.TEMPORAL_PROXIMITY.value
                    # STRICT INTEGRITY: Circumstantial temporal proximity alone is NEVER grouping eligible.
                    # Only group if there is a concrete shared IP anchor (source or destination).
                    has_endpoint_anchor = bool(
                        ev.get("shared_entities", {}).get("source_ip")
                        or ev.get("shared_entities", {}).get("destination_ip")
                    )
                    grouping = has_endpoint_anchor
                    weight = 0.5
                    desc = (
                        f"Circumstantial temporal co-occurrence within {ev.get('time_delta_seconds')}s "
                        f"({ev.get('proximity_classification', 'proximity')})."
                    )

                graph.add_relationship(
                    AlertRelationship(
                        relationship_id=f"rel_temp_{trel.relationship_id}",
                        source_alert_id=trel.source_alert_id,
                        target_alert_id=trel.target_alert_id,
                        relationship_type=rtype,
                        weight=weight,
                        grouping_eligible=grouping,
                        evidence=ev,
                        description=desc,
                    )
                )

        return graph
