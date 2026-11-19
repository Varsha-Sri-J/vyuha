"""Feature completeness and security observation validator for live Zeek telemetry.

Implements strict validation ensuring that raw Zeek connection logs are NOT
passed to the 63-feature ML models with synthetic zeros, while providing
transparent heuristic security observation for real-time traffic visibility.
"""

from typing import Dict, Any, Tuple, Optional
import datetime

from src.models.preprocessor import CANONICAL_63_FEATURES as REQUIRED_FEATURES

# Sensitive or administrative destination ports for heuristic visibility
SENSITIVE_PORTS = {
    21: "FTP",
    22: "SSH",
    23: "Telnet",
    25: "SMTP",
    135: "RPC",
    139: "NetBIOS",
    445: "SMB",
    1433: "MSSQL",
    1521: "Oracle",
    3306: "MySQL",
    3389: "RDP",
    5432: "PostgreSQL",
    5900: "VNC",
}

# Standard benign service ports
BENIGN_STANDARD_PORTS = {
    53: "DNS",
    67: "DHCP",
    68: "DHCP",
    80: "HTTP",
    123: "NTP",
    443: "HTTPS",
    5353: "mDNS",
    3000: "Vite Dev",
    8000: "FastAPI",
}


class LiveFeatureValidator:
    """Validates live connection records and determines ML applicability."""

    @staticmethod
    def validate_ml_eligibility(flow_dict: Dict[str, Any]) -> Tuple[bool, list, str]:
        """Checks whether all 63 required CICIDS2017 features are legitimately present.

        Returns:
            Tuple of (is_eligible: bool, missing_features: list, explanation: str)
        """
        missing = [f for f in REQUIRED_FEATURES if f not in flow_dict and f.lower() not in flow_dict]
        if missing:
            return (
                False,
                missing,
                f"Connection record contains {len(flow_dict)} attributes but lacks {len(missing)} "
                "required statistical dispersion features (e.g. IAT, packet lengths, window flags) "
                "required by the trained scikit-learn models. Flow is preserved as live telemetry "
                "without synthetic padding.",
            )
        return True, [], "All 63 statistical flow features are present for ML evaluation."

    @staticmethod
    def classify_connection(
        src_ip: str,
        dst_ip: str,
        src_port: Optional[int],
        dst_port: Optional[int],
        proto: str,
        conn_state: str,
        duration: Optional[float] = None,
        orig_bytes: Optional[int] = None,
        resp_bytes: Optional[int] = None,
        orig_pkts: Optional[int] = None,
        resp_pkts: Optional[int] = None,
        service: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """Performs rule-based heuristic observation on raw Zeek connection telemetry.

        Labels:
            - NORMAL: Standard established connections or legitimate local services.
            - SUSPICIOUS: Rejected/unanswered attempts on sensitive ports or unusual flags.
            - POTENTIAL THREAT: Volumetric flooding or repeated scanning patterns.

        Returns:
            Tuple of (detection_status, severity, description)
        """
        port = dst_port or 0
        state = (conn_state or "").upper()
        p = (proto or "").lower()

        # 1. Localhost / Loopback benign traffic
        if src_ip in ("127.0.0.1", "::1") and dst_ip in ("127.0.0.1", "::1"):
            return (
                "NORMAL",
                "INFO",
                f"Local loopback communication on port {port} ({service or 'internal'}).",
            )

        # 2. Rejected or unanswered connection attempts on sensitive ports
        if port in SENSITIVE_PORTS and state in ("REJ", "S0", "RSTOS0", "RSTO"):
            srv_name = SENSITIVE_PORTS[port]
            return (
                "SUSPICIOUS",
                "MEDIUM",
                f"Unsuccessful connection attempt ({state}) targeting sensitive service {srv_name} (port {port}).",
            )

        # 3. Telnet or unencrypted legacy administrative access
        if port == 23:
            return (
                "SUSPICIOUS",
                "MEDIUM",
                "Observed unencrypted Telnet traffic (port 23). Modern hosts should disable Telnet.",
            )

        # 4. SMB / RPC probing across networks
        if port in (135, 139, 445) and not (src_ip.startswith("192.168.") or src_ip.startswith("10.")):
            return (
                "POTENTIAL THREAT",
                "HIGH",
                f"External SMB/RPC probe detected on port {port} ({SENSITIVE_PORTS.get(port, 'Windows Service')}).",
            )

        # 5. Volumetric flood or asymmetric traffic surge
        if orig_pkts and orig_pkts > 5000 and (duration or 0) < 5.0:
            return (
                "POTENTIAL THREAT",
                "HIGH",
                f"High-frequency packet surge: {orig_pkts} packets sent within {duration or 0:.2f}s.",
            )

        # 6. Standard established connections (SF: Normal SYN/FIN completion)
        if state == "SF":
            if port in BENIGN_STANDARD_PORTS:
                return (
                    "NORMAL",
                    "INFO",
                    f"Standard {BENIGN_STANDARD_PORTS[port]} connection completed normally ({state}).",
                )
            return (
                "NORMAL",
                "INFO",
                f"Established {p.upper()} connection to port {port} completed normally.",
            )

        # 7. Unanswered SYN probes (S0) on non-standard high ports
        if state == "S0" and port > 1024:
            return (
                "SUSPICIOUS",
                "LOW",
                f"Outbound connection attempt to port {port} received no response (SYN sent, no SYN-ACK).",
            )

        # Default fallback
        return (
            "NORMAL",
            "INFO",
            f"Observed {p.upper()} traffic with state {state or 'UNKNOWN'}.",
        )
