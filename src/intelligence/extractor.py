"""VYUHA Entity Extraction Module (Phase 4.1).

Extracts and strictly validates network entities from raw or processed flow records:
- Source and Destination IP addresses (IPv4 / IPv6 validation)
- Source and Destination Ports (0-65535 range validation)
- Timestamps (ISO-8601 standardization)
- Transport Protocol (strictly extracted only when explicitly available)
- Deterministic Flow Identifiers
- Flow Duration

Enforces strict separation between missing data and observed values, avoiding
unsupported heuristics or fabricated attributes.
"""

from dataclasses import dataclass, field
import datetime
import hashlib
import ipaddress
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical candidate column names across various flow meter exports
SRC_IP_CANDIDATES = ["Source IP", "source_ip", "src_ip", "Src IP", "src_addr", "ip_src"]
DST_IP_CANDIDATES = ["Destination IP", "destination_ip", "dst_ip", "Dst IP", "dst_addr", "ip_dst"]
SRC_PORT_CANDIDATES = ["Source Port", "source_port", "src_port", "Src Port", "sport"]
DST_PORT_CANDIDATES = ["Destination Port", "destination_port", "dst_port", "Dst Port", "dport"]
TIMESTAMP_CANDIDATES = ["Timestamp", "timestamp", "Flow Timestamp", "time", "date_time"]
PROTOCOL_CANDIDATES = ["Protocol", "protocol", "proto", "ip_proto"]
FLOW_ID_CANDIDATES = ["Flow ID", "flow_id", "FlowId", "id"]
DURATION_CANDIDATES = ["Flow Duration", "flow_duration", "duration"]

# Standard IANA numeric protocol mappings
NUMERIC_PROTOCOL_MAP = {
    1: "ICMP",
    6: "TCP",
    17: "UDP",
    47: "GRE",
    50: "ESP",
    51: "AH",
    58: "ICMPv6",
    89: "OSPF",
}


@dataclass
class ExtractedEntities:
    """Encapsulates strictly validated entities extracted from a network flow record."""

    flow_id: str
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    source_port: Optional[int] = None
    destination_port: Optional[int] = None
    protocol: Optional[str] = None
    timestamp: Optional[str] = None
    flow_duration: Optional[float] = None
    validation_errors: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts extracted entities to dictionary."""
        return {
            "flow_id": self.flow_id,
            "source_ip": self.source_ip,
            "destination_ip": self.destination_ip,
            "source_port": self.source_port,
            "destination_port": self.destination_port,
            "protocol": self.protocol,
            "timestamp": self.timestamp,
            "flow_duration": self.flow_duration,
            "validation_errors": self.validation_errors,
        }


class EntityExtractor:
    """Robust extractor and validator for network flow entities."""

    @staticmethod
    def _find_field_value(record: Union[Dict[str, Any], pd.Series], candidates: List[str]) -> Any:
        """Finds the first matching candidate key present in the record."""
        if isinstance(record, pd.Series):
            for cand in candidates:
                if cand in record.index:
                    val = record[cand]
                    if pd.notna(val):
                        return val
        elif isinstance(record, dict):
            for cand in candidates:
                if cand in record and record[cand] is not None:
                    val = record[cand]
                    if not (isinstance(val, float) and np.isnan(val)):
                        return val
        return None

    @staticmethod
    def validate_ip(ip_val: Any, strict: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """Validates an IPv4 or IPv6 address string.

        Args:
            ip_val: Raw IP value (string, or None).
            strict: If True, raises ValueError on invalid syntax.

        Returns:
            Tuple of (normalized_ip_string, error_message).
        """
        if ip_val is None or (isinstance(ip_val, float) and np.isnan(ip_val)):
            return None, None

        ip_str = str(ip_val).strip()
        if not ip_str or ip_str.lower() in ("nan", "none", "null"):
            return None, None

        try:
            ip_obj = ipaddress.ip_address(ip_str)
            return str(ip_obj), None
        except ValueError as err:
            err_msg = f"Invalid IP syntax: '{ip_str}'"
            if strict:
                raise ValueError(err_msg) from err
            return None, err_msg

    @staticmethod
    def validate_port(port_val: Any, field_name: str = "port", strict: bool = False) -> Tuple[Optional[int], Optional[str]]:
        """Validates a network port number within the range [0, 65535].

        Args:
            port_val: Raw port value (int, float, string, or None).
            field_name: Name of field for error reporting.
            strict: If True, raises ValueError on invalid port numbers.

        Returns:
            Tuple of (port_int, error_message).
        """
        if port_val is None or (isinstance(port_val, float) and np.isnan(port_val)):
            return None, None

        # Attempt numeric conversion first
        try:
            port_num = float(port_val)
        except (ValueError, TypeError) as err:
            err_msg = f"{field_name} cannot be converted to integer: {port_val}"
            if strict:
                raise ValueError(err_msg) from err
            return None, err_msg

        # Check for fractional numbers (e.g. 80.5)
        if not port_num.is_integer() and not isinstance(port_val, int):
            if abs(port_num - round(port_num)) > 1e-6:
                err_msg = f"{field_name} must be an integer, got float {port_val}"
                if strict:
                    raise ValueError(err_msg)
                return None, err_msg

        port_int = int(round(port_num))
        if not (0 <= port_int <= 65535):
            err_msg = f"{field_name} out of bounds [0, 65535]: {port_int}"
            if strict:
                raise ValueError(err_msg)
            return None, err_msg

        return port_int, None

    @staticmethod
    def validate_timestamp(ts_val: Any, strict: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """Standardizes a timestamp into an ISO-8601 string representation.

        Args:
            ts_val: Timestamp value (datetime, pd.Timestamp, string, or None).
            strict: If True, raises ValueError on parsing failures.

        Returns:
            Tuple of (iso_timestamp_string, error_message).
        """
        if ts_val is None or (isinstance(ts_val, float) and np.isnan(ts_val)):
            return None, None

        if isinstance(ts_val, (datetime.datetime, pd.Timestamp)):
            return ts_val.isoformat(), None

        ts_str = str(ts_val).strip()
        if not ts_str or ts_str.lower() in ("nan", "none", "null"):
            return None, None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                dt = pd.to_datetime(ts_str)
            return dt.isoformat(), None
        except Exception as err:
            err_msg = f"Failed to parse timestamp '{ts_str}'"
            if strict:
                raise ValueError(err_msg) from err
            return None, err_msg

    @staticmethod
    def validate_protocol(proto_val: Any, strict: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """Validates and standardizes a transport protocol name from explicit numeric or string fields.

        Important: In datasets like CICIDS2017 MachineLearningCVE, no explicit protocol column
        exists. This function returns None when missing, rather than inferring from heuristics.

        Args:
            proto_val: Protocol value (int, string, or None).
            strict: If True, raises ValueError on unrecognized protocol types.

        Returns:
            Tuple of (standardized_protocol_string, error_message).
        """
        if proto_val is None or (isinstance(proto_val, float) and np.isnan(proto_val)):
            return None, None

        # Check numeric protocol
        try:
            proto_num = int(round(float(proto_val)))
            if proto_num in NUMERIC_PROTOCOL_MAP:
                return NUMERIC_PROTOCOL_MAP[proto_num], None
        except (ValueError, TypeError):
            pass

        # Check string protocol
        proto_str = str(proto_val).strip().upper()
        if proto_str in ("TCP", "UDP", "ICMP", "GRE", "ESP", "AH", "OSPF"):
            return proto_str, None

        err_msg = f"Unknown or unparseable protocol: '{proto_val}'"
        if strict:
            raise ValueError(err_msg)
        return None, err_msg

    @classmethod
    def generate_flow_id(
        cls,
        record: Union[Dict[str, Any], pd.Series],
        index: Optional[Union[int, str]] = None,
    ) -> str:
        """Generates a reliable, deterministic flow identifier.

        Prioritizes existing flow identifier fields; falls back to index or hash.

        Args:
            record: Flow record dictionary or pandas Series.
            index: Row index or sequence number if available.

        Returns:
            Flow identifier string.
        """
        # 1. Native flow ID takes absolute precedence
        existing_id = cls._find_field_value(record, FLOW_ID_CANDIDATES)
        if existing_id is not None:
            clean_id = str(existing_id).strip()
            if clean_id and clean_id.lower() not in ("nan", "none", "null"):
                return clean_id

        # 2. Prefer explicit positional index
        if index is not None:
            return f"flow_{index}"

        # 3. Detect pandas Series row index (record.name)
        if isinstance(record, pd.Series) and record.name is not None:
            return f"flow_{record.name}"

        # 4. Fallback: Hash non-target features (Limitation: Collides on identical flow records)
        # Callers must provide an index or native flow ID for guaranteed distinctness.
        if isinstance(record, pd.Series):
            items = record.to_dict()
        else:
            items = dict(record)

        # Exclude ground-truth targets from hash to prevent leakage
        filtered_items = {
            str(k): str(v)
            for k, v in sorted(items.items())
            if not str(k).startswith("label_") and k not in ("label_raw", "label_binary", "label_category")
        }
        digest = hashlib.sha256(str(filtered_items).encode("utf-8")).hexdigest()[:12]
        return f"flow_hash_{digest}"

    @classmethod
    def extract(
        cls,
        record: Union[Dict[str, Any], pd.Series],
        index: Optional[Union[int, str]] = None,
        strict: bool = False,
    ) -> ExtractedEntities:
        """Extracts and validates all available entities from a flow record.

        Args:
            record: Flow record dictionary or Series.
            index: Optional sequence or row index for flow identification.
            strict: If True, raises exceptions on validation failures.

        Returns:
            ExtractedEntities dataclass with validated fields and any logged errors.
        """
        validation_errors: Dict[str, str] = {}

        # 1. Flow ID
        flow_id = cls.generate_flow_id(record, index=index)

        # 2. Source & Destination IP
        raw_src_ip = cls._find_field_value(record, SRC_IP_CANDIDATES)
        src_ip, src_ip_err = cls.validate_ip(raw_src_ip, strict=strict)
        if src_ip_err:
            validation_errors["source_ip"] = src_ip_err

        raw_dst_ip = cls._find_field_value(record, DST_IP_CANDIDATES)
        dst_ip, dst_ip_err = cls.validate_ip(raw_dst_ip, strict=strict)
        if dst_ip_err:
            validation_errors["destination_ip"] = dst_ip_err

        # 3. Source & Destination Port
        raw_src_port = cls._find_field_value(record, SRC_PORT_CANDIDATES)
        src_port, src_port_err = cls.validate_port(raw_src_port, field_name="source_port", strict=strict)
        if src_port_err:
            validation_errors["source_port"] = src_port_err

        raw_dst_port = cls._find_field_value(record, DST_PORT_CANDIDATES)
        dst_port, dst_port_err = cls.validate_port(raw_dst_port, field_name="destination_port", strict=strict)
        if dst_port_err:
            validation_errors["destination_port"] = dst_port_err

        # 4. Protocol (Strictly explicit only)
        raw_proto = cls._find_field_value(record, PROTOCOL_CANDIDATES)
        protocol, proto_err = cls.validate_protocol(raw_proto, strict=strict)
        if proto_err:
            validation_errors["protocol"] = proto_err

        # 5. Timestamp
        raw_ts = cls._find_field_value(record, TIMESTAMP_CANDIDATES)
        timestamp, ts_err = cls.validate_timestamp(raw_ts, strict=strict)
        if ts_err:
            validation_errors["timestamp"] = ts_err

        # 6. Flow Duration
        raw_dur = cls._find_field_value(record, DURATION_CANDIDATES)
        flow_duration = None
        if raw_dur is not None:
            try:
                flow_duration = float(raw_dur)
            except (ValueError, TypeError):
                validation_errors["flow_duration"] = f"Invalid duration value: {raw_dur}"

        return ExtractedEntities(
            flow_id=flow_id,
            source_ip=src_ip,
            destination_ip=dst_ip,
            source_port=src_port,
            destination_port=dst_port,
            protocol=protocol,
            timestamp=timestamp,
            flow_duration=flow_duration,
            validation_errors=validation_errors,
        )

    @classmethod
    def extract_batch(
        cls,
        df: pd.DataFrame,
        strict: bool = False,
    ) -> List[ExtractedEntities]:
        """Extracts entities for a batch of flow records in a DataFrame.

        Args:
            df: Input DataFrame.
            strict: If True, raises exceptions on validation failures.

        Returns:
            List of ExtractedEntities objects aligned with rows in df.
        """
        results: List[ExtractedEntities] = []
        for idx, row in df.iterrows():
            entities = cls.extract(row, index=idx, strict=strict)
            results.append(entities)
        return results
