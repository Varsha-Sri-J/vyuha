"""Lightweight, non-blocking Zeek connection log collector.

Safely tails, parses, and deduplicates JSON and TSV formatted Zeek conn.log files,
maintaining an in-memory ring buffer of live network observations.
"""

import os
import json
import logging
import threading
from collections import deque, Counter
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set

from src.monitoring.validator import LiveFeatureValidator

logger = logging.getLogger("vyuha.monitoring.collector")


class ZeekLogCollector:
    """Tails Zeek conn.log incrementally, parses records, and computes live statistics."""

    def __init__(self, log_path: Optional[str] = None, max_records: int = 1000):
        self.log_path = log_path
        self.max_records = max_records

        # Synchronization lock
        self._lock = threading.Lock()

        # Incremental file tracking
        self._file_offset: int = 0
        self._last_inode: Optional[int] = None
        self._seen_uids: Set[str] = set()
        self._seen_uids_queue: deque = deque(maxlen=10000)

        # In-memory connection ring buffer
        self._connections: deque = deque(maxlen=max_records)

        # Telemetry aggregates
        self._total_connections: int = 0
        self._protocol_counts: Counter = Counter()
        self._port_counts: Counter = Counter()
        self._status_counts: Counter = Counter()
        self._latest_timestamp: Optional[float] = None
        self._malformed_records_count: int = 0

    def reset(self) -> None:
        """Clears in-memory buffers and resets file read offset."""
        with self._lock:
            self._file_offset = 0
            self._last_inode = None
            self._seen_uids.clear()
            self._seen_uids_queue.clear()
            self._connections.clear()
            self._total_connections = 0
            self._protocol_counts.clear()
            self._port_counts.clear()
            self._status_counts.clear()
            self._latest_timestamp = None
            self._malformed_records_count = 0

    def parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parses a single log line (JSON or TSV) into a normalized connection record.

        Returns None if the line is empty, comment, or malformed.
        """
        line = line.strip()
        if not line or line.startswith("#"):
            return None

        # 1. Attempt JSON parsing (Zeek LogAscii::use_json=T)
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                return self._normalize_json_record(data)
            except Exception as exc:
                self._malformed_records_count += 1
                logger.debug("Failed to parse JSON Zeek line: %s (%s)", line[:100], exc)
                return None

        # 2. Attempt TSV parsing (standard Zeek ASCII logs)
        parts = line.split("\t")
        if len(parts) >= 10:
            try:
                return self._normalize_tsv_record(parts)
            except Exception as exc:
                self._malformed_records_count += 1
                logger.debug("Failed to parse TSV Zeek line: %s (%s)", line[:100], exc)
                return None

        self._malformed_records_count += 1
        return None

    def _normalize_json_record(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extracts standard connection fields from a raw Zeek JSON object."""
        uid = str(raw.get("uid") or "")
        ts_raw = raw.get("ts")
        try:
            ts = float(ts_raw) if ts_raw is not None else datetime.now(timezone.utc).timestamp()
        except (ValueError, TypeError):
            ts = datetime.now(timezone.utc).timestamp()

        # IP addresses and ports
        src_ip = str(raw.get("id.orig_h") or raw.get("src_ip") or "0.0.0.0")
        dst_ip = str(raw.get("id.resp_h") or raw.get("dst_ip") or "0.0.0.0")

        def _to_int(val: Any) -> Optional[int]:
            if val is None or val == "-":
                return None
            try:
                return int(val)
            except (ValueError, TypeError):
                return None

        def _to_float(val: Any) -> Optional[float]:
            if val is None or val == "-":
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        src_port = _to_int(raw.get("id.orig_p") or raw.get("src_port"))
        dst_port = _to_int(raw.get("id.resp_p") or raw.get("dst_port"))
        proto = str(raw.get("proto") or "tcp").lower()
        service = raw.get("service")
        if service == "-":
            service = None

        duration = _to_float(raw.get("duration"))
        orig_bytes = _to_int(raw.get("orig_bytes")) or 0
        resp_bytes = _to_int(raw.get("resp_bytes")) or 0
        orig_pkts = _to_int(raw.get("orig_pkts")) or 0
        resp_pkts = _to_int(raw.get("resp_pkts")) or 0
        conn_state = str(raw.get("conn_state") or raw.get("state") or "SF")
        history = raw.get("history")
        if history == "-":
            history = None

        # Determine heuristic status and check ML eligibility
        detection_status, severity, description = LiveFeatureValidator.classify_connection(
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            proto=proto,
            conn_state=conn_state,
            duration=duration,
            orig_bytes=orig_bytes,
            resp_bytes=resp_bytes,
            orig_pkts=orig_pkts,
            resp_pkts=resp_pkts,
            service=service,
        )

        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        time_str = dt.strftime("%H:%M:%S")

        return {
            "uid": uid or f"flow_{int(ts * 1000)}_{src_port}_{dst_port}",
            "timestamp": time_str,
            "iso_time": dt.isoformat(),
            "ts": ts,
            "src_ip": src_ip,
            "src_port": src_port or 0,
            "dst_ip": dst_ip,
            "dst_port": dst_port or 0,
            "proto": proto.upper(),
            "service": service,
            "duration": round(duration or 0.0, 4),
            "orig_bytes": orig_bytes,
            "resp_bytes": resp_bytes,
            "total_bytes": orig_bytes + resp_bytes,
            "orig_pkts": orig_pkts,
            "resp_pkts": resp_pkts,
            "conn_state": conn_state,
            "history": history,
            "detection_status": detection_status,
            "detection_severity": severity,
            "detection_reason": description,
            "ml_evaluated": False,
            "ml_status": "INSUFFICIENT_FEATURES",
            "missing_features_count": 63,
            "explanation": (
                "Raw Zeek connection record. Contains 13 transport metadata fields; "
                "lacks the 63 packet-level dispersion and inter-arrival time statistics "
                "required by the trained ML models. Preserved as raw live telemetry."
            ),
        }

    def _normalize_tsv_record(self, parts: List[str]) -> Optional[Dict[str, Any]]:
        """Maps standard Zeek TSV fields to a normalized dictionary."""
        # Standard default Zeek conn.log column ordering:
        # ts uid id.orig_h id.orig_p id.resp_h id.resp_p proto service duration orig_bytes resp_bytes conn_state ...
        raw_dict = {
            "ts": parts[0] if len(parts) > 0 else None,
            "uid": parts[1] if len(parts) > 1 else "",
            "id.orig_h": parts[2] if len(parts) > 2 else "0.0.0.0",
            "id.orig_p": parts[3] if len(parts) > 3 else "0",
            "id.resp_h": parts[4] if len(parts) > 4 else "0.0.0.0",
            "id.resp_p": parts[5] if len(parts) > 5 else "0",
            "proto": parts[6] if len(parts) > 6 else "tcp",
            "service": parts[7] if len(parts) > 7 else "-",
            "duration": parts[8] if len(parts) > 8 else "0",
            "orig_bytes": parts[9] if len(parts) > 9 else "0",
            "resp_bytes": parts[10] if len(parts) > 10 else "0",
            "conn_state": parts[11] if len(parts) > 11 else "SF",
        }
        return self._normalize_json_record(raw_dict)

    def ingest_records(self, records: List[Dict[str, Any]]) -> int:
        """Ingests a list of pre-parsed records with deduplication."""
        added = 0
        with self._lock:
            for rec in records:
                if not rec:
                    continue
                uid = rec.get("uid")
                if uid and uid in self._seen_uids:
                    continue

                if uid:
                    self._seen_uids.add(uid)
                    self._seen_uids_queue.append(uid)
                    if len(self._seen_uids) > 12000:
                        # Prune seen_uids to match queue
                        self._seen_uids = set(self._seen_uids_queue)

                self._connections.append(rec)
                self._total_connections += 1
                proto = rec.get("proto", "TCP").upper()
                self._protocol_counts[proto] += 1
                dst_port = rec.get("dst_port", 0)
                if dst_port:
                    self._port_counts[dst_port] += 1
                status = rec.get("detection_status", "NORMAL")
                self._status_counts[status] += 1
                ts = rec.get("ts", 0.0)
                if self._latest_timestamp is None or ts > self._latest_timestamp:
                    self._latest_timestamp = ts
                added += 1
        return added

    def poll_file(self) -> int:
        """Reads newly appended lines from log_path incrementally without blocking."""
        if not self.log_path or not os.path.exists(self.log_path):
            return 0

        new_records = []
        try:
            stat = os.stat(self.log_path)
            # Detect log file truncation / rotation
            if self._last_inode is not None and (stat.st_ino != self._last_inode or stat.st_size < self._file_offset):
                self._file_offset = 0

            self._last_inode = stat.st_ino

            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._file_offset)
                lines = f.readlines()
                self._file_offset = f.tell()

            for line in lines:
                rec = self.parse_log_line(line)
                if rec:
                    new_records.append(rec)

        except Exception as exc:
            logger.error("Error reading Zeek log file %s: %s", self.log_path, exc)
            return 0

        if new_records:
            return self.ingest_records(new_records)
        return 0

    def get_connections(self, limit: int = 50, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns the most recent connections up to limit, newest first."""
        with self._lock:
            items = list(self._connections)

        if status_filter and status_filter.upper() != "ALL":
            items = [c for c in items if c.get("detection_status", "").upper() == status_filter.upper()]

        # Return newest first
        items.reverse()
        return items[:limit]

    def get_summary(self) -> Dict[str, Any]:
        """Returns high-level connection aggregates and telemetry metrics."""
        with self._lock:
            top_ports = [{"port": p, "count": c} for p, c in self._port_counts.most_common(5)]
            proto_dist = dict(self._protocol_counts)
            status_dist = dict(self._status_counts)
            latest_ts = self._latest_timestamp
            total = self._total_connections
            malformed = self._malformed_records_count

        return {
            "total_connections": total,
            "protocol_distribution": proto_dist,
            "status_distribution": status_dist,
            "top_destination_ports": top_ports,
            "latest_timestamp": latest_ts,
            "malformed_records_count": malformed,
            "buffered_records_count": len(self._connections),
        }
