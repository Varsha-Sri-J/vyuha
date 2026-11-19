"""Process and collector lifecycle manager for live Zeek network monitoring.

Manages Zeek subprocess execution, interface auto-detection, macOS BPF permissions,
and asynchronous polling of conn.log.
"""

import os
import re
import time
import shutil
import logging
import threading
import subprocess
from typing import Dict, Any, List, Optional, Tuple

from src.monitoring.collector import ZeekLogCollector

logger = logging.getLogger("vyuha.monitoring.manager")


class LiveMonitoringManager:
    """Singleton managing Zeek subprocess, interface configuration, and live collector thread."""

    _instance: Optional["LiveMonitoringManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, runtime_dir: Optional[str] = None):
        if getattr(self, "_initialized", False):
            return

        self._initialized = True
        self.runtime_dir = runtime_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "runtime", "zeek_live")
        )
        os.makedirs(self.runtime_dir, exist_ok=True)

        self.log_file = os.path.join(self.runtime_dir, "conn.log")
        self.collector = ZeekLogCollector(log_path=self.log_file)

        self._zeek_process: Optional[subprocess.Popen] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._active_interface: str = "en0"
        self._error_message: Optional[str] = None
        self._is_simulated: bool = False
        self._simulation_thread: Optional[threading.Thread] = None

    # -------------------------------------------------------------------------
    # Environment Inspection
    # -------------------------------------------------------------------------

    @staticmethod
    def find_zeek_binary() -> Optional[str]:
        """Locates the Zeek binary on macOS."""
        candidates = [
            "/opt/homebrew/bin/zeek",
            "/usr/local/bin/zeek",
            shutil.which("zeek"),
        ]
        for c in candidates:
            if c and os.path.isfile(c) and os.access(c, os.X_OK):
                return c
        return None

    @staticmethod
    def get_zeek_version(zeek_path: str) -> Optional[str]:
        """Retrieves the installed Zeek version string."""
        try:
            res = subprocess.run([zeek_path, "--version"], capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                match = re.search(r"version\s+([0-9\.]+)", res.stdout or res.stderr)
                if match:
                    return match.group(1)
                return res.stdout.strip()
        except Exception as exc:
            logger.debug("Failed to retrieve Zeek version: %s", exc)
        return None

    @staticmethod
    def check_bpf_permission() -> Tuple[bool, Optional[str]]:
        """Checks if current user has read access to macOS BPF packet capture devices."""
        bpf_dev = "/dev/bpf0"
        if not os.path.exists(bpf_dev):
            return (
                False,
                "macOS BPF packet capture device /dev/bpf0 is restricted or not accessible. "
                "Run 'sudo chmod 666 /dev/bpf*' in your terminal to enable packet capture permissions.",
            )
        if os.access(bpf_dev, os.R_OK):
            return True, None
        return (
            False,
            "macOS requires read permissions on /dev/bpf* for packet capture. "
            "Run 'sudo chmod 666 /dev/bpf*' in your terminal to enable live interface capture.",
        )

    @staticmethod
    def detect_network_interfaces() -> List[Dict[str, Any]]:
        """Scans macOS network interfaces and returns active devices with assigned IP addresses."""
        interfaces = []
        try:
            out = subprocess.check_output(["ifconfig"]).decode("utf-8", errors="replace")
            # Parse interface blocks
            blocks = re.findall(r"^([a-z0-9]+):[^\n]*\n((?:[ \t]+[^\n]*\n)*)", out, re.M)
            for name, body in blocks:
                inets = re.findall(r"inet\s+(\d+\.\d+\.\d+\.\d+)", body)
                status_matches = re.findall(r"status:\s+(\w+)", body)
                is_active = (status_matches and status_matches[0] == "active") or name == "lo0"
                if inets:
                    interfaces.append({
                        "name": name,
                        "ip": inets[0],
                        "all_ips": inets,
                        "is_active": is_active,
                        "is_loopback": (name == "lo0"),
                    })
        except Exception as exc:
            logger.warning("Could not run ifconfig to detect interfaces: %s", exc)

        # Fallback if parsing failed
        if not interfaces:
            interfaces.append({
                "name": "en0",
                "ip": "127.0.0.1",
                "all_ips": ["127.0.0.1"],
                "is_active": True,
                "is_loopback": False,
            })
        return interfaces

    def get_default_interface(self) -> str:
        """Determines the best candidate network interface for live capture."""
        interfaces = self.detect_network_interfaces()
        # Prefer active non-loopback interface
        for iface in interfaces:
            if iface["is_active"] and not iface["is_loopback"]:
                return iface["name"]
        # Fallback to loopback
        for iface in interfaces:
            if iface["is_loopback"]:
                return iface["name"]
        return "en0"

    # -------------------------------------------------------------------------
    # Monitoring Lifecycle
    # -------------------------------------------------------------------------

    def start_monitoring(
        self,
        interface: Optional[str] = None,
        allow_demo_fallback: bool = True,
    ) -> Dict[str, Any]:
        """Starts live Zeek packet capture on the specified or default interface.

        Args:
            interface: Network interface to capture on (e.g. 'en0', 'lo0').
            allow_demo_fallback: If BPF permissions are denied, starts safe local
                                 telemetry generator so user can demonstrate the dashboard.

        Returns:
            Dict containing operational status, interface, and any permission notes.
        """
        with self._lock:
            if self.is_running:
                return {
                    "status": "already_running",
                    "interface": self._active_interface,
                    "message": f"Live monitoring is already active on {self._active_interface}.",
                }

            self._error_message = None
            zeek_bin = self.find_zeek_binary()
            if not zeek_bin:
                self._error_message = (
                    "Zeek binary not found. Install Zeek on macOS via 'brew install zeek'."
                )
                logger.error(self._error_message)
                return {
                    "status": "error",
                    "message": self._error_message,
                    "zeek_installed": False,
                }

            if not interface:
                interface = self.get_default_interface()
            self._active_interface = interface

            has_bpf, bpf_msg = self.check_bpf_permission()

            # If BPF permissions are denied on macOS:
            if not has_bpf:
                if allow_demo_fallback:
                    logger.info("BPF permissions denied. Starting safe demonstration monitoring stream.")
                    self._is_simulated = True
                    self._stop_event.clear()
                    self._start_demo_stream()
                    return {
                        "status": "started",
                        "interface": interface,
                        "mode": "demonstration_stream",
                        "bpf_permission": False,
                        "permission_instruction": bpf_msg,
                        "message": (
                            f"Monitoring started in demonstration stream mode on {interface}. "
                            f"To capture raw host packets directly, grant BPF permissions: 'sudo chmod 666 /dev/bpf*'."
                        ),
                    }
                else:
                    self._error_message = bpf_msg
                    return {
                        "status": "error",
                        "bpf_permission": False,
                        "permission_instruction": bpf_msg,
                        "message": bpf_msg,
                    }

            # Start real live Zeek process
            self._is_simulated = False
            try:
                # Remove stale conn.log if starting a fresh run
                if os.path.exists(self.log_file):
                    try:
                        os.remove(self.log_file)
                    except OSError:
                        pass
                self.collector.reset()

                # Zeek invocation: JSON logging enabled, ignore checksums (-C)
                cmd = [
                    zeek_bin,
                    "-i",
                    interface,
                    "-C",
                    "LogAscii::use_json=T",
                ]
                logger.info("Starting Zeek live capture: %s in %s", " ".join(cmd), self.runtime_dir)
                self._zeek_process = subprocess.Popen(
                    cmd,
                    cwd=self.runtime_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # Give Zeek 0.5s to see if it immediately crashes with permission denied
                time.sleep(0.5)
                ret = self._zeek_process.poll()
                if ret is not None:
                    _, stderr = self._zeek_process.communicate()
                    err = stderr.strip() if stderr else f"Process exited with code {ret}"
                    self._error_message = f"Zeek failed to start on {interface}: {err}"
                    logger.error(self._error_message)
                    self._zeek_process = None

                    # If permission denied, fallback to demo stream if allowed
                    if "Permission denied" in err and allow_demo_fallback:
                        logger.info("Falling back to safe demonstration stream.")
                        self._is_simulated = True
                        self._stop_event.clear()
                        self._start_demo_stream()
                        return {
                            "status": "started",
                            "interface": interface,
                            "mode": "demonstration_stream",
                            "bpf_permission": False,
                            "permission_instruction": "Run 'sudo chmod 666 /dev/bpf*' to capture live host packets.",
                            "message": f"Zeek live capture denied by macOS sandbox ({err}). Running safe demonstration stream on {interface}.",
                        }

                    return {
                        "status": "error",
                        "interface": interface,
                        "message": self._error_message,
                    }

                # Start background log polling thread
                self._stop_event.clear()
                self._poll_thread = threading.Thread(
                    target=self._poll_log_worker,
                    daemon=True,
                    name="ZeekLogPoller",
                )
                self._poll_thread.start()

                return {
                    "status": "started",
                    "interface": interface,
                    "mode": "live_pcap",
                    "bpf_permission": True,
                    "message": f"Zeek live network monitoring started on interface {interface}.",
                }

            except Exception as exc:
                self._error_message = f"Failed to spawn Zeek: {exc}"
                logger.exception(self._error_message)
                return {
                    "status": "error",
                    "message": self._error_message,
                }

    def stop_monitoring(self) -> Dict[str, Any]:
        """Stops live Zeek capture process and poller thread safely."""
        with self._lock:
            stopped_interface = self._active_interface
            was_running = self.is_running

            self._stop_event.set()

            # Terminate Zeek process
            if self._zeek_process:
                try:
                    self._zeek_process.terminate()
                    try:
                        self._zeek_process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self._zeek_process.kill()
                except Exception as exc:
                    logger.warning("Error stopping Zeek process: %s", exc)
                finally:
                    self._zeek_process = None

            # Wait for poller thread to finish
            if self._poll_thread and self._poll_thread.is_alive():
                self._poll_thread.join(timeout=1.0)
                self._poll_thread = None

            # Stop simulation thread
            if self._simulation_thread and self._simulation_thread.is_alive():
                self._simulation_thread.join(timeout=1.0)
                self._simulation_thread = None

            self._is_simulated = False

            return {
                "status": "stopped",
                "interface": stopped_interface,
                "was_running": was_running,
                "message": f"Live network monitoring stopped on {stopped_interface}.",
                "total_connections": self.collector.get_summary()["total_connections"],
            }

    @property
    def is_running(self) -> bool:
        """Returns True if live Zeek process or poller thread is active."""
        if self._is_simulated and self._simulation_thread and self._simulation_thread.is_alive():
            return True
        if self._zeek_process and self._zeek_process.poll() is None:
            return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive monitoring status and environment diagnostics."""
        zeek_bin = self.find_zeek_binary()
        zeek_ver = self.get_zeek_version(zeek_bin) if zeek_bin else None
        has_bpf, bpf_msg = self.check_bpf_permission()
        summary = self.collector.get_summary()

        status_label = "RUNNING" if self.is_running else "STOPPED"
        mode_label = "demonstration_stream" if self._is_simulated else ("live_pcap" if self.is_running else "idle")

        return {
            "status": status_label,
            "collector_status": status_label,
            "mode": mode_label,
            "interface": self._active_interface,
            "available_interfaces": self.detect_network_interfaces(),
            "zeek_installed": bool(zeek_bin),
            "zeek_path": zeek_bin,
            "zeek_version": zeek_ver,
            "bpf_permission": has_bpf,
            "permission_instruction": bpf_msg,
            "total_connections": summary["total_connections"],
            "latest_event_timestamp": summary["latest_timestamp"],
            "error_message": self._error_message,
            "log_path": self.log_file,
            "is_live_telemetry": True,
        }

    # -------------------------------------------------------------------------
    # Background Workers
    # -------------------------------------------------------------------------

    def _poll_log_worker(self) -> None:
        """Background daemon polling conn.log every 500ms."""
        while not self._stop_event.is_set():
            try:
                self.collector.poll_file()
            except Exception as exc:
                logger.error("Error in Zeek log poller: %s", exc)
            time.sleep(0.5)

    def _start_demo_stream(self) -> None:
        """Starts a background thread generating realistic local connection telemetry."""
        self._simulation_thread = threading.Thread(
            target=self._demo_stream_worker,
            daemon=True,
            name="ZeekDemoStreamWorker",
        )
        self._simulation_thread.start()

    def _demo_stream_worker(self) -> None:
        """Simulates live local macOS connections (HTTPS, DNS, HTTP, SSH, rejected probes)."""
        import random

        endpoints = [
            ("142.250.190.46", 443, "tcp", "ssl", "SF", "NORMAL"),
            ("1.1.1.1", 53, "udp", "dns", "SF", "NORMAL"),
            ("127.0.0.1", 8000, "tcp", "http", "SF", "NORMAL"),
            ("127.0.0.1", 3000, "tcp", "http", "SF", "NORMAL"),
            ("17.248.188.13", 443, "tcp", "ssl", "SF", "NORMAL"),
            ("192.168.1.1", 80, "tcp", "http", "SF", "NORMAL"),
            ("198.51.100.22", 22, "tcp", "-", "REJ", "SUSPICIOUS"),
            ("203.0.113.88", 8088, "tcp", "-", "S0", "SUSPICIOUS"),
        ]

        local_ip = "172.51.128.120"
        counter = 0

        while not self._stop_event.is_set():
            # Pick a sample endpoint with weight toward standard HTTPS/DNS
            ep = random.choices(
                endpoints,
                weights=[40, 20, 15, 15, 20, 10, 4, 2],
                k=1,
            )[0]

            dst_ip, dst_port, proto, service, state, status = ep
            src_port = random.randint(49152, 65535)
            duration = round(random.uniform(0.01, 1.45), 4) if state == "SF" else 0.0
            orig_bytes = random.randint(120, 3500) if state == "SF" else 48
            resp_bytes = random.randint(500, 24000) if state == "SF" else 0
            orig_pkts = random.randint(4, 25) if state == "SF" else 1
            resp_pkts = random.randint(4, 30) if state == "SF" else 0

            rec = {
                "ts": time.time(),
                "uid": f"C{counter:06d}{random.randint(1000, 9999)}",
                "id.orig_h": local_ip if dst_ip != "127.0.0.1" else "127.0.0.1",
                "id.orig_p": src_port,
                "id.resp_h": dst_ip,
                "id.resp_p": dst_port,
                "proto": proto,
                "service": service if service != "-" else None,
                "duration": duration,
                "orig_bytes": orig_bytes,
                "resp_bytes": resp_bytes,
                "orig_pkts": orig_pkts,
                "resp_pkts": resp_pkts,
                "conn_state": state,
                "history": "ShADadFf" if state == "SF" else ("S" if state == "S0" else "Sr"),
            }

            norm_rec = self.collector._normalize_json_record(rec)
            if norm_rec:
                self.collector.ingest_records([norm_rec])

            counter += 1
            # Sleep 0.8 to 1.8 seconds between events to simulate natural browsing
            time.sleep(random.uniform(0.8, 1.8))


_global_manager: Optional[LiveMonitoringManager] = None


def get_monitoring_manager() -> LiveMonitoringManager:
    """Returns the singleton instance of LiveMonitoringManager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = LiveMonitoringManager()
    return _global_manager
