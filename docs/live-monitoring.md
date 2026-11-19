# VYUHA Phase 7: Live Network Monitoring with Zeek

## 1. Executive Summary

Phase 7 delivers the final development milestone for **VYUHA**: transitioning the platform from static dataset benchmarks and on-demand preset inference into a **functional, demonstrable live network monitoring system** running on macOS (Apple Silicon).

The implementation combines:
1. **Zeek (Bro) 9.0.1** network security monitor capturing connection-level traffic.
2. A **lightweight, non-blocking log collector** tailing and parsing JSON-formatted `conn.log` records.
3. A **feature completeness validator** that strictly enforces empirical data boundaries (rejecting synthetic zero-padding for unsupported ML features).
4. **FastAPI REST endpoints** for monitoring lifecycle, telemetry streaming, and aggregation.
5. An **interactive Live Monitoring view** in the React dashboard with real-time polling, heuristic classification, and connection inspection.

> [!IMPORTANT]
> **Academic Scope & Operational Boundaries**: VYUHA is a solo college engineering project designed to demonstrate accessible network security monitoring principles. It is not an enterprise Security Operations Centre (SOC) and does not perform active intrusion prevention or automated packet blocking.

---

## 2. Architecture & Telemetry Pipeline

```
+----------------------------------------------------------------------------------------------------+
|                                    MACOS HOST NETWORK INTERFACE                                    |
|              Active Wi-Fi / Ethernet (en0: 172.51.128.120)  |  Loopback (lo0: 127.0.0.1)            |
+----------------------------------------------------------------------------------------------------+
                                                   │
                                      BPF Filter (/dev/bpf*)
                                                   ▼
+----------------------------------------------------------------------------------------------------+
|                                      ZEEK 9.0.1 ENGINE (daemon)                                    |
|   zeek -i <interface> -C LogAscii::use_json=T (runtime/zeek_live/)                                |
|   Outputs connection telemetry to structured JSON: conn.log                                       |
+----------------------------------------------------------------------------------------------------+
                                                   │
                                        Incremental File Tailing
                                                   ▼
+----------------------------------------------------------------------------------------------------+
|                             VYUHA LOG COLLECTOR (src/monitoring/collector.py)                       |
|   * Non-blocking incremental tailing with byte-offset tracking & inode detection                    |
|   * Duplicate prevention via Zeek UID ring buffer                                                  |
|   * Fault-tolerant JSON & TSV record parser                                                        |
+----------------------------------------------------------------------------------------------------+
                                                   │
                                         Feature Validation
                                                   ▼
+----------------------------------------------------------------------------------------------------+
|                           FEATURE INTEGRITY VALIDATOR (src/monitoring/validator.py)                 |
|   * Strict 63-Feature Completeness Check (Missing features flagged, zero-padding rejected)         |
|   * Status: INSUFFICIENT_FEATURES (13/63 features available from standard conn.log)                |
|   * Rule-Based Heuristic Triage: NORMAL | SUSPICIOUS | POTENTIAL THREAT                           |
+----------------------------------------------------------------------------------------------------+
                                                   │
                                        FastAPI Service Layer
                                                   ▼
+----------------------------------------------------------------------------------------------------+
|                              REST API ENDPOINTS (src/api/routes.py)                                |
|   GET  /api/live/status       | GET  /api/live/connections   | GET  /api/live/summary              |
|   POST /api/live/start        | POST /api/live/stop                                                |
+----------------------------------------------------------------------------------------------------+
                                                   │
                                            HTTP / Vite Proxy
                                                   ▼
+----------------------------------------------------------------------------------------------------+
|                            REACT DASHBOARD (/live — frontend/src/App.tsx)                          |
|   * Real-time polling indicator (2.5s auto-refresh)                                                |
|   * Capture interface selection & Start/Stop controls                                              |
|   * Live connection table with source/dest IPs, ports, protocols, state, and volume                |
|   * Heuristic observation badges & transparent ML constraint callout                               |
+----------------------------------------------------------------------------------------------------+
```

---

## 3. Zeek Installation & macOS Setup

### 3.1 Installation via Homebrew
Zeek is installed via Homebrew for macOS Apple Silicon (`arm64`):
```bash
brew install zeek
```
* **Binary Path**: `/opt/homebrew/bin/zeek`
* **Installed Version**: Zeek 9.0.1

### 3.2 macOS Packet Capture Permissions (`/dev/bpf*`)
By default, macOS restricts packet capture devices (`/dev/bpf*`) to `root:wheel` with permissions `600` (`crw-------`). An unprivileged user running `zeek -i <interface>` will encounter:
```text
pcap_error: (cannot open BPF device) /dev/bpf0: Permission denied
```

#### Permission Resolution:
To grant read/write access to BPF devices for local packet capture without running the whole server as root:
```bash
sudo chmod 666 /dev/bpf*
```

#### Resilient Dual-Mode Operation:
If BPF permissions have not been granted, VYUHA automatically switches to **Safe Demonstration Stream Mode**:
1. It reports `bpf_permission: false` in `/api/live/status`.
2. It surfaces a clear, copyable permission banner in the React dashboard.
3. It generates local loopback and outbound connection telemetry (HTTPS, DNS, HTTP, SSH rejected probes) so that evaluators can test and demonstrate the live dashboard immediately without requiring root access.
4. When `sudo chmod 666 /dev/bpf*` is executed, clicking "Start Live Monitoring" immediately captures raw host packets directly from the hardware interface.

---

## 4. The Critical ML Feature Constraint & Data Integrity

A core development principle of VYUHA is **zero fabricated results and strict empirical honesty**.

### 4.1 The Feature Gap
* **Trained ML Models (Phase 6)**: The supervised Binary Random Forest, Multiclass Random Forest, and unsupervised Isolation Forest models require **all 63 statistical flow features** from the CICIDS2017 dataset (e.g. forward/backward packet inter-arrival times, packet length standard deviation, TCP window size sentinel flags, subflow byte rates).
* **Standard Zeek `conn.log`**: Provides **13 transport-layer fields** (`ts`, `uid`, `id.orig_h`, `id.orig_p`, `id.resp_h`, `id.resp_p`, `proto`, `service`, `duration`, `orig_bytes`, `resp_bytes`, `conn_state`, `history`).

### 4.2 Prohibited Workarounds
* **DO NOT** fill missing statistical features with arbitrary zeros.
* **DO NOT** pass incomplete 13-feature vectors to the 63-feature models.
* **DO NOT** claim that basic connection logs reproduce deep bidirectional flow statistics.

### 4.3 VYUHA's Solution
1. **Raw Telemetry Display**: Real connection records are presented with full fidelity in the dashboard.
2. **Explicit Labeling**: Every live connection record displays `ml_status: "INSUFFICIENT_FEATURES"` and `missing_features_count: 63` with an inline explanation:
   > *"Raw Zeek connection record contains 13 transport metadata fields; lacks the 63 packet-level dispersion and inter-arrival time statistics required by the trained models. Preserved as raw live telemetry."*
3. **Rule-Based Heuristic Observation**: Provides practical security visibility without fabricating machine learning predictions:
   * **`NORMAL`**: Standard established connections (`conn_state == 'SF'`) on legitimate services (443 HTTPS, 80 HTTP, 53 DNS, local loopback).
   * **`SUSPICIOUS`**: Rejected (`REJ`) or unanswered (`S0`) attempts on sensitive administrative ports (22 SSH, 23 Telnet, 445 SMB, 3389 RDP).
   * **`POTENTIAL THREAT`**: Volumetric flooding (e.g. `>5,000` packets within seconds) or external probes.

---

## 5. Supported Zeek Log Fields

The collector (`ZeekLogCollector`) parses and normalizes the following fields:

| Field Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `uid` | `string` | Unique Zeek connection identifier | `CHhAvV3ug9PqY1u4` |
| `ts` | `float` | Epoch timestamp of connection start | `1795067848.017` |
| `timestamp` | `string` | Formatted local time string | `11:27:28` |
| `src_ip` | `string` | Originator IP address | `172.51.128.120` |
| `src_port` | `integer` | Originator ephemeral port | `53288` |
| `dst_ip` | `string` | Responder IP address | `142.250.190.46` |
| `dst_port` | `integer` | Responder service port | `443` |
| `proto` | `string` | Transport protocol (`TCP`, `UDP`, `ICMP`) | `TCP` |
| `service` | `string \| null`| Application protocol if detected | `ssl`, `http`, `dns` |
| `duration` | `float` | Connection duration in seconds | `0.4523` |
| `orig_bytes` | `integer` | Payload bytes sent by originator | `1420` |
| `resp_bytes` | `integer` | Payload bytes sent by responder | `8920` |
| `total_bytes`| `integer` | Total connection bytes | `10340` |
| `orig_pkts` | `integer` | Packets sent by originator | `12` |
| `resp_pkts` | `integer` | Packets sent by responder | `16` |
| `conn_state` | `string` | Zeek connection state summary | `SF`, `S0`, `REJ`, `RSTO` |
| `history` | `string \| null`| TCP handshake and state flags | `ShADadFf` |
| `detection_status` | `string` | Heuristic status: `NORMAL`, `SUSPICIOUS`, `POTENTIAL THREAT` | `NORMAL` |
| `ml_status` | `string` | Machine learning eligibility | `INSUFFICIENT_FEATURES` |

---

## 6. API Endpoints Reference

### 6.1 `GET /api/live/status`
Returns Zeek installation details, collector state, active interface, and BPF permissions.
* **Status**: `200 OK`
* **Response Body**:
  ```json
  {
    "status": "RUNNING",
    "collector_status": "RUNNING",
    "mode": "live_pcap",
    "interface": "en0",
    "available_interfaces": [
      { "name": "lo0", "ip": "127.0.0.1", "is_active": true, "is_loopback": true },
      { "name": "en0", "ip": "172.51.128.120", "is_active": true, "is_loopback": false }
    ],
    "zeek_installed": true,
    "zeek_path": "/opt/homebrew/bin/zeek",
    "zeek_version": "9.0.1",
    "bpf_permission": true,
    "permission_instruction": null,
    "total_connections": 142,
    "latest_event_timestamp": 1795067848.017,
    "error_message": null,
    "is_live_telemetry": true
  }
  ```

### 6.2 `GET /api/live/connections`
Returns recently observed connection records with pagination and status filtering.
* **Query Parameters**:
  - `limit` (integer, default: 50, max: 500)
  - `status` (string, optional: `NORMAL`, `SUSPICIOUS`, `POTENTIAL THREAT`, `ALL`)
* **Response Body**:
  ```json
  {
    "connections": [
      {
        "uid": "C0000018492",
        "timestamp": "11:27:28",
        "src_ip": "172.51.128.120",
        "src_port": 53288,
        "dst_ip": "142.250.190.46",
        "dst_port": 443,
        "proto": "TCP",
        "service": "ssl",
        "duration": 0.4523,
        "total_bytes": 10340,
        "conn_state": "SF",
        "detection_status": "NORMAL",
        "detection_severity": "INFO",
        "detection_reason": "Standard HTTPS connection completed normally (SF).",
        "ml_evaluated": false,
        "ml_status": "INSUFFICIENT_FEATURES",
        "missing_features_count": 63
      }
    ],
    "total_count": 142,
    "returned_count": 1,
    "interface": "en0",
    "is_monitoring": true
  }
  ```

### 6.3 `GET /api/live/summary`
Returns aggregated protocol distribution, top destination ports, and status counts.
* **Status**: `200 OK`
* **Response Body**:
  ```json
  {
    "total_connections": 142,
    "protocol_distribution": { "TCP": 118, "UDP": 24 },
    "status_distribution": { "NORMAL": 136, "SUSPICIOUS": 6 },
    "top_destination_ports": [
      { "port": 443, "count": 92 },
      { "port": 53, "count": 24 },
      { "port": 80, "count": 18 },
      { "port": 22, "count": 6 }
    ],
    "latest_timestamp": 1795067848.017,
    "interface": "en0",
    "is_monitoring": true
  }
  ```

### 6.4 `POST /api/live/start`
Starts live packet capture via Zeek on the host.
* **Request Body**:
  ```json
  {
    "interface": "en0",
    "allow_demo_fallback": true
  }
  ```
* **Response Body**:
  ```json
  {
    "status": "started",
    "interface": "en0",
    "mode": "live_pcap",
    "bpf_permission": true,
    "permission_instruction": null,
    "message": "Zeek live network monitoring started on interface en0."
  }
  ```

### 6.5 `POST /api/live/stop`
Stops Zeek subprocess execution safely.
* **Response Body**:
  ```json
  {
    "status": "stopped",
    "interface": "en0",
    "was_running": true,
    "message": "Live network monitoring stopped on en0.",
    "total_connections": 142
  }
  ```

---

## 7. How to Demonstrate Live Monitoring

1. **Start the FastAPI Backend**:
   ```bash
   cd vyuha
   source .venv/bin/activate
   uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
   ```

2. **Start the Frontend Dashboard**:
   ```bash
   cd vyuha/frontend
   npm run dev
   ```

3. **Navigate to Live Monitoring**:
   * Open `http://localhost:3000` in Chrome/Safari.
   * Click **Live Monitoring** in the sidebar navigation or browse directly to `http://localhost:3000/live`.

4. **Start Packet Observation**:
   * Select your active network interface (e.g. `en0` or `lo0`).
   * Click **Start Live Monitoring**.
   * If BPF permissions are restricted, review the banner and run `sudo chmod 666 /dev/bpf*` in Terminal.

5. **Generate Local Traffic**:
   * Open browser tabs (e.g. visiting HTTPS sites).
   * Or run local HTTP requests:
     ```bash
     curl -s http://127.0.0.1:8000/api/health
     ```

6. **Inspect Telemetry**:
   * Watch live connections appear in the table with 2.5s auto-refresh.
   * Filter by `NORMAL`, `SUSPICIOUS`, or `POTENTIAL THREAT`.
   * Click any row to expand connection metadata, TCP flags, heuristic rationale, and ML eligibility explanation.

---

## 8. Verification & Test Suite

Phase 7 is validated by **18 automated unit and integration tests** in `tests/test_live_monitoring.py`:
* Zeek installation detection and version extraction.
* BPF permission checks on `/dev/bpf*`.
* Network interface auto-discovery.
* Strict 63-feature validation (`INSUFFICIENT_FEATURES` rejection).
* Rule-based heuristic classification.
* JSON and TSV log parsing.
* Non-blocking incremental tailing with file offset tracking.
* Duplicate prevention using Zeek connection `uid`.
* Malformed log line safety.
* Process manager startup/shutdown lifecycle.
* All 5 FastAPI REST endpoints (`/api/live/*`).

Run the automated test suite:
```bash
pytest tests/test_live_monitoring.py
```
Total project test suite passes: **135 passed tests**.
