# VYUHA Frontend–Backend API Integration Guide

## 1. Executive Summary

This document specifies the integration architecture between the **VYUHA Incident Intelligence backend** (FastAPI) and the **editorial security dashboard** (Vite, React, TypeScript, Tailwind CSS).

The integration delivers an operational, explainable security analytics interface that exposes the four core phases of VYUHA:
* **Phase 1 & 2**: Cleaned CICIDS2017 network flow features and balanced training distributions.
* **Phase 3**: Supervised Binary Classification (Random Forest), Multiclass Attack-Family Classification (Random Forest), and Unsupervised Anomaly Detection (Isolation Forest).
* **Phase 4**: Canonical Alert Normalization, Temporal Sliding-Window Correlation, and Deterministic Graph Incident Grouping.

All data exposed via the API strictly complies with empirical data integrity constraints: telemetry is clearly labeled as offline benchmark evaluations and demonstration flows, preserving honest attribution without claiming live packet capture.

---

## 2. System Architecture

```
+-------------------------------------------------------------------------+
|                              CLIENT TIER                                |
|   Vite + React 18 + TypeScript + Tailwind CSS (Port 3000)               |
|   Manus Editorial Design System (Playfair Display / DM Sans)            |
+-------------------------------------------------------------------------+
                                    |
                            HTTP / Proxy (/api)
                                    v
+-------------------------------------------------------------------------+
|                              BACKEND API                                |
|   FastAPI Application (Port 8000)                                       |
|   - Uvicorn ASGI Server                                                 |
|   - Pydantic v2 Schema Validation                                       |
|   - Route Handlers: src/api/routes.py                                   |
|   - Service Layer:  src/api/service.py                                  |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                        VYUHA INTELLIGENCE CORE                          |
|   - Phase 4.1 Schema & Normalizer (src/intelligence/schema.py, adapters)|
|   - Phase 4.2 Temporal Correlation (src/intelligence/temporal.py)       |
|   - Phase 4.3 Incident Clustering (src/intelligence/incident.py, graph) |
|   - Phase 3 Benchmark Reports & Model Metrics (src/models/benchmarks)   |
+-------------------------------------------------------------------------+
```

### 2.1 Directory Structure
```
vyuha/
├── src/
│   ├── api/
│   │   ├── __init__.py       # API module exports
│   │   ├── app.py            # FastAPI entry point & CORS configuration
│   │   ├── routes.py         # REST route handlers
│   │   ├── schemas.py        # Pydantic v2 request/response models
│   │   └── service.py        # Business logic & Phase 4 intelligence bridge
│   ├── intelligence/         # Phase 4 Alert, Temporal & Incident engines
│   ├── models/               # Phase 3 Binary, Multiclass & Anomaly models
│   └── preprocessing/        # Phase 1 & 2 cleaning and feature selection
├── tests/
│   ├── test_api.py           # 13 FastAPI integration tests
│   └── ...                   # 81 Phase 1–4 unit/integration tests
├── docs/                     # Technical specifications and audit reports
├── reports/                  # Benchmark JSON evaluations
└── frontend/                 # Vite + React + TypeScript Dashboard
    ├── src/
    │   ├── types/api.ts      # TypeScript interfaces matching Pydantic schemas
    │   ├── services/api.ts   # API client with fallback handlers
    │   ├── App.tsx           # Root component, routing, and view state
    │   ├── index.css         # Editorial typography & theme rules
    │   └── main.tsx          # React DOM entry point
    ├── index.html
    ├── package.json
    ├── package-lock.json
    ├── tailwind.config.ts
    ├── tsconfig.json
    └── vite.config.ts        # Vite config with /api reverse proxy to 8000
```

---

## 3. Data Integrity & Provenance Principles

1. **No Live Packet Capture Claims**:
   The system explicitly declares through headers, topbars, and API payloads (`is_live_telemetry: false`, `data_provenance: "CICIDS2017 Verified Model Benchmarks & Demonstrations"`) that network telemetry reflects offline CICIDS2017 benchmark evaluations and demonstration flow models.
2. **Metadata Fidelity (CICIDS2017 Limitations)**:
   Earlier dataset audits confirmed that the CICIDS2017 `MachineLearningCVE` CSVs lack Source IP, Destination IP, Source Port, Timestamp, and Protocol columns. The API and frontend preserve this reality:
   - Unavailable entity fields are kept `null` or labeled as unrecorded.
   - Incidents do not invent false multi-host progression when IP addresses are unobserved.
   - Flow grouping relies strictly on verified features (e.g. `Destination Port`, observed protocol flags, temporal sequence offsets).
3. **No Fabricated Probabilities**:
   - Supervised models (`RandomForestClassifier`) provide calibrated posterior probabilities via `predict_proba()`.
   - Unsupervised models (`IsolationForest`) supply tree path length anomaly scores and decision thresholds; confidence is set to `None`.
4. **Deterministic Flow Identification**:
   Flow identifiers use dataset positional indexing (`flow_{index}`) to preserve discrete network events even when feature vectors are identical.

---

## 4. API Endpoints Reference

### 4.1 `GET /api/health`
Checks backend operational status, active model artifact readiness, and declares data provenance.
* **Status**: `200 OK`
* **Response Body**:
  ```json
  {
    "status": "healthy",
    "service": "VYUHA Incident Intelligence API",
    "version": "0.1.0",
    "data_provenance": "CICIDS2017 Verified Model Benchmarks & On-Demand Inference",
    "is_live_telemetry": false,
    "models_loaded": true,
    "models_status": {
      "binary_rf": true,
      "multiclass_rf": true,
      "iforest": true
    }
  }
  ```

### 4.2 `GET /api/overview`
Supplies high-level operational metrics, diurnal traffic summary, recent alerts, and actionable attention cards for the **Overview** page.
* **Response Model**: `OverviewResponse`
* **Key Fields**:
  - `posture`: Overall assessment (`"Heightened Alert"`, `"Active Monitoring"`), severity counts (high, medium, low), and candidate incident counts.
  - `flow_totals`: Total evaluated flows, threats flagged, and benign baseline count.
  - `traffic_data`: 8 diurnal observation windows (`"00:00 - 03:00"` through `"21:00 - 24:00"`) with normal vs flagged flow volumes.
  - `attention_items`: Dynamic attention cards containing `title`, `copy`, `tone` (`"sage" | "terracotta" | "lavender" | "yellow"`), `target` (`"/incidents" | "/activity" | "/insights"`), and `icon`.
  - `recent_alerts`: Latest canonical security alerts.

### 4.3 `GET /api/network-activity`
Provides volumetric flow breakdown across 3-hour diurnal windows, weekly bar chart distributions, and chronological traffic observation moments.
* **Response Model**: `NetworkActivityResponse`
* **Key Fields**:
  - `flow_totals`: Aggregate flow telemetry.
  - `traffic_data`: 8 diurnal windows with flow volumes.
  - `week_data`: 7 daily observation buckets (`"Mon"` through `"Sun"`).
  - `moments`: Chronological traffic moments detailing observed security activity.

### 4.4 `GET /api/alerts`
Returns normalized security alerts generated by binary, multiclass, and anomaly detection engines.
* **Query Parameters**:
  - `query` (string, optional): Keyword search matching alert title, summary, or category.
  - `severity` (string, optional): Filter by `"High"`, `"Medium"`, `"Low"`, or `"All"`.
* **Response Model**: `SecurityAlertsResponse`
* **Key Fields**:
  - `alerts`: List of `AlertItem` objects (`id`, `title`, `category`, `severity`, `timestamp`, `flowId`, `destinationPort`, `threatStatus`, `whyFlagged`, `evidence`).
  - `total_count`: Total alerts matching criteria.

### 4.5 `GET /api/incidents`
Returns candidate security incidents synthesized by the Phase 4 Incident Intelligence clustering engine.
* **Query Parameters**:
  - `status` (string, optional): Filter by `"Review"`, `"Monitoring"`, `"Closed candidate"`, or `"All"`.
* **Response Model**: `IncidentsResponse`
* **Key Fields**:
  - `incidents`: List of `IncidentItem` objects with synthesized headlines, severities, grouped alert counts, investigation next steps, and grouping explanations.
  - `total_count`: Number of candidate incidents returned.

### 4.6 `GET /api/incidents/{incident_id}`
Retrieves an in-depth candidate incident dossier by its ID.
* **Path Parameters**: `incident_id` (string)
* **Response Codes**:
  - `200 OK`: Returns full `IncidentItem` with detailed `timeline`, `evidence`, `alerts`, and `confidence`.
  - `404 Not Found`: Returns error detail if the incident does not exist. The frontend renders a dedicated *"Incident Not Found"* view rather than silently substituting mock data.

### 4.7 `GET /api/insights`
Supplies attack category distributions, detection engine participation metrics, weekly attack rhythm, and model benchmark scores.
* **Response Model**: `InsightsResponse`
* **Key Fields**:
  - `category_data`: Family distribution (`PortScan`, `DDoS`, `DoS`, `Web Attack`, `Botnet`, `Infiltration`, `Brute Force`).
  - `engine_data`: Model evaluation counts (Supervised Binary RF, Multiclass RF, Anomaly Isolation Forest).
  - `benchmark_metrics`: Verified model metrics (`binary_rf_accuracy`, `binary_rf_precision`, `binary_rf_recall`, `binary_rf_f1`, `anomaly_iforest_roc_auc`, `anomaly_iforest_f1`).

### 4.8 `GET /api/analyze-flow/presets`
Retrieves calibrated 63-feature network flow presets representing characteristic traffic behaviors for testing the ML inference engine.
* **Response Model**: `dict` containing `presets` key
* **Preset Definitions**:
  - `benign`: Benign HTTPS Web Browsing (Destination Port 443, standard TCP window, zero loss).
  - `portscan`: Stealth SYN PortScan (Destination Port 22, 0 duration, -1 backward window, single packet).
  - `dos`: Volumetric DoS/DDoS Flood (Destination Port 80, high packet volume, compressed inter-arrival times).
  - `anomaly`: Statistical Novelty / Zero-Day (Destination Port 8088, abnormal TCP flag distribution URG/CWE/ECE, out-of-distribution header sizes).
* **Response Structure**:
  ```json
  {
    "presets": {
      "benign": {
        "name": "Benign Web Browsing",
        "category": "BENIGN",
        "description": "Standard HTTPS TLS flow on port 443 with balanced payload and legitimate TCP window parameters.",
        "flow": { "Destination Port": 443, "Flow Duration": 115000, ... }
      },
      ...
    }
  }
  ```

### 4.9 `POST /api/analyze-flow`
Evaluates one or more 63-feature network flow vectors on demand through genuine Phase 6 machine learning inference (`Binary RF`, `Multiclass RF`, `Isolation Forest`) and Phase 4 incident clustering.
* **Request Model**: `AnalyzeFlowRequest`
  - `flows`: `List[Dict[str, Any]]` — List of flow dictionaries containing the required 63 features (accepts both clean snake_case and raw CICIDS2017 feature names).
  - `alert_on_threat_only`: `bool` (default: `false`) — When true, benign flows that trigger no threat alerts are omitted from the canonical alerts list.
  - `anomaly_threshold_tier`: `Optional[str]` (default: `"fpr_1.0pct"`) — Calibrated False Positive Rate threshold tier for Isolation Forest (`"fpr_5.0pct"`, `"fpr_1.0pct"`, `"fpr_0.50pct"`, `"fpr_0.10pct"`).
* **Response Model**: `AnalyzeFlowResponse`
  - `alerts`: `List[AlertItem]` — Normalized security alerts produced from flagged flows.
  - `incidents`: `List[IncidentItem]` — Candidate incidents synthesized by clustering correlated alerts.
  - `evaluated_flows_count`: `int` — Count of evaluated flow records.
  - `threats_count`: `int` — Count of flows flagged as threats (`ATTACK` or `ANOMALY`).
  - `evaluations`: `List[FlowEvaluationItem]` — Detailed multi-engine breakdown for each flow:
    - `flow_index`: Positional index in the request.
    - `flow_id`: Deterministic flow identifier (e.g. `flow_0`).
    - `consensus_status`: Multi-engine consensus verdict:
      - `CORROBORATED_ATTACK`: Binary RF Attack (`>=0.50`) **AND** Anomaly Score `>= Threshold`.
      - `SUPERVISED_CONFLICT`: Binary RF Attack (`>=0.50`) **AND** Anomaly Score `< Threshold`.
      - `NOVEL_ANOMALY`: Binary RF Benign (`<0.50`) **AND** Anomaly Score `>= Threshold`.
      - `CORROBORATED_BENIGN`: Binary RF Benign (`<0.50`) **AND** Anomaly Score `< Threshold`.
    - `consensus_threat`: Aggregated status (`"ATTACK"`, `"ANOMALY"`, `"BENIGN"`).
    - `is_attack`: Supervised binary attack classification boolean.
    - `flagged_as_threat`: Whether the flow met threat criteria under consensus.
    - `primary_category`: Categorical prediction (e.g. `"PortScan"`, `"DoS_DDoS"`, `"BENIGN"`).
    - `provisional_severity`: Calculated severity (`"CRITICAL"`, `"HIGH"`, `"MEDIUM"`, `"LOW"`, `"INFO"`).
    - `explanation`: Human-readable explanation of the multi-engine findings.
    - `binary`: `BinaryEvaluationDetail` (`prediction`, `label`, `is_attack`, `confidence`, `attack_probability`, `benign_probability`, `model_name`).
    - `multiclass`: `MulticlassEvaluationDetail` (`predicted_category`, `is_attack`, `confidence`, `class_probabilities`, `model_name`).
    - `anomaly`: `AnomalyEvaluationDetail` (`anomaly_score`, `calibrated_threshold`, `threshold_name`, `is_anomaly`, `threat_status`, `is_attack`, `model_name`).
  - `models_used`: `Dict[str, str]` — Map of model types to loaded joblib artifact filenames.
  - `anomaly_threshold_tier`: `str` — Active FPR threshold tier used.
* **Status Codes & Error Responses**:
  - `200 OK`: Successful inference evaluation.
  - `422 Unprocessable Entity`: Raised on `IncompleteFlowError` when one or more required features are missing from the 63-feature schema:
    ```json
    {
      "detail": {
        "error": "IncompleteFlowError",
        "message": "Flow record 0 is missing 2 required features: ['Destination Port', 'Flow Duration']",
        "record_index": 0,
        "missing_features": ["Destination Port", "Flow Duration"]
      }
    }
    ```
  - `503 Service Unavailable`: Raised on `ModelLoadError` if model artifacts cannot be loaded from disk or are corrupted.

### 4.10 `GET /api/live/status` (Phase 7)
Returns Zeek installation details, collector state, active interface, and macOS BPF permissions.
* **Status**: `200 OK`
* **Response Model**: `LiveStatusResponse`
* **Key Fields**: `status`, `collector_status`, `mode`, `interface`, `available_interfaces`, `zeek_installed`, `zeek_path`, `zeek_version`, `bpf_permission`, `permission_instruction`, `total_connections`, `latest_event_timestamp`.

### 4.11 `GET /api/live/connections` (Phase 7)
Returns recently observed connection records with pagination and status filtering.
* **Query Parameters**: `limit: int = 50`, `status: Optional[str]`
* **Response Model**: `LiveConnectionsResponse`
* **Key Fields**: `connections` (list of `LiveConnectionItem` with `src_ip`, `dst_ip`, `src_port`, `dst_port`, `proto`, `service`, `duration`, `orig_bytes`, `resp_bytes`, `conn_state`, `detection_status`, `ml_status`), `total_count`, `returned_count`, `interface`.

### 4.12 `GET /api/live/summary` (Phase 7)
Returns aggregated protocol distribution, top destination ports, and status counts.
* **Response Model**: `LiveSummaryResponse`
* **Key Fields**: `total_connections`, `protocol_distribution`, `status_distribution`, `top_destination_ports`, `latest_timestamp`.

### 4.13 `POST /api/live/start` (Phase 7)
Starts live packet capture via Zeek on the host.
* **Request Model**: `LiveStartRequest` (`interface: Optional[str]`, `allow_demo_fallback: bool = true`)
* **Response Model**: `LiveStartResponse` (`status`, `interface`, `mode`, `bpf_permission`, `permission_instruction`, `message`).

### 4.14 `POST /api/live/stop` (Phase 7)
Stops Zeek subprocess execution safely.
* **Response Model**: `LiveStopResponse` (`status`, `interface`, `was_running`, `message`, `total_connections`).

---

## 5. Frontend Integration & View Mapping

| Route / View | Frontend Component | API Endpoint | Dynamic Elements Bound |
|---|---|---|---|
| `/` or `/overview` | `Overview` | `GET /api/overview` | Posture status pill, flow counters, diurnal SVG area chart, attention cards, recent alerts table |
| `/live` | `LiveMonitoring` | `GET /api/live/status`<br>`GET /api/live/connections`<br>`GET /api/live/summary`<br>`POST /api/live/start`<br>`POST /api/live/stop` | Real-time connection table, auto-polling (2.5s), Start/Stop controls, capture interface selector, BPF permission banner, protocol breakdown, heuristic observation badges (NORMAL, SUSPICIOUS, THREAT), transparent ML constraint callout |
| `/activity` | `NetworkActivity` | `GET /api/network-activity` | Flow summary cards, diurnal volume chart, weekly bar chart, traffic moment cards |
| `/alerts` | `SecurityAlerts` | `GET /api/alerts` | Search bar, severity filter pills, alert count, expandable alert rows with multi-engine evidence, "Test ML Inference" launcher |
| `/workbench` | `FlowInferenceWorkbench` | `GET /api/analyze-flow/presets`<br>`POST /api/analyze-flow` | 4 Preset selector cards (Benign, PortScan, DoS, Novelty Anomaly), calibrated FPR threshold selector, 63-feature inspection, live multi-engine consensus badge, probability distribution gauges, one-click jump to candidate incident details |
| `/incidents` | `Incidents` | `GET /api/incidents` | Status filter pills, candidate incident cards with alert count and next steps |
| `/details` | `IncidentDetails` | `GET /api/incidents/{id}` | Dynamic incident title, relationship explanation (with summary fallback), timeline events, evidence cards, 404 state |
| `/insights` | `Insights` | `GET /api/insights` | Attack category progress bars, engine review badges, weekly rhythm chart, model benchmark guidance cards |

---

## 6. Verification and Test Suite

The integration is verified by 135 automated backend tests and production frontend bundling:

1. **Backend Automated Tests** (`vyuha/tests/` — 135 tests passing):
   * **Live Monitoring Tests** (`tests/test_live_monitoring.py` — 18 tests):
     - Zeek installation and version detection.
     - macOS BPF permission check and copyable instructions.
     - Host network interface auto-discovery (`lo0`, `en0`).
     - Strict 63-feature validation (`INSUFFICIENT_FEATURES` rejection).
     - Heuristic classification rules (NORMAL, SUSPICIOUS, POTENTIAL THREAT).
     - JSON and TSV log parsing.
     - Non-blocking incremental tailing with file offset tracking.
     - Duplicate prevention using Zeek connection `uid`.
     - Malformed log line safety.
     - Process manager startup/shutdown lifecycle.
     - Live REST endpoints (`/api/live/*`).
   * **API Integration Tests** (`tests/test_api.py`):
     - Health check provenance and model status validation (`models_loaded: true`).
     - Overview payload consistency, diurnal activity, and attention item contract.
     - Security alerts retrieval, severity filtering, and keyword search.
     - Candidate incident clustering, status filtering, and strict 404 enforcement.
     - Preset flow retrieval (`GET /api/analyze-flow/presets`).
     - Live on-demand ML flow inference (`POST /api/analyze-flow`).
   * **ML Inference Engine Tests** (`tests/test_ml_inference.py`):
     - Flow preprocessor 63-feature schema verification and alias mapping.
     - TCP window sentinel value preservation (`-1` and `65535` preserved intact).
     - Model registry thread-safe singleton lifecycle and warm cache access.
     - Multi-engine consensus matrix (`CORROBORATED_ATTACK`, `SUPERVISED_CONFLICT`, `NOVEL_ANOMALY`, `CORROBORATED_BENIGN`).
     - Unsupervised novelty handling (anomaly-only flows are never flagged as confirmed attacks).
     - Strict HTTP 422 enforcement on incomplete feature vectors.
     - Strict HTTP 503 error handling on missing or corrupt model artifacts.
   * **Phase 1–4 Core Tests**:
     - 81 unit and integration tests covering alert normalization, sliding-window temporal correlation, target graph clustering, and dataset transforms.
2. **Frontend Production Build**:
   ```bash
   cd frontend && npm run build
   ```
   Ensures zero TypeScript compiler errors and produces an optimized production bundle (`dist/`).
3. **Running the Full System**:
   - Backend:
     ```bash
     uvicorn src.api.app:app --host 127.0.0.1 --port 8000
     ```
   - Frontend:
     ```bash
     cd frontend && npm run dev
     ```
   - Access the dashboard at `http://localhost:3000`.
