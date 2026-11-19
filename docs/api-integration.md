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
Checks backend operational status and declares data provenance.
* **Status**: `200 OK`
* **Response Body**:
  ```json
  {
    "status": "healthy",
    "service": "VYUHA Incident Intelligence API",
    "version": "0.1.0",
    "data_provenance": "CICIDS2017 Verified Model Benchmarks & Demonstrations",
    "is_live_telemetry": false
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

### 4.8 `POST /api/analyze-flow`
Evaluates dynamic network flow records on demand through Phase 4 alert normalization and temporal incident clustering.
* **Request Model**: `AnalyzeFlowRequest` (`flows: List[dict]`, `alert_on_threat_only: bool`)
* **Response Model**: `AnalyzeFlowResponse` (`alerts`, `incidents`, `evaluated_flows_count`, `threats_count`)

---

## 5. Frontend Integration & View Mapping

| Route / View | Frontend Component | API Endpoint | Dynamic Elements Bound |
|---|---|---|---|
| `/` or `/overview` | `Overview` | `GET /api/overview` | Posture status pill, flow counters, diurnal SVG area chart, attention cards, recent alerts table |
| `/activity` | `NetworkActivity` | `GET /api/network-activity` | Flow summary cards, diurnal volume chart, weekly bar chart, traffic moment cards |
| `/alerts` | `SecurityAlerts` | `GET /api/alerts` | Search bar, severity filter pills, alert count, expandable alert rows with multi-engine evidence |
| `/incidents` | `Incidents` | `GET /api/incidents` | Status filter pills, candidate incident cards with alert count and next steps |
| `/details` | `IncidentDetails` | `GET /api/incidents/{id}` | Dynamic incident title, relationship explanation (with summary fallback), timeline events, evidence cards, 404 state |
| `/insights` | `Insights` | `GET /api/insights` | Attack category progress bars, engine review badges, weekly rhythm chart, model benchmark guidance cards |

---

## 6. Verification and Test Suite

The integration is verified by 94 automated backend tests and production frontend bundling:

1. **Backend Integration Tests** (`vyuha/tests/test_api.py`):
   - Health check provenance validation.
   - Overview payload consistency and attention item contract.
   - Diurnal and weekly network activity data structure.
   - Alert filtering by severity and keyword search.
   - Candidate incident retrieval and status filtering.
   - Incident lookup by ID and strict 404 enforcement.
   - Dynamic non-empty relationship fallback validation.
   - Model benchmark metrics verification in insights.
   - Dynamic flow evaluation and on-demand incident synthesis.
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
