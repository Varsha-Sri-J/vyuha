# VYUHA

**AI-Powered Cybersecurity for Small Industries**  
*Making network security monitoring more accessible, intelligent and explainable.*

---

## About VYUHA

**VYUHA** is an AI-powered network security monitoring and threat intelligence platform designed to help small manufacturing industries and businesses identify suspicious network activity and understand potential cybersecurity threats.

Large enterprises often have dedicated cybersecurity teams and Security Operations Centres (SOCs) to monitor their networks. However, smaller industries may lack the resources and infrastructure required for continuous security monitoring.

VYUHA aims to bridge this gap by combining machine learning-based intrusion detection, network behaviour analysis, anomaly detection, and intelligent alert correlation into a unified, accessible platform.

Instead of simply displaying isolated security alerts, VYUHA provides a clearer, contextualized picture of what is happening across the network.

---

## Problem Statement

Small manufacturing industries often lack affordable, reliable, and accessible cybersecurity infrastructure. Without dedicated security teams or continuous network monitoring:
* Suspicious network behaviour can go unnoticed.
* Unauthorised access attempts may not be detected.
* Malicious traffic can remain unidentified.
* Security alerts are difficult to triage and investigate without deep security expertise.

The challenge is to develop an accessible AI-powered solution that helps these industries monitor network activity, identify potential threats, and understand security incidents without depending on expensive enterprise-level infrastructure.

---

## Proposed Solution

VYUHA provides an integrated platform that:
* **Analyses network traffic** and communication patterns.
* **Identifies potentially malicious network activity** using supervised machine learning.
* **Detects unusual behaviour and traffic anomalies** through unsupervised learning.
* **Explores network congestion indicators** and unusual traffic surges.
* **Correlates potentially related security alerts** using temporal and topological features.
* **Generates explainable candidate security incidents** with supporting technical rationale.
* **Provides investigation timelines** and actionable analyst guidance.

---

## Core Features

### 1. Network Traffic Analysis
Analyses network-flow characteristics, including:
* Source and destination IP addresses
* Source and destination ports
* Network protocols
* Flow duration
* Packet and byte counts
* Network flow rates

### 2. Network Congestion Analysis
Explores unusual traffic spikes and network behaviour that may indicate congestion. Potential indicators include traffic volume, packet rates, latency, packet loss, and throughput. *(Direct network performance measurements may require additional telemetry beyond the primary dataset).*

### 3. Machine Learning-Based Intrusion Detection
Explores supervised machine learning models (Random Forest, Decision Tree) to identify traffic patterns associated with:
* Port scanning
* DoS and DDoS attacks
* Brute-force attempts (FTP / SSH)
* Suspicious web traffic (Brute Force, XSS, SQL Injection)
* Other malicious communication patterns

### 4. Anomaly Detection
Identifies network behaviour that deviates significantly from established traffic patterns using unsupervised learning (Isolation Forest).

### 5. Alert Correlation and Campaign Intelligence
Rather than treating every suspicious event as an isolated alert, the platform examines relationships between alerts using:
* Temporal proximity
* Shared source and destination IP addresses
* Port relationships
* Protocol similarities
* Attack-category compatibility
* Repeated connection patterns

Potentially related alerts are grouped into candidate security incidents supported by empirical evidence.

### 6. Explainable Threat Intelligence
Provides clear analyst context for detected activity:
* **What** was detected?
* **Why** was it flagged?
* **Which** network endpoints were involved?
* **Which** alerts may be related?
* **What** evidence supports the correlation?
* **What** should an analyst investigate next?

---

## Dataset

* **Primary Dataset**: CICIDS2017 — MachineLearningCVE
* **Developer**: Canadian Institute for Cybersecurity at the University of New Brunswick (UNB), Canada
* **Official Source**: [UNB CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)

### Dataset Overview

| Attribute | Details |
| :--- | :--- |
| **Dataset** | CICIDS2017 |
| **Version** | MachineLearningCVE |
| **CSV Files** | 8 |
| **Total Records** | 2,830,743 |
| **Network-Flow Features** | 78 |
| **Original Traffic Classes** | 15 (1 Benign + 14 Attack scenarios) |
| **Raw Dataset Size** | ~843.66 MB |

The dataset has undergone an initial inspection and data-quality audit. The raw dataset is maintained outside this repository and is not tracked in Git.

---

## System Architecture

```text
             CICIDS2017 DATASET
                     |
                     v
              TRAFFIC INGESTION
                     |
                     v
             DATA PREPROCESSING
                     |
                     v
             ML DETECTION ENGINE
                     |
          +----------+----------+
          |                     |
          v                     v
    TRAFFIC ANALYSIS      ALERT GENERATION
          |                     |
          v                     v
    ANOMALY /             ALERT CORRELATION
    CONGESTION                  |
    ANALYSIS                    v
                       CAMPAIGN INTELLIGENCE
                                |
                                v
                       INCIDENT GENERATION
                                |
                                v
                       EXPLANATION ENGINE
                                |
                                v
                          SOC DASHBOARD
                                |
                                v
                       INVESTIGATION REPORTS
```

---

## Technology Stack

| Component | Technology |
| :--- | :--- |
| **Programming Language** | Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **Backend API** | FastAPI, Uvicorn |
| **Frontend** | React, TypeScript |
| **Visualisation** | Recharts |
| **Relationship Analysis** | NetworkX |

---

## Implementation Roadmap

| Phase | Description | Status |
| :--- | :--- | :--- |
| **Phase 1** | Project setup, environment verification & CICIDS2017 dataset audit | Completed |
| **Phase 2** | Feature selection (63 features) and data preprocessing pipeline | Completed |
| **Phase 3** | Supervised (Binary/Multiclass RF) & Unsupervised (Isolation Forest) models | Completed |
| **Phase 4** | Alert normalization, temporal correlation & candidate incident grouping | Completed |
| **Phase 5** | FastAPI REST API backend & React editorial security dashboard integration | Completed |
| **Phase 6** | Genuine on-demand machine learning inference engine & interactive workbench | Completed |
| **Phase 7** | Live network monitoring with Zeek on macOS & real-time connection observation | Completed |
| **Phase 8** | Industrial protocol expansion (Modbus, DNP3, OPC-UA telemetry) | Planned |
| **Phase 9** | Automated response playbooks & incident remediation workflows | Planned |

---

## Setup & Startup Guide

### Prerequisites
* **Python**: 3.9+ (Python 3.11 recommended)
* **Node.js**: 18+ and `npm`
* **Zeek**: 9.0+ (`brew install zeek` on macOS)

### 1. Backend Setup & Startup
```bash
# Navigate to repository root
cd vyuha

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install backend dependencies
pip install -r requirements.txt

# Start the FastAPI backend service (port 8000)
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```
* **API Documentation & Swagger UI**: `http://127.0.0.1:8000/docs`
* **Health Check**: `http://127.0.0.1:8000/api/health`
* **Live Monitoring Status**: `http://127.0.0.1:8000/api/live/status`

### 2. Frontend Setup & Startup
```bash
# Navigate to the frontend directory
cd vyuha/frontend

# Install frontend dependencies
npm install

# Start the Vite development server (port 3000, proxies /api requests to 8000)
npm run dev
```
* **Security Dashboard**: Access `http://localhost:3000`
* **Live Network Monitoring**: `http://localhost:3000/live`
* **ML Inference Workbench**: Accessible via **Security Alerts** or `/workbench`

### 3. Verification & Testing

#### Run Backend Test Suite
```bash
# Run all 135 automated unit and integration tests
pytest tests/
```

#### Run Frontend Production Build
```bash
# Verify TypeScript types and compile production bundle
cd frontend && npm run build
```

---

## Live Network Monitoring with Zeek (Phase 7)

Phase 7 transitions VYUHA into an active, demonstrable network security observer on macOS:

* **Zeek 9.0 Integration**: Ingests live connection events from host network interfaces (`en0`, `lo0`) formatted as structured JSON (`conn.log`).
* **Non-Blocking Ingestion**: The `ZeekLogCollector` incrementally tails logs using byte-offset tracking, preventing duplicate processing with an in-memory UID ring buffer.
* **macOS BPF Permission Handling**: By default, macOS requires read access to `/dev/bpf*` for packet capture. If unprivileged, VYUHA surfaces clear instructions (`sudo chmod 666 /dev/bpf*`) while operating in safe demonstration stream mode so evaluators can inspect telemetry immediately.
* **Strict ML Feature Integrity**: Standard Zeek `conn.log` provides 13 transport-layer fields but lacks the 63 statistical dispersion features of CICIDS2017. To prevent fabricated predictions, live connection records are classified with transparent rule-based heuristics (`NORMAL`, `SUSPICIOUS`, `POTENTIAL THREAT`) and marked as **INSUFFICIENT_FEATURES** for ML rather than padding missing columns with synthetic zeros.
* **Comprehensive Documentation**: Complete guide available in [`docs/live-monitoring.md`](docs/live-monitoring.md).

---

## Project Structure

```text
vyuha/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/                     # Engineering specifications & phase audit reports
│   ├── api-integration.md
│   ├── ml-inference.md
│   ├── live-monitoring.md
│   ├── incident-intelligence.md
│   └── ...
│
├── models/                   # Trained scikit-learn model artifacts (untracked)
│   ├── rf_full.joblib
│   ├── multiclass_rf_full.joblib
│   └── iforest_full_100k.joblib
│
├── src/
│   ├── api/                  # FastAPI routes, Pydantic schemas, and service layer
│   ├── models/               # ModelRegistry, FlowPreprocessor, and InferenceEngine
│   ├── monitoring/           # ZeekLogCollector, LiveMonitoringManager, FeatureValidator
│   ├── intelligence/         # Canonical alert normalization & temporal correlation
│   └── preprocessing/        # 63-feature selection & pipeline transforms
│
├── tests/                    # 135 automated pytest unit and integration tests
│   ├── test_api.py
│   ├── test_ml_inference.py
│   ├── test_live_monitoring.py
│   └── ...
│
└── frontend/                 # React + TypeScript + Vite + Tailwind CSS dashboard
    ├── src/
    │   ├── App.tsx           # Dashboard view state, Live Monitoring, ML Workbench
    │   ├── services/         # API client & offline preset fallbacks
    │   └── types/            # TypeScript schemas matching backend Pydantic models
    ├── package.json
    └── vite.config.ts
```

---

## Development Principles

* **Build incrementally** and validate each component before proceeding.
* **Preserve original datasets**: raw data is immutable and kept outside version control.
* **Memory-efficient processing**: chunked processing for large volume flows.
* **Zero data leakage**: strict deduplication and train/test separation.
* **Rigorous evaluation**: evaluate using Precision, Recall, F1-score, and PR-AUC instead of misleading accuracy.
* **Explainable decisions**: clear rationale for detections and correlation groupings.
* **Evidence-based reporting**: correlated alerts are candidate incidents, not unfalsifiable conclusions.
* **Honest metrics**: zero fabricated results.

---

## Current Development Status

* [x] Project workspace established and monorepo structure configured.
* [x] GitHub repository initialized.
* [x] Python virtual environment configured.
* [x] CICIDS2017 dataset verified and inspected.
* [x] Empirical dataset intelligence & quality audit completed (2.83M rows, 78 features).
* [x] Preprocessing and 63-feature selection pipeline implemented.
* [x] Supervised binary intrusion detector (Random Forest, F1 = 0.9991) trained and validated.
* [x] Supervised multiclass attack classifier (Random Forest, 8 classes, Macro F1 = 0.8872) trained.
* [x] Unsupervised anomaly detector (Isolation Forest on 100k benign baseline flows, ROC-AUC = 0.8023) calibrated.
* [x] Phase 4 alert normalization, sliding-window temporal correlation, and target graph clustering completed.
* [x] Phase 5 FastAPI backend and editorial React dashboard integrated.
* [x] Phase 6 genuine machine learning inference engine implemented with `ModelRegistry`, 63-feature schema validation, multi-engine consensus, calibrated presets, and interactive frontend workbench.
* [x] Phase 7 live network monitoring implemented with Zeek 9.0.1, non-blocking log collector, 135 automated tests, and live React dashboard view.
* [x] Final development milestone completed; core architecture frozen.

---

## Vision

> **Detect. Connect. Explain.**  
> Building more accessible cybersecurity intelligence for industries that need it.
