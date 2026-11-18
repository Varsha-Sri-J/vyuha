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
| **Phase 1** | Project setup and environment verification | Completed |
| **Phase 2** | CICIDS2017 dataset inspection and audit | Completed |
| **Phase 3** | Data preprocessing and feature engineering | Planned |
| **Phase 4** | Machine learning detection engine | Planned |
| **Phase 5** | Network behaviour and traffic analysis | Planned |
| **Phase 6** | Alert generation and correlation | Planned |
| **Phase 7** | Campaign intelligence and explainability | Planned |
| **Phase 8** | SOC dashboard development | Planned |
| **Phase 9** | Dataset replay and demonstration | Planned |

---

## Project Structure

```text
vyuha/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── analysis/
│   ├── correlation/
│   ├── explainability/
│   └── api/
│
├── tests/
│
├── docs/
│
└── frontend/
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

* [x] Project workspace established.
* [x] GitHub repository initialized.
* [x] Python virtual environment configured.
* [x] CICIDS2017 dataset verified and inspected.
* [x] Empirical dataset intelligence & quality audit completed (2.83M rows, 78 features).
* [ ] Preprocessing and machine learning implementation (Upcoming Milestone).

---

## Vision

> **Detect. Connect. Explain.**  
> Building more accessible cybersecurity intelligence for industries that need it.
