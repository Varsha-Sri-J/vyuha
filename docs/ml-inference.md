# VYUHA Phase 6: Machine Learning Inference Engine

## 1. Architecture Overview

Phase 6 implements genuine on-demand Machine Learning inference within VYUHA, transitioning the platform from precomputed benchmarks and demonstration stubs to real-time execution of trained scikit-learn models.

The Phase 6 inference engine bridges raw network flow feature vectors with the Phase 4 Incident Intelligence engine through a multi-engine consensus architecture:

```
+----------------------------------------------------------------------------------------------------+
|                                      VYUHA API / User Ingestion                                     |
|             POST /api/analyze-flow  |  FlowFeatureVector(s)  |  Presets (Benign/Attack/Novelty)     |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                         FlowInferencePreprocessor (src/models/preprocessor.py)                      |
|  * Strict 63-Feature Schema Validation & IncompleteFlowError (HTTP 422)                            |
|  * Bi-directional Column Alias Resolution (Clean vs Raw CICIDS2017 Headers)                         |
|  * TCP Window Sentinel Preservation (-1 / 65535 intact)                                            |
|  * Infinite Rate & Division-by-Zero Sanitization (clipped to safe max representations)             |
|  * Data-Type Coercion to float64/int64                                                             |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                         ModelRegistry Singleton (src/models/registry.py)                            |
|  * Thread-Safe Lazy Loading & In-Memory Model Caching                                              |
|  * Primary Models: rf_full.joblib, multiclass_rf_full.joblib, iforest_full_100k.joblib              |
|  * Strict Pipeline & Estimator Validation (check_is_fitted, FeatureSelector integrity)             |
|  * Clean Degradation & Diagnostics (ModelNotFoundError, ModelCorruptError)                          |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                             InferenceEngine (src/models/inference.py)                              |
|  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌────────────────────────────────────┐  |
|  │   Engine 1: Binary RF   │  │ Engine 2: Multiclass RF │  │    Engine 3: Isolation Forest      │  |
|  │  - predict_proba()      │  │  - predict_proba()      │  │  - A(x) = -score_samples(x)        │  |
|  │  - P(Attack) >= 0.50    │  │  - 8-Class Distribution │  │  - Calibrated FPR Thresholds:      │  |
|  │  - Binary Decision      │  │  - Confidence Ranking   │  │    1.0%, 5.0%, 0.5%, 0.1%          │  |
|  └────────────┬────────────┘  └────────────┬────────────┘  └─────────────────┬──────────────────┘  |
+───────────────┼────────────────────────────┼─────────────────────────────────┼─────────────────────+
                └────────────────────────────┼─────────────────────────────────┘
                                             ▼
+----------------------------------------------------------------------------------------------------+
|                                    Multi-Engine Consensus Fusion                                    |
|  * CORROBORATED_ATTACK:   Binary Attack (>=0.5) AND Anomaly Score >= Threshold                      |
|  * SUPERVISED_CONFLICT:   Binary Attack (>=0.5) AND Anomaly Score < Threshold                       |
|  * NOVEL_ANOMALY:         Binary Benign (<0.5)  AND Anomaly Score >= Threshold                      |
|  * CORROBORATED_BENIGN:   Binary Benign (<0.5)  AND Anomaly Score < Threshold                       |
+----------------------------------------------------------------------------------------------------+
                                             │
                                             ▼
+----------------------------------------------------------------------------------------------------+
|                                     Phase 4 Incident Intelligence                                  |
|  * Canonical Alert Normalization (AlertItem with whyFlagged, technical, severity)                   |
|  * Temporal Correlation & Target Graph Clustering                                                  |
|  * Candidate Incident Generation (Incident dossiers with confidence & evidence)                    |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Model Inventory & Artifact Provenance

VYUHA utilizes three complementary machine learning artifacts trained during Phases 2, 3, and 4 on the CICIDS2017 benchmark dataset:

| Model ID | Artifact Filename | Estimator Type | Serialization Format | Target Task | Performance Metric |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `binary_rf` | `rf_full.joblib` | `Pipeline(FeatureSelector, RandomForestClassifier)` | Joblib (`Pipeline`) | Binary intrusion detection (Benign vs Attack) | Precision: 0.9998, Recall: 0.9984, F1: 0.9991 |
| `multiclass_rf` | `multiclass_rf_full.joblib` | `Pipeline(FeatureSelector, RandomForestClassifier)` | Joblib (`Pipeline`) | 8-Class threat categorization | Macro F1: 0.8872, Accuracy: 0.9986 |
| `iforest` | `iforest_full_100k.joblib` | `Pipeline(FeatureSelector, IsolationForest)` + Metadata | Joblib (`dict` with pipeline and metadata) | Unsupervised novelty & anomaly detection | Calibrated Thresholds: 1.0% FPR = 0.634684 |

### Artifact Provenance & Checksums
All models reside in `vyuha/models/`. During training, each model was fitted using the exact 63 features selected in Phase 1 (`reports/feature_selection_results.json`):
* `rf_full.joblib` (Binary): 100 estimators, max depth 20, class-balanced weights.
* `multiclass_rf_full.joblib` (Multiclass): 100 estimators, max depth 20, class weights computed via scikit-learn.
* `iforest_full_100k.joblib` (Isolation Forest): 100 estimators, trained on 100,000 pure benign baseline flows to establish the nominal enterprise network profile.

---

## 3. Loading & Registry Lifecycle

The `ModelRegistry` class (`src/models/registry.py`) implements a thread-safe singleton pattern providing controlled access to model artifacts.

### Key Lifecycle Principles
1. **Thread-Safe Lazy Initialization**: Models are loaded on first access or explicitly preloaded during FastAPI application startup via a `threading.Lock`.
2. **Strict Validation**: Upon loading, each pipeline is inspected for:
   * Presence of required pipeline steps (`selector`, `classifier` / `anomaly_detector`).
   * Scikit-learn estimator fitness (`check_is_fitted`).
   * Threshold dictionary presence in Isolation Forest artifacts.
3. **No Mock Fallbacks**: If an artifact is missing or corrupted, `ModelRegistry` raises `ModelNotFoundError` or `ModelCorruptError`. The service maps this to HTTP 503 Service Unavailable rather than silently serving mock predictions.
4. **Fast Warm-Startup**: Models remain resident in RAM across API requests, eliminating disk I/O and unpickling overhead during high-frequency flow evaluation.

---

## 4. Feature Preprocessing Pipeline & Sentinel Handling

The `FlowInferencePreprocessor` class (`src/models/preprocessor.py`) bridges heterogeneous input sources with the strict 63-feature matrix expected by the models.

### Strict Schema Enforcement
The preprocessor enforces the exact 63 selected features. If any required feature is missing from an input record, the preprocessor aborts and raises `IncompleteFlowError` with a detailed list of missing feature names.

### Aliases & Header Normalization
The preprocessor supports both sanitized snake_case names and raw CICIDS2017 column names:
* `destination_port` $\leftrightarrow$ `Destination Port`
* `flow_duration` $\leftrightarrow$ `Flow Duration`
* `total_fwd_packets` $\leftrightarrow$ `Total Fwd Packets`
* `flow_bytes_s` $\leftrightarrow$ `Flow Bytes/s`
* `init_win_bytes_forward` $\leftrightarrow$ `Init_Win_bytes_forward`

### Sentinel & Outlier Safeguards
1. **TCP Window Sentinels**: TCP window sizes of `-1` (indicating non-TCP traffic or unavailable window measurements) and `65535` are preserved as valid domain signals and never imputed with means or zeros.
2. **Division-by-Zero / Infinite Rates**: When flow duration is zero, packet and byte rates (`Flow Bytes/s`, `Flow Packets/s`) become infinite. The preprocessor clamps these to a domain-safe ceiling ($10^9$) to prevent scikit-learn tree splitting errors.
3. **NaN Sanitization**: Any NaN or null numerical attributes are cleanly coerced to zero.

---

## 5. Binary Inference Engine

The binary detection engine executes `rf_full.joblib` to determine whether a flow represents normal network operations or malicious activity.

### Mechanics
1. **Probability Estimation**: Calls `predict_proba(X)`, producing $P(\text{Benign})$ and $P(\text{Attack})$.
2. **Calibrated Decision Boundary**: Uses a standard decision threshold of $\tau = 0.50$:
   $$\text{is\_attack} = P(\text{Attack}) \ge 0.50$$
3. **Confidence Scoring**: Confidence is defined as the distance from maximum uncertainty ($0.5$):
   $$\text{Confidence} = 2 \cdot |P(\text{Attack}) - 0.5| \in [0.0, 1.0]$$

---

## 6. Multiclass Inference Engine

The multiclass engine executes `multiclass_rf_full.joblib` across 8 canonical traffic categories:

1. `BENIGN`: Legitimate client-server network interactions.
2. `PortScan`: Reconnaissance SYN/FIN sweeps and service discovery probes.
3. `DoS_DDoS`: Volumetric denial-of-service floods (Slowloris, Hulk, GoldenEye, LOIC).
4. `BruteForce`: Password brute-force targeting SSH, FTP, or Telnet services.
5. `WebAttack`: SQL injection, cross-site scripting (XSS), and directory traversal attempts.
6. `Botnet`: Command-and-control (C2) beaconing and automated coordination.
7. `Infiltration`: Lateral movement, internal privilege escalation, and network pivots.
8. `Heartbleed`: OpenSSL memory extraction exploits targeting TLS implementations.

### Confidence Ranking
The multiclass engine computes the full probability distribution across all 8 classes. The predicted class is the argmax:
$$C^* = \arg\max_k P(Y = C_k \mid X)$$
$$\text{Confidence} = P(Y = C^* \mid X)$$

---

## 7. Anomaly Scoring & Calibrated Thresholds

The unsupervised engine executes `iforest_full_100k.joblib` to detect statistical shifts, novel exploitation patterns, and zero-day behavior that lack supervised signatures.

### Score Inversion
Scikit-learn's `IsolationForest.score_samples(X)` returns negative values where lower numbers indicate anomalies. VYUHA converts this to an intuitive **Anomaly Score** $A(x) \in [0, 1]$ where higher values indicate higher deviation:
$$A(x) = -\text{score\_samples}(x)$$

### Calibrated False Positive Rate (FPR) Tiers
Isolation Forest decision boundaries are calibrated against nominal validation traffic at four operational sensitivity tiers:

| Tier Key | Description | Calibrated Threshold $\tau$ | False Positive Rate | Recommended Operational Use Case |
| :--- | :--- | :--- | :--- | :--- |
| `fpr_1.0pct` | Standard / Default | `0.634684` | 1.0% | General enterprise manufacturing operations |
| `fpr_5.0pct` | Sensitive | `0.538771` | 5.0% | High-security zones, critical PLC / SCADA segments |
| `fpr_0.50pct` | Conservative | `0.681326` | 0.5% | Noisy office environments, automated backups |
| `fpr_0.10pct` | Strict | `0.712724` | 0.1% | High-throughput perimeter egress gateways |

A flow is flagged as an anomaly if:
$$\text{is\_anomaly} = A(x) \ge \tau_{\text{tier}}$$

> [!IMPORTANT]
> **Data Integrity Rule**: In accordance with VYUHA core principles, an unsupervised anomaly flag alone is NEVER labeled as a confirmed attack (`is_attack=False`, `threat_status="ANOMALY"`). It represents statistical novelty requiring human review, not verified malicious intent.

---

## 8. Multi-Engine Prediction Fusion

The `InferenceEngine` correlates outputs from all three models into a unified consensus verdict:

| Binary Verdict | Multiclass Predicted Family | Isolation Forest ($A(x) \ge \tau$) | Consensus Status | Threat Status | Operational Action |
| :---: | :---: | :---: | :--- | :---: | :--- |
| **Attack** | Malicious Category | **Anomaly** | `CORROBORATED_ATTACK` | `ATTACK` | High-priority alert; correlate immediately into candidate incident. |
| **Attack** | Malicious Category | Normal | `SUPERVISED_CONFLICT` | `ATTACK` | Medium-priority alert; supervised signature matches known attack but flow volume/timing is subtle. |
| Benign | `BENIGN` | **Anomaly** | `NOVEL_ANOMALY` | `ANOMALY` | Review signal; flow does not match known attack patterns but diverges strongly from baseline. |
| Benign | `BENIGN` | Normal | `CORROBORATED_BENIGN` | `BENIGN` | Normal operation; discarded from threat alerting unless verbose logging enabled. |

---

## 9. Alert Generation & Phase 4 Integration

When a flow is evaluated as `ATTACK` or `ANOMALY`, it is passed directly into the Phase 4 Incident Intelligence engine:

1. **Alert Normalization**:
   * Uses `MultiEngineAlertNormalizer` to construct an `AlertItem`.
   * Severity is computed objectively:
     * `High`: High-confidence corroborated attacks ($\ge 85\%$ probability).
     * `Medium`: Moderate-confidence attacks ($50\text{--}84\%$) or high-scoring novel anomalies ($A(x) \ge 0.70$).
     * `Low`: Baseline anomalies ($A(x) < 0.70$).
   * Explanatory text (`whyFlagged` and `technical`) is generated dynamically from the specific model outputs.
2. **Temporal Correlation**:
   * Alert timestamps and target ports are ingested by `AlertCorrelationWindow`.
3. **Graph Construction & Incident Clustering**:
   * The `CandidateIncidentEngine` links related alerts into unified incidents based on port adjacency, attack family consistency, and temporal proximity.

---

## 10. API Request & Response Specifications

### `POST /api/analyze-flow`
Evaluates a batch of network flow feature vectors.

#### Request Schema
```json
{
  "flows": [
    {
      "Destination Port": 22,
      "Flow Duration": 0,
      "Total Fwd Packets": 1,
      "Total Backward Packets": 0,
      "Total Length of Fwd Packets": 0,
      "Total Length of Bwd Packets": 0,
      "Fwd Packet Length Max": 0,
      "Fwd Packet Length Min": 0,
      "Fwd Packet Length Mean": 0.0,
      "Fwd Packet Length Std": 0.0,
      "Bwd Packet Length Max": 0,
      "Bwd Packet Length Min": 0,
      "Bwd Packet Length Mean": 0.0,
      "Bwd Packet Length Std": 0.0,
      "Flow Bytes/s": 0.0,
      "Flow Packets/s": 0.0,
      "Flow IAT Mean": 0.0,
      "Flow IAT Std": 0.0,
      "Flow IAT Max": 0,
      "Flow IAT Min": 0,
      "Fwd IAT Total": 0,
      "Fwd IAT Mean": 0.0,
      "Fwd IAT Std": 0.0,
      "Fwd IAT Max": 0,
      "Fwd IAT Min": 0,
      "Bwd IAT Total": 0,
      "Bwd IAT Mean": 0.0,
      "Bwd IAT Std": 0.0,
      "Bwd IAT Max": 0,
      "Bwd IAT Min": 0,
      "Fwd PSH Flags": 0,
      "Fwd URG Flags": 0,
      "Fwd Header Length": 32,
      "Bwd Header Length": 0,
      "Fwd Packets/s": 0.0,
      "Bwd Packets/s": 0.0,
      "Min Packet Length": 0,
      "Max Packet Length": 0,
      "Packet Length Mean": 0.0,
      "Packet Length Std": 0.0,
      "Packet Length Variance": 0.0,
      "FIN Flag Count": 0,
      "SYN Flag Count": 1,
      "RST Flag Count": 0,
      "PSH Flag Count": 0,
      "ACK Flag Count": 0,
      "URG Flag Count": 0,
      "CWE Flag Count": 0,
      "ECE Flag Count": 0,
      "Down/Up Ratio": 0,
      "Average Packet Size": 0.0,
      "Avg Fwd Segment Size": 0.0,
      "Avg Bwd Segment Size": 0.0,
      "Subflow Fwd Packets": 1,
      "Subflow Fwd Bytes": 0,
      "Subflow Bwd Packets": 0,
      "Subflow Bwd Bytes": 0,
      "Init_Win_bytes_forward": 29200,
      "Init_Win_bytes_backward": -1,
      "act_data_pkt_fwd": 0,
      "min_seg_size_forward": 32,
      "Active Mean": 0.0,
      "Active Std": 0.0,
      "Active Max": 0,
      "Active Min": 0,
      "Idle Mean": 0.0,
      "Idle Std": 0.0,
      "Idle Max": 0,
      "Idle Min": 0
    }
  ],
  "alert_on_threat_only": false,
  "anomaly_threshold_tier": "fpr_1.0pct"
}
```

#### Response Schema
```json
{
  "alerts": [
    {
      "id": "ALT-20261119-0001",
      "title": "PortScan probe activity flagged",
      "category": "PortScan",
      "source": "Flow Evaluation Engine",
      "severity": "High",
      "summary": "Stealth SYN probe targeting SSH (Port 22) confirmed by Random Forest and Isolation Forest.",
      "time": "Just now",
      "detail": "Flow exhibited single SYN probe with zero duration and inactive backward window.",
      "evidence": "Destination Port 22, SYN Flag Count 1, Backward Window -1",
      "whyFlagged": "Random Forest detected PortScan with 98.4% probability; Isolation Forest confirmed score 0.6521 >= 0.6347 threshold.",
      "technical": "Model consensus: CORROBORATED_ATTACK across binary and anomaly engines.",
      "threat_status": "ATTACK",
      "flow_id": "flow_0",
      "destination_port": 22,
      "confidence": 0.984,
      "anomaly_score": 0.6521,
      "is_attack": true
    }
  ],
  "incidents": [
    {
      "id": "INC-001",
      "title": "PortScan Reconnaissance Sweep",
      "summary": "Correlated probe activity targeting port 22.",
      "severity": "High",
      "alerts": 1,
      "categories": ["PortScan"],
      "status": "Review",
      "updated": "Just now",
      "engines": ["Binary RF", "Multiclass RF", "Isolation Forest"],
      "relationship": "Single-host port sweep targeting standard SSH services.",
      "timeline": [],
      "nextSteps": ["Verify firewall ingress rules for port 22.", "Check SSH authentication logs."],
      "confidence": "Strong",
      "evidence": []
    }
  ],
  "evaluated_flows_count": 1,
  "threats_count": 1,
  "evaluations": [
    {
      "flow_id": "flow_0",
      "destination_port": 22,
      "binary": {
        "is_attack": true,
        "attack_probability": 0.984,
        "confidence": 0.968,
        "decision_threshold": 0.50
      },
      "multiclass": {
        "predicted_category": "PortScan",
        "category_probabilities": {
          "BENIGN": 0.016,
          "PortScan": 0.984,
          "DoS_DDoS": 0.0,
          "BruteForce": 0.0,
          "WebAttack": 0.0,
          "Botnet": 0.0,
          "Infiltration": 0.0,
          "Heartbleed": 0.0
        },
        "confidence": 0.984
      },
      "anomaly": {
        "anomaly_score": 0.6521,
        "threshold": 0.634684,
        "threshold_tier": "fpr_1.0pct",
        "is_anomaly": true,
        "calibrated_thresholds": {
          "fpr_5.0pct": 0.538771,
          "fpr_1.0pct": 0.634684,
          "fpr_0.50pct": 0.681326,
          "fpr_0.10pct": 0.712724
        }
      },
      "consensus_status": "CORROBORATED_ATTACK",
      "threat_status": "ATTACK",
      "is_attack": true,
      "flow_summary": "Supervised Attack (PortScan, 98.4%) corroborated by Anomaly Outlier (score 0.6521)"
    }
  ],
  "models_used": [
    "models/rf_full.joblib",
    "models/multiclass_rf_full.joblib",
    "models/iforest_full_100k.joblib"
  ],
  "anomaly_threshold_tier": "fpr_1.0pct"
}
```

### `GET /api/analyze-flow/presets`
Returns four verified synthetic 63-feature flow vectors (`benign`, `portscan`, `dos`, `anomaly`) for interactive inference testing.

---

## 11. Error Handling & Validation

VYUHA provides strict, diagnostic HTTP error handling:

| Error Class | HTTP Status | Response Payload | Trigger Condition |
| :--- | :---: | :--- | :--- |
| `IncompleteFlowError` | `422 Unprocessable Entity` | `{"error": "IncompleteFlowError", "message": "...", "record_index": 0, "missing_features": [...]}` | Input flow is missing one or more of the 63 required features. |
| `InvalidFeatureValueError` | `422 Unprocessable Entity` | `{"error": "InvalidFeatureValueError", "message": "...", "record_index": 0, "feature": "..."}` | Feature contains non-numerical string or invalid format. |
| `ModelNotFoundError` | `503 Service Unavailable` | `{"error": "ModelLoadError", "message": "Trained ML models could not be loaded: ..."}` | Required model `.joblib` artifact is absent from disk. |
| `ModelCorruptError` | `503 Service Unavailable` | `{"error": "ModelLoadError", "message": "..."}` | Artifact is unreadable or fails scikit-learn fitness check. |

---

## 12. Local Execution Guide

### Running Programmatic Inference via Python
```python
import sys
sys.path.insert(0, "/Users/varshasri/VYUHA/vyuha")

from src.models.registry import ModelRegistry
from src.models.preprocessor import FlowInferencePreprocessor, get_sample_flow_preset
from src.models.inference import InferenceEngine

# 1. Load registry and inference engine
registry = ModelRegistry.get_instance()
engine = InferenceEngine(registry=registry)

# 2. Obtain flow and preprocess
raw_flow = get_sample_flow_preset("portscan")
preprocessor = FlowInferencePreprocessor()
X = preprocessor.transform([raw_flow])

# 3. Execute multi-engine inference
evaluations = engine.evaluate(X, anomaly_threshold_tier="fpr_1.0pct")
ev = evaluations[0]

print(f"Consensus Status: {ev.consensus_status}")
print(f"Binary Attack Prob: {ev.binary.attack_probability:.4f}")
print(f"Multiclass Category: {ev.multiclass.predicted_category} ({ev.multiclass.confidence:.4f})")
print(f"Anomaly Score: {ev.anomaly.anomaly_score:.4f} (Threshold: {ev.anomaly.threshold:.4f})")
```

### Running Test Suite
Execute the 117 tests covering unit, integration, and API validation:
```bash
cd /Users/varshasri/VYUHA/vyuha
pytest tests/test_ml_inference.py tests/test_api.py -v
```

---

## 13. Known Limitations & Data Integrity Notes

1. **Absence of Host Identifiers in Benchmark Data**: The CICIDS2017 `MachineLearningCVE` dataset does not provide source IP addresses, destination IP addresses, source ports, timestamps, or protocol numbers. VYUHA strictly preserves this constraint and NEVER fabricates synthetic IP addresses or false timestamps.
2. **Offline Flow Feature Reliance**: All 63 features are flow-level statistical metrics (packet lengths, inter-arrival times, window sizes). Deep packet inspection (payload inspection) is not performed.
3. **Unsupervised Novelty Disambiguation**: An anomaly detected by Isolation Forest is an indicator of statistical distribution divergence, which can be caused by legitimate network changes (e.g., firmware updates, new hardware) as well as zero-day attacks. Human review is always preserved.

---

## 14. Zeek Integration Roadmap (Phase 7 Bridge)

Phase 7 will bridge VYUHA with live enterprise network taps using [Zeek Network Security Monitor](https://zeek.org/).

### Zeek Log to VYUHA Feature Mapping Architecture

```
+----------------------------------------------------------------------------------------------------+
|                                      Enterprise Network Tap / SPAN                                 |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                    Zeek Network Security Monitor                                   |
|   conn.log (Flow metadata) | http.log (Web traffic) | ssl.log (TLS handshakes) | dns.log (Queries) |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                VYUHA Zeek Flow Extractor (Phase 7)                                  |
|  * Stream parsing via Zeek JSON log exporter or Kafka/Redis broker                                 |
|  * Bi-directional Flow Aggregation (Fwd/Bwd packets, bytes, durations)                              |
|  * Statistical Accumulator: Mean, Std, Min, Max packet lengths & IATs                              |
|  * TCP Header Flag Parser: SYN, ACK, FIN, RST, PSH, URG, CWE, ECE counts                            |
|  * Extraction of genuine Source & Destination IPs, Ports, Protocols, and Real Wall-Clock Timestamps |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼ (Mapped 63-Feature Vector + Real Metadata)
+----------------------------------------------------------------------------------------------------+
|                           VYUHA Phase 6 Genuine Inference Engine & Dashboard                       |
|   POST /api/analyze-flow  -->  Real-Time Detection  -->  Incident Intelligence Dossier             |
+----------------------------------------------------------------------------------------------------+
```

### Primary Zeek Field Mappings
* `conn.log:duration` $\rightarrow$ `Flow Duration` (converted to microseconds)
* `conn.log:orig_bytes` $\rightarrow$ `Total Length of Fwd Packets`
* `conn.log:resp_bytes` $\rightarrow$ `Total Length of Bwd Packets`
* `conn.log:orig_pkts` $\rightarrow$ `Total Fwd Packets`
* `conn.log:resp_pkts` $\rightarrow$ `Total Backward Packets`
* `conn.log:id.resp_p` $\rightarrow$ `Destination Port`
* Real Zeek metadata (`id.orig_h`, `id.resp_h`, `id.orig_p`, `proto`, `ts`) will populate alert headers, replacing the offline benchmark placeholder notices with genuine enterprise telemetry.
