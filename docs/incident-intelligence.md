# VYUHA Incident Intelligence: Alert Normalisation and Entity Extraction (Phase 4.1)

## 1. Executive Summary

Phase 4.1 establishes the foundational data architecture for **VYUHA Incident Intelligence**. Prior phases engineered discrete detection engines:
* **Phase 3B**: Supervised Binary Classification (BENIGN vs ATTACK).
* **Phase 3C**: Supervised Multiclass Attack-Family Classification (8 canonical categories).
* **Phase 3D**: Unsupervised Anomaly Detection (Novelty isolation trees fitted strictly on benign flows).

In isolation, raw model predictions cannot drive security operations center (SOC) triage or automated defense. Each engine produces distinct inference primitives (class probabilities, categorical strings, continuous tree-depth anomaly scores). Furthermore, downstream incident grouping and campaign intelligence require structured network entities (IP addresses, ports, timestamps, flow identifiers) rather than tabular feature vectors.

Phase 4.1 introduces:
1. **Common Alert Schema**: A canonical, serializable representation capturing detections from all three engines with unambiguous operational threat semantics.
2. **Explicit Threat Status Semantics**: Clean separation of `threat_status` (`BENIGN`, `ATTACK`, `ANOMALY`, `UNKNOWN`) from `is_attack`.
3. **Entity Extraction & Validation Engine**: Strict parser and validator for IPv4/IPv6 addresses, port bounds $[0, 65535]$, ISO-8601 timestamps, and flow identifiers.
4. **Flow Identifier Integrity**: Deterministic positional identification preventing the accidental merger of statistically identical flows.
5. **Engine-Specific Normalization Adapters**: Dedicated adapters translating model outputs into standardized alerts without fabricating unobserved probabilities.
6. **Multi-Engine Consolidation & Negative Assessment Preservation**: Consolidation mechanism that eliminates duplicate alert generation for identical flows while strictly preserving engine-specific evidence and non-flagging evaluations.

---

## 2. Architecture Decisions

### 2.1 Threat Status Semantics
A key defect in traditional intrusion schemas is conflating anomalies with confirmed attacks or benign traffic.
* **Statistical Novelty vs. Confirmed Attack**: An unsupervised anomaly is an out-of-distribution pattern relative to normal traffic; it is **not** an established malicious attack. Flagging an anomaly as `is_attack=True` creates false positives and misleads analysts. Conversely, setting `is_attack=False` without context can cause it to be confused with benign traffic.
* **Explicit `threat_status`**: VYUHA introduces four explicit states:
  * `BENIGN`: Verified or predicted normal traffic.
  * `ATTACK`: Confirmed malicious threat identified by supervised detection.
  * `ANOMALY`: Statistical distribution novelty identified by unsupervised isolation.
  * `UNKNOWN`: Missing, uninitialized, or unclassifiable traffic.
* **Exact Meaning of `is_attack`**: `is_attack: bool` is strictly a convenience flag indicating whether the flow is a confirmed supervised attack (`threat_status == ThreatStatus.ATTACK`). For `threat_status == ThreatStatus.ANOMALY`, `is_attack` is strictly `False`. Schema-level assertions enforce this invariant.

### 2.2 Probabilistic Integrity: No Fabricated Confidences
Inferences must never pretend to possess statistical properties they lack:
* **Tree Classifiers with Probabilities** (`RandomForestClassifier`, `DecisionTreeClassifier`): Provide calibrated posterior probabilities via `predict_proba()`. The adapter extracts class probability as `confidence \in [0.0, 1.0]`.
* **Models without Probabilities**: If a model does not support class probability estimation, `confidence` is explicitly set to `None`. It is never fabricated.
* **Unsupervised Anomaly Detection** (`IsolationForest`): Isolation Forest computes an anomaly score $A(x) = -\text{score\_samples}(x)$ based on average tree path length. Path length is a non-probabilistic ranking metric. Consequently, anomaly alerts set `confidence = None` and record `anomaly_score` and `calibrated_threshold` directly.

### 2.3 Flow Identifier Integrity on Statistically Identical Flows
Network traffic naturally generates identical feature vectors (e.g. repeated ping packets, periodic health-check beacons, or mass SYN probes hitting port 80 with duration 0 and 1 packet).
* **The Vulnerability of Feature Hashing**: If a system generates flow IDs by hashing non-target feature values, distinct packets with identical features generate the exact same hash. Downstream consolidation would silently collapse distinct physical network events into a single alert!
* **VYUHA's Deterministic Identifier Strategy**:
  1. *Native Flow ID Priority*: If a native flow ID exists (`Flow ID`), it is used directly.
  2. *Positional Row Index*: In datasets lacking native IDs (such as CICIDS2017 `MachineLearningCVE`), VYUHA uses the dataset row/sequence index `f"flow_{index}"` (or `f"flow_{series.name}"`). Because row $i$ is stable across repeated runs and shared across all adapters, all engines produce the exact same `flow_id` for row $i$, while row $i$ and row $j$ remain strictly distinct even if their feature vectors are 100% identical.
  3. *Stateless Hashing Limitation*: If neither native flow ID nor row index is provided, hash-based identification is a lossy fallback that cannot distinguish identical flows. Callers must provide an index or native ID for strict uniqueness.

### 2.4 Multi-Engine Consolidation & Negative Assessment Preservation
When an enterprise pipeline deploys binary, multiclass, and anomaly detection concurrently against incoming flows, naive alerting would emit up to three separate alerts for a single flow, causing severe alert fatigue.

Conversely, naive deduplication that only merges alerts when an engine flags a threat can discard critical diagnostic context. If Engine A flags ATTACK, but Engine B evaluates the flow and predicts BENIGN:
* VYUHA ensures that **engines that do not flag a flow are not represented as missing**.
* In `MultiEngineAlertNormalizer`, all active engines record their evaluations in `evidence['engines']`.
* Non-flagging engines are explicitly marked with `flagged_as_threat: False` and their predictions (`BENIGN` or `NORMAL`) and confidences are preserved.
* Triage analysts can inspect `consensus_summary` to see which engines flagged the threat and which engines evaluated it as normal.

### 2.5 Provisional Severity Semantics
The `severity` field (`INFO`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) at Phase 4.1 is **strictly provisional**. A standalone model prediction on a single flow cannot establish real-world incident priority. True incident severity requires host asset criticality (e.g., domain controller vs test lab), temporal frequency (single flow probe vs sustained saturation), blast radius, and multi-host graph correlation. These capabilities are developed in subsequent Phase 4 milestones.

---

## 3. Common Alert Schema Specification

The canonical schema is implemented as a Python dataclass (`src.intelligence.schema.Alert`) supporting bidirectional JSON serialization.

### 3.1 Field Definitions

| Field Name | Type | Nullable | Description |
| :--- | :--- | :---: | :--- |
| `alert_id` | `str` | No | Unique alert identifier (e.g. `alt_bin_...`, `alt_multi_...`). |
| `flow_id` | `str` | No | Persistent, unique reference to the originating flow record. |
| `detection_engine` | `str` | No | Generating engine: `binary`, `multiclass`, `anomaly`, or `multi_engine`. |
| `model_name` | `str` | No | Specific model artifact identifier (e.g. `rf_full`, `iforest_port_100k`). |
| `prediction` | `Union[str, int, bool]` | No | Primary prediction (`ATTACK`, `BENIGN`, category string, or `ANOMALOUS`). |
| `threat_status` | `str` | No | Operational threat status: `BENIGN`, `ATTACK`, `ANOMALY`, or `UNKNOWN`. |
| `is_attack` | `bool` | No | True strictly for confirmed supervised attacks; strictly False for anomalies. |
| `attack_category` | `Optional[str]` | Yes | Standardized threat family (e.g. `DoS_DDoS`, `Botnet`); `None` for benign/anomaly. |
| `source_ip` | `Optional[str]` | Yes | Validated IPv4 or IPv6 source address; `None` if missing. |
| `destination_ip` | `Optional[str]` | Yes | Validated IPv4 or IPv6 destination address; `None` if missing. |
| `source_port` | `Optional[int]` | Yes | Source transport port $[0, 65535]$; `None` if missing. |
| `destination_port` | `Optional[int]` | Yes | Destination transport port $[0, 65535]$; `None` if missing. |
| `protocol` | `Optional[str]` | Yes | Transport protocol name (`TCP`, `UDP`, `ICMP`); `None` if missing. |
| `timestamp` | `Optional[str]` | Yes | Standardized ISO-8601 UTC timestamp; `None` if missing. |
| `flow_duration` | `Optional[float]` | Yes | Flow duration in microseconds. |
| `anomaly_score` | `Optional[float]` | Yes | Continuous Isolation Forest anomaly score $A(x) \in [0.0, 1.0]$. |
| `anomaly_threshold`| `Optional[float]` | Yes | Calibrated decision threshold $\tau_\alpha$ used to flag the flow. |
| `confidence` | `Optional[float]` | Yes | Model posterior class probability $\in [0.0, 1.0]$; `None` if non-probabilistic. |
| `severity` | `str` | No | Provisional triage severity level: `INFO`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`. |
| `evidence` | `Dict[str, Any]` | No | Engine-specific diagnostic evidence, probabilities, and thresholds. |
| `metadata` | `Dict[str, Any]` | No | Pipeline metadata, consolidation provenance, and validation logs. |

### 3.2 Provisional Severity Hierarchy

```
CRITICAL (Rank 4): Severe destructive threats (Heartbleed, Infiltration)
HIGH     (Rank 3): High-volume / automated threats (DoS_DDoS, Botnet), extreme anomalies (A(x) >= 0.70), high-confidence binary attacks (P >= 0.90)
MEDIUM   (Rank 2): Targeted attacks (WebAttack, BruteForce), clear statistical anomalies (A(x) >= 0.64), moderate-confidence binary attacks
LOW      (Rank 1): Network reconnaissance (PortScan), marginal anomalies (A(x) >= tau_alpha)
INFO     (Rank 0): Normal baseline traffic (BENIGN flows)
```

---

## 4. Detection Engine Adapters

### 4.1 Binary Alert Adapter (`BinaryAlertAdapter`)
* **Source**: Phase 3B Random Forest and Decision Tree models (`rf_full`, `rf_port_agnostic`).
* **Threat Status**: `ATTACK` if prediction is 1; `BENIGN` if prediction is 0.
* **Confidence Handling**: Maps $P(\hat{y}|x)$ to `confidence` and records $P(\text{ATTACK}|x)$ in `evidence['attack_probability']`.
* **Provisional Severity**: `HIGH` ($P \ge 0.90$), `MEDIUM` ($P \ge 0.70$), `LOW` ($P < 0.70$), `INFO` (Benign).

### 4.2 Multiclass Alert Adapter (`MulticlassAlertAdapter`)
* **Source**: Phase 3C Random Forest and Decision Tree models (`multiclass_rf_full`, etc.).
* **Threat Status**: `ATTACK` for non-BENIGN categories; `BENIGN` for BENIGN.
* **Confidence Handling**: Maps class probability of predicted category to `confidence`. Preserves full distribution in `evidence['class_probabilities']`.
* **Provisional Severity**: Mapped by attack category (`Heartbleed` $\to$ `CRITICAL`, `DoS_DDoS` $\to$ `HIGH`, etc.).

### 4.3 Anomaly Alert Adapter (`AnomalyAlertAdapter`)
* **Source**: Phase 3D Isolation Forest novelty models (`iforest_port_100k`, etc.).
* **Threat Status**: `ANOMALY` if $A(x) \ge \tau_\alpha$; `BENIGN` if $A(x) < \tau_\alpha$.
* **Fidelity Constraints**:
  * `is_attack = False` (statistical novelty is not a confirmed attack).
  * `attack_category = None` (unsupervised engine has no threat labels).
  * `confidence = None` (non-probabilistic tree isolation score).
* **Evidence**: Records `anomaly_score`, `calibrated_threshold`, `threshold_name`, and `threshold_margin` ($A(x) - \tau_\alpha$).

---

## 5. Entity Extraction & Data Limitations

### 5.1 Entity Validation Rules
The `EntityExtractor` applies strict validation across network flow attributes:
1. **IP Addresses**: Validated via Python `ipaddress.ip_address`. Supports full IPv4 dotted-decimal and IPv6 hex formats. Rejects malformed strings (e.g. `999.999.999.999`).
2. **Ports**: Validated within integer range $[0, 65535]$. Safely converts floating-point port representations from parquet (e.g. `80.0` $\to$ `80`) while rejecting fractional values (e.g. `80.5`).
3. **Timestamps**: Parsed into standardized ISO-8601 strings (`YYYY-MM-DDTHH:MM:SS`).
4. **Protocols**: Mapped from explicit IANA numeric values ($6 \to$ TCP, $17 \to$ UDP, $1 \to$ ICMP) or validated string tokens.

### 5.2 Dataset Limitations in CICIDS2017 MachineLearningCVE
An exhaustive audit of the 8 canonical CSV files in CICIDS2017 `MachineLearningCVE` confirms key architectural limitations:
* **Missing Entity Headers**: The CSV exports generated by CICFlowMeter omit `Source IP`, `Destination IP`, `Source Port`, and `Timestamp`. Only `Destination Port` and statistical flow features are present.
* **No Numeric Protocol Column**: The dataset lacks an explicit numeric protocol column. While presence of TCP window parameters (`Init_Win_bytes_forward`) or TCP flags implies TCP, these are indirect heuristics. **VYUHA strictly refuses to claim definitive TCP/UDP protocol identification from heuristics**. The `protocol` field remains `None` unless an explicit protocol column is supplied.
* **Flow Identification**: Because `Flow ID` is omitted in MachineLearningCVE, VYUHA generates deterministic positional flow identifiers based on row index (`flow_{index}`), ensuring identical feature vectors at distinct positions are never merged.

---

## 6. Example Normalized Alerts

### 6.1 Supervised Binary Alert (High Confidence Attack)
```json
{
  "alert_id": "alt_bin_4a7f9c1e2b3d",
  "flow_id": "flow_42901",
  "detection_engine": "binary",
  "model_name": "rf_full",
  "prediction": "ATTACK",
  "threat_status": "ATTACK",
  "is_attack": true,
  "attack_category": "ATTACK",
  "source_ip": "192.168.10.50",
  "destination_ip": "172.16.0.1",
  "source_port": 51234,
  "destination_port": 80,
  "protocol": "TCP",
  "timestamp": "2026-11-19T00:40:15",
  "flow_duration": 45200.0,
  "anomaly_score": null,
  "anomaly_threshold": null,
  "confidence": 0.985,
  "severity": "HIGH",
  "evidence": {
    "binary_prediction": "ATTACK",
    "raw_prediction": 1,
    "threat_status": "ATTACK",
    "confidence": 0.985,
    "attack_probability": 0.985,
    "validation_errors": {}
  },
  "metadata": {
    "engine_type": "supervised_binary"
  }
}
```

### 6.2 Supervised Multiclass Alert (DoS_DDoS Attack)
```json
{
  "alert_id": "alt_mc_7e8b2a1c9f0d",
  "flow_id": "flow_88124",
  "detection_engine": "multiclass",
  "model_name": "multiclass_rf_full",
  "prediction": "DoS_DDoS",
  "threat_status": "ATTACK",
  "is_attack": true,
  "attack_category": "DoS_DDoS",
  "source_ip": "192.168.10.52",
  "destination_ip": "172.16.0.1",
  "source_port": 49812,
  "destination_port": 80,
  "protocol": "TCP",
  "timestamp": "2026-11-19T00:41:00",
  "flow_duration": 1250000.0,
  "anomaly_score": null,
  "anomaly_threshold": null,
  "confidence": 0.942,
  "severity": "HIGH",
  "evidence": {
    "predicted_category": "DoS_DDoS",
    "threat_status": "ATTACK",
    "is_attack": true,
    "confidence": 0.942,
    "class_probabilities": {
      "BENIGN": 0.005,
      "DoS_DDoS": 0.942,
      "PortScan": 0.041,
      "BruteForce": 0.008,
      "WebAttack": 0.002,
      "Botnet": 0.001,
      "Infiltration": 0.001,
      "Heartbleed": 0.0
    },
    "validation_errors": {}
  },
  "metadata": {
    "engine_type": "supervised_multiclass"
  }
}
```

### 6.3 Unsupervised Anomaly Alert (Statistical Novelty)
```json
{
  "alert_id": "alt_ano_9c3d1f7e4a2b",
  "flow_id": "flow_10452",
  "detection_engine": "anomaly",
  "model_name": "iforest_port_100k",
  "prediction": "ANOMALOUS",
  "threat_status": "ANOMALY",
  "is_attack": false,
  "attack_category": null,
  "source_ip": null,
  "destination_ip": null,
  "source_port": null,
  "destination_port": 443,
  "protocol": null,
  "timestamp": null,
  "flow_duration": 3400.0,
  "anomaly_score": 0.6782,
  "anomaly_threshold": 0.643,
  "confidence": null,
  "severity": "MEDIUM",
  "evidence": {
    "anomaly_score": 0.6782,
    "calibrated_threshold": 0.643,
    "threshold_name": "fpr_1.0pct",
    "threshold_margin": 0.0352,
    "threat_status": "ANOMALY",
    "is_anomaly": true,
    "assessment": "Statistical novelty detected; unconfirmed as malicious attack",
    "validation_errors": {}
  },
  "metadata": {
    "engine_type": "unsupervised_anomaly",
    "calibration_subset_size": 50000
  }
}
```

### 6.4 Consolidated Multi-Engine Alert with Negative Assessment Preservation
```json
{
  "alert_id": "alt_multi_3f1a8c9e4b7d",
  "flow_id": "flow_88124",
  "detection_engine": "multi_engine",
  "model_name": "iforest_port_100k+multiclass_rf_full+rf_full",
  "prediction": "DoS_DDoS",
  "threat_status": "ATTACK",
  "is_attack": true,
  "attack_category": "DoS_DDoS",
  "source_ip": "192.168.10.52",
  "destination_ip": "172.16.0.1",
  "source_port": 49812,
  "destination_port": 80,
  "protocol": "TCP",
  "timestamp": "2026-11-19T00:41:00",
  "flow_duration": 1250000.0,
  "anomaly_score": 0.6845,
  "anomaly_threshold": 0.643,
  "confidence": 0.942,
  "severity": "HIGH",
  "evidence": {
    "participating_engines": ["binary", "multiclass", "anomaly"],
    "engine_count": 3,
    "engines": {
      "binary": {
        "alert_id": "alt_bin_4a7f9c1e2b3d",
        "model_name": "rf_full",
        "prediction": "ATTACK",
        "threat_status": "ATTACK",
        "flagged_as_threat": true,
        "is_attack": true,
        "confidence": 0.985,
        "anomaly_score": null,
        "anomaly_threshold": null,
        "severity": "HIGH",
        "evidence": {
          "binary_prediction": "ATTACK",
          "raw_prediction": 1,
          "confidence": 0.985,
          "attack_probability": 0.985
        }
      },
      "multiclass": {
        "alert_id": "alt_mc_7e8b2a1c9f0d",
        "model_name": "multiclass_rf_full",
        "prediction": "DoS_DDoS",
        "threat_status": "ATTACK",
        "flagged_as_threat": true,
        "is_attack": true,
        "confidence": 0.942,
        "anomaly_score": null,
        "anomaly_threshold": null,
        "severity": "HIGH",
        "evidence": {
          "predicted_category": "DoS_DDoS",
          "is_attack": true,
          "confidence": 0.942
        }
      },
      "anomaly": {
        "alert_id": "alt_ano_9c3d1f7e4a2b",
        "model_name": "iforest_port_100k",
        "prediction": "NORMAL",
        "threat_status": "BENIGN",
        "flagged_as_threat": false,
        "is_attack": false,
        "confidence": null,
        "anomaly_score": 0.421,
        "anomaly_threshold": 0.643,
        "severity": "INFO",
        "evidence": {
          "anomaly_score": 0.421,
          "calibrated_threshold": 0.643,
          "threshold_name": "fpr_1.0pct",
          "threshold_margin": -0.222,
          "is_anomaly": false
        }
      }
    },
    "consensus_summary": {
      "threat_status": "ATTACK",
      "engines_flagging_threat": ["binary", "multiclass"],
      "engines_not_flagging": ["anomaly"],
      "supervised_attack_detected": true,
      "anomaly_detected": false,
      "all_engines_agreed_attack": true
    }
  },
  "metadata": {
    "consolidated_from_alerts": [
      "alt_bin_4a7f9c1e2b3d",
      "alt_mc_7e8b2a1c9f0d",
      "alt_ano_9c3d1f7e4a2b"
    ],
    "engine_count": 3,
    "severity_semantics": "provisional_flow_heuristic"
  }
}
```

---

## 7. Known Limitations & Forward Roadmap

1. **Temporal Context in Offline MachineLearningCVE**:
   Because `MachineLearningCVE` does not bundle packet-level timestamps, temporal correlation cannot reconstruct microburst sequences in historical benchmarking. In live deployment, network sensors (Zeek, Suricata) supply nanosecond timestamps, enabling time-window aggregation.
2. **Flow-Level vs. Campaign-Level Scope**:
   Phase 4.1 standardizes discrete single-flow alerts. A sophisticated APT campaign spans hundreds of coordinated flows across multiple hosts. Milestone 4.2 will introduce **Temporal Correlation & Relationship Graphs** to cluster related alerts into unified incidents.
3. **Transport Protocol Explicitness**:
   As established, heuristic protocol tagging is disabled by default to maintain data integrity. Future integrations with raw PCAP collectors will ingest explicit IP protocol numbers.
