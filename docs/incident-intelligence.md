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
   Phase 4.1 standardizes discrete single-flow alerts, and Phase 4.2 establishes temporal correlation. A sophisticated APT campaign spans multiple hosts, protocols, and days. Subsequent milestones will introduce **Relationship Graphs** and **Incident Clustering** to aggregate related alerts across entities and topology into unified campaign dossiers.
3. **Transport Protocol Explicitness**:
   As established, heuristic protocol tagging is disabled by default to maintain data integrity. Future integrations with raw PCAP collectors will ingest explicit IP protocol numbers.

---

## 8. Temporal Correlation Engine (Phase 4.2)

### 8.1 Architecture & Objectives
Phase 4.2 implements the **Temporal Correlation Engine** (`src.intelligence.temporal.TemporalCorrelationEngine`), which identifies temporal proximity, repeated detection patterns, and chronological progression stages across normalized `Alert` streams.

```
       Normalized Alerts Stream (Alert objects)
                          │
                          ▼
            [ Timestamp Normalizer ]
             - UTC timezone normalization
             - Missing/Invalid check
                 ├── Ineligible Alerts ──► [ IneligibleAlertRecord ]
                 │                          (reason: missing_timestamp)
                 ▼
         Chronological Sorting (t_0 <= t_1 <= ... <= t_M-1)
                          │
                          ▼
        [ Sliding-Window Pairwise Evaluator ] (dt <= W)
         ├── TemporalProximityRule
         ├── RepeatedAlertRule
         ├── RepeatedCategoryRule
         └── SequentialOrderingRule
                          │
                          ▼
             [ Temporal Cluster Former ]
              (gap > max_cluster_gap => new cluster)
                          │
                          ▼
        [ TemporalCorrelationResult ]
         - Ordered Alerts
         - Discovered TemporalRelationships
         - Formed TemporalClusters
         - Full Ineligibility Audit
```

### 8.2 Strict Cybersecurity Semantics & Causation Guarantees
1. **Candidate Attack-Pattern Sequences vs. Confirmed Kill Chains**:
   Sequential progressions discovered by `SequentialOrderingRule` are strictly categorized as **candidate attack-pattern hypotheses**, NOT confirmed multi-stage kill chains or coordinated campaigns.
   - Temporal precedence ($t_A < t_B$) is a necessary but entirely insufficient condition for causal execution.
   - An adversary performing reconnaissance (`PortScan`) followed by another automated host attempting brute force (`BruteForce`) against the same subnet does not constitute proof of identical attacker attribution without cryptographic, payload, or command-and-control corroboration.
   - Every sequential relationship explicitly preserves a non-causation caveat, earlier/later alert identifiers, individual categories, elapsed time $\Delta t$, observation window $W$, shared entity evidence, and missing entity information.
2. **Temporal Proximity vs. Attack Causation**:
   Events occurring close in time are circumstantial co-occurrences. The engine differentiates:
   - `time_only_proximity`: Alerts co-occur within window $W$ but share no source or destination endpoints.
   - `shared_source_ip`: Alerts share the identical source IP address.
   - `shared_destination_ip`: Alerts target the same destination IP address.
   - `shared_network_entity`: Alerts target the identical destination host and service port, or share multiple endpoints.
   - **Entity Absence Guarantee**: Missing entity values (`None`) **never** count as shared entities. A flow lacking IP metadata evaluated against another flow lacking IP metadata will never evaluate `None == None` as a shared endpoint; both are tracked explicitly under `missing_entities`.
3. **Anomalies and Attacks Remain Distinct**:
   Temporal correlation preserves the `threat_status` of each constituent alert. An unsupervised `ANOMALY` occurring within window $W$ of a supervised `ATTACK` is documented as a temporal co-occurrence, but is **never** converted into a confirmed attack without entity or graph corroboration.
4. **Dataset Limitation Disclosure (CICIDS2017 MachineLearningCVE)**:
   The benchmark dataset `MachineLearningCVE` (CICIDS2017) does not contain flow timestamp columns.
   - Executing `TemporalCorrelationEngine.correlate()` on raw alerts extracted from `MachineLearningCVE` will mark 100% of alerts as ineligible with `reason="missing_timestamp"`.
   - VYUHA strictly refuses to fabricate synthetic timestamps, interpolate timestamps, or substitute dataset row indices as timestamps.
   - Synthetic timestamps are restricted strictly to isolated unit test fixtures in `tests/test_temporal_correlation.py`.
   - In production deployment, VYUHA ingests genuine ISO-8601 timestamps from network telemetry (e.g. Zeek `ts`, Suricata `timestamp`, NetFlow `FIRST_SWITCHED`/`LAST_SWITCHED`).

### 8.3 Correlation Rules Suite

| Rule Name | Class Name | Trigger Condition | Primary Evidence Extracted |
| :--- | :--- | :--- | :--- |
| **Temporal Proximity** | `TemporalProximityRule` | $0 \le \Delta t \le W$ | $\Delta t$, $W$, `proximity_classification` (`time_only_proximity`, `shared_source_ip`, `shared_destination_ip`, `shared_network_entity`), shared entities, missing entities, non-causation caveat. |
| **Repeated Alert** | `RepeatedAlertRule` | $0 \le \Delta t \le W$ AND identical model/prediction or identical category | Repetition type (`identical_model_prediction` vs `identical_category`), repeated prediction string, shared entities, missing entities, non-causation caveat. |
| **Repeated Category** | `RepeatedCategoryRule` | $0 \le \Delta t \le W$ AND identical non-benign `attack_category` | Threat family, burst interval ($\Delta t$), target ports, shared entities, missing entities, non-causation caveat. |
| **Sequential Ordering** | `SequentialOrderingRule` | Strict $0 < \Delta t \le W$ ($t_A < t_B$) AND recognized candidate attack progression | Earlier/later alert IDs, earlier/later categories, candidate progression stage, shared target indicator, shared entities, missing entities, non-causation caveat. |

#### Supported Candidate Attack-Pattern Sequences (`SequentialOrderingRule`):
* **Reconnaissance to Exploitation / DoS (`candidate_reconnaissance_to_exploitation`)**:
  - `PortScan` $\to$ `BruteForce`
  - `PortScan` $\to$ `WebAttack`
  - `PortScan` $\to$ `DoS_DDoS`
  - `PortScan` $\to$ `Infiltration`
  - `PortScan` $\to$ `Heartbleed`
* **Exploitation to C2 / Persistence (`candidate_exploitation_to_c2`)**:
  - `WebAttack` $\to$ `Botnet`
  - `Infiltration` $\to$ `Botnet`
  - `BruteForce` $\to$ `Botnet`
* **Exploitation Sequence (`candidate_sequential_attack_pattern`)**:
  - `Heartbleed` $\to$ `Infiltration`

Inverted progressions (e.g. `BruteForce` followed by `PortScan`) and simultaneous events ($\Delta t = 0$) are strictly rejected by the sequential rule.

### 8.4 Transitive Single-Linkage Temporal Clustering

The correlation engine implements **transitive single-linkage temporal clustering** (`clustering_method="transitive_single_linkage"`).

#### Mathematical Behavior & Transitive Chaining:
Alerts are clustered chronologically. Two consecutive alerts $e_i$ and $e_{i+1}$ belong to the same cluster if and only if:
$$\Delta t_{i, i+1} = t_{i+1} - t_i \le \text{max\_cluster\_gap\_seconds}$$

Because clustering is transitive, chains of adjacent alerts are grouped into a single cluster even if the total duration of the cluster exceeds the pairwise observation window $W$ or maximum gap:
- **Concrete Scenario**:
  - Alert A at `10:00:00`
  - Alert B at `10:00:09` ($\Delta t_{AB} = 9\text{s} \le 10\text{s}$)
  - Alert C at `10:00:18` ($\Delta t_{BC} = 9\text{s} \le 10\text{s}$)
  - Configured `window_seconds = 10.0` and `max_cluster_gap_seconds = 10.0`.
- **Clustering Outcome**:
  - All three alerts $[A, B, C]$ are grouped into **one single cluster**.
  - `first_timestamp`: `2026-11-19T10:00:00+00:00`
  - `last_timestamp`: `2026-11-19T10:00:18+00:00`
  - `total_duration_seconds`: $18.0\text{s}$ (strictly greater than the 10.0s window).
  - `max_cluster_gap_seconds`: $10.0\text{s}$.
  - `alerts_count`: $3$.
  - `relationships`: Contains $(A, B)$ with $\Delta t = 9\text{s}$ and $(B, C)$ with $\Delta t = 9\text{s}$.
  - Pair $(A, C)$ does **not** have a direct temporal relationship because $\Delta t_{AC} = 18\text{s} > 10\text{s}$ exceeds the pairwise observation boundary.
  - `relationship_count`: $2$.

### 8.5 Temporal Relationship & Cluster Evidence Model

#### Canonical Temporal Relationship JSON:
```json
{
  "relationship_id": "trel_seq_4b9a1e8c2d7f",
  "source_alert_id": "scan_01",
  "target_alert_id": "brute_02",
  "rule_name": "sequential_ordering",
  "source_timestamp": "2026-11-19T10:00:00+00:00",
  "target_timestamp": "2026-11-19T10:00:30+00:00",
  "time_delta_seconds": 30.0,
  "window_seconds": 300.0,
  "is_sequential": true,
  "evidence": {
    "earlier_alert_id": "scan_01",
    "later_alert_id": "brute_02",
    "earlier_category": "PortScan",
    "later_category": "BruteForce",
    "time_delta_seconds": 30.0,
    "window_seconds": 300.0,
    "candidate_stage": "candidate_reconnaissance_to_exploitation",
    "is_candidate_pattern_sequence": true,
    "shared_target_detected": true,
    "shared_entity_evidence": {
      "source_ip": "192.168.10.50",
      "destination_ip": "172.16.0.1",
      "destination_port": 22
    },
    "has_shared_entities": true,
    "proximity_classification": "shared_network_entity",
    "missing_entity_information": {
      "alert_a": [],
      "alert_b": []
    },
    "causation_caveat": "Observed sequential pattern is a candidate attack-pattern hypothesis ('PortScan' -> 'BruteForce' within 30.0s); temporal ordering does NOT establish causal connection, coordinated multi-stage execution, or common attacker identity."
  },
  "metadata": {}
}
```

#### Canonical Temporal Cluster JSON:
```json
{
  "cluster_id": "tclust_e8c2d7f4b9a1",
  "first_timestamp": "2026-11-19T10:00:00+00:00",
  "last_timestamp": "2026-11-19T10:00:18+00:00",
  "total_duration_seconds": 18.0,
  "max_cluster_gap_seconds": 10.0,
  "alerts_count": 3,
  "alert_ids": ["alt_a", "alt_b", "alt_c"],
  "relationships": [
    {
      "relationship_id": "trel_prox_111",
      "source_alert_id": "alt_a",
      "target_alert_id": "alt_b",
      "rule_name": "temporal_proximity",
      "time_delta_seconds": 9.0
    },
    {
      "relationship_id": "trel_prox_222",
      "source_alert_id": "alt_b",
      "target_alert_id": "alt_c",
      "rule_name": "temporal_proximity",
      "time_delta_seconds": 9.0
    }
  ],
  "relationship_count": 2,
  "dominant_categories": ["PortScan", "BruteForce"],
  "clustering_method": "transitive_single_linkage"
}
```

---

## 9. Relationship Graph Architecture

The **Relationship Graph** (`src.intelligence.graph.RelationshipGraph`) provides a lightweight, explainable topology that connects alerts using corroborating evidence rather than assumption or coincidence.

```
                  ┌──────────────────────┐
                  │ Normalized Alert A   │
                  └──────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
            │       Relationship Edge         │
            │  - relationship_type            │
            │  - weight                       │
            │  - grouping_eligible (bool)     │
            │  - evidence dictionary          │
            │  - human explanation            │
            └────────────────┬────────────────┘
                             │
                  ┌──────────▼───────────┐
                  │ Normalized Alert B   │
                  └──────────────────────┘
```

### 9.1 Supported Relationship Types

| Relationship Type | Class / Enum | Description | Grouping Eligible |
| :--- | :--- | :--- | :--- |
| `same_flow` | `RelationshipType.SAME_FLOW` | Identical flow identifier (`flow_id`); typically multi-engine evaluations of the same event. | **Yes** |
| `shared_source_ip` | `RelationshipType.SHARED_SOURCE_IP` | Exact matching non-None source IP address; links multi-target or scanning activity from the same origin. | **Yes** |
| `shared_destination_ip` | `RelationshipType.SHARED_DESTINATION_IP` | Exact matching non-None destination IP address; links multi-vector attacks against the same host. | **Yes** |
| `shared_network_entity` | `RelationshipType.SHARED_NETWORK_ENTITY` | Exact matching destination IP and destination port; links targeted attacks against the same service. | **Yes** |
| `shared_destination_port` | `RelationshipType.SHARED_DESTINATION_PORT` | Matching destination port across distinct/unverified destination IPs. | **No** (Informational only) |
| `repeated_category` | `RelationshipType.REPEATED_CATEGORY` | Bursts of identical threat categories (e.g. repeated `DoS_DDoS`). | **Yes** (if anchored by endpoint/flow) |
| `detection_engine_agreement` | `RelationshipType.DETECTION_ENGINE_AGREEMENT` | Supervised model and unsupervised anomaly engine both flagging the same flow as a threat. | **Yes** |
| `temporal_sequence` | `RelationshipType.TEMPORAL_SEQUENCE` | Candidate multi-stage attack progression (e.g. `PortScan` $\to$ `BruteForce`) within time window $W$. | **Yes** (with shared target) |
| `temporal_recurrence` | `RelationshipType.TEMPORAL_RECURRENCE` | Rapid burst of alerts within time window $W$. | **Yes** (with shared endpoint) |
| `temporal_proximity` | `RelationshipType.TEMPORAL_PROXIMITY` | Circumstantial co-occurrence within time window $W$. | **No** (Informational only unless endpoints match) |

### 9.2 Strict Integrity & Boundary Principles
1. **No Entity Creation from Missing Metadata**:
   Missing attributes (`None`) **never** evaluate as equal (`None == None` is rejected). If two flows lack source IPs, no `shared_source_ip` edge is created.
2. **Time Proximity Alone is Never Sufficient for Grouping**:
   Events occurring close in time without shared endpoints are classified as `time_only_proximity`. These edges are retained in the graph for contextual observation, but carry `grouping_eligible=False`. They **never** merge unrelated hosts into a single incident.
3. **Unrelated Alerts Remain Separate**:
   Connected components for incident generation traverse **only** grouping-eligible edges. Unrelated alerts form independent, single-alert candidate incidents, preventing false campaign conflation.

---

## 10. Candidate Incident Generation Engine

The **Candidate Incident Generator** (`src.intelligence.incident.CandidateIncidentGenerator`) transforms discrete alerts and graph components into explainable, prioritized `CandidateIncident` dossiers.

```
       Normalized Alerts Stream (Alert objects)
                          │
                          ▼
             [ Temporal Correlation ] (optional)
              - Computes candidate sequences & bursts
                          │
                          ▼
            [ Relationship Graph Builder ]
              - Pairwise entity evaluation
              - Ingests temporal edges
                          │
                          ▼
         [ Connected Component Traversal ]
           (Traverses grouping_eligible edges only)
           ├── Component 1 (Alerts A, B, C) ──► Candidate Incident 1
           ├── Component 2 (Alert D)       ──► Candidate Incident 2 (Isolated)
           └── Component 3 (Alerts E, F)   ──► Candidate Incident 3
                          │
                          ▼
       [ Multi-Factor Severity & Explainability ]
         - Explainable score calculation
         - Human-readable prose summary
         - Actionable investigation playbooks
                          │
                          ▼
               List[ CandidateIncident ]
               (Prioritized by severity descending)
```

### 10.1 Operational Status & Semantics
All automatically generated incidents are assigned the status:
$$\text{status} = \text{"CANDIDATE"}$$
* A **Candidate Incident** is a machine-generated hypothesis indicating that specific network alerts exhibit structural, entity, or temporal connections warranting unified human investigation.
* It is **not** a confirmed breach, verified attacker campaign, or closed ticket. Confirmation requires security analyst triage.

---

## 11. Explainable Provisional Severity Assessment

VYUHA rejects opaque model confidence scores as proxies for incident severity. Model confidence ($P(\text{class})$) reflects statistical certainty on flow features, not operational business impact.

### 11.1 Multi-Factor Scoring Matrix
Provisional severity is evaluated using a transparent point-based scoring rubric:

$$\text{Severity Score} = \text{Base Category Score} + \sum \text{Corroborating Modifiers}$$

#### Base Category Weights:
* **High-Impact Exploitation / Persistence (+3)**: `Infiltration`, `Botnet`, `Heartbleed`.
* **Active Exploitation / Disruption (+2)**: `WebAttack`, `DoS_DDoS`, `BruteForce`.
* **Reconnaissance / Novelty (+1)**: `PortScan`, generic `ATTACK`, unsupervised `ANOMALY`.
* **Normal / Verified Traffic (0)**: `BENIGN` (strictly results in `INFO`).

#### Corroborating Modifiers:
* **Detection Engine Consensus (+1)**: Supervised detection (Binary/Multiclass) corroborated by unsupervised Anomaly Isolation Forest.
* **Alert Burst / High Volume (+1)**: Repeated activity across 3 or more alerts against the same entity.
* **Sensitive Service Port Targeted (+1)**: Traffic directed at well-known enterprise service ports (e.g. 21/FTP, 22/SSH, 80/HTTP, 443/HTTPS, 445/SMB, 3389/RDP, 8080).
* **Candidate Attack Progression Sequence (+1)**: Observed sequential progression (e.g. `PortScan` $\to$ `BruteForce`).

#### Discrete Severity Mapping:
* **CRITICAL** ($\ge 5$ points): High-impact threat with multiple corroborating factors (e.g. Botnet beaconing with cross-engine agreement, or multi-stage exploitation).
* **HIGH** ($3 - 4$ points): Active exploitation against sensitive ports or repeated brute force bursts.
* **MEDIUM** ($2$ points): Moderate single-engine attack or reconnaissance targeting a known port.
* **LOW** ($1$ point): Isolated anomaly, low-impact probe, or single uncorroborated event.
* **INFO** ($0$ points): Exclusively benign traffic or informational notifications.

#### Environmental Impact Disclaimer:
> [!NOTE]
> Provisional severity measures technical threat signals. True operational severity requires environmental business context (asset criticality, data classification, and compensating controls), which offline network datasets (such as CICIDS2017) do not provide.

---

## 12. Human-Centric Explainability & Investigation Recommendations

### 12.1 Explainability Architecture
Every `CandidateIncident` contains a structured `explanation` dictionary and a human-readable `summary_text` designed for rapid triage by security operations center (SOC) analysts and college hackathon evaluators.

The explanation explicitly answers:
1. **Why alerts were grouped**: Documents the exact entity, port, flow, or sequence anchors that connected the alerts.
2. **Which relationships were identified**: Lists all active edge types (e.g. `shared_network_entity`, `temporal_sequence`).
3. **Which detection engines contributed**: Discloses participating algorithms (Random Forest, Decision Tree, Isolation Forest).
4. **Which attack categories were observed**: Lists confirmed and candidate threat classifications.
5. **What evidence is missing**: Transparently discloses absent metadata (e.g. missing timestamps or source IPs).
6. **Why provisional severity was assigned**: Details the exact additive scoring factors and domain justifications.

### 12.2 Actionable Investigation Recommendations
Every candidate incident generates tailored, rule-based triage steps grounded strictly in observed evidence.

#### Core Investigation Playbooks:
* **Originating Host**:
  - *When Source IP is known*: "Investigate originating host [IP] for unauthorized software, active scanning tools, or indicators of external compromise."
  - *When Source IP is missing*: "Identify originating host IP address via perimeter firewall, NAT session tables, or NetFlow logs."
* **Targeted Host & Service**:
  - "Inspect service and authentication logs on destination host [IP] on targeted port(s) [PORTS] for repeated failed authentications or unexpected service restarts."
* **Threat-Specific Actions**:
  - *BruteForce*: "Audit authentication logs for account lockout events and inspect successful logins immediately following the alert timeframe."
  - *PortScan*: "Check network firewall logs for horizontal port sweeps or host discovery probes originating from the same network segment."
  - *WebAttack*: "Examine web application firewall (WAF) and HTTP access logs for SQL injection, cross-site scripting, or path traversal URI patterns."
  - *Botnet*: "Isolate suspicious internal hosts immediately from the local VLAN and inspect DNS query logs for domain generation algorithms (DGA) or known C2 beaconing."
  - *Infiltration*: "Initiate host-level forensic analysis on target systems: examine active user sessions, recently created scheduled tasks/cron jobs, and modified binary executables."
* **Activity Continuity**:
  - "Review network flow telemetry immediately following the incident window to verify whether suspicious connection attempts ceased or persisted."

> [!CAUTION]
> Recommendations strictly reflect triage procedures. VYUHA never claims that an attack was blocked, mitigated, or resolved at the detection phase.

---

## 13. Dataset Limitations & Offline Telemetry Handling

The canonical benchmark dataset CICIDS2017 `MachineLearningCVE` (and the processed parquet splits derived from it) lacks:
* **Source IP**
* **Destination IP**
* **Source Port**
* **Timestamp**
* **Numeric Transport Protocol**

The **only** network entity column genuinely present in `MachineLearningCVE` is **`Destination Port`** (along with `Flow Duration` and 76 statistical flow metrics).

VYUHA's Incident Intelligence engine is explicitly architected to support partial evidence with strict empirical honesty:

1. **No Synthetic Telemetry**:
   Source IPs, Destination IPs, and timestamps are **never** fabricated, imputed, or guessed. When absent, `source_ips` and `destination_ips` remain strictly empty lists (`[]`), and `first_timestamp` / `last_timestamp` remain `None`.
2. **Explicit Missing Evidence Tracking**:
   When evaluating offline `MachineLearningCVE` records, absent fields are explicitly documented in the incident payload:
   `missing_evidence: ["source_ip", "destination_ip", "source_port", "timestamp"]`
   The structured explanation notes:
   - *"Source IP is absent in dataset; attacker attribution cannot be established from flow records."*
   - *"Destination IP is absent in dataset; target host attribution cannot be established from flow records."*
   - *"Packet timestamps are absent in offline CICIDS2017 dataset; temporal duration was not evaluated."*
3. **Genuine Offline Evidence Correlation (Single-Flow Scope)**:
   Because IP addresses and timestamps are absent offline, VYUHA strictly refuses to claim IP-based, temporal, or cross-flow campaign correlation across separate records. Instead, offline incident correlation operates strictly on genuinely available evidence:
   - **Same-Flow Multi-Engine Identity (`same_flow`)**: Consolidating multiple detection engine assessments (e.g. supervised Random Forest, multiclass Decision Tree, and unsupervised Isolation Forest) that evaluated the exact same physical flow record (aligned by positional row index `flow_{index}`, native flow ID when available, or hash-based fallback only when indices are absent).
   - **Cross-Engine Consensus (`detection_engine_agreement`)**: Linking assessments where supervised and unsupervised engines both corroborate that the identical flow is anomalous or malicious.
   - **Targeted Service Port (`destination_port`)**: Identifying the destination service port (e.g. port 21/FTP, 22/SSH, 80/HTTP, 443/HTTPS) on that specific flow.
4. **No Cross-Flow Campaign Linkage Without Endpoint Telemetry**:
   Distinct records from `MachineLearningCVE` that lack IP addresses are **never** merged into a synthetic campaign incident solely because they share a common destination port (e.g. port 80) or threat category. Two separate flows targeting port 80 cannot be verified as belonging to the same campaign without IP or temporal corroboration. Sharing `destination_port` or `attack_category` alone across records without IP anchors carries `grouping_eligible=False`. Distinct offline flow records strictly remain separate, honest candidate incidents.
5. **Live Telemetry Multi-Flow Correlation**:
   When deployed in live environments (e.g. ingesting Zeek `conn.log`, Suricata EVE JSON, or NetFlow v9/IPFIX records with genuine IP headers and nanosecond timestamps), the engine automatically activates multi-flow relationship graphs (`shared_source_ip`, `shared_destination_ip`, `shared_network_entity`) and sliding-window temporal correlation across distinct flows without modifying model interfaces.

---

## 14. End-to-End Candidate Incident Dossier Example

The following JSON illustrates a fully populated `CandidateIncident` generated from live telemetry (where full IP headers and timestamps are available from network sensors such as Zeek or Suricata), demonstrating how the engine correlates multi-stage reconnaissance and brute-force activity against an internal SSH server:

> [!NOTE]
> In offline benchmarking on CICIDS2017 `MachineLearningCVE`, IP addresses and timestamps are absent (`source_ips: []`, `destination_ips: []`, `first_timestamp: None`). Offline incidents correlate based on deterministic same-flow multi-engine assessments, cross-engine agreement, and target service ports (`destination_ports`), explicitly recording absent fields in `missing_evidence`.

```json
{
  "incident_id": "inc_7f8a9b1c2d3e",
  "alert_ids": [
    "alt_scan_001",
    "alt_brute_002"
  ],
  "status": "CANDIDATE",
  "severity": "HIGH",
  "first_timestamp": "2026-11-19T10:00:00Z",
  "last_timestamp": "2026-11-19T10:00:30Z",
  "source_ips": [
    "192.168.10.50"
  ],
  "destination_ips": [
    "172.16.0.1"
  ],
  "source_ports": [
    45000,
    45002
  ],
  "destination_ports": [
    22
  ],
  "attack_categories": [
    "BruteForce",
    "PortScan"
  ],
  "threat_statuses": [
    "ATTACK"
  ],
  "detection_engines": [
    "multiclass"
  ],
  "alerts_count": 2,
  "relationships": [
    {
      "relationship_id": "rel_ent_4a1b2c",
      "source_alert_id": "alt_scan_001",
      "target_alert_id": "alt_brute_002",
      "relationship_type": "shared_network_entity",
      "weight": 1.8,
      "grouping_eligible": true,
      "evidence": {
        "destination_ip": "172.16.0.1",
        "destination_port": 22
      },
      "description": "Both alerts target the exact same service endpoint 172.16.0.1:22."
    },
    {
      "relationship_id": "rel_temp_trel_seq_99",
      "source_alert_id": "alt_scan_001",
      "target_alert_id": "alt_brute_002",
      "relationship_type": "temporal_sequence",
      "weight": 1.6,
      "grouping_eligible": true,
      "evidence": {
        "earlier_category": "PortScan",
        "later_category": "BruteForce",
        "time_delta_seconds": 30.0,
        "shared_target_detected": true
      },
      "description": "Sequential candidate attack progression ('PortScan' -> 'BruteForce' within 30.0s)."
    }
  ],
  "relationship_count": 2,
  "explanation": {
    "grouping_rationale": "Grouped 2 alerts based on targeting the same destination service (172.16.0.1:22); originating from shared source host (192.168.10.50); demonstrating candidate multi-stage attack sequence.",
    "relationships_identified": [
      "shared_destination_ip",
      "shared_network_entity",
      "shared_source_ip",
      "temporal_sequence"
    ],
    "detection_engines_contributing": [
      "multiclass"
    ],
    "attack_categories_observed": [
      "BruteForce",
      "PortScan"
    ],
    "missing_evidence": [],
    "missing_evidence_notes": [
      "All primary network telemetry fields (IPs, ports, timestamps) were available."
    ],
    "severity_assigned": "HIGH",
    "severity_score": 4,
    "severity_justification": "Active exploitation category detected: BruteForce (+2); Targeted well-known service port(s): [22] (+1); Observed chronological progression matching candidate attack-pattern sequence (+1)",
    "business_impact_caveat": "Provisional severity reflects technical anomaly and attack metrics. True business impact requires asset criticality and environment context, which are absent in offline network benchmark datasets."
  },
  "summary_text": "Candidate Incident groups 2 alert(s) observed between 2026-11-19T10:00:00Z and 2026-11-19T10:00:30Z. Observed threat activity involves 'BruteForce, PortScan' identified by multiclass. Grouped 2 alerts based on targeting the same destination service (172.16.0.1:22); originating from shared source host (192.168.10.50); demonstrating candidate multi-stage attack sequence. Provisional severity is assessed as HIGH (Active exploitation category detected: BruteForce (+2); Targeted well-known service port(s): [22] (+1); Observed chronological progression matching candidate attack-pattern sequence (+1)). Note: All primary network telemetry fields (IPs, ports, timestamps) were available.",
  "recommendations": [
    "Investigate originating host(s) 192.168.10.50 for unauthorized software, active scanning tools, or indicators of external compromise.",
    "Inspect service and authentication logs on destination host(s) 172.16.0.1 on targeted port(s) 22 for repeated failed authentications or unexpected service restarts.",
    "Audit authentication logs for account lockout events and inspect successful logins immediately following the alert timeframe to verify if credentials were compromised.",
    "Check network firewall logs for horizontal port sweeps or host discovery probes originating from the same network segment.",
    "Candidate reconnaissance-to-exploitation progression observed: correlate endpoint logs to determine if subsequent lateral movement or persistence attempts occurred.",
    "Review network flow telemetry immediately following the incident window to verify whether suspicious connection attempts ceased or persisted."
  ],
  "missing_evidence": [],
  "metadata": {
    "scoring_points": 4,
    "dataset_source": "live_telemetry"
  }
}
```


