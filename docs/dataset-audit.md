# VYUHA: CICIDS2017 Dataset Intelligence & Audit Report

**Dataset**: Canadian Institute for Cybersecurity (UNB) - CICIDS2017 (MachineLearningCVE)  
**Processing Method**: Memory-efficient chunked streaming (`chunksize=100,000`)  
**Total Records Audited**: 2,830,743 rows across 8 CSV files (~843.66 MB uncompressed)  
**Audit Date**: November 2026

---

## 1. File Inventory and Physical Metrics

All 8 official MachineLearningCVE CSV files located in `../datasets/raw/CICIDS2017/` were verified on disk with valid headers and data streams:

| CSV Filename | File Size | Total Rows | Nulls | Infs | Duplicates | Process Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 73.55 MB | 225,745 | 4 | 64 | 1,422 | 1.23s |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 73.34 MB | 286,467 | 15 | 727 | 42,191 | 1.51s |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 55.62 MB | 191,033 | 28 | 216 | 9,135 | 1.26s |
| `Monday-WorkingHours.pcap_ISCX.csv` | 168.73 MB | 529,918 | 64 | 810 | 20,235 | 5.05s |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 79.25 MB | 288,602 | 18 | 396 | 28,826 | 2.74s |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 49.61 MB | 170,366 | 20 | 250 | 12,901 | 1.88s |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 128.82 MB | 445,909 | 201 | 327 | 29,181 | 5.85s |
| `Wednesday-workingHours.pcap_ISCX.csv` | 214.74 MB | 692,703 | 1,008 | 1,586 | 94,201 | 9.39s |
| **Total / Aggregate** | **843.66 MB** | **2,830,743** | **1,358** | **4,376** | **238,092** | **28.9s** |

---

## 2. Attack Label Distribution & Class Imbalance

The dataset consists of **15 distinct classes** (1 Benign baseline + 14 specific attack scenarios).

### Binary Split
* **BENIGN Traffic**: `2,273,097` records (**80.30%**)
* **MALICIOUS Traffic**: `557,646` records (**19.70%**)

### Label Frequency Breakdown

| Original Label | Sample Count | % of Dataset | Threat Category | Typical Network Signature |
| :--- | :--- | :--- | :--- | :--- |
| **BENIGN** | 2,273,097 | 80.300% | Normal Traffic | Standard HTTP/HTTPS browsing, routine DNS, legitimate internal RPC |
| **DoS Hulk** | 231,073 | 8.163% | Denial of Service | High-rate HTTP request flood with obfuscated User-Agents |
| **PortScan** | 158,930 | 5.614% | Reconnaissance | Rapid sequential SYN/Connect requests across multiple destination ports |
| **DDoS** | 128,027 | 4.523% | Distributed DoS | Multi-source volumetric packet and bandwidth flood |
| **DoS GoldenEye** | 10,293 | 0.364% | Denial of Service | HTTP Keep-Alive header exhaustion consuming server worker pools |
| **FTP-Patator** | 7,938 | 0.280% | Brute Force | High-frequency dictionary credential attempts targeting Port 21 |
| **SSH-Patator** | 5,897 | 0.208% | Brute Force | High-frequency dictionary credential attempts targeting Port 22 |
| **DoS slowloris** | 5,796 | 0.205% | Denial of Service | Slow, partial HTTP header transmission keeping connections open |
| **DoS Slowhttptest** | 5,499 | 0.194% | Denial of Service | Slow HTTP read/post starvation exhausting server resources |
| **Bot** | 1,966 | 0.069% | Malware / C2 | Ares botnet IRC command & control communication |
| **Web Attack - Brute Force** | 1,507 | 0.053% | Application Attack | Repeated HTTP POST credential stuffing against web login forms |
| **Web Attack - XSS** | 652 | 0.023% | Application Attack | Cross-site scripting payload injection in web request parameters |
| **Infiltration** | 36 | 0.001% | Advanced Threat | Multi-stage lateral network exploitation following initial payload drop |
| **Web Attack - Sql Injection** | 21 | 0.0007% | Application Attack | Malicious SQL syntax in parameters attempting database extraction |
| **Heartbleed** | 11 | 0.0004% | Vulnerability Exploit | OpenSSL TLS heartbeat buffer over-read targeting Port 443 |

### Class Imbalance Considerations
* **Extreme Long-Tail Distribution**: Rare classes like Heartbleed (11 flows), SQL Injection (21 flows), and Infiltration (36 flows) represent less than 0.002% of the dataset.
* **Evaluation Implication**: Standard overall accuracy is misleading (a trivial dummy classifier predicting `BENIGN` achieves 80.3% accuracy while completely failing to detect real attacks). Evaluation must use Precision, Recall, F1-Score (macro-averaged and per-class), and Precision-Recall AUC (PR-AUC).
* **Grouping Strategy**: To ensure statistical validity during machine learning, attacks are grouped into 7 coherent threat categories:
  1. `BENIGN`
  2. `DoS_DDoS` (`DoS Hulk`, `DDoS`, `DoS GoldenEye`, `DoS slowloris`, `DoS Slowhttptest`)
  3. `PortScan` (`PortScan`)
  4. `BruteForce` (`FTP-Patator`, `SSH-Patator`)
  5. `WebAttack` (`Web Attack – Brute Force`, `Web Attack – XSS`, `Web Attack – Sql Injection`)
  6. `Botnet` (`Bot`)
  7. `Infiltration` (`Infiltration`, `Heartbleed`)

---

## 3. Data Quality Issues & Preprocessing Decisions

### A. Column Header Normalization
* **Observation**: Every raw column header contains leading or trailing whitespace (e.g. `' Destination Port'`, `' Flow Duration'`, `' Label'`).
* **Decision**: Normalize column names by applying `.str.strip()` across all headers during CSV parsing.

### B. Duplicate Column Names
* **Observation**: Column index 34 is named `Fwd Header Length` and column index 63 is named `Fwd Header Length.1` (an exact redundant duplicate of the forward header length metric).
* **Decision**: Remove the redundant column index 63 (`Fwd Header Length.1`) during preprocessing to avoid multicollinearity.

### C. Missing & Infinite Values
* **Observation**:
  * Missing values: Exactly **1,358** nulls, occurring strictly in `Flow Bytes/s`.
  * Infinite values: Exactly **1,509** in `Flow Bytes/s` and **2,867** in `Flow Packets/s`.
* **Root Cause**: When network flows consist of a single packet or microburst where start and end timestamps match within the microsecond precision window, `Flow Duration == 0`. Dividing byte or packet count by 0 produces `+inf` or `NaN`.
* **Decision**:
  1. Impute null values in `Flow Bytes/s` with the median of valid finite values.
  2. For `Flow Duration == 0`, recalculate the rate using an epsilon time interval (1 microsecond = $10^{-6}$ s), preserving the true high-rate burst signal without causing mathematical overflow or numerical errors.
  3. Cap remaining extreme positive outliers at the 99.9th percentile to ensure stability for machine learning models.

### D. Negative Metric Values
* **Observation**:
  * `Flow Duration` contains negative values down to `-13` in 115 records.
  * Negative flow duration is physically impossible; it occurs in multi-threaded packet capture systems when the finishing packet timestamp is recorded with slight clock jitter or clock drift relative to the initiating packet.
* **Decision**:
  * Filter out records with negative flow duration (`Flow Duration < 0`) during data cleaning. Because only 115 records out of 2.83 million are affected (<0.005%), removing these demonstrably invalid capture artifacts preserves data integrity without discarding meaningful attack patterns.

### E. Meaningful Sentinel Values (-1) in TCP Window Features
* **Observation**:
  * `Init_Win_bytes_forward` and `Init_Win_bytes_backward` contain `-1` values.
* **Technical Explanation**:
  * In socket telemetry and flow collectors (such as CICFlowMeter), `-1` is an intentional protocol sentinel indicating that no SYN/ACK packet containing a TCP Window Scale option was observed for that direction (e.g. half-open connections, UDP stateless traffic, or raw sockets).
* **Decision**:
  * **Preserve `-1` as a valid discrete sentinel value.** Do not treat it as an erroneous negative measurement or impute it with mean/median, as the absence of a TCP window is a strong discriminatory feature between standard OS TCP stacks and lightweight attack scripts.

### F. Exact Duplicate Records & Data Leakage Prevention
* **Observation**:
  * Initial chunked stream audit identified **238,092 duplicate records** (8.41%).
  * Full-population dataset deduplication across all 77 flow features and labels revealed the true total: **331,215 duplicate records** (**11.70%**).
  * *Methodology Difference*: The initial streaming script checked duplicates by tracking a hash set across chunk boundaries. Repeated records that occurred multiple times *within the same initial 100,000-row chunk* were added to the hash set on the first pass and were not counted as duplicates. The full-dataset deduplication performed by `FlowPreprocessor.deduplicate()` evaluated the entire population simultaneously, capturing all intra-chunk and inter-chunk duplicates.
  * *Root Cause*: Automated attack tools (e.g., DoS Hulk flooders, automated port scanners) generate thousands of identical SYN packets with fixed window sizes, identical header lengths, and identical inter-arrival distributions within bursts spanning mere seconds.
* **Risk**:
  * If duplicate flows are randomly partitioned into training and testing sets, identical records will exist in both splits, causing severe **data leakage** and artificially inflating model evaluation metrics.
* **Decision**:
  * Deduplicate identical flow records prior to train/test partitioning to guarantee strict independence between training and evaluation splits.

---

## 4. Key Network-Flow Features Profile

The 77 network-flow metrics capture three essential operational dimensions:

1. **Volume & Rate Metrics (Congestion & Volumetric Attacks)**:
   * `Flow Bytes/s`, `Flow Packets/s`, `Flow Duration`, `Flow IAT Mean/Std`.
   * Essential for distinguishing network saturation (DoS/DDoS) from organic high-throughput transfers.
2. **Directional Asymmetry (Reconnaissance & Scan Detection)**:
   * `Total Fwd Packets`, `Total Backward Packets`, `Total Length of Fwd Packets`, `Total Length of Bwd Packets`, `Down/Up Ratio`.
   * Unanswered forward packets (`Bwd Packets == 0`) strongly correlate with port scanning or dropped SYN floods.
3. **TCP Connection State & Protocol Signatures (Brute Force & Targeted Exploits)**:
   * `Destination Port` (identifying target services e.g., 21 FTP, 22 SSH, 80 HTTP, 443 HTTPS).
   * `SYN Flag Count`, `FIN Flag Count`, `RST Flag Count`, `ACK Flag Count`, `PSH Flag Count`.
   * `Init_Win_bytes_forward`, `Init_Win_bytes_backward` (initial TCP receive window sizes).

---

## 5. Storage and Export Specifications

* **Raw Input Directory**: `datasets/raw/CICIDS2017/` (Outside Git repository, read-only).
* **Processed Output Directory**: `datasets/processed/` (Outside Git repository).
* **Processed Output Formats**:
  * High-performance compressed Parquet (`train.parquet`: 1,999,530 rows, `test.parquet`: 499,883 rows) for fast, typed ingestion.
  * Stratified development sample (`dev_sample.parquet`: 200,000 rows) strictly extracted from the training partition.

---

## 6. Preprocessing Decisions & Limitations Analysis

### A. Rate Feature 99.9th-Percentile Clipping
* **Decision**: `Flow Bytes/s` is capped at $1.317 \times 10^8$ B/s (~131.7 MB/s, representing ~1 Gbps line speed); `Flow Packets/s` is capped at $2.0 \times 10^6$ pkts/s (standard 10 Gbps minimum-frame wire capacity).
* **Attack Signal Preservation Verification**:
  * Empirical analysis of raw attack classes confirms that 99% of DoS Hulk traffic operates below $863.4$ KB/s, and 99% of DDoS traffic operates below $601.1$ KB/s.
  * Extreme values above the 99.9th percentile ($>131.7$ MB/s) represent single-packet microburst calculation artifacts ($tot\_bytes / 10^{-6}$ s) and mathematical infinities rather than sustained volumetric attacks.
  * Clipping preserves the true high-rate signal (clamped at massive line capacity) while preventing numerical overflow and gradient instability in ML algorithms.

### B. Limitations of Random Stratified Splitting
* **Temporal and Topological Capture-Level Leakage**:
  * CICIDS2017 attacks were executed on specific days (e.g., DoS on Wednesday, DDoS/PortScan on Friday) between fixed IP pairs (`172.16.0.1` and `192.168.10.50`).
  * While exact row deduplication eliminates identical flow instances, random stratified partitioning splits flows from the *same capture window* across both train and test splits.
  * **Consequence**: Models evaluate primarily on *in-distribution signature recognition* (identifying an attack tool's behavior within the captured environment) rather than *cross-temporal zero-shot generalization* (identifying the attack on a new network with different topology). This limitation is documented for academic transparency.

### C. Development Sample Partition Isolation
* **Verification**: `dev_sample.parquet` (200,000 rows) was generated strictly by sampling from `train.parquet`.
* **Empirical Overlap**: Exactly **0 overlapping records** exist between `dev_sample.parquet` and `test.parquet`, guaranteeing that rapid development and hyperparameter tuning on the dev sample will never leak test-set evaluation data.
