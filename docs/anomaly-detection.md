# VYUHA: Unsupervised Anomaly Detection (Phase 3D Report)

**Project**: VYUHA — AI-Powered Cybersecurity for Small Industries  
**Phase**: Phase 3D: Unsupervised Novelty & Anomaly Detection Engine  
**Algorithm**: Isolation Forest (`sklearn.ensemble.IsolationForest`)  
**Training Regime**: Benign-Only Novelty Detection (Zero Attack Labels During Fitting)  
**Dataset**: Canadian Institute for Cybersecurity (UNB) - CICIDS2017 (MachineLearningCVE)  
**Status**: Completed, Benchmarked & Audited  
**Date**: November 2026  

---

## 1. Executive Summary & Experimental Scope

Phase 3D implements and evaluates an **unsupervised anomaly detection engine** for VYUHA. Unlike the supervised classification models from Phase 3B (Binary Detection) and Phase 3C (Multiclass Attack-Family Classification), this engine learns statistical profiles of legitimate network traffic **strictly from benign flows** without utilizing attack labels during model fitting.

### The Fundamental Distinction: Anomaly vs. Attack
A core principle of network security is that **an anomaly is not automatically malicious**. 
* Legitimate but unusual network events (such as bulk data transfers, automated database backups, or network maintenance) exhibit high anomaly scores.
* Conversely, stealthy malicious activity (such as slow dictionary brute-forcing, SQL injection inside standard HTTP requests, or low-volume IRC botnet beacons) frequently exhibits flow statistics that sit directly in the heart of normal traffic distributions.

### Key Empirical Findings
1. **The Anomaly Detection Ranking Capability**:
   * Evaluated across the complete held-out validation set ($N_{val} = 399,906$), Isolation Forest achieves a **ROC-AUC of $0.8023$** and a **PR-AUC of $0.5239$** (Full 63-feature model, $100\text{k}$ benign training flows).
   * Removing `Destination Port` (Port-Agnostic 62 features) increases ROC-AUC to **$0.8191$** and PR-AUC to **$0.5252$**, demonstrating that excluding port numbers prevents trees from splitting on arbitrary port values and forces reliance on behavioral transport metrics.
2. **Strict Scaling Invariance Empirically Confirmed**:
   * Comparing unscaled features against `RobustScaler` produced **identical ROC-AUC ($0.8023$) and identical PR-AUC ($0.5239$) to four decimal places**. 
   * This confirms the theoretical property of Isolation Forest: because isolation trees perform random axis-aligned univariate splits ($x_j \le \theta$), strictly monotonic feature transformations do not alter sample partitioning or path lengths. Feature scaling is unnecessary.
3. **High Fidelity of Benign Threshold Calibration**:
   * Using a separate held-out benign calibration subset ($N_{\text{cal}} = 50,000$), decision thresholds were frozen at pre-defined false-positive rate (FPR) targets.
   * When evaluated on the 331,768 benign validation flows, the empirical false-positive rates matched the targets with high fidelity:
     - **Target 5.0% FPR** $\to$ Empirical Validation Benign FPR: **4.94%** (Full) / **5.01%** (Port-Agnostic)
     - **Target 1.0% FPR** $\to$ Empirical Validation Benign FPR: **1.06%** (Full) / **1.04%** (Port-Agnostic)
     - **Target 0.5% FPR** $\to$ Empirical Validation Benign FPR: **0.50%** (Full) / **0.51%** (Port-Agnostic)
     - **Target 0.1% FPR** $\to$ Empirical Validation Benign FPR: **0.12%** (Full) / **0.13%** (Port-Agnostic)
4. **Attack Detection Divergence (The Visibility Boundary)**:
   * **Volumetric Outliers & Buffer Exploits Are Detected**:
     - At a calibrated $1.0\%$ FPR, the anomaly detector catches **$100.0\%$ of Heartbleed** ($N=2$) and **$50.0\%$ of Infiltration** ($N=6$). At $5.0\%$ FPR, it catches **$100.0\%$ of Heartbleed**, **$100.0\%$ of Infiltration**, and **$54.1\%$ of DoS/DDoS** ($27,888$ flows).
   * **Stealthy & Application-Layer Attacks Are Completely Missed**:
     - At $1.0\%$ FPR, detection rates for `BruteForce`, `WebAttack`, and `Botnet` are **$0.00\%$** (and only $0.02\%$ for `PortScan`).
     - **Root Cause**: These attacks utilize standard TCP/HTTP dialogs with small packet sizes and ordinary timing intervals that are statistically indistinguishable from legitimate browsing and administrative traffic without ground-truth labels.
5. **Computational Scalability**:
   * Training on 100,000 benign flows takes only **0.18 seconds**, with serialized model artifacts consuming just **0.71 MB**.
   * Amortized batch inference latency is **$1.81\ \mu\text{s/flow}$** ($\sim 552,000\text{ flows/s}$ batch throughput).

---

## 2. Dataset Constraints & Benign-Only Partitioning Architecture

### Strict Test Set Isolation
* Source partition: `datasets/processed/train.parquet` ($1,999,530$ records).
* **Held-Out Test Set Untouched**: `datasets/processed/test.parquet` ($499,883$ records) remained completely sealed, unopened, and unreferenced.

### Three-Fold Benign/Validation Architecture
To ensure zero data leakage and guarantee valid threshold calibration, the processed training dataset was partitioned using a strict three-fold architecture:

```
train.parquet (1,999,530 flows)
  │
  ├── 80% Training Partition (1,599,624 flows)
  │     ├── Attack Flows (272,552 flows) ─── [DISCARDED - NEVER USED IN TRAINING]
  │     └── Benign Training Pool (1,327,072 flows)
  │           ├── Calibration Set (50,000 benign flows) ─── [Threshold Calibration Only]
  │           └── Training Pool (1,277,072 benign flows) ─── [Subsampled: 50k, 100k, 200k]
  │
  └── 20% Stratified Validation Partition (399,906 flows) ─── [Final Evaluation Only]
        ├── BENIGN (331,768 flows)
        └── ATTACK (68,138 flows across all 7 families)
```

1. **Stratified Split**: An initial 80/20 stratified split on `label_category` preserves the exact validation partition ($N_{val} = 399,906$) used in Phase 3B and Phase 3C, enabling direct comparative analysis.
2. **Attack Label Purging**: All 272,552 attack records in the training partition were purged. Only `BENIGN` flows were retained.
3. **Calibration Isolation**: A fixed, disjoint slice of 50,000 benign flows was set aside strictly for threshold calibration. The remaining 1,277,072 benign flows formed the training pool.
4. **Target Column Exclusion**: All target columns (`label_raw`, `label_binary`, `label_category`) were dropped prior to model fitting via `extract_features_and_targets()`.

---

## 3. Anomaly Score Formulation & Threshold Calibration Strategy

### Mathematical Score Definition vs. Observed Range
In scikit-learn, `IsolationForest.score_samples(X)` calculates the opposite of the anomaly score defined by Liu et al. (2008):
$$s_{\text{sklearn}}(x) = - 2^{-\frac{E(h(x))}{c(n)}}$$
Where $E(h(x))$ is the average path length of observation $x$ across all trees, and $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree constructed on $n$ samples ($c(256) \approx 9.67$):
$$c(n) = 2 \ln(n - 1) + 0.5772156649 - \frac{2(n - 1)}{n}$$

VYUHA defines the **normalized Anomaly Score $A(x)$**:
$$A(x) = - s_{\text{sklearn}}(x) = 2^{-\frac{E(h(x))}{c(n)}}$$

> [!IMPORTANT]
> **Observed Range vs. Theoretical Bounds**:
> While $2^{-x}$ is bounded mathematically within $(0, 1]$, **the empirical anomaly scores do not span the entire $[0, 1]$ interval**. 
> With `max_samples=256`, scikit-learn sets the maximum tree depth to:
> $$\text{max\_depth} = \lceil \log_2(\text{max\_samples}) \rceil = \lceil \log_2(256) \rceil = 8$$
> When an evaluation path terminates at a leaf containing $e > 1$ samples, scikit-learn adds an adjustment $c(e)$ to the path length.
> Because trees have a finite depth limit ($\text{max\_depth} = 8$) and random root splits rarely isolate points in a single step:
> * Across all $399,906$ validation flows, observed scores range strictly between **$0.3164$ and $0.7593$**.
> * Deeply normal flows cluster around **$0.34\text{--}0.40$**.
> * Volumetric and structural outliers reach **$0.63\text{--}0.75$**.
> * An observation never achieves $0.0$ or $1.0$.

### Threshold Calibration on the Calibration Subset
Rather than using arbitrary contamination values, VYUHA implements a calibrated percentile-based thresholding strategy:
1. For a pre-defined target false-positive rate $\alpha \in \{0.05, 0.01, 0.005, 0.001\}$:
2. The decision threshold $\tau_\alpha$ is computed as the $(1 - \alpha)$-quantile of anomaly scores on the held-out benign calibration subset ($X_{\text{cal}}$, $N=50,000$):
   $$\tau_\alpha = \text{Percentile}_{100(1 - \alpha)}\big(\{A(x) : x \in X_{\text{cal}}\}\big)$$
3. The threshold $\tau_\alpha$ is **frozen**.
4. For any flow $x$ in the validation set:
   $$\hat{y}(x) = \begin{cases} 1 \ (\text{ANOMALY}) & \text{if } A(x) \ge \tau_\alpha \\ 0 \ (\text{NORMAL}) & \text{if } A(x) < \tau_\alpha \end{cases}$$

This mathematical formulation guarantees that the expected false-alarm rate on normal traffic is bounded by $\alpha$.

---

## 4. Computational Scalability & Sub-Sampling Mechanics

Controlled experiments were evaluated on 50,000, 100,000, and 200,000 benign training flows.

### Computational Scaling Benchmarks

| Model Identifier | Training Pool Size | Feature Set | Training Time | Model Size | Amortized Batch Latency | Batch Throughput | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `iforest_full_50k` | 50,000 | Full (63) | **0.14 s** | **0.705 MB** | **$1.794\ \mu\text{s/flow}$** | **$557,493\ \text{flows/s}$** | 0.7985 | 0.5203 |
| **`iforest_full_100k`** | **100,000** | **Full (63)** | **0.18 s** | **0.708 MB** | **$1.810\ \mu\text{s/flow}$** | **$552,495\ \text{flows/s}$** | **0.8023** | **0.5239** |
| `iforest_full_200k` | 200,000 | Full (63) | 0.26 s | 0.743 MB | $1.925\ \mu\text{s/flow}$ | $519,548\ \text{flows/s}$ | 0.8015 | 0.5205 |

> [!NOTE]
> **Batch Throughput vs. Live Streaming Latency**:
> The reported latency ($1.81\ \mu\text{s/flow}$) and throughput ($\sim 550,000\text{ flows/s}$) represent vectorized batch evaluation across pre-extracted in-memory arrays. They do not include live packet capture, TCP reassembly, or feature extraction overhead, and do not demonstrate line-rate readiness on live hardware.

### Clarification of Isolation Forest Sub-Sampling Dynamics
The role of the training pool and the mechanism of tree construction operate as follows:
* **Estimator Count**: `n_estimators = 100` trees are grown.
* **Samples Drawn per Tree**: Each estimator samples up to `max_samples = 256` observations without replacement from the available training pool.
* **Role of the Overall Training Pool**: The training pool ($N_{\text{train}} \in \{50k, 100k, 200k\}$) represents the population from which these 100 sub-samples are drawn. Changing the size of the training pool changes the specific observations available for sampling by individual trees.
* **Experimental Findings**: What the controlled experiments establish:
  - Scaling from **50k to 100k** benign flows improved ROC-AUC from **$0.7985$ to $0.8023$** and PR-AUC from **$0.5203$ to $0.5239$**.
  - Scaling from **100k to 200k** benign flows produced no measured improvement (ROC-AUC: **$0.8015$**, PR-AUC: **$0.5205$**).
* **Experimental Selection**: **100,000 benign flows** is the selected experimental configuration based on the observed performance and computational trade-off.

---

## 5. Feature Configurations & Port-Agnostic Threshold Audit

### 1. Empirical Confirmation of Scaling Invariance
* Comparing `iforest_full_100k` (Unscaled) and `iforest_full_scaled_100k` (`RobustScaler`):
  - **Unscaled**: $\text{ROC-AUC} = 0.80231$, $\text{PR-AUC} = 0.52391$
  - **RobustScaler**: $\text{ROC-AUC} = 0.80231$, $\text{PR-AUC} = 0.52391$
* **Mathematical Rationale**: Isolation Forest selects a random feature $q$ and a random split value $p$ uniformly between $\min(q)$ and $\max(q)$. Because linear/quantile scaling is a strictly monotonic transformation, it preserves relative sample ordering and split probability density. Scaling is mathematically and empirically redundant for tree-based anomaly isolation.

### 2. Comprehensive Port-Agnostic Threshold Audit (`iforest_port_100k`)
Threshold calibration was executed on the disjoint 50k benign calibration set and evaluated across all 399,906 validation flows for the port-agnostic model:

| Target FPR | Calibrated Threshold ($\tau_\alpha$) | Empirical Val Benign FPR | Attack Recall | Precision | F1-Score | True Positives (TP) | False Positives (FP) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5.0%** | $\tau = 0.5490$ | 5.01% | **40.43%** | 62.39% | **0.4906** | 27,549 | 16,610 |
| **1.0%** | $\tau = 0.6430$ | 1.04% | **25.46%** | **83.47%** | **0.3902** | 17,346 | 3,435 |
| **0.5%** | $\tau = 0.6861$ | 0.51% | 0.20% | 7.44% | 0.0039 | 135 | 1,679 |
| **0.1%** | $\tau = 0.7101$ | 0.13% | 0.00% | 0.23% | 0.0000 | 1 | 432 |

#### Per-Attack Recall Comparison at 1.0% FPR: Full vs. Port-Agnostic

| Traffic Category | Validation Support ($N$) | Full Model Recall (`iforest_full_100k`) | Port-Agnostic Recall (`iforest_port_100k`) | Performance Difference ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **BENIGN (FPR)** | 331,768 | 1.06% ($3,520$ FPs) | **1.04% ($3,435$ FPs)** | **-85 False Alarms** |
| **DoS_DDoS** | 51,480 | 30.21% ($15,553$) | **33.68% ($17,337$)** | **+1,784 Detections (+3.47%)** |
| **Heartbleed** | 2 | 100.00% (2) | 100.00% (2) | Unchanged (Descriptive, $N=2$) |
| **Infiltration** | 6 | 50.00% (3) | 50.00% (3) | Unchanged (Descriptive, $N=6$) |
| **PortScan** | 14,531 | 0.02% (3) | 0.00% (0) | Invisible to flow isolation |
| **BruteForce** | 1,464 | 0.00% (0) | 0.00% (0) | Invisible to flow isolation |
| **WebAttack** | 343 | 0.00% (0) | 0.00% (0) | Invisible to flow isolation |
| **Botnet** | 312 | 0.00% (0) | 0.00% (0) | Invisible to flow isolation |

**Why Port-Agnostic Modeling Improves Detection**: Destination port is an arbitrary categorical coordinate. Excluding port numbers forces the trees to split purely on transport dynamics (flow durations, packet sizes, inter-arrival times), yielding higher DoS recall (+1,784 flows) and fewer false alarms (-85 FPs) at 1% FPR.

---

## 6. Detailed Evaluation & Per-Attack-Family Anomaly Diagnostics

### Score Distributions by Category (`iforest_full_100k`)

| Traffic Category | Support ($N$) | Score Median ($A(x)$) | Score Max ($A(x)$) | Recall @ 5.0% FPR ($\tau=0.539$) | Recall @ 1.0% FPR ($\tau=0.635$) | Recall @ 0.5% FPR ($\tau=0.681$) | Evaluation Sample Context |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **BENIGN** | 331,768 | 0.349 | 0.739 | *4.94% (FPR)* | *1.06% (FPR)* | *0.50% (FPR)* | Larger evaluation sample |
| **DoS_DDoS** | 51,480 | **0.627** | 0.715 | **54.10%** | **30.21%** | 0.30% | Larger evaluation sample |
| **PortScan** | 14,531 | 0.361 | 0.666 | 0.16% | 0.02% | 0.00% | Larger evaluation sample |
| **BruteForce** | 1,464 | 0.389 | 0.471 | 0.00% | 0.00% | 0.00% | Moderate evaluation sample |
| **WebAttack** | 343 | 0.371 | 0.556 | 0.00% | 0.00% | 0.00% | Moderate evaluation sample |
| **Botnet** | 312 | 0.352 | 0.606 | 1.92% | 0.00% | 0.00% | Moderate evaluation sample |
| **Infiltration** | 6 | **0.625** | 0.667 | **100.00%** | **50.00%** | 0.00% | Small sample; descriptive/uncertain |
| **Heartbleed** | 2 | **0.697** | 0.697 | **100.00%** | **100.00%** | 0.00% | Small sample; descriptive/uncertain |

---

### In-Depth Investigation: Why Anomaly Detection Diverges Across Attacks

```
Anomaly Score Density Distribution:
Normal Traffic Peak: ~0.35            Outlier Region: >0.63
  ┌───────────────────────┐             ┌──────────────┐
  │ BENIGN, Botnet,       │             │ DoS_DDoS,    │
  │ BruteForce, WebAttack │             │ Heartbleed,  │
  │ (Indistinguishable)   │             │ Infiltration │
──┴───────────────────────┴─────────────┴──────────────┴───► Anomaly Score
  0.30        0.40        0.50          0.60        0.70
```

#### 1. Why DoS/DDoS, Heartbleed, and Infiltration Are Caught
* **DoS/DDoS**: High packet counts, abnormal asymmetry, and massive forward/backward packet length aggregates place volumetric attacks far outside normal clustering boundaries. The median DoS score is **$0.627$** (compared to $0.349$ for benign).
* **Heartbleed**: The OpenSSL TLS heartbeat over-read payload generates a distinct packet length pattern with high backward-to-forward byte ratios. Both validation flows have an anomaly score of **$0.697$**, cleanly crossing the 1.0% threshold ($\tau = 0.635$).
* **Infiltration**: Multistage lateral compromise involves unusual flow durations and irregular inter-arrival timing statistics, producing a median score of **$0.625$**.

#### 2. Why Botnet, BruteForce, and WebAttack Are Invisible to Anomaly Detection
* **Botnet**: Ares IRC botnet traffic has a median anomaly score of **$0.352$**—almost identical to the benign median ($0.349$). Because C2 keep-alives consist of standard 6-byte packets, they appear completely typical.
* **BruteForce**: Dictionary attacks against SSH/FTP consist of standard protocol handshake exchanges. The maximum anomaly score observed for BruteForce was only **$0.471$**, never reaching even the 5% threshold ($\tau = 0.539$).
* **WebAttack**: SQL injection and XSS attacks occur in the HTTP payload text. The summary flow features (packet counts, duration, header lengths) mirror ordinary web browsing. Without deep packet inspection (DPI) or supervised target labels, Isolation Forest cannot isolate them.

---

## 7. Comparison with Supervised Intrusion Detection

### Supervised Benchmark Consistency & Verification
To ensure methodological rigor, the supervised reference metrics were verified directly against Phase 3B records:
* **Phase 3B Baseline Model (`rf_full`) on Phase 3B Binary Validation Set**:
  - Precision: **0.99504**
  - Recall: **0.99852**
  - F1-Score: **0.99678**
* **Phase 3B Baseline Model (`rf_full`) Evaluated on Phase 3D Multiclass Validation Set ($N_{val}=399,906$)**:
  - Precision: **0.99505**
  - Recall: **0.99947**
  - F1-Score: **0.99725**
  - Per-class recall: DoS_DDoS = $99.97\%$, PortScan = $99.99\%$, BruteForce = $99.93\%$, WebAttack = $99.13\%$, Botnet = $95.19\%$, Infiltration = $100.0\%$, Heartbleed = $100.0\%$.

### Side-by-Side Comparison

| Evaluation Dimension | Supervised Binary (`rf_full`, Phase 3B) | Unsupervised Anomaly (`iforest_full_100k`, Phase 3D @ 1% FPR) |
| :--- | :--- | :--- |
| **Training Labels** | Requires extensive, labeled attack instances | **Requires zero attack labels** (fits strictly on benign baseline) |
| **Overall Attack Recall** | **99.85%** (Phase 3B) / **99.95%** (on Phase 3D val) | 22.83% (Concentrated in volumetric anomalies) |
| **Overall Precision** | **99.50%** | 81.55% |
| **Botnet Recall** | **95.19%** | **0.00%** (Missed completely) |
| **BruteForce Recall** | **99.93%** | **0.00%** (Missed completely) |
| **WebAttack Recall** | **99.13%** | **0.00%** (Missed completely) |
| **DoS/DDoS Recall** | **99.97%** | 30.21% (at 1% FPR) / 54.10% (at 5% FPR) |
| **Evaluation Scope** | In-distribution evaluation on known attack patterns | Unsupervised ranking on in-distribution flow deviations |

> [!CAUTION]
> **No Zero-Day Resilience Claim**:
> While unsupervised anomaly detection does not require attack labels during fitting, **this experiment does not establish zero-day resilience or generalisation to unseen real-world attacks**. Anomaly detection detects statistical outliers on the CICIDS2017 dataset; novel real-world threats that mimic benign flow dynamics will remain undetected without application-layer telemetry.

---

## 8. Technical Limitations & Experimental Realities

1. **In-Distribution Offline Flow Dataset**:
   Evaluations reflect pre-computed flow summary statistics from the 2017 synthetic network. Live deployment requires streaming feature extraction and continuous baseline drift adaptation.
2. **Anomalies Are Not Necessarily Attacks**:
   Unusual legitimate network activity (e.g. database backups, large file transfers) will produce high anomaly scores. Anomaly detection outputs must be treated as telemetry indicators rather than definitive security alerts.
3. **Severe Small-Sample Uncertainty for Rare Classes**:
   * **Heartbleed**: Exactly 2 validation samples. While detection was $100\%$, point estimates carry wide statistical variance.
   * **Infiltration**: Exactly 6 validation samples. A single misclassified flow shifts recall by $\pm 16.7\%$.
4. **Batch Throughput vs. Streaming Reality**:
   The measured inference throughput ($550,000\text{ flows/s}$) reflects vectorized batch execution on cached memory. Real-world inline deployment requires dedicated streaming pipeline integration.

---

## 9. Deliverables & Verification Artifacts

### Machine-Readable Results File
* [`vyuha/reports/anomaly_benchmark_results.json`](file:///Users/varshasri/VYUHA/vyuha/reports/anomaly_benchmark_results.json): Contains complete numerical results, calibrated thresholds, per-class detection rates, ROC/PR curve points, and latency measurements across all 5 experimental models.

### Publication-Quality Diagnostic Figures
All generated plots are stored in [`vyuha/reports/figures/`](file:///Users/varshasri/VYUHA/vyuha/reports/figures/):
1. [`17_anomaly_score_distributions.png`](file:///Users/varshasri/VYUHA/vyuha/reports/figures/17_anomaly_score_distributions.png): Horizontal distribution chart showing median, mean, range, and calibrated threshold boundaries across all 8 traffic categories.
2. [`18_anomaly_roc_pr_curves.png`](file:///Users/varshasri/VYUHA/vyuha/reports/figures/18_anomaly_roc_pr_curves.png): Dual ROC and Precision-Recall curves with calibrated operating points marked.
3. [`19_anomaly_per_attack_recall.png`](file:///Users/varshasri/VYUHA/vyuha/reports/figures/19_anomaly_per_attack_recall.png): Grouped bar chart illustrating per-attack detection rates across 5%, 1%, and 0.5% FPR thresholds.
4. [`20_anomaly_feature_and_size_comparison.png`](file:///Users/varshasri/VYUHA/vyuha/reports/figures/20_anomaly_feature_and_size_comparison.png): Comparative bar chart evaluating training subset sizes (50k/100k/200k), feature sets (Full vs Port-Agnostic), and scaling invariance.

### Serialized Model Artifacts
Trained model artifacts are persisted in `vyuha/models/` (gitignored):
* `models/iforest_full_50k.joblib` ($0.705\text{ MB}$)
* `models/iforest_full_100k.joblib` ($0.708\text{ MB}$)
* `models/iforest_full_200k.joblib` ($0.743\text{ MB}$)
* `models/iforest_port_100k.joblib` ($0.721\text{ MB}$)
* `models/iforest_full_scaled_100k.joblib` ($0.709\text{ MB}$)

### Source Code & Unit Tests
* [`vyuha/src/models/train_anomaly.py`](file:///Users/varshasri/VYUHA/vyuha/src/models/train_anomaly.py): Benign-only partitioning, pipeline construction, and threshold calibration.
* [`vyuha/src/models/evaluate_anomaly.py`](file:///Users/varshasri/VYUHA/vyuha/src/models/evaluate_anomaly.py): Scoring, threshold evaluation, and diagnostic plotting.
* [`vyuha/scripts/run_anomaly_experiments.py`](file:///Users/varshasri/VYUHA/vyuha/scripts/run_anomaly_experiments.py): Benchmark orchestration script.
* [`vyuha/tests/test_anomaly_models.py`](file:///Users/varshasri/VYUHA/vyuha/tests/test_anomaly_models.py): Complete test suite (All 34 project tests passing).
