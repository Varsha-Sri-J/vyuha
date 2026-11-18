# VYUHA: Binary Intrusion Detection (Phase 3B Report)

**Project**: VYUHA — AI-Powered Cybersecurity for Small Industries  
**Phase**: Phase 3B: Supervised Binary Intrusion Detection  
**Target Variable**: `label_binary` (`0` = `BENIGN`, `1` = `ATTACK`)  
**Dataset**: Canadian Institute for Cybersecurity (UNB) - CICIDS2017 (MachineLearningCVE)  
**Status**: Technical Audit & Verification Completed  
**Date**: November 2026  

---

## 1. Executive Summary & Experimental Scope

Phase 3B constructs, validates, and diagnoses supervised machine learning models that classify network flows into `BENIGN` ($0$) or `ATTACK` ($1$). 

Four distinct model configurations were benchmarked across identical training and validation partitions:
1. **Random Forest (Full Feature Set, 63 features)**: Evaluates ensemble performance with complete flow dynamics including `Destination Port`.
2. **Random Forest (Port-Agnostic, 62 features)**: Evaluates ensemble performance strictly on behavioural timing, packet size, and flag metrics with `Destination Port` pruned.
3. **Decision Tree (Full Feature Set, 63 features)**: Evaluates a single-tree architecture with complete flow dynamics.
4. **Decision Tree (Port-Agnostic, 62 features)**: Evaluates single-tree behavioural detection without port shortcuts.

### Key Empirical Findings
* **High In-Distribution Detection**: All four configurations achieve $F_1 > 0.995$ and $\text{PR-AUC} > 0.997$ on the 400,000-flow validation set.
* **Port-Agnostic Viability**: Removing `Destination Port` incurs only a minor performance change ($\Delta F_1 = -0.00099$ for Random Forest; $\Delta F_1 = -0.00040$ for Decision Tree). Network flow dynamics alone (inter-arrival times, packet variance, TCP flags) demonstrate high discriminative power on this benchmark.
* **Inference Latency Profile**: Batch inference divides total batch time over hundreds of thousands of samples, achieving sub-microsecond throughput ($0.17\ \mu\text{s/flow}$ for DT; $0.93\ \mu\text{s/flow}$ for RF). However, single-flow streaming inference ($N=1$) exhibits Python/pipeline invocation overhead ($592\ \mu\text{s}$ median for DT; $14.3\text{ ms}$ median for RF), illustrating the critical distinction between offline batch processing and per-packet streaming monitoring.
* **Vulnerability Analysis**: Missed attacks (False Negatives) are concentrated in low-volume, stealthy attack types: `Botnet` (Ares IRC C2, $8\%\text{--}16.5\%$ miss rate) and application-layer `Web Attacks` ($2.5\%\text{--}6.3\%$ miss rate), where packet headers lack volumetric anomalies.

---

## 2. Experimental Setup & Leakage-Safe Validation Strategy

### Strict Test Set Isolation
* Source partition: `datasets/processed/train.parquet` ($1,999,530$ records).
* **Held-Out Test Set Untouched**: `datasets/processed/test.parquet` ($499,883$ records) remained completely sealed, unopened, and unreferenced throughout Phase 3B.

### Stratified Validation Architecture
To evaluate models at scale without data leakage, `train.parquet` was partitioned into an 80/20 in-distribution stratified split:
* **Training Partition ($X_{tr}, y_{tr}$)**: $1,599,624$ flows ($82.96\%$ Benign, $17.04\%$ Malicious).
* **Validation Partition ($X_{val}, y_{val}$)**: $399,906$ flows ($331,768$ Benign, $68,138$ Malicious).
* **Independent Preprocessing & Feature Selection Fitting**:
  The `FeatureSelector` was embedded inside each Scikit-Learn `Pipeline` and fitted **exclusively on $X_{tr}$** during `pipeline.fit(X_tr, y_tr)`. All variance thresholds (pruning 8 constants), collinear twin masks (pruning 6 subflow twins), and optional port exclusions were computed strictly from the 1.6M training slice. The validation slice ($X_{val}$) was transformed using the training-learned mask during `pipeline.predict(X_val)` with zero parameter leakage.
* **Target Isolation**: Target columns (`label_raw`, `label_binary`, `label_category`) were extracted into separate series and asserted absent from $X$ prior to pipeline ingestion.

### Class Imbalance Handling
The training set exhibits an $\sim 83:17$ imbalance. Rather than applying synthetic oversampling (such as SMOTE), which interpolates non-physical continuous values in high-dimensional flow space and incurs severe memory overhead on 1.6M rows, models were trained using **algorithmic cost-sensitive weighting** (`class_weight="balanced"`):
$$w_c = \frac{N}{2 \cdot N_c}$$
This penalizes false negatives proportionally ($\sim 2.93\times$ higher penalty for attack misclassifications than benign misclassifications) while preserving true network manifold geometry.

---

## 3. Comprehensive Benchmark Results & Methodology Audit

### Model Hyperparameters
All models were trained with fixed seeds (`random_state=42`) and the following explicit configurations:

| Parameter | Random Forest (`rf_full` / `rf_port_agnostic`) | Decision Tree (`dt_full` / `dt_port_agnostic`) | Rationale |
| :--- | :--- | :--- | :--- |
| **Estimators (`n_estimators`)** | 100 | 1 (Single Tree) | Standard ensemble depth vs. single-tree baseline |
| **Max Depth (`max_depth`)** | 20 | 20 | Constrains model capacity to prevent leaf memorization |
| **Class Weight (`class_weight`)** | `"balanced"` | `"balanced"` | Inversely proportional cost-weighting for ~83/17 skew |
| **Min Samples Split** | 2 (Default) | 2 (Default) | Standard split criterion |
| **Min Samples Leaf** | 1 (Default) | 1 (Default) | Standard terminal leaf size |
| **Max Features** | `"sqrt"` ($\sqrt{d} \approx 8$ features) | `None` (Evaluates all $d$ features) | Random subspace projection in RF vs. exhaustive split in DT |
| **Splitting Criterion** | Gini Impurity | Gini Impurity | Consistent impurity metric across architectures |
| **Multiprocessing (`n_jobs`)** | `-1` (8 cores) | 1 (Single-threaded) | Parallelized tree induction across CPU cores |
| **Random State** | `42` | `42` | Guarantees exact reproducibility |

---

### Master Performance Comparison Table ($N_{val} = 399,906$)

| Metric | Random Forest (Full) | Random Forest (Port-Agnostic) | Decision Tree (Full) | Decision Tree (Port-Agnostic) |
| :--- | :--- | :--- | :--- | :--- |
| **Feature Count** | 63 | 62 | 63 | 62 |
| **Accuracy** | 99.890% | 99.856% | 99.894% | 99.880% |
| **Balanced Accuracy** | 99.875% | 99.784% | 99.906% | 99.876% |
| **Precision** | **99.504%** | 99.486% | 99.454% | 99.429% |
| **Recall (TPR)** | 99.852% | 99.673% | **99.925%** | 99.871% |
| **F1-Score (Binary)** | **0.99678** | 0.99579 | 0.99689 | 0.99649 |
| **F1-Score (Macro)** | 0.99806 | 0.99746 | 0.99812 | 0.99788 |
| **PR-AUC (Avg Precision)** | **0.99984** | **0.99978** | 0.99852 | 0.99729 |
| **ROC-AUC** | **0.99997** | 0.99995 | 0.99937 | 0.99918 |
| **False Positive Rate (FPR)** | **0.1022%** | **0.1058%** | 0.1127% | 0.1179% |
| **False Negative Rate (FNR)** | 0.1482% | 0.3273% | **0.0748%** | 0.1291% |
| **Training Time (s)** | 66.33s | 72.31s | 53.34s | **47.55s** |
| **Vectorized Batch Latency** | $0.926\ \mu\text{s/flow}$ | $0.944\ \mu\text{s/flow}$ | $0.173\ \mu\text{s/flow}$ | **$0.177\ \mu\text{s/flow}$** |
| **Single-Flow Streaming Latency** | $14,334\ \mu\text{s/flow}$ ($14.3\text{ ms}$) | $14,334\ \mu\text{s/flow}$ ($14.3\text{ ms}$) | $602\ \mu\text{s/flow}$ ($0.60\text{ ms}$) | **$592\ \mu\text{s/flow}$ ($0.59\text{ ms}$)** |
| **Serialized Model Size** | 7.01 MB | 7.79 MB | **0.04 MB** | **0.05 MB** |

### Confusion Matrix Breakdown ($N_{val} = 399,906$)

| Model Configuration | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest (Full)** | 331,429 ($99.90\%$) | 339 ($0.10\%$) | 101 ($0.15\%$) | 68,037 ($99.85\%$) |
| **Random Forest (Port-Agnostic)** | 331,417 ($99.89\%$) | 351 ($0.11\%$) | 223 ($0.33\%$) | 67,915 ($99.67\%$) |
| **Decision Tree (Full)** | 331,394 ($99.89\%$) | 374 ($0.11\%$) | 51 ($0.07\%$) | 68,087 ($99.93\%$) |
| **Decision Tree (Port-Agnostic)** | 331,377 ($99.88\%$) | 391 ($0.12\%$) | 88 ($0.13\%$) | 68,050 ($99.87\%$) |

---

### Latency Benchmark Audit: Batch Throughput vs. Streaming Latency

A controlled inference benchmark was executed to prevent conflating batch throughput with per-packet streaming latency:
* **Warm-up**: 3 warm-up iterations executed on 10,000 samples prior to timing.
* **Scope**: In-memory models (model deserialization excluded); pipeline preprocessing (`FeatureSelector.transform`) is fully included.
* **Vectorized Batch Mode**: Measured over 5 repetitions across $N = 399,906$ validation flows. Per-flow latency is the median run time divided by $N$.
* **Single-Flow Streaming Mode**: Evaluated by passing single individual flow rows ($N = 1$) across 1,000 distinct iterations, reporting Median, Mean, 95th, and 99th percentiles.

#### Batch Size Scaling & Streaming Latency Comparison

| Inference Mode | Batch Size | RF Full (`rf_full`) | DT Full (`dt_full`) | Architectural Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **Large Batch** | 399,906 | $0.926\ \mu\text{s/flow}$ | $0.173\ \mu\text{s/flow}$ | C-level Cython loops, SIMD vectorization, threadpool amortized over 400k flows. |
| **Medium Batch** | 10,000 | $2.875\ \mu\text{s/flow}$ | $0.225\ \mu\text{s/flow}$ | Efficient multi-sample vectorization with minor threadpool overhead. |
| **Small Batch** | 1,000 | $14.550\ \mu\text{s/flow}$ | $0.760\ \mu\text{s/flow}$ | Thread dispatch and Python function invocation start to dominate. |
| **Micro Batch** | 100 | $145.701\ \mu\text{s/flow}$ | $6.626\ \mu\text{s/flow}$ | Python slicing overhead exceeds tree traversal computation. |
| **Streaming ($N=1$) Median** | 1 | **$14,340.5\ \mu\text{s}$ ($14.34\text{ ms}$)** | **$601.7\ \mu\text{s}$ ($0.60\text{ ms}$)** | Python DataFrame indexing + OpenMP threadpool synchronization for 100 trees in RF. |
| **Streaming ($N=1$) P95** | 1 | $14,654.9\ \mu\text{s}$ ($14.65\text{ ms}$) | $658.0\ \mu\text{s}$ ($0.66\text{ ms}$) | 95th percentile worst-case single-flow evaluation latency. |

> [!IMPORTANT]
> Sub-microsecond figures represent **amortized batch throughput** during bulk offline processing. They do **not** represent single-packet wire latency. In streaming network architectures, passing individual flows through Python scikit-learn incurs $\sim 0.60\text{ ms}$ for a Decision Tree and $\sim 14.3\text{ ms}$ for Random Forest. For true microsecond streaming inspection, decision logic must be compiled to native C, eBPF, or hardware FPGA tables.

---

## 4. Model Architecture & Feature Set Trade-offs

Rather than declaring a single model "best", the architectures present distinct engineering trade-offs:

### Random Forest vs. Decision Tree Trade-off Matrix

| Criterion | Random Forest (100 Trees) | Decision Tree (Single Tree) | Operational Significance |
| :--- | :--- | :--- | :--- |
| **Probability Calibration** | **Superior ($\text{PR-AUC} = 0.9998$)** | Good ($\text{PR-AUC} = 0.9985$) | RF enables continuous threshold adjustment to tune FP vs FN rates. |
| **Variance & Generalization** | **High stability (Bagging + Feature Subspaces)** | Higher variance (Greedy splits) | RF resists overfitting to noise and peculiar traffic spikes. |
| **Streaming Latency ($N=1$)** | $14.3\text{ ms}$ (High threadpool overhead) | **$0.60\text{ ms}$ ($24\times$ faster)** | DT handles individual real-time events with far lower latency. |
| **Memory / Disk Size** | $7.0\text{--}7.8\text{ MB}$ | **$40\text{--}55\text{ KB}$ ($150\times$ smaller)** | DT can be embedded on micro-controllers or low-RAM industrial routers. |
| **Explainability** | Black-box voting ensemble | **White-box transparent tree** | DT rules can be audited and directly exported as Snort/Suricata rules. |

### Impact of Excluding `Destination Port`
* **Behavioural Independence**: In both model families, the $F_1$ drop is negligible ($\Delta F_1 = -0.00099$ for RF, $\Delta F_1 = -0.00040$ for DT). This confirms that flow duration, inter-arrival times, packet variance, and TCP flags contain sufficient behavioural information to detect attacks.
* **Brute-Force Dependency**: The primary vulnerability introduced by port-agnostic modeling occurs in credential brute-forcing: in `rf_full`, only 2 `SSH-Patator` flows were missed, whereas in `rf_port_agnostic`, misses jumped to 24 (and `Web Attack Brute Force` misses jumped from 2 to 14). Because individual password attempts mimic legitimate small-payload TCP exchanges, destination ports 22 and 80 act as strong contextual priors in the Full model. In deployment, port-agnostic models require temporal rate-limiting rules to detect distributed credential stuffing.

---

## 5. False-Positive Projections & The Base Rate Fallacy

The False Positive Rate ($FPR = \frac{FP}{TN + FP}$) applies strictly to **benign flows**, not total network volume. To understand operational impact in production, two deployment scenarios are presented alongside Bayes' Theorem calculations:

### Scenario A: 10 Million Benign Flows per Day
Assuming a monitoring environment processing 10,000,000 benign flows daily:

$$\text{Daily False Positives} = 10,000,000 \times FPR$$

* **`rf_full` ($FPR = 0.1022\%$)**: **10,220 false alarms/day** ($\sim 426\text{ alarms/hour}$, $\sim 7.1\text{ alarms/min}$).
* **`rf_port_agnostic` ($FPR = 0.1058\%$)**: **10,580 false alarms/day** ($\sim 441\text{ alarms/hour}$, $\sim 7.3\text{ alarms/min}$).
* **`dt_full` ($FPR = 0.1127\%$)**: **11,270 false alarms/day** ($\sim 470\text{ alarms/hour}$, $\sim 7.8\text{ alarms/min}$).
* **`dt_port_agnostic` ($FPR = 0.1179\%$)**: **11,790 false alarms/day** ($\sim 491\text{ alarms/hour}$, $\sim 8.2\text{ alarms/min}$).

### Scenario B: 10 Million Total Flows per Day (Illustrative Benign Prevalence)
Assuming a network processing 10,000,000 total flows per day, using the validation set's observed benign prevalence ($82.96\%$) as an illustrative baseline ($8,296,150$ benign flows):

$$\text{Daily False Positives} = 8,296,150 \times FPR$$

* **`rf_full`**: **8,479 false alarms/day** ($\sim 353\text{ alarms/hour}$).
* **`rf_port_agnostic`**: **8,777 false alarms/day** ($\sim 366\text{ alarms/hour}$).
* **`dt_full`**: **9,350 false alarms/day** ($\sim 390\text{ alarms/hour}$).
* **`dt_port_agnostic`**: **9,781 false alarms/day** ($\sim 408\text{ alarms/hour}$).

---

### Expected Precision Across Hypothetical Attack Prevalences (Bayes' Theorem)

In production networks, true malicious traffic rarely accounts for $17\%$ of total traffic. Under varying hypothetical attack prevalences $\alpha \in \{10\%, 1\%, 0.1\%, 0.01\%\}$, expected operational precision is governed by Bayes' Theorem:

$$\text{Precision} = \frac{\text{Recall} \cdot \alpha}{\text{Recall} \cdot \alpha + \text{FPR} \cdot (1 - \alpha)}$$

#### Mathematical Sensitivity Table

| Model Configuration | Measured Recall | Measured FPR | Prev = 10.0% | Prev = 1.0% | Prev = 0.1% (1 in 1k) | Prev = 0.01% (1 in 10k) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest (Full)** | 0.9985 | 0.001022 | **99.09%** | **90.80%** | **49.44%** | **8.90%** |
| **Random Forest (Port-Agnostic)** | 0.9967 | 0.001058 | 99.05% | 90.49% | 48.53% | 8.61% |
| **Decision Tree (Full)** | 0.9993 | 0.001127 | 99.00% | 89.96% | 47.02% | 8.15% |
| **Decision Tree (Port-Agnostic)** | 0.9987 | 0.001179 | 98.95% | 89.54% | 45.89% | 7.81% |

> [!NOTE]
> **Why Production Precision Collapses (The Base Rate Fallacy)**:
> In benchmark evaluation where attack prevalence is $17\%$, precision is $99.5\%$. However, if attack prevalence drops to $0.1\%$ (1 attack per 1,000 flows), precision falls to $\sim 48\%$, meaning **more than half of all alerts are false alarms**. At $0.01\%$ prevalence (1 in 10,000 flows, common in quiet industrial automation networks), precision collapses to $< 9\%$, meaning over **91% of alerts are false alarms**. These calculations demonstrate why raw model accuracy is insufficient for real-world operations and justify VYUHA's upcoming alert aggregation and correlation layer (Phase 4).

---

## 6. Error Analysis Audit & Small-Sample Caveats

An exact audit of the $68,138$ attack records in the validation partition was conducted against model predictions:

### Ground-Truth Validation Attack Distribution ($N_{attack} = 68,138$)

| Attack Family / Category | Validation Count ($N_{val}$) | RF Full Misses (FNR) | RF Port-Agnostic Misses (FNR) | DT Full Misses (FNR) | DT Port-Agnostic Misses (FNR) | Evaluation Sample Context & Failure Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DoS / DDoS** | 51,424 | 41 ($0.08\%$) | 120 ($0.23\%$) | 19 ($0.04\%$) | 39 ($0.08\%$) | **Larger evaluation sample ($N=51,424$)**. Volumetric floods are near-trivially detected. Misses occur in slow DoS (`DoS Slowhttptest`, `slowloris`). |
| **PortScan** | 14,572 | 2 ($0.01\%$) | 2 ($0.01\%$) | 3 ($0.02\%$) | 2 ($0.01\%$) | **Larger evaluation sample ($N=14,572$)**. SYN scan durations and flag distributions are near-unanimously recognized. |
| **Brute Force** | 1,467 | 2 ($0.14\%$) | 26 ($1.77\%$) | 1 ($0.07\%$) | 5 ($0.34\%$) | **Moderate evaluation sample ($N=1,467$)**. SSH/FTP attempts mimic benign handshakes when port numbers are excluded. |
| **Web Attack** | 363 | 9 ($2.48\%$) | 23 ($6.34\%$) | 4 ($1.10\%$) | 11 ($3.03\%$) | **Moderate evaluation sample ($N=363$)**. Payload text (XSS/SQLi) resides inside HTTP parameters; flow headers appear completely normal. |
| **Botnet (`Bot`)** | 302 | **46 (15.23%)** | **50 (16.56%)** | **24 (7.95%)** | **30 (9.93%)** | **Moderate evaluation sample ($N=302$)**. Ares IRC C2 keep-alives generate normal packet sizes with zero burst volume. |
| **Infiltration** | 5 | 1 ($20.00\%$) | 2 ($40.00\%$) | 0 ($0.00\%$) | 1 ($20.00\%$) | **Small evaluation sample; results are descriptive and uncertain ($N=5$)**. A single misclassification shifts the error rate by $\pm 20\%$. |
| **Heartbleed** | 5 | 0 ($0.00\%$) | 0 ($0.00\%$) | 0 ($0.00\%$) | 0 ($0.00\%$) | **Small evaluation sample; results are descriptive and uncertain ($N=5$)**. 100% recall on 5 samples cannot confirm real-world exploit generalization. |

### False Positive Root Cause
* False Positives ($339\text{--}391$ flows) are primarily triggered by legitimate traffic on **Port 8080** ($\sim 85\text{ FPs}$), **Port 80** ($40\text{--}88\text{ FPs}$), and **Port 443** ($19\text{--}55\text{ FPs}$). Rapid client browsing, proxy multiplexing, and parallel asset fetching generate bursty inter-arrival times that mirror application-layer HTTP DoS attacks.

---

## 7. Limitations of CICIDS2017 & Production Realities

1. **In-Distribution Evaluation**:
   The validation split evaluates traffic from the exact same synthetic testbed environment recorded during the 5-day capture in 2017. High laboratory accuracy does **not** demonstrate zero-day generalization to unseen corporate or industrial networks.
2. **Offline Flow Summaries vs. Streaming Packets**:
   CICIDS2017 features are post-facto flow summaries computed by CICFlowMeter after connection termination. This phase validates offline flow classification, not real-time streaming packet capture or inline deep packet inspection (DPI).
3. **Topology and Protocol Artifacts**:
   The victim and attacking subnets in CICIDS2017 represent a synthetic laboratory network. Features like destination port and local IP subnet configurations reflect the testbed's specific topology rather than universal industrial realities.

---

## 8. Deliverables & Diagnostic Artifacts

### Machine-Readable Results File
* [`vyuha/reports/binary_benchmark_results.json`](file:///Users/varshasri/VYUHA/vyuha/reports/binary_benchmark_results.json): Contains complete numerical results, confusion matrices, latency benchmarks, and error breakdowns.

### Publication-Quality Diagnostic Plots
All generated figures are stored in [`vyuha/reports/figures/`](file:///Users/varshasri/VYUHA/vyuha/reports/figures/):
1. [`09_binary_confusion_matrices.png`](file:///Users/varshasri/VYUHA/vyuha/reports/figures/09_binary_confusion_matrices.png): 2x2 grid displaying raw flow counts and normalized percentages for all 4 models.
2. [`10_binary_precision_recall_curves.png`](file:///Users/varshasri/VYUHA/vyuha/reports/figures/10_binary_precision_recall_curves.png): Precision-Recall and ROC curves demonstrating high PR-AUC across decision thresholds.
3. [`11_port_vs_agnostic_comparison.png`](file:///Users/varshasri/VYUHA/vyuha/reports/figures/11_port_vs_agnostic_comparison.png): Side-by-side grouped bar charts comparing F1, PR-AUC, Recall, Precision, and the latency vs. FPR operational trade-off.
4. [`12_error_analysis_breakdown.png`](file:///Users/varshasri/VYUHA/vyuha/reports/figures/12_error_analysis_breakdown.png): Miss rate breakdown across attack categories, highlighting botnet and web attack vulnerabilities.

### Serialized Model Artifacts
Trained models are persisted in `vyuha/models/` (gitignored, outside dataset directories):
* `models/rf_full.joblib` ($7.01\text{ MB}$)
* `models/rf_port_agnostic.joblib` ($7.79\text{ MB}$)
* `models/dt_full.joblib` ($0.04\text{ MB}$)
* `models/dt_port_agnostic.joblib` ($0.05\text{ MB}$)

### Source Code & Unit Tests
* [`vyuha/src/models/train_binary.py`](file:///Users/varshasri/VYUHA/vyuha/src/models/train_binary.py)
* [`vyuha/src/models/evaluate_binary.py`](file:///Users/varshasri/VYUHA/vyuha/src/models/evaluate_binary.py)
* [`vyuha/scripts/run_binary_experiments.py`](file:///Users/varshasri/VYUHA/vyuha/scripts/run_binary_experiments.py)
* [`vyuha/tests/test_binary_models.py`](file:///Users/varshasri/VYUHA/vyuha/tests/test_binary_models.py) (All 22 unit tests passing).
