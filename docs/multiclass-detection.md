# VYUHA: Multiclass Intrusion Detection & Class-Imbalance Audit (Phase 3C Report)

**Project**: VYUHA — AI-Powered Cybersecurity for Small Industries  
**Phase**: Phase 3C: Multiclass Attack-Family Classification & Class-Imbalance Audit  
**Target Variable**: `label_category` (8 classes: `BENIGN` + 7 Attack Families)  
**Dataset**: Canadian Institute for Cybersecurity (UNB) - CICIDS2017 (MachineLearningCVE)  
**Status**: Completed, Audited & Verified  
**Date**: November 2026  

---

## 1. Executive Summary & Experimental Scope

Phase 3C extends VYUHA from binary detection (`BENIGN` vs `ATTACK`) to **granular attack-family classification**. The objective is to evaluate supervised machine-learning models classifying network flow records into their specific threat families:

1. `BENIGN`: Legitimate baseline network traffic
2. `DoS_DDoS`: Volumetric and resource-starvation denial-of-service attacks
3. `PortScan`: Reconnaissance and host scanning
4. `BruteForce`: Automated dictionary credential attempts (FTP and SSH)
5. `WebAttack`: Web vulnerability injection (SQLi, XSS, HTTP Brute Force)
6. `Botnet`: Ares IRC Command-and-Control (C2) communication
7. `Infiltration`: Multi-stage internal network compromise and lateral movement
8. `Heartbleed`: OpenSSL TLS buffer over-read vulnerability exploitation

### Initial Four-Model Benchmark vs. Additional Six Audit Experiments
The experimental investigation was conducted in two structured stages:
1. **Original Four-Model Baseline Benchmark** (documented in Section 3): Evaluated Random Forest (RF) and Decision Tree (DT) across Full (63 features) and Port-Agnostic (62 features) sets using standard cost-sensitive balanced weighting (`class_weight="balanced"`).
2. **Additional Six Class-Weighting Audit Experiments** (documented in Section 4): Investigated the extreme sample skews (ranging from $1.3\text{M}$ benign flows down to $7$ Heartbleed flows in the training split) by benchmarking:
   - **No Class Weighting** (`class_weight=None` across RF Full, RF Port-Agnostic, DT Full, and DT Port-Agnostic).
   - **Balanced Subsample Weighting** (`class_weight="balanced_subsample"` across RF Full and RF Port-Agnostic).
   These six new experiments, combined with the four baseline models, form the comprehensive 10-model audit matrix.

### Key Empirical Findings
* **The Multiclass Gap**: While binary detection achieved $>99.6\%$ F1 across all models in Phase 3B, multiclass attack-family identification reveals substantial difficulty: baseline Macro-F1 under balanced weighting ranged from $0.7977$ to $0.9188$.
* **The Deceptive Nature of Weighted Metrics**: Overall Accuracy ($>99.4\%$) and Weighted-F1 ($>0.995\%$) remain misleadingly high because majority classes (`BENIGN` and `DoS_DDoS`) represent $>95\%$ of all flows. **Macro-F1 and per-class metrics are the only honest indicators of performance.**
* **The Botnet False-Alarm Explosion Under Balanced Weighting**: With `class_weight="balanced"`, the theoretical sample weight assigned to Botnet ($w \approx 160.0$) is $1,061\times$ larger than `BENIGN` ($w \approx 0.15$). This massive loss penalty for missed botnets caused models to aggressively misclassify routine benign small-packet flows as botnet traffic:
  - `multiclass_rf_full (balanced)`: **1,087 BENIGN flows** misclassified as Botnet (Precision: $21.97\%$).
  - `multiclass_rf_port_agnostic (balanced)`: **1,905 BENIGN flows** misclassified as Botnet (Precision: $13.84\%$).
* **Unweighted Training Mitigates the False-Alarm Flood**:
  - In `rf_full_none` (`class_weight=None`), Botnet false alarms plummeted by **$98.3\%$ (from 1,087 down to 18)**.
  - Botnet Precision increased from **$21.97\%$ to $90.67\%$**.
  - Overall Macro-F1 reached **$0.9574$**—the highest multiclass Macro-F1 observed in the project.
  - The trade-off: Botnet Recall dropped from $98.08\%$ to $56.09\%$ ($175$ of $312$ detected).
* **Port-Agnostic Collapse in Single Decision Trees vs. Ensemble Resilience**:
  - In single Decision Trees without destination port, `dt_port_balanced` assigned $w_{\text{Infiltration}} \approx 8,693.6$, causing the tree to overfit noisy timing features and flag **75 BENIGN flows as Infiltration** (Precision: $2.60\%$, F1: $0.0482$).
  - Unweighted training (`dt_port_none`) eliminated this collapse (false positives dropped from 75 to 2; Precision: $66.67\%$, F1: $0.6667$).
  - Random Forest resisted this collapse entirely across all weighting modes, maintaining $100\%$ precision on Infiltration due to feature subsampling and bootstrap bagging.
* **Scope Limitation**: Because CICIDS2017 is an offline, pre-computed benchmark dataset from a 2017 synthetic network environment, these results provide an experimental comparison of model behavior and do not establish production readiness or live deployment viability.

---

## 2. Dataset Constraints & Stratified Validation Architecture

### Strict Test Set Isolation
* Source partition: `datasets/processed/train.parquet` ($1,999,530$ records).
* **Held-Out Test Set Untouched**: `datasets/processed/test.parquet` ($499,883$ records) remained completely sealed, unopened, and unreferenced throughout Phase 3C.

### Stratified Multiclass Split Breakdown
The training dataset was partitioned into an 80/20 stratified split based on `label_category`:
* **Training Partition ($X_{tr}, y_{tr}$)**: $1,599,624$ flows.
* **Validation Partition ($X_{val}, y_{val}$)**: $399,906$ flows.

#### Per-Class Split Counts & Sample-Size Context

| Target Category | Total in `train.parquet` | Training Slice ($X_{tr}, 80\%$) | Validation Slice ($X_{val}, 20\%$) | Validation Prevalence | Evaluation Sample Context |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BENIGN** | 1,658,840 | 1,327,072 | 331,768 | 82.9615% | Larger evaluation sample |
| **DoS_DDoS** | 257,399 | 205,919 | 51,480 | 12.8730% | Larger evaluation sample |
| **PortScan** | 72,655 | 58,124 | 14,531 | 3.6336% | Larger evaluation sample |
| **BruteForce** | 7,322 | 5,858 | 1,464 | 0.3661% | Moderate evaluation sample |
| **WebAttack** | 1,714 | 1,371 | 343 | 0.0858% | Moderate evaluation sample |
| **Botnet** | 1,562 | 1,250 | 312 | 0.0780% | Moderate evaluation sample |
| **Infiltration** | 29 | 23 | 6 | 0.0015% | Small evaluation sample; descriptive/uncertain |
| **Heartbleed** | 9 | 7 | 2 | 0.0005% | Small evaluation sample; descriptive/uncertain |
| **Total** | **1,999,530** | **1,599,624** | **399,906** | **100.0000%** | **8 Distinct Classes Evaluated** |

### Leakage Prevention
Feature selection (`FeatureSelector`) was encapsulated inside each model pipeline and fitted **strictly on $X_{tr}$**. All variance thresholds (pruning 8 constants) and collinear twin masks (pruning 6 subflow twins) were learned exclusively from the training slice. The validation partition was transformed using the training-learned mask with zero parameter leakage.

### Mathematical Derivation of Effective Class Weights
In cost-sensitive learning (`class_weight="balanced"`), scikit-learn computes sample weights inversely proportional to class frequencies:
$$w_c = \frac{N}{K \cdot N_c}$$
Where $N = 1,599,624$ (total training samples) and $K = 8$ (number of classes).

Evaluating this equation on the exact training counts yields:
* **BENIGN**: $N_c = 1,327,072 \implies \mathbf{w = 0.15067}$
* **DoS_DDoS**: $N_c = 205,919 \implies \mathbf{w = 0.97103}$
* **PortScan**: $N_c = 58,124 \implies \mathbf{w = 3.44011}$
* **BruteForce**: $N_c = 5,858 \implies \mathbf{w = 34.13332}$
* **WebAttack**: $N_c = 1,371 \implies \mathbf{w = 145.84464}$
* **Botnet**: $N_c = 1,250 \implies \mathbf{w = 159.96240}$ ($1,061.7\times$ BENIGN)
* **Infiltration**: $N_c = 23 \implies \mathbf{w = 8,693.60870}$ ($57,699.7\times$ BENIGN)
* **Heartbleed**: $N_c = 7 \implies \mathbf{w = 28,564.71429}$ ($189,584.6\times$ BENIGN)

**Algorithmic Impact of These Ratios**: In tree-based split criteria (Gini impurity or entropy), sample weights multiply each flow's contribution to node loss. Under balanced weighting:
- Misclassifying a single Botnet flow is penalized identically to misclassifying **1,061 Benign flows**.
- Misclassifying a single Infiltration flow is penalized identically to misclassifying **57,700 Benign flows**.

---

## 3. Master Multiclass Benchmark Results (Baseline: Balanced Weighting)

All models were evaluated on the held-out $399,906$-flow validation partition with fixed seed `random_state=42`.

### Aggregate Multiclass Performance Comparison

| Metric | Random Forest (Full) | Random Forest (Port-Agnostic) | Decision Tree (Full) | Decision Tree (Port-Agnostic) |
| :--- | :--- | :--- | :--- | :--- |
| **Feature Count** | 63 | 62 | 63 | 62 |
| **Class Weighting** | `balanced` | `balanced` | `balanced` | `balanced` |
| **Macro F1-Score** | **0.90492** | **0.87448** | **0.91884** | **0.79772** |
| **Balanced Accuracy** | 0.97207 | 0.94741 | **0.99309** | 0.90954 |
| **Accuracy** | 99.643% | 99.405% | **99.780%** | 99.671% |
| **Weighted F1-Score** | 0.99730 | 0.99584 | **0.99811** | 0.99725 |
| **Macro Precision** | **0.90002** | 0.88924 | 0.88428 | 0.76478 |
| **Macro Recall** | 0.97207 | 0.94741 | **0.99309** | 0.90954 |
| **Training Time (s)** | 71.28s | 77.80s | 54.33s | **42.16s** |
| **Batch Inference Latency (amortized)** | $1.174\ \mu\text{s/flow}$ | $1.213\ \mu\text{s/flow}$ | **$0.177\ \mu\text{s/flow}$** | **$0.177\ \mu\text{s/flow}$** |
| **Batch Inference Throughput** | $851,833\ \text{flows/s}$ | $824,244\ \text{flows/s}$ | **$5,651,989\ \text{flows/s}$** | **$5,662,904\ \text{flows/s}$** |
| **Serialized Model Size** | 8.35 MB | 9.42 MB | **0.05 MB** | **0.05 MB** |

> [!NOTE]
> **Batch Throughput vs. Single-Flow Streaming Latency**:
> The throughput figures above ($5.65\text{M flows/s}$ for Decision Tree and $\sim 850\text{k flows/s}$ for Random Forest) were measured via **vectorized batch inference** on pre-extracted feature arrays residing in host RAM. This amortized batch measurement does **not** reflect single-flow sequential execution latency in a live streaming packet pipeline, nor does it include packet capture, flow reassembly, or feature extraction overhead. These figures do not demonstrate line-rate readiness or embedded switch deployment feasibility.

---

### Per-Class Performance Breakdown ($N_{val} = 399,906$)

The table below details Precision ($P$), Recall ($R$), and F1-score ($F_1$) across all 8 canonical categories:

| Target Category | Support | RF Full ($P / R / F_1$) | RF Port-Agnostic ($P / R / F_1$) | DT Full ($P / R / F_1$) | DT Port-Agnostic ($P / R / F_1$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BENIGN** | 331,768 | $1.000\ /\ 0.996\ /\ \mathbf{0.9979}$ | $1.000\ /\ 0.993\ /\ \mathbf{0.9964}$ | $1.000\ /\ 0.998\ /\ \mathbf{0.9987}$ | $1.000\ /\ 0.996\ /\ \mathbf{0.9980}$ |
| **DoS_DDoS** | 51,480 | $0.998\ /\ 0.999\ /\ \mathbf{0.9986}$ | $0.997\ /\ 0.998\ /\ \mathbf{0.9975}$ | $0.998\ /\ 0.999\ /\ \mathbf{0.9987}$ | $0.996\ /\ 0.999\ /\ \mathbf{0.9974}$ |
| **PortScan** | 14,531 | $0.989\ /\ 0.999\ /\ \mathbf{0.9939}$ | $0.989\ /\ 0.999\ /\ \mathbf{0.9939}$ | $0.989\ /\ 0.998\ /\ \mathbf{0.9933}$ | $0.989\ /\ 0.999\ /\ \mathbf{0.9939}$ |
| **BruteForce** | 1,464 | $1.000\ /\ 0.997\ /\ \mathbf{0.9986}$ | $0.999\ /\ 0.985\ /\ \mathbf{0.9921}$ | $1.000\ /\ 0.999\ /\ \mathbf{0.9993}$ | $0.984\ /\ 0.997\ /\ \mathbf{0.9902}$ |
| **WebAttack** | 343 | $0.994\ /\ 0.971\ /\ \mathbf{0.9823}$ | $0.991\ /\ 0.956\ /\ \mathbf{0.9733}$ | $0.977\ /\ 0.977\ /\ \mathbf{0.9767}$ | $0.815\ /\ 0.965\ /\ \mathbf{0.8838}$ |
| **Botnet** | 312 | $0.220\ /\ 0.981\ /\ \mathbf{0.3589}$ | $0.138\ /\ 0.981\ /\ \mathbf{0.2426}$ | $0.361\ /\ 0.974\ /\ \mathbf{0.5269}$ | $0.309\ /\ 0.987\ /\ \mathbf{0.4702}$ |
| **Infiltration** | 6 | $1.000\ /\ 0.833\ /\ \mathbf{0.9091}$ | $1.000\ /\ 0.667\ /\ \mathbf{0.8000}$ | $0.750\ /\ 1.000\ /\ \mathbf{0.8571}$ | $0.026\ /\ 0.333\ /\ \mathbf{0.0482}$ |
| **Heartbleed** | 2 | $1.000\ /\ 1.000\ /\ \mathbf{1.0000}$ | $1.000\ /\ 1.000\ /\ \mathbf{1.0000}$ | $1.000\ /\ 1.000\ /\ \mathbf{1.0000}$ | $1.000\ /\ 1.000\ /\ \mathbf{1.0000}$ |

---

## 4. Focused Class-Imbalance & Cost-Weighting Audit

To address the high Botnet false-positive rate and the port-agnostic Infiltration collapse observed in the initial benchmark, a controlled class-weighting audit was executed across the identical 80/20 stratified validation split.

### Experimental Lineage: Baseline vs. Audit Configurations
The overall Phase 3C research encompasses ten distinct configurations clearly divided into two sets:
1. **Original Four Baseline Experiments** (documented in Section 3):
   * `rf_full_balanced` (`class_weight="balanced"`, Full 63 features)
   * `rf_port_balanced` (`class_weight="balanced"`, Port-Agnostic 62 features)
   * `dt_full_balanced` (`class_weight="balanced"`, Full 63 features)
   * `dt_port_balanced` (`class_weight="balanced"`, Port-Agnostic 62 features)
2. **Additional Six Class-Weighting Audit Experiments** (introduced in this audit):
   * `rf_full_none` (`class_weight=None`, Full 63 features — *Preferred Low-False-Positive Candidate*)
   * `rf_full_subsample` (`class_weight="balanced_subsample"`, Full 63 features)
   * `dt_full_none` (`class_weight=None`, Full 63 features)
   * `rf_port_none` (`class_weight=None`, Port-Agnostic 62 features)
   * `rf_port_subsample` (`class_weight="balanced_subsample"`, Port-Agnostic 62 features)
   * `dt_port_none` (`class_weight=None`, Port-Agnostic 62 features)

### Controlled 10-Model Comparison Matrix

All counts and metrics have been verified directly against validation predictions:

| Model Identifier | Experiment Origin | Weighting Strategy | Feats | Macro-F1 | W-F1 | Bal. Acc. | Botnet TP | Botnet FP (Tot / Benign) | Botnet P | Botnet R | Infil TP | Infil FP (Tot / Benign) | Infil P | Infil R |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`rf_full_none`** | **Audit (New)** | **`None`** | **63** | **0.9574** | **0.9987** | 0.9393 | **175** | **18 / 18** | **0.9067** | 0.5609 | **6** | **0 / 0** | **1.0000** | **1.0000** |
| `rf_full_balanced` | Baseline | `balanced` | 63 | 0.9049 | 0.9973 | 0.9721 | 306 | 1,087 / 1,087 | 0.2197 | **0.9808** | 5 | 0 / 0 | 1.0000 | 0.8333 |
| `rf_full_subsample`| Audit (New) | `balanced_subsample` | 63 | 0.8847 | 0.9969 | 0.9515 | 307 | 1,371 / 1,371 | 0.1830 | 0.9840 | 4 | 0 / 0 | 1.0000 | 0.6667 |
| `dt_full_none` | Audit (New) | `None` | 63 | 0.9355 | 0.9987 | 0.9588 | 230 | 45 / 45 | 0.8364 | 0.7372 | 6 | 4 / 4 | 0.6000 | 1.0000 |
| `dt_full_balanced` | Baseline | `balanced` | 63 | 0.9188 | 0.9981 | **0.9931** | 304 | 538 / 538 | 0.3610 | 0.9744 | 6 | 2 / 2 | 0.7500 | 1.0000 |
| **`rf_port_none`** | **Audit (New)** | **`None`** | **62** | **0.9058** | **0.9983** | 0.8598 | **157** | **11 / 11** | **0.9345** | 0.5032 | **3** | **0 / 0** | **1.0000** | 0.5000 |
| `rf_port_balanced` | Baseline | `balanced` | 62 | 0.8745 | 0.9958 | 0.9474 | 306 | 1,905 / 1,905 | 0.1384 | 0.9808 | 4 | 0 / 0 | 1.0000 | 0.6667 |
| `rf_port_subsample`| Audit (New) | `balanced_subsample` | 62 | 0.8932 | 0.9964 | 0.9677 | 305 | 1,531 / 1,531 | 0.1661 | 0.9776 | 5 | 0 / 0 | 1.0000 | 0.8333 |
| `dt_port_none` | Audit (New) | `None` | 62 | 0.8897 | 0.9982 | 0.8699 | 120 | 33 / 32 | 0.7843 | 0.3846 | 4 | 2 / 2 | 0.6667 | 0.6667 |
| `dt_port_balanced` | Baseline | `balanced` | 62 | 0.7977 | 0.9973 | 0.9095 | 308 | 690 / 690 | 0.3086 | 0.9872 | 2 | 75 / 75 | 0.0260 | 0.3333 |

---

### In-Depth Investigation: The Botnet False-Alarm Avalanche

#### 1. The Mathematical Mechanism of Distortion Under Balanced Weighting
In decision tree induction, when evaluating split candidates at node $m$, the impurity reduction $\Delta I$ is weighted by sample weight:
$$\Delta I = I(m) - \frac{N_L^w}{N_m^w} I(L) - \frac{N_R^w}{N_m^w} I(R)$$
Where $N^w = \sum_{i \in \text{node}} w_{y_i}$.

Because $w_{\text{Botnet}} \approx 160.0$ and $w_{\text{BENIGN}} \approx 0.1507$, the loss ratio is:
$$\frac{w_{\text{Botnet}}}{w_{\text{BENIGN}}} = \frac{159.9624}{0.15067} \approx 1,061.67$$

**The Leaf Prediction Consequence**: A terminal leaf predicts class $c$ maximizing the weighted sample sum. Consider a leaf containing **2 True Botnet flows** and **1,000 Benign flows**:
* $\text{Weighted Botnet mass} = 2 \times 159.9624 = 319.92$
* $\text{Weighted Benign mass} = 1,000 \times 0.15067 = 150.67$

Because $319.92 > 150.67$, the leaf declares **`Botnet`** for every sample in that partition. Even though **$99.8\%$ of the actual flows in that leaf are benign**, all 1,000 benign flows are tagged as Botnet.

#### 2. Measured Flow Feature Parity vs. Inferred Explanations
An inspection was conducted on the 1,087 benign flows misclassified as Botnet by `rf_full_balanced`.

##### Directly Measured Summary Flow Features
The table below displays the numerical median values recorded directly in the CICFlowMeter feature matrix:

| Measured Flow Feature | True Botnet ($N=306$) | Benign $\to$ Botnet FP ($N=1,087$) | True Benign ($N=330,681$) | Measured Numerical Finding |
| :--- | :--- | :--- | :--- | :--- |
| **Total Length of Fwd Packets** | Median: **6.0 bytes** | Median: **6.0 bytes** | Median: 68.0 bytes | Identical tiny forward length |
| **Fwd Packet Length Mean** | Median: **6.0 bytes** | Median: **6.0 bytes** | Median: 38.0 bytes | Identical small forward packets |
| **Bwd Packet Length Mean** | Median: **6.0 bytes** | Median: **6.0 bytes** | Median: 85.0 bytes | Identical small backward packets |
| **Packet Length Mean** | Median: **6.0 bytes** | Median: **6.0 bytes** | Median: 61.9 bytes | Identical overall packet size |
| **Init_Win_bytes_backward** | Median: 237.0 bytes | Median: 256.0 bytes | Median: -1.0 (no bwd flow) | Bidirectional TCP established |
| **Destination Port** | Mode: **8080** ($64.4\%$) | Mode: **8080** ($10.9\%$) | Mode: 80 / 443 | Port 8080 numerical overlap |

##### Separation of Inferred Domain Context
> [!IMPORTANT]
> **No Packet Payloads or IRC Contents Were Inspected**:
> We did **not** inspect raw packet payloads, application-layer IRC messages, or raw packet byte sequences. CICIDS2017 documentation identifies the botnet scenario as Ares Botnet IRC traffic. However, our classification pipeline operates solely on tabular summary flow statistics. The confusion occurs because routine benign network events (such as proxy health checks, polling, or keep-alives) generate statistical flow summaries with 6-byte average packet lengths and destination port 8080 that numerically overlap with the botnet flow records.

#### 3. Impact of Unweighted Training (`class_weight=None`)
Under unweighted training ($w_c = 1.0 \ \forall c$), the tree optimizes raw misclassification rate. In leaves where benign flows vastly outnumber botnet flows, the leaf correctly predicts `BENIGN`:
* In `rf_full_none`, Botnet false alarms drop from **1,087 down to 18** ($98.3\%$ reduction).
* Botnet Precision increases from **$21.97\%$ to $90.67\%$**.
* In `rf_port_none`, Botnet false alarms drop from **1,905 down to 11** ($99.4\%$ reduction), with Precision reaching **$93.45\%$**.
* **The Trade-off**: Botnet Recall drops from $98.08\%$ to $56.09\%$ ($175$ detected out of $312$). The 137 missed flows are small-packet records that numerically overlap with benign background noise.

#### 4. Why `balanced_subsample` Fails to Alleviate False Alarms
`class_weight="balanced_subsample"` recomputes class weights independently for each bootstrap sample of each tree. Because rare classes like Botnet ($1,250$ in $1.6\text{M}$) experience Poisson sampling variance across bootstrap draws, individual trees occasionally draw even fewer botnets, assigning even more extreme local weights ($w > 200$). This amplifies split volatility, resulting in **1,371 Botnet false alarms** in Full RF and **1,531** in Port-Agnostic RF.

---

### Infiltration Disintegration vs. Ensemble Resilience

#### 1. The Single Decision Tree Collapse
In `dt_port_balanced`, removing `Destination Port` stripped the single tree of its primary structural separator for Infiltration flows (which targeted port 4444 / SSH). Facing $w_{\text{Infiltration}} \approx 8,693.6$ ($57,700\times$ benign weight), the greedy recursive splitter isolated leaves on noisy microsecond timing variations (`Flow IAT Mean`, `Bwd Packets/s`).
* **The Result**: **75 Benign flows** were classified as Infiltration, causing Precision to disintegrate to **$2.60\%$**, Recall to fall to **$33.33\%$**, and F1 to collapse to **$0.0482$**.
* **The Effect of Unweighted Training**: Under `dt_port_none`, unweighted loss penalizes false alarms equally. Infiltration false alarms dropped from 75 to **2**, restoring Precision to **$66.67\%$**, Recall to **$66.67\%$**, and F1 to **$0.6667$**.

#### 2. The Random Forest Ensemble Defense
In contrast to the single tree, Random Forest (`rf_port_balanced`) maintained **1.0000 Precision** (0 false alarms) and an F1 of **$0.8000$** for Infiltration under balanced weighting, and **1.0000 Precision** (0 false alarms) under unweighted training.
* **Ensemble Explanation**: Random Forest considers only $\sqrt{p} \approx 8$ random features at each split and averages 100 decorrelated trees. A noisy timing split that creates false alarms in one tree is outvoted by the remaining trees, providing structural robustness against extreme class weights.

---

### Rare-Class Empirical Reality & Synthetic Oversampling Discussion

#### 1. Sample Size Constraints and Statistical Uncertainty
* **Heartbleed**: Exactly 7 training flows and **2 validation flows** in the stratified split ($N_{val} = 2$). While all models achieved $100\%$ precision and recall on these 2 flows, this outcome is strictly descriptive. Two flows cannot establish generalizable performance claims against unseen OpenSSL exploit variations.
* **Infiltration**: Exactly 23 training flows and **6 validation flows** in the stratified split ($N_{val} = 6$). A single misclassified flow shifts recall by **$\pm 16.67\%$**. Results carry wide statistical uncertainty and must be treated as descriptive indicators.

#### 2. Synthetic Oversampling (SMOTE) Considerations
Synthetic oversampling techniques (such as SMOTE or ADASYN) were deliberately avoided for the following reasons:
1. **Potential for Unrealistic Feature Combinations**: The CICIDS2017 dataset provides summary statistical flow features pre-computed by CICFlowMeter, rather than raw packet-level captures. In a continuous high-dimensional flow summary space ($p=63$), linear interpolation between sparse data points risks generating synthetic feature combinations (e.g. artificial pairings of flow duration, packet counts, and inter-arrival timing statistics) that do not correspond to physically plausible network flows.
2. **Geometric Artifacts in High Dimensions**: For ultra-rare classes with only 7 or 23 examples, interpolating along line segments in 63-dimensional space populates arbitrary lines in empty space rather than reflecting genuine exploit variability.
3. **Computational Overhead on Millions of Flows**: Generating synthetic records across $1.6\text{M}$ training rows introduces substantial memory and compute overhead without empirical validation that synthetic points improve true detection fidelity.

---

## 5. In-Depth Error Analysis & Threat Confusion Diagnostics

Examining cross-category confusion matrices across all configurations reveals consistent inter-attack boundary dynamics:

### 1. Benign vs. Attack Separation
* Volumetric attacks (`DoS_DDoS`, `PortScan`, `BruteForce`) maintain strong separation from `BENIGN` ($F_1 > 0.992$ across all evaluated models).
* The primary source of multiclass error is concentrated in low-volume attacks (`Botnet` and `WebAttack`), where statistical flow summaries overlap with benign web traffic.

### 2. Inter-Attack Confusions
* **DoS vs. DDoS**: Merged into `DoS_DDoS` during Phase 1 preprocessing because both exhibit identical volumetric flow profiles.
* **WebAttack vs. DoS_DDoS**: In port-agnostic configurations, 6 Web Attack flows were misclassified as `DoS_DDoS` due to low-bandwidth HTTP connection-holding patterns resembling certain DoS tools.
* **PortScan vs. DoS_DDoS**: In `dt_port_balanced`, 5 PortScan flows were classified as `DoS_DDoS`, reflecting rapid SYN scanning that resembles volumetric SYN flooding when destination port information is absent.

---

## 6. Experimental Model-Selection Candidates and Operational Trade-Offs

Because these experiments were conducted on an offline benchmark dataset, the results do **not** establish production readiness or live deployment viability. Instead, they provide an experimental comparison of model behavior under different operational objectives.

### Multi-Criteria Experimental Comparison

| Operational Criterion | `rf_full_none` (Low-False-Positive Candidate) | `rf_full_balanced` (High-Sensitivity Candidate) | `rf_port_none` (Port-Agnostic Candidate) | `dt_full_none` (Fastest Decision Candidate) |
| :--- | :--- | :--- | :--- | :--- |
| **Macro F1-Score** | **0.9574** | 0.9049 | 0.9058 | 0.9355 |
| **Botnet False Alarms ($N_{val}$)** | **18** ($90.7\%$ Precision) | 1,087 ($22.0\%$ Precision) | **11** ($93.5\%$ Precision) | 45 ($83.6\%$ Precision) |
| **Botnet Recall** | 56.09% ($175 / 312$) | **98.08%** ($306 / 312$) | 50.32% ($157 / 312$) | 73.72% ($230 / 312$) |
| **Infiltration F1 ($N_{val}=6$)** | **1.0000** (0 FPs) | 0.9091 (0 FPs) | 0.6667 (0 FPs) | 0.7500 (4 FPs) |
| **Amortized Batch Latency** | $1.23\ \mu\text{s/flow}$ | $1.17\ \mu\text{s/flow}$ | $1.22\ \mu\text{s/flow}$ | **$0.18\ \mu\text{s/flow}$** |
| **Batch Inference Throughput** | $810,000\ \text{flows/s}$ | $852,000\ \text{flows/s}$ | $824,000\ \text{flows/s}$ | **$5,650,000\ \text{flows/s}$** |
| **Model Storage Size** | 8.35 MB | 8.35 MB | 9.42 MB | **0.05 MB** |

---

### Two Primary Experimental Configurations & The Precision/Recall Trade-off

Rather than declaring either configuration universally superior, two primary candidate configurations are documented to illustrate the core precision/recall trade-off:

#### Configuration A: `rf_full_none` (Preferred Low-False-Positive Experimental Candidate)
* **Optimization Objective**: Minimizing false alarms while maximizing overall multiclass Macro-F1 ($0.9574$).
* **Measured Strengths**: Achieves **$90.67\%$ precision on Botnet with only 18 false alarms across $399,906$ validation flows**, $1.0000$ precision on Infiltration ($0$ false alarms), and $1.0000$ precision on Heartbleed ($0$ false alarms).
* **Measured Trade-off**: Botnet recall drops to **$56.09\%$** ($175$ detected out of $312$), leaving $137$ low-volume botnet flows unflagged because they blend statistically into benign background traffic.
* **Experimental Context**: Suitable for experimental evaluations that prioritize alert trust and where false positives incur high operational costs.

#### Configuration B: `rf_full_balanced` (High-Sensitivity Experimental Candidate)
* **Optimization Objective**: Maximizing threat capture rate across minority attack classes.
* **Measured Strengths**: Detects **$98.08\%$ of Botnet flows** ($306$ out of $312$) and achieves a Balanced Accuracy of **$0.9721$**.
* **Measured Trade-off**: Generates **1,087 false alarms on benign flows** attributed to Botnet, resulting in a Botnet precision of **$21.97\%$**.
* **Experimental Context**: Suitable for experimental threat-hunting evaluations where missing potential indicators of compromise carries a higher penalty than triaging false alarms.

---

## 7. Technical Limitations & Experimental Scope Realities

1. **The Multiclass Base Rate Challenge**:
   In high-volume network traffic, false-alarm rates across 8 classes compound. Even a $0.1\%$ false-positive rate generates 1,000 false alarms per million flows. Unweighted training demonstrates one mechanism to manage this skew within an offline benchmark.
2. **Small-Sample Uncertainty for Rare Classes**:
   * **Heartbleed**: Exactly 2 validation samples. While detection was $100\%$, point estimates carry wide statistical variance.
   * **Infiltration**: Exactly 6 validation samples. A single misclassified flow shifts recall by $\pm 16.67\%$.
3. **Offline Pre-Computed Flow Summary Benchmark**:
   Evaluations are conducted on pre-computed flow summary statistics from the synthetic CICIDS2017 network. These experiments demonstrate relative algorithmic trade-offs, but do not represent online streaming packet capture, live protocol decoding, or modern threat landscapes.

---

## 8. Deliverables & Verification Artifacts

### Machine-Readable Benchmark & Audit Results
* [`vyuha/reports/multiclass_benchmark_results.json`](file:///Users/varshasri/VYUHA/vyuha/reports/multiclass_benchmark_results.json): Contains numerical results, 8x8 confusion matrices, per-class metrics, and the audit summary.
* [`vyuha/reports/class_weighting_audit_results.json`](file:///Users/varshasri/VYUHA/vyuha/reports/class_weighting_audit_results.json): Contains complete per-class and confusion metrics across all 10 weighting experiments.

### Publication-Quality Diagnostic Plots
All figures are stored in [`vyuha/reports/figures/`](file:///Users/varshasri/VYUHA/vyuha/reports/figures/):
1. [`13_multiclass_confusion_matrices.png`](file:///Users/varshasri/VYUHA/vyuha/reports/figures/13_multiclass_confusion_matrices.png): 2x2 grid displaying 8x8 normalized heatmaps with raw count overlays.
2. [`14_multiclass_per_class_f1_comparison.png`](file:///Users/varshasri/VYUHA/vyuha/reports/figures/14_multiclass_per_class_f1_comparison.png): Grouped bar chart comparing per-class F1-scores across all 8 categories for the 4 models.
3. [`15_multiclass_port_vs_agnostic.png`](file:///Users/varshasri/VYUHA/vyuha/reports/figures/15_multiclass_port_vs_agnostic.png): Aggregate metrics comparison and per-class F1 drops ($\Delta$) when `Destination Port` is pruned.
4. [`16_multiclass_error_confusion_analysis.png`](file:///Users/varshasri/VYUHA/vyuha/reports/figures/16_multiclass_error_confusion_analysis.png): Miss rate breakdown of attacks escaping detection into BENIGN.

### Implementation Scripts & Test Suite
* [`vyuha/src/models/train_multiclass.py`](file:///Users/varshasri/VYUHA/vyuha/src/models/train_multiclass.py): Multiclass training pipeline with encapsulated feature selection.
* [`vyuha/src/models/evaluate_multiclass.py`](file:///Users/varshasri/VYUHA/vyuha/src/models/evaluate_multiclass.py): Evaluation metrics, 8x8 confusion matrices, and figure generation.
* [`vyuha/scripts/run_multiclass_experiments.py`](file:///Users/varshasri/VYUHA/vyuha/scripts/run_multiclass_experiments.py): Master training and benchmarking script.
* [`vyuha/scripts/audit_class_weighting.py`](file:///Users/varshasri/VYUHA/vyuha/scripts/audit_class_weighting.py): Controlled 10-experiment class-weighting audit harness.
* [`vyuha/tests/test_multiclass_models.py`](file:///Users/varshasri/VYUHA/vyuha/tests/test_multiclass_models.py): Complete test suite (All 28 project tests passing).
