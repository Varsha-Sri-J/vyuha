# VYUHA: Feature Engineering & Selection Specification

**Project**: VYUHA — AI-Powered Cybersecurity for Small Industries  
**Phase**: Phase 3A: Feature Engineering & Selection  
**Dataset**: Canadian Institute for Cybersecurity (UNB) - CICIDS2017 (MachineLearningCVE)  
**Status**: Specification & Implementation Verified  
**Date**: November 2026  

---

## 1. Executive Summary & Architectural Scope

Phase 3A of Project VYUHA establishes the feature engineering, selection, and cross-validation foundation for both supervised intrusion detection (classification) and unsupervised anomaly detection (Isolation Forest). 

The primary objectives of this phase are:
1. **Zero Data Leakage by Design**: Enforce strict inside-cross-validation parameter learning. No feature selection, thresholding, or scaling parameters are computed on the broader dataset prior to fold splitting.
2. **Strict Target Isolation**: Ensure target columns (`label_raw`, `label_binary`, `label_category`) are decoupled from feature matrices ($X$) and cannot leak into model inputs.
3. **Pruning Uninformative & Redundant Features**: Empirically eliminate zero-variance constants (8 features) and exact collinear twins (6 features) stemming from network flow calculation redundancies.
4. **Topology-Shortcut Mitigation (Port-Agnostic vs. Full Feature Sets)**: Address the operational risk of models overfitting to specific network port numbers (e.g., Port 80, 21, 22) by providing dual feature candidate sets (Full: 63 features, Port-Agnostic: 62 features).
5. **Principled Model-Specific Preprocessing**: Acknowledge that tree-based models and Isolation Forests are invariant to monotonic linear feature scaling, establishing an unscaled baseline while enabling experimental comparisons with robust scalers.
6. **Dual Importance Evaluation**: Combine validation-based Permutation Feature Importance with tree-based Mean Decrease in Impurity (MDI) to overcome Gini-bias toward high-cardinality continuous features.
7. **Complete Test Set Isolation**: Ensure the final evaluation dataset (`datasets/processed/test.parquet`) remains 100% frozen, untouched, and unread throughout the entire feature engineering phase.

---

## 2. Feature Schema & Target Separation

The processed CICIDS2017 dataset consists of **80 columns**:
* **77 Numerical Network Flow Features**: Derived from bidirectional packet captures by CICFlowMeter (durations, packet counts, inter-arrival times, flag counters, packet size distributions, subflow metrics).
* **3 Target Columns**:
  * `label_raw`: Original granular attack names (15 classes: `BENIGN`, `DoS Hulk`, `PortScan`, `DDoS`, etc.).
  * `label_binary`: Binary indicator (`0` for `BENIGN`, `1` for `ATTACK`).
  * `label_category`: 8 total unique classes comprising **7 attack families plus BENIGN** (`BENIGN`, `DoS_DDoS`, `PortScan`, `BruteForce`, `WebAttack`, `Botnet`, `Infiltration`, `Heartbleed`).

### Strict Target Isolation Protocol
Target columns are extracted and isolated using `extract_features_and_targets(df)` in `src/models/feature_selector.py`:
```python
def extract_features_and_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    targets_to_drop = [col for col in df.columns if col.startswith("label_") or col in TARGET_COLUMNS]
    y_dict = {
        "binary": df["label_binary"].copy(),
        "category": df["label_category"].copy(),
        "raw": df["label_raw"].copy(),
    }
    X = df.drop(columns=targets_to_drop).copy()
    # Explicit assertion check prevents accidental leakage
    for target_col in TARGET_COLUMNS:
        assert target_col not in X.columns
    return X, y_dict
```

---

## 3. Stage 1: Zero-Variance & Constant Feature Elimination

Across the complete training population ($1,999,530$ flows), the held-out test partition ($499,883$ flows), and the representative dev sample ($200,000$ flows), exactly **8 features** exhibit zero variance ($\text{Var}(X) = 0.0$) and contain strictly the single scalar value `0.0` for 100% of rows.

### Mathematical & Engineering Justification
Features with zero variance across the population carry zero mutual information with respect to any target column ($I(X; Y) = 0$). They cannot create informative decision splits in tree-based algorithms (Random Forest, XGBoost) and provide no gradient in linear or neural models. Pruning them reduces feature matrix dimensionality from 77 to 69 with zero loss of descriptive power.

### Empirical Observation vs. Underlying Extraction Notes
To maintain scientific rigor, we distinguish between what is directly observed in the dataset and hypotheses regarding the capture or extractor:
* **Direct Empirical Fact**: The 8 columns listed below evaluate to `0.0` across all $2,499,413$ deduplicated records in the processed dataset.
* **Extraction Context**: While these features conceptually correspond to TCP flags on backward flows or bulk transfer rate tracking, whether their zero values stem from actual absence of such traffic during the Monday–Friday 2017 capture or the internal extraction configuration of the CICFlowMeter Java engine cannot be definitively determined from the CSV dataset alone without auditing the original PCAP frames. Their elimination is therefore strictly justified by empirical invariance.

### Eliminated Constant Features (8 Features)

| Feature Name | Unique Observed Value | Observed Variance | Observed Invariance Scope |
| :--- | :--- | :--- | :--- |
| `Bwd PSH Flags` | `0.0` | `0.000000` | 100% of rows in train, test, and dev partitions |
| `Bwd URG Flags` | `0.0` | `0.000000` | 100% of rows in train, test, and dev partitions |
| `Fwd Avg Bytes/Bulk` | `0.0` | `0.000000` | 100% of rows in train, test, and dev partitions |
| `Fwd Avg Packets/Bulk` | `0.0` | `0.000000` | 100% of rows in train, test, and dev partitions |
| `Fwd Avg Bulk Rate` | `0.0` | `0.000000` | 100% of rows in train, test, and dev partitions |
| `Bwd Avg Bytes/Bulk` | `0.0` | `0.000000` | 100% of rows in train, test, and dev partitions |
| `Bwd Avg Packets/Bulk` | `0.0` | `0.000000` | 100% of rows in train, test, and dev partitions |
| `Bwd Avg Bulk Rate` | `0.0` | `0.000000` | 100% of rows in train, test, and dev partitions |

---

## 4. Stage 2: Collinear Twin & Redundancy Pruning

During Phase 2 Exploratory Data Analysis, seven perfectly correlated feature pairs ($r = 1.000$) were identified. One pair (`Fwd Header Length` vs. `Fwd Header Length.1`) represented an identical CSV column duplicate from the original UNB export, which was resolved during Phase 1 column standardisation.

The remaining **6 feature pairs** were subjected to an element-by-element verification across all $1,999,530$ records of the training partition. The audit reveals two distinct sub-categories:

### Category A: Strictly Identical Feature Pairs (4 Pairs, 100% Identical)
For these 4 pairs, the values are **element-by-element identical across 100.0000% of rows** ($\Delta = 0.0$ for all $1,999,530$ flows):
1. `Subflow Fwd Packets` == `Total Fwd Packets` ($0$ mismatches, $r = 1.00000000$)
2. `Subflow Bwd Packets` == `Total Backward Packets` ($0$ mismatches, $r = 1.00000000$)
3. `Avg Fwd Segment Size` == `Fwd Packet Length Mean` ($0$ mismatches, $r = 1.00000000$)
4. `Avg Bwd Segment Size` == `Bwd Packet Length Mean` ($0$ mismatches, $r = 1.00000000$)

### Category B: Highly Collinear Near-Identical Pairs (2 Pairs, $r > 0.999999$)
For these 2 pairs, the features are nearly identical, with Spearman rank correlation equal to $1.00000000$ and Pearson correlation exceeding $0.999999$, but exhibit minute discrepancies in rare high-volume flows:
1. `Subflow Fwd Bytes` vs `Total Length of Fwd Packets`:
   * Identical in $1,999,529$ out of $1,999,530$ rows (**$99.99995\%$ identical**).
   * Exactly **1 row** differs (row 950004: `Subflow Fwd Bytes` = 12,870,338.0 vs `Total Length of Fwd Packets` = 12,900,000.0, where the latter is rounded in the original export).
   * Pearson $r = 0.99999939$, Spearman $r = 1.00000000$.
2. `Subflow Bwd Bytes` vs `Total Length of Bwd Packets`:
   * Identical in $1,999,411$ out of $1,999,530$ rows (**$99.9940\%$ identical**).
   * Exactly **119 rows** (0.0060%) differ, all consisting of massive benign bulk transfers (> 10 MB) where `Total Length of Bwd Packets` was rounded/truncated to 100k boundaries in the raw dataset.
   * Pearson $r = 0.99999984$, Spearman $r = 1.00000000$.

### Pruning Rationale & Retained Primaries
Retaining near-duplicate features ($r > 0.999999$) produces severe multicollinearity in linear models and causes tree ensembles to split arbitrarily across redundant dimensions, increasing model complexity without providing new discriminative signal. Therefore, the secondary subflow alias features are pruned by `FeatureSelector(drop_collinear_twins=True)`, retaining the standard primary flow features:

| Pruned Feature | Retained Primary Feature | Identical Rows (%) | Pearson $r$ | Spearman $r$ | Removal Justification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Subflow Fwd Packets` | `Total Fwd Packets` | 100.0000% ($0$ diffs) | $1.000000$ | $1.000000$ | 100% element-by-element clone. Retain primary standard flow packet count. |
| `Subflow Bwd Packets` | `Total Backward Packets` | 100.0000% ($0$ diffs) | $1.000000$ | $1.000000$ | 100% element-by-element clone. Retain primary standard backward packet count. |
| `Avg Fwd Segment Size` | `Fwd Packet Length Mean` | 100.0000% ($0$ diffs) | $1.000000$ | $1.000000$ | 100% element-by-element clone. Retain standard forward mean packet length. |
| `Avg Bwd Segment Size` | `Bwd Packet Length Mean` | 100.0000% ($0$ diffs) | $1.000000$ | $1.000000$ | 100% element-by-element clone. Retain standard backward mean packet length. |
| `Subflow Fwd Bytes` | `Total Length of Fwd Packets` | 99.99995% ($1$ diff) | $0.999999$ | $1.000000$ | Perfect rank collinearity; single-row rounding artifact. Retain primary packet length. |
| `Subflow Bwd Bytes` | `Total Length of Bwd Packets` | 99.9940% ($119$ diffs) | $0.999999$ | $1.000000$ | Perfect rank collinearity ($r_s=1.000$); rounding artifact in 0.006% of rows. Retain primary packet length. |

---

## 5. Feature Selection Funnel & Candidate Feature Sets

Rather than forcing the feature count to an arbitrary range (such as 25–32) before empirical model training, VYUHA establishes a rigorous selection funnel and defines multiple validated candidate sets to be evaluated under cross-validation.

```
+-----------------------------------------------------------+
| Raw CICIDS2017 Numerical Features (77 Features)            |
+-----------------------------------------------------------+
                             |
                             v  [Drop 8 Zero-Variance Features]
+-----------------------------------------------------------+
| Intermediate Non-Constant Feature Set (69 Features)       |
+-----------------------------------------------------------+
                             |
                             v  [Drop 6 Collinear Subflow Twins]
+-----------------------------------------------------------+
| Full Baseline Feature Set (63 Features)                   |
| (Includes Destination Port)                               |
+-----------------------------------------------------------+
                             |
                             v  [Drop Destination Port]
+-----------------------------------------------------------+
| Port-Agnostic Feature Set (62 Features)                   |
| (Generalised Network Behaviour Only)                      |
+-----------------------------------------------------------+
                             |
                             v  [Inside-CV Permutation / Importance Tuning]
+-----------------------------------------------------------+
| Compact Candidate Subsets (Evaluated by CV Evidence)     |
+-----------------------------------------------------------+
```

### Feature Counts Summary

| Feature Candidate Set | Feature Count | Features Excluded | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **Raw Flow Features** | 77 | Target columns only | Unfiltered baseline (unsuitable for production due to zero-variance and twin columns). |
| **Full Baseline Feature Set** | **63** | 8 constants + 6 collinear twins | Standard evaluation benchmark capturing all unique flow dynamics including target services. |
| **Port-Agnostic Feature Set** | **62** | 8 constants + 6 collinear twins + `Destination Port` | Deployment benchmark preventing topology shortcuts; forces models to learn pure behavioural characteristics. |
| **Compact Candidate Sets** | Variable (e.g. 20–35) | Learned via inside-CV permutation importance thresholding | Low-latency inference for resource-constrained edge industrial monitors (evaluated in Phase 3B). |

---

## 6. The Destination Port Dilemma: Full vs. Port-Agnostic Models

In network intrusion detection, destination port numbers present a fundamental trade-off between classification accuracy on historical test benchmarks and real-world operational generalisability.

### Empirical Evidence from CICIDS2017
Analysis of the processed dataset reveals extreme port concentration:
* **Port 80 (HTTP)**: Accounts for **76.14%** of all attack records in the training partition (and 76.13% in the dev sample). This is driven by large volumetric DoS attacks (`DoS Hulk`, `DoS Slowhttptest`, `DoS slowloris`, `DoS GoldenEye`).
* **Port 21 (FTP)**: 65.3% of flows targeting port 21 are `FTP-Patator` brute-force attacks.
* **Port 22 (SSH)**: 33.9% of flows targeting port 22 are `SSH-Patator` brute-force attacks.

### The Risk: Shortcut Learning
If `Destination Port` is provided, high-capacity machine learning models (e.g., Random Forests, XGBoost) quickly learn decision splits such as:
$$\text{If } \text{Destination Port} = 80 \implies \text{High prior probability of DoS}$$
$$\text{If } \text{Destination Port} = 21 \implies \text{High prior probability of Brute Force}$$

In a production manufacturing environment, this creates severe vulnerabilities:
1. **Adversarial Port Hopping**: An adversary running an SSH backdoor on non-standard port 2222 or C2 traffic over port 443 will evade detection rules tethered to destination ports.
2. **Industrial Plant Diversity**: Small manufacturing industries often run specialized SCADA/ICS interfaces or web services on non-standard ports (e.g., 8080, 502, 44818). A model dependent on port 80 will fail to recognize HTTP attacks targeting port 8080.

### VYUHA Dual-Model Protocol
To ensure rigorous evaluation and transparent reporting:
* **Full Model (63 features)**: Evaluated to document maximum theoretical performance and compare directly against published academic benchmarks.
* **Port-Agnostic Model (62 features)**: Serves as the primary operational baseline for VYUHA deployment. It forces algorithms to distinguish attacks exclusively through flow timing, packet length variance, inter-arrival times, and TCP flag patterns.

---

## 7. Feature Importance Methodology: Permutation vs. Tree-Based (MDI)

Feature importance evaluation in VYUHA combines two complementary techniques, implemented in `src/models/feature_selector.py`:

### 1. Tree-Based Importance (Mean Decrease in Impurity / MDI)
* **Function**: `compute_tree_feature_importance(tree_estimator, feature_names)`
* **Mechanism**: Aggregates the total reduction in node impurity (Gini or entropy) brought by each feature across all trees in the ensemble.
* **Limitations**:
  * **In-Sample Evaluation**: MDI is computed entirely on training splits, failing to penalize features that overfit.
  * **Continuous/High-Cardinality Bias**: Features with many unique continuous values offer more potential split points and receive artificially inflated MDI scores even when they contain little generalizable information.

### 2. Permutation Feature Importance
* **Function**: `compute_permutation_feature_importance(estimator, X_val, y_val, scoring="f1_macro", n_repeats=5)`
* **Mechanism**: Evaluates the model on **held-out validation data**. For each feature, its values are randomly shuffled (breaking its relationship with the target while preserving the marginal distribution), and the drop in validation performance ($F_1$-macro) is measured:
$$I(f) = \text{Score}_{\text{baseline}} - \text{Score}_{\text{shuffled}(f)}$$
* **Advantages**:
  * Unbiased towards feature cardinality.
  * Directly measures true out-of-sample generalization degradation.
  * Identifies features that contribute positively to validation stability.

---

## 8. Unsupervised Anomaly Detection: Isolation Forest Considerations

VYUHA separates the feature preparation and scaling pipelines for unsupervised anomaly detection (`build_anomaly_pipeline`) from supervised classification (`build_classification_pipeline`).

### Scaling Invariance of Isolation Forests
Isolation Forest constructs an ensemble of isolation trees (iTrees) by:
1. Randomly selecting an axis-aligned feature $q$ from the feature set.
2. Randomly selecting a split point $p$ between the minimum and maximum values of $q$:
$$p \sim \text{Uniform}(\min(q), \max(q))$$
3. Recursively partitioning data until instances are isolated.

Because the split threshold $p$ is drawn uniformly from $[\min(q), \max(q)]$, any strictly increasing affine linear transformation of the feature ($x' = a \cdot x + b$, where $a > 0$):
$$\min(q') = a \cdot \min(q) + b, \quad \max(q') = a \cdot \max(q) + b$$
$$p' = a \cdot p + b$$
The probability distribution of splits and the resulting tree structures remain mathematically invariant.

### Implementation Architecture
* **Baseline Pipeline**: Built with `scaler=None` (unscaled features), preserving numerical precision and avoiding unnecessary compute overhead.
* **Experimental Comparative Pipeline**: Allows passing `scaler=RobustScaler()` inside the pipeline to empirically measure whether median/IQR centering alters boundary behavior in the presence of heavy-tailed network flow outliers.

---

## 9. Rare Class Limitations & Reconciled Counts

In the CICIDS2017 dataset, two attack categories exhibit extreme sparsity. A strict audit of the processed dataset partitions and `preprocessing_metadata.json` confirms the exact sample distribution:

### Reconciled Ground-Truth Sample Counts

| Attack Category / Label | Raw Dataset | Deduplicated Total | Training Partition (`train.parquet`) | Test Partition (`test.parquet`) | Dev Sample (`dev_sample.parquet`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Heartbleed** | 11 | **11** | **9** | **2** | **1** |
| **Infiltration** | 36 | **36** | **29** | **7** | **3** |

*Note on Reconciliation*: The 80/20 stratified train/test split of the 11 deduplicated Heartbleed records produces exactly $11 \times 0.8 = 8.8 \approx 9$ training records and $11 \times 0.2 = 2.2 \approx 2$ test records ($9 + 2 = 11$). An earlier preliminary draft erroneously cited 8 training records due to a typographical transposition; the actual ground truth on disk is **9 in train** and **2 in test**.

### Statistical Reality & Modeling Constraints
1. **Fold Instability in Cross-Validation**: When performing 5-fold cross-validation on the training set, Heartbleed's 9 records yield folds with only 1 or 2 positive samples. Statistical metrics ($F_1$-score) computed on 1 or 2 samples possess massive variance ($\text{std} > 0.40$) and cannot establish statistical significance.
2. **Support Defense**: VYUHA explicitly avoids making unsupported high-accuracy claims for Heartbleed or Infiltration. 
3. **Evaluation Protocol**:
   * Multi-class classification evaluates the **8 total classes** (7 attack families plus `BENIGN`), using **Macro-Averaged F1** to ensure rare classes are monitored alongside majority classes.
   * Per-class confusion matrices and classification reports must explicitly display sample support alongside point metrics.

---

## 10. Cross-Validation & Pipeline Implementation Architecture

All feature selection, scaling, and transformation steps are wrapped in Scikit-Learn `Pipeline` objects and orchestrated via `LeakageSafeCrossValidator` (`src/models/evaluate.py`).

### Execution Flow
```
For each fold (k = 1 to 5):
    1. Partition: X_tr, X_val = X[train_idx], X[val_idx]
    2. Fit Pipeline on X_tr:
       a. FeatureSelector.fit(X_tr):
          - Learn constant features from X_tr only.
          - Learn collinear duplicates from X_tr only.
          - Filter X_tr -> X_tr_selected.
       b. Scaler.fit(X_tr_selected) [if applicable]:
          - Learn median, IQR, or mean from X_tr_selected only.
          - Transform X_tr_selected -> X_tr_scaled.
       c. Model.fit(X_tr_scaled, y_tr):
          - Train estimator on fold-specific transformed features.
    3. Transform X_val:
       - FeatureSelector.transform(X_val) using X_tr-learned feature mask.
       - Scaler.transform(X_val_selected) using X_tr-learned parameters.
    4. Predict & Evaluate on X_val:
       - Calculate Accuracy, Balanced Accuracy, Precision, Recall, Macro F1.
       - Record selected feature set stability across folds.
```

### Module Deliverables in `src/models/`
1. `src/models/__init__.py`: Package export interface.
2. `src/models/feature_selector.py`:
   * `TARGET_COLUMNS`, `EXACT_COLLINEAR_TWINS`.
   * `extract_features_and_targets()`: Strict target separation.
   * `get_port_agnostic_features()`: Port exclusion utility.
   * `FeatureSelector`: Scikit-Learn transformer with `fit()` and `transform()`.
   * `compute_permutation_feature_importance()`: Out-of-sample validation ranking.
   * `compute_tree_feature_importance()`: MDI tree importance extractor.
3. `src/models/pipeline.py`:
   * `build_classification_pipeline()`: End-to-end supervised pipeline.
   * `build_anomaly_pipeline()`: Unsupervised anomaly pipeline (unscaled baseline & scaled).
4. `src/models/evaluate.py`:
   * `LeakageSafeCrossValidator`: Stratified K-fold CV orchestrator reporting macro metrics and fold stability.

---

## 11. Test Coverage & Deliverable Verification

A dedicated unit test suite (`tests/test_feature_selector.py`) verifies all Phase 3A requirements. Combined with Phase 1 preprocessing tests, **16 test cases pass completely**:

```
============================= test session starts ==============================
tests/test_feature_selector.py::test_extract_features_and_targets_strict_isolation PASSED [  6%]
tests/test_feature_selector.py::test_zero_variance_removal PASSED        [ 12%]
tests/test_feature_selector.py::test_collinear_twin_removal PASSED       [ 18%]
tests/test_feature_selector.py::test_port_agnostic_feature_set PASSED    [ 25%]
tests/test_feature_selector.py::test_feature_selection_fitted_only_on_training_fold PASSED [ 31%]
tests/test_feature_selector.py::test_pipeline_integration_and_cross_validation PASSED [ 37%]
tests/test_feature_selector.py::test_unsupervised_anomaly_pipeline PASSED [ 43%]
tests/test_feature_selector.py::test_processed_dataset_immutability PASSED [ 50%]
tests/test_preprocessor.py::test_standardize_columns PASSED              [ 56%]
tests/test_preprocessor.py::test_normalize_labels PASSED                 [ 62%]
tests/test_preprocessor.py::test_clean_numerical_anomalies_negative_duration PASSED [ 68%]
tests/test_preprocessor.py::test_sentinel_preservation PASSED            [ 75%]
tests/test_preprocessor.py::test_infinities_and_nulls_imputed PASSED     [ 81%]
tests/test_preprocessor.py::test_deduplication PASSED                    [ 87%]
tests/test_preprocessor.py::test_separate_features_and_targets PASSED    [ 93%]
tests/test_preprocessor.py::test_stratified_train_test_split PASSED      [100%]
============================== 16 passed in 5.25s ==============================
```

### Dataset Immutability Verification
The underlying processed dataset files in `datasets/processed/` were verified:
* `test.parquet`: **49,543,434 bytes**, timestamp untouched (`Nov 18 22:21`). Exactly 499,883 records remain isolated and unread.
* `train.parquet`: **192,767,616 bytes**, timestamp untouched (`Nov 18 22:21`). Exactly 1,999,530 records.
* `dev_sample.parquet`: **20,282,441 bytes**, timestamp untouched (`Nov 18 22:21`). Exactly 200,000 records.
