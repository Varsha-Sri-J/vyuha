"""VYUHA Class Weighting Comparison & Imbalance Audit Script.

Compares:
1. No class weighting (class_weight=None)
2. Balanced class weighting (class_weight="balanced")
3. Balanced subsample weighting (class_weight="balanced_subsample") [Random Forest only]

Evaluates on the exact same 80/20 stratified validation split derived strictly from train.parquet.
Analyzes trade-offs between minority recall and false-positive explosion.
"""

import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.train_multiclass import (
    CANONICAL_CATEGORIES,
    load_training_data,
    split_multiclass_data,
)
from src.models.pipeline import build_classification_pipeline
from src.models.feature_selector import FeatureSelector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("==================================================================")
    logger.info("   VYUHA PHASE 3C: CLASS IMBALANCE & WEIGHTING AUDIT             ")
    logger.info("==================================================================")

    data_path = Path("datasets/processed/train.parquet")
    if not data_path.exists():
        data_path = Path("../datasets/processed/train.parquet")

    df = load_training_data(data_path)
    X_train, X_val, y_train, y_val, val_meta = split_multiclass_data(
        df,
        test_size=0.2,
        random_state=42,
    )

    N_tr = len(X_train)
    K = len(CANONICAL_CATEGORIES)
    logger.info("Training samples: %d, Classes: %d", N_tr, K)

    # Compute exact theoretical class weights for class_weight="balanced"
    class_weights_dict = {}
    for cat in CANONICAL_CATEGORIES:
        count = (y_train == cat).sum()
        w = N_tr / (K * count)
        class_weights_dict[cat] = round(w, 5)
    logger.info("Theoretical 'balanced' class weights: %s", class_weights_dict)

    # Experimental matrix
    experiments = [
        # Full feature set experiments
        {
            "name": "rf_full_none",
            "model_type": "rf",
            "exclude_ports": False,
            "class_weight": None,
            "desc": "Random Forest — Full (class_weight=None)",
        },
        {
            "name": "rf_full_balanced",
            "model_type": "rf",
            "exclude_ports": False,
            "class_weight": "balanced",
            "desc": "Random Forest — Full (class_weight='balanced')",
        },
        {
            "name": "rf_full_subsample",
            "model_type": "rf",
            "exclude_ports": False,
            "class_weight": "balanced_subsample",
            "desc": "Random Forest — Full (class_weight='balanced_subsample')",
        },
        {
            "name": "dt_full_none",
            "model_type": "dt",
            "exclude_ports": False,
            "class_weight": None,
            "desc": "Decision Tree — Full (class_weight=None)",
        },
        {
            "name": "dt_full_balanced",
            "model_type": "dt",
            "exclude_ports": False,
            "class_weight": "balanced",
            "desc": "Decision Tree — Full (class_weight='balanced')",
        },
        # Port-agnostic experiments
        {
            "name": "rf_port_none",
            "model_type": "rf",
            "exclude_ports": True,
            "class_weight": None,
            "desc": "Random Forest — Port-Agnostic (class_weight=None)",
        },
        {
            "name": "rf_port_balanced",
            "model_type": "rf",
            "exclude_ports": True,
            "class_weight": "balanced",
            "desc": "Random Forest — Port-Agnostic (class_weight='balanced')",
        },
        {
            "name": "rf_port_subsample",
            "model_type": "rf",
            "exclude_ports": True,
            "class_weight": "balanced_subsample",
            "desc": "Random Forest — Port-Agnostic (class_weight='balanced_subsample')",
        },
        {
            "name": "dt_port_none",
            "model_type": "dt",
            "exclude_ports": True,
            "class_weight": None,
            "desc": "Decision Tree — Port-Agnostic (class_weight=None)",
        },
        {
            "name": "dt_port_balanced",
            "model_type": "dt",
            "exclude_ports": True,
            "class_weight": "balanced",
            "desc": "Decision Tree — Port-Agnostic (class_weight='balanced')",
        },
    ]

    results = {
        "metadata": {
            "validation_samples": len(X_val),
            "training_samples": len(X_train),
            "categories": CANONICAL_CATEGORIES,
            "effective_class_weights": class_weights_dict,
        },
        "experiments": {},
    }

    for exp in experiments:
        name = exp["name"]
        logger.info(">>> Running Experiment: %s (%s)", name, exp["desc"])

        selector = FeatureSelector(
            exclude_ports=exp["exclude_ports"],
            variance_threshold=0.0,
            drop_collinear_twins=True,
        )

        if exp["model_type"] == "rf":
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                class_weight=exp["class_weight"],
                random_state=42,
                n_jobs=-1,
            )
        else:
            clf = DecisionTreeClassifier(
                max_depth=20,
                class_weight=exp["class_weight"],
                random_state=42,
            )

        pipeline = build_classification_pipeline(feature_selector=selector, classifier=clf)

        # Fit strictly on X_train
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        # Predict on X_val
        t0 = time.perf_counter()
        y_pred = pipeline.predict(X_val)
        pred_time = time.perf_counter() - t0

        inf_us = (pred_time / len(X_val)) * 1e6

        # Metrics
        acc = accuracy_score(y_val, y_pred)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        macro_prec = precision_score(y_val, y_pred, labels=CANONICAL_CATEGORIES, average="macro", zero_division=0)
        macro_rec = recall_score(y_val, y_pred, labels=CANONICAL_CATEGORIES, average="macro", zero_division=0)
        macro_f1 = f1_score(y_val, y_pred, labels=CANONICAL_CATEGORIES, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_val, y_pred, labels=CANONICAL_CATEGORIES, average="weighted", zero_division=0)

        per_prec = precision_score(y_val, y_pred, labels=CANONICAL_CATEGORIES, average=None, zero_division=0)
        per_rec = recall_score(y_val, y_pred, labels=CANONICAL_CATEGORIES, average=None, zero_division=0)
        per_f1 = f1_score(y_val, y_pred, labels=CANONICAL_CATEGORIES, average=None, zero_division=0)

        cm = confusion_matrix(y_val, y_pred, labels=CANONICAL_CATEGORIES)

        # Specific diagnostics for Botnet and Infiltration
        botnet_idx = CANONICAL_CATEGORIES.index("Botnet")
        infil_idx = CANONICAL_CATEGORIES.index("Infiltration")
        benign_idx = CANONICAL_CATEGORIES.index("BENIGN")

        botnet_tp = int(cm[botnet_idx, botnet_idx])
        botnet_fp_total = int(cm[:, botnet_idx].sum() - botnet_tp)
        botnet_fp_from_benign = int(cm[benign_idx, botnet_idx])

        infil_tp = int(cm[infil_idx, infil_idx])
        infil_fp_total = int(cm[:, infil_idx].sum() - infil_tp)
        infil_fp_from_benign = int(cm[benign_idx, infil_idx])

        per_class_data = {}
        for idx, cat in enumerate(CANONICAL_CATEGORIES):
            per_class_data[cat] = {
                "precision": round(float(per_prec[idx]), 6),
                "recall": round(float(per_rec[idx]), 6),
                "f1_score": round(float(per_f1[idx]), 6),
                "support": int(cm[idx].sum()),
            }

        exp_result = {
            "name": name,
            "description": exp["desc"],
            "model_type": exp["model_type"],
            "class_weight": str(exp["class_weight"]),
            "exclude_ports": exp["exclude_ports"],
            "feature_count": len(selector.selected_features_),
            "training_time_seconds": round(train_time, 2),
            "inference_microseconds_per_flow": round(inf_us, 3),
            "accuracy": round(acc, 6),
            "balanced_accuracy": round(bal_acc, 6),
            "macro_precision": round(macro_prec, 6),
            "macro_recall": round(macro_rec, 6),
            "macro_f1": round(macro_f1, 6),
            "weighted_f1": round(weighted_f1, 6),
            "botnet_tp": botnet_tp,
            "botnet_fp_total": botnet_fp_total,
            "botnet_fp_from_benign": botnet_fp_from_benign,
            "botnet_f1": round(float(per_f1[botnet_idx]), 6),
            "botnet_precision": round(float(per_prec[botnet_idx]), 6),
            "botnet_recall": round(float(per_rec[botnet_idx]), 6),
            "infiltration_tp": infil_tp,
            "infiltration_fp_total": infil_fp_total,
            "infiltration_fp_from_benign": infil_fp_from_benign,
            "infiltration_f1": round(float(per_f1[infil_idx]), 6),
            "per_class": per_class_data,
        }

        results["experiments"][name] = exp_result
        logger.info(
            "[%s] Macro-F1=%.4f | Botnet: P=%.4f, R=%.4f, FP=%d | Infiltration: P=%.4f, R=%.4f, FP=%d",
            name,
            macro_f1,
            per_prec[botnet_idx],
            per_rec[botnet_idx],
            botnet_fp_total,
            per_prec[infil_idx],
            per_rec[infil_idx],
            infil_fp_total,
        )

    # Save results JSON
    out_file = Path("reports/class_weighting_audit_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved class weighting audit results to: %s", out_file)

    # Print summary comparative table
    print("\n" + "=" * 125)
    print(f"{'Experiment':<25} | {'Weighting':<20} | {'Macro-F1':<9} | {'Botnet P':<9} | {'Botnet R':<9} | {'Botnet FP':<10} | {'Infil P':<8} | {'Infil R':<8} | {'Infil FP':<9}")
    print("-" * 125)
    for name, r in results["experiments"].items():
        print(
            f"{name:<25} | "
            f"{r['class_weight']:<20} | "
            f"{r['macro_f1']:<9.4f} | "
            f"{r['botnet_precision']:<9.4f} | "
            f"{r['botnet_recall']:<9.4f} | "
            f"{r['botnet_fp_total']:<10d} | "
            f"{r['per_class']['Infiltration']['precision']:<8.4f} | "
            f"{r['per_class']['Infiltration']['recall']:<8.4f} | "
            f"{r['infiltration_fp_total']:<9d}"
        )
    print("=" * 125 + "\n")


if __name__ == "__main__":
    main()
