"""Master Experiment Runner for VYUHA Phase 3D: Unsupervised Anomaly Detection.

Executes controlled benchmarks for Isolation Forest:
1. Training-size comparisons: 50k vs 100k vs 200k benign flows.
2. Feature set comparisons: Full (63 features) vs Port-Agnostic (62 features).
3. Scaling invariance check: Unscaled vs RobustScaler.
4. Threshold calibrations: 5.0%, 1.0%, 0.5%, 0.1% target benign FPR.
5. Saves results to reports/anomaly_benchmark_results.json.
6. Generates Figures 17, 18, 19, 20 in reports/figures/.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.train_anomaly import (
    build_anomaly_model,
    calibrate_anomaly_thresholds,
    load_training_data,
    partition_anomaly_data,
    sample_benign_training_subset,
    save_anomaly_artifact,
    train_anomaly_model,
)
from src.models.evaluate_anomaly import (
    evaluate_anomaly_detector,
    generate_anomaly_diagnostic_plots,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_anomaly_experiments")


def main():
    logger.info("==================================================================")
    logger.info("  VYUHA PHASE 3D: UNSUPERVISED ANOMALY DETECTION BENCHMARKS       ")
    logger.info("==================================================================")

    # 1. Load data and partition strictly inside train.parquet
    df = load_training_data()
    X_train_benign_pool, X_cal_benign, X_val, y_val, val_meta = partition_anomaly_data(
        df, test_size=0.2, random_state=42, cal_size=50000
    )

    experiments_to_run = [
        # (exp_id, description, n_train, exclude_ports, scaler_type)
        ("iforest_full_50k", "Isolation Forest (50k Benign, Full 63)", 50000, False, None),
        ("iforest_full_100k", "Isolation Forest (100k Benign, Full 63) - Primary", 100000, False, None),
        ("iforest_full_200k", "Isolation Forest (200k Benign, Full 63)", 200000, False, None),
        ("iforest_port_100k", "Isolation Forest (100k Benign, Port-Agnostic 62)", 100000, True, None),
        ("iforest_full_scaled_100k", "Isolation Forest (100k Benign, Full 63, RobustScaler)", 100000, False, "robust"),
    ]

    target_fprs = (0.05, 0.01, 0.005, 0.001)
    results = {
        "metadata": {
            "validation_samples": len(X_val),
            "calibration_samples": len(X_cal_benign),
            "total_benign_training_pool": len(X_train_benign_pool),
            "random_seed": 42,
            "target_fprs": list(target_fprs),
            "anomaly_score_definition": "A(x) = -score_samples(x) where higher = more anomalous",
        },
        "experiments": {},
    }

    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for exp_id, desc, n_train, exclude_ports, scaler_type in experiments_to_run:
        logger.info("\n--------------------------------------------------------------")
        logger.info("Running Experiment: %s (%s)", exp_id, desc)
        logger.info("--------------------------------------------------------------")

        # Sample benign training records
        X_tr_sub = sample_benign_training_subset(
            X_train_benign_pool, n_samples=n_train, random_state=42
        )

        # Build pipeline
        pipeline = build_anomaly_model(
            exclude_ports=exclude_ports,
            scaler_type=scaler_type,
            n_estimators=100,
            max_samples=256,
            random_state=42,
        )

        # Fit model and measure training duration
        t_train_start = time.perf_counter()
        pipeline = train_anomaly_model(pipeline, X_tr_sub)
        t_train = time.perf_counter() - t_train_start

        # Calibrate thresholds on held-out calibration set
        cal_res = calibrate_anomaly_thresholds(
            pipeline, X_cal_benign, target_fprs=target_fprs
        )

        # Evaluate on validation dataset
        eval_res = evaluate_anomaly_detector(
            pipeline, X_val, y_val, cal_res, benchmark_latency=True
        )

        # Save model binary
        model_save_path = models_dir / f"{exp_id}.joblib"
        save_anomaly_artifact(
            pipeline,
            model_save_path,
            metadata={
                "exp_id": exp_id,
                "n_train": n_train,
                "exclude_ports": exclude_ports,
                "scaler_type": scaler_type,
                "calibrated_thresholds": cal_res,
            },
        )
        file_size_mb = model_save_path.stat().st_size / (1024 * 1024)

        # Pack results
        results["experiments"][exp_id] = {
            "name": exp_id,
            "description": desc,
            "n_train_samples": n_train,
            "exclude_ports": exclude_ports,
            "scaler_type": str(scaler_type),
            "training_time_seconds": round(t_train, 4),
            "serialized_model_size_mb": round(file_size_mb, 3),
            "roc_auc": round(eval_res["roc_auc"], 5),
            "pr_auc": round(eval_res["pr_auc"], 5),
            "latency": eval_res["latency"],
            "threshold_evaluations": eval_res["threshold_evaluations"],
            "category_score_stats": eval_res["category_score_stats"],
            "compact_roc": eval_res["compact_roc"],
            "compact_pr": eval_res["compact_pr"],
        }

        # Print summary line
        cal_1pct = eval_res["threshold_evaluations"].get("fpr_1.0pct", {})
        logger.info(
            "%s Complete | Train: %.2fs | ROC-AUC: %.4f | PR-AUC: %.4f | 1%% FPR Val Benign FPR: %.2f%%, Recall: %.2f%%, P: %.2f%%, F1: %.4f",
            exp_id,
            t_train,
            eval_res["roc_auc"],
            eval_res["pr_auc"],
            cal_1pct.get("validation_benign_fpr", 0.0) * 100,
            cal_1pct.get("recall", 0.0) * 100,
            cal_1pct.get("precision", 0.0) * 100,
            cal_1pct.get("f1_score", 0.0),
        )

    # 3. Generate diagnostic plots
    figures_dir = repo_root / "reports" / "figures"
    logger.info("\nGenerating diagnostic plots in %s...", figures_dir)
    generated_figures = generate_anomaly_diagnostic_plots(
        results["experiments"], output_dir=figures_dir
    )
    results["generated_figures"] = generated_figures

    # 4. Save JSON results
    json_path = repo_root / "reports" / "anomaly_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved benchmark results to: %s", json_path)

    logger.info("\n==================================================================")
    logger.info("  PHASE 3D ANOMALY BENCHMARKS SUCCESSFULLY COMPLETED!              ")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
