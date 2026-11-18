"""VYUHA Binary Detection Experiment Runner.

Executes the complete Phase 3B benchmark across the 4 configurations:
1. Random Forest — Full feature set (63 features)
2. Random Forest — Port-agnostic feature set (62 features)
3. Decision Tree — Full feature set (63 features)
4. Decision Tree — Port-agnostic feature set (62 features)

Ensures zero test set leakage, computes comprehensive metrics, performs error analysis,
generates diagnostic figures, and outputs machine-readable results.
"""

import logging
from pathlib import Path
import sys
import time

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.train_binary import run_training_suite
from src.models.evaluate_binary import run_evaluation_suite

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("==================================================================")
    logger.info("   VYUHA PHASE 3B: BINARY INTRUSION DETECTION BENCHMARK SUITE    ")
    logger.info("==================================================================")

    data_path = Path("datasets/processed/train.parquet")
    if not data_path.exists():
        data_path = Path("../datasets/processed/train.parquet")

    output_models_dir = Path("models")
    reports_file = Path("reports/binary_benchmark_results.json")
    figures_dir = Path("reports/figures")

    t_total_start = time.perf_counter()

    # Step 1: Train all 4 configurations on the training partition (80/20 split)
    logger.info("Step 1/2: Training all 4 model configurations (RF & DT; Full & Port-Agnostic)...")
    train_results = run_training_suite(
        data_path=data_path,
        output_dir=output_models_dir,
        dev_mode=False,
        n_estimators=100,
        random_state=42,
    )

    # Step 2: Evaluate on the held-out validation set
    logger.info("Step 2/2: Evaluating models, analyzing errors, and generating figures...")
    eval_results = run_evaluation_suite(
        train_results=train_results,
        output_report_path=reports_file,
        figures_dir=figures_dir,
    )

    t_total = time.perf_counter() - t_total_start
    logger.info("==================================================================")
    logger.info("   EXPERIMENT COMPLETE IN %.2f SECONDS                           ", t_total)
    logger.info("==================================================================")

    # Print summary table
    print("\n" + "=" * 115)
    print(f"{'Model Configuration':<32} | {'Feat':<5} | {'F1-Score':<9} | {'PR-AUC':<8} | {'Precision':<10} | {'Recall':<8} | {'FPR (%)':<8} | {'Train(s)':<8} | {'Latency(us)':<11}")
    print("-" * 115)
    for name, metrics in eval_results["models"].items():
        print(
            f"{name:<32} | "
            f"{metrics['feature_count']:<5} | "
            f"{metrics['f1_score']:<9.5f} | "
            f"{metrics['pr_auc']:<8.5f} | "
            f"{metrics['precision']:<10.5f} | "
            f"{metrics['recall']:<8.5f} | "
            f"{metrics['false_positive_rate']*100:<8.4f} | "
            f"{metrics['training_time_seconds']:<8.2f} | "
            f"{metrics['inference_microseconds_per_flow']:<11.3f}"
        )
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
