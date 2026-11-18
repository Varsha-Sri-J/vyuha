"""VYUHA Multiclass Intrusion Detection Experiment Runner.

Executes the complete Phase 3C benchmark across the 4 configurations:
1. Random Forest — Full feature set (63 features)
2. Random Forest — Port-agnostic feature set (62 features)
3. Decision Tree — Full feature set (63 features)
4. Decision Tree — Port-agnostic feature set (62 features)

Evaluates multiclass attack-family identification across the 8 canonical classes:
BENIGN, DoS_DDoS, PortScan, BruteForce, WebAttack, Botnet, Infiltration, Heartbleed.
"""

import logging
from pathlib import Path
import sys
import time

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.train_multiclass import run_multiclass_training_suite
from src.models.evaluate_multiclass import run_multiclass_evaluation_suite

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("==================================================================")
    logger.info("  VYUHA PHASE 3C: MULTICLASS INTRUSION DETECTION BENCHMARK SUITE  ")
    logger.info("==================================================================")

    data_path = Path("datasets/processed/train.parquet")
    if not data_path.exists():
        data_path = Path("../datasets/processed/train.parquet")

    output_models_dir = Path("models")
    reports_file = Path("reports/multiclass_benchmark_results.json")
    figures_dir = Path("reports/figures")

    t_total_start = time.perf_counter()

    # Step 1: Train all 4 configurations on the training partition (80/20 split)
    logger.info("Step 1/2: Training all 4 multiclass model configurations (RF & DT; Full & Port-Agnostic)...")
    train_results = run_multiclass_training_suite(
        data_path=data_path,
        output_dir=output_models_dir,
        dev_mode=False,
        n_estimators=100,
        random_state=42,
    )

    # Step 2: Evaluate on the held-out validation set
    logger.info("Step 2/2: Evaluating multiclass models, analyzing errors, and generating figures...")
    eval_results = run_multiclass_evaluation_suite(
        train_results=train_results,
        output_report_path=reports_file,
        figures_dir=figures_dir,
    )

    t_total = time.perf_counter() - t_total_start
    logger.info("==================================================================")
    logger.info("  EXPERIMENT COMPLETE IN %.2f SECONDS                             ", t_total)
    logger.info("==================================================================")

    # Print summary table
    print("\n" + "=" * 110)
    print(f"{'Model Configuration':<32} | {'Feat':<5} | {'Macro-F1':<9} | {'Bal-Acc':<8} | {'Accuracy':<9} | {'Weighted-F1':<12} | {'Train(s)':<8} | {'Latency(us)':<11}")
    print("-" * 110)
    for name, metrics in eval_results["models"].items():
        print(
            f"{name:<32} | "
            f"{metrics['feature_count']:<5} | "
            f"{metrics['macro_f1']:<9.5f} | "
            f"{metrics['balanced_accuracy']:<8.5f} | "
            f"{metrics['accuracy']:<9.5f} | "
            f"{metrics['weighted_f1']:<12.5f} | "
            f"{metrics['training_time_seconds']:<8.2f} | "
            f"{metrics['inference_microseconds_per_flow']:<11.3f}"
        )
    print("=" * 110 + "\n")

    # Print per-class F1 table
    categories = eval_results["metadata"]["categories"]
    print("=" * 120)
    print("PER-CLASS F1-SCORE BREAKDOWN")
    print("=" * 120)
    cat_header = f"{'Model Configuration':<32} | " + " | ".join([f"{c[:10]:<10}" for c in categories])
    print(cat_header)
    print("-" * 120)
    for name, metrics in eval_results["models"].items():
        row_str = f"{name:<32} | "
        f1_strs = [f"{metrics['per_class'][c]['f1_score']:<10.4f}" for c in categories]
        row_str += " | ".join(f1_strs)
        print(row_str)
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
