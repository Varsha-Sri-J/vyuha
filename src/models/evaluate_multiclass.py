"""VYUHA Multiclass Intrusion Detection Evaluation & Diagnostics Engine.

Evaluates the four Phase 3C multiclass classification models across the 8 attack families:
- RF Full (63 features) vs RF Port-Agnostic (62 features)
- DT Full (63 features) vs DT Port-Agnostic (62 features)

Computes:
- Overall: Accuracy, Balanced Accuracy, Macro Precision/Recall/F1, Weighted F1.
- Per-Class: Precision, Recall, F1, Support for each of the 8 canonical categories.
- Confusion Matrices: Raw counts and normalized percentages.
- Diagnostic plots and machine-readable JSON exports.
- Detailed inter-attack confusion analysis and rare-class caveats.
"""

import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Configure non-interactive backend for headless execution
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.train_multiclass import CANONICAL_CATEGORIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_single_multiclass_model(
    pipeline: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str,
    training_time: float = 0.0,
    categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Computes exhaustive multiclass classification metrics and latency.

    Args:
        pipeline: Trained Scikit-Learn Pipeline.
        X_val: Validation feature matrix.
        y_val: Validation ground truth Series (category strings).
        model_name: Unique identifier for the configuration.
        training_time: Recorded training duration in seconds.
        categories: Ordered list of categories (defaults to CANONICAL_CATEGORIES).

    Returns:
        Dictionary of comprehensive evaluation metrics.
    """
    if categories is None:
        categories = CANONICAL_CATEGORIES

    n_samples = len(X_val)
    logger.info("Evaluating multiclass model: %s on %d validation flows...", model_name, n_samples)

    # Measure inference latency with microsecond precision
    t_start = time.perf_counter()
    y_pred = pipeline.predict(X_val)
    t_pred = time.perf_counter() - t_start

    inf_throughput_flows_sec = n_samples / t_pred if t_pred > 0 else 0.0
    inf_microseconds_per_flow = (t_pred / n_samples) * 1e6 if n_samples > 0 else 0.0

    # Macro & Overall Metrics
    acc = accuracy_score(y_val, y_pred)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    macro_prec = precision_score(y_val, y_pred, labels=categories, average="macro", zero_division=0)
    macro_rec = recall_score(y_val, y_pred, labels=categories, average="macro", zero_division=0)
    macro_f1 = f1_score(y_val, y_pred, labels=categories, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_val, y_pred, labels=categories, average="weighted", zero_division=0)

    # Per-Class Metrics
    per_class_prec = precision_score(y_val, y_pred, labels=categories, average=None, zero_division=0)
    per_class_rec = recall_score(y_val, y_pred, labels=categories, average=None, zero_division=0)
    per_class_f1_vals = f1_score(y_val, y_pred, labels=categories, average=None, zero_division=0)

    # Confusion Matrix (8x8)
    cm_raw = confusion_matrix(y_val, y_pred, labels=categories)
    cm_norm = np.zeros_like(cm_raw, dtype=float)
    row_sums = cm_raw.sum(axis=1)
    for i, s in enumerate(row_sums):
        if s > 0:
            cm_norm[i] = cm_raw[i] / s

    per_class_metrics = {}
    for idx, cat in enumerate(categories):
        support = int(row_sums[idx])
        per_class_metrics[cat] = {
            "precision": round(float(per_class_prec[idx]), 6),
            "recall": round(float(per_class_rec[idx]), 6),
            "f1_score": round(float(per_class_f1_vals[idx]), 6),
            "support": support,
        }

    selector = pipeline.named_steps.get("selector")
    feature_count = len(selector.selected_features_) if selector else X_val.shape[1]

    metrics = {
        "model_name": model_name,
        "feature_count": feature_count,
        "training_time_seconds": round(training_time, 4),
        "inference_time_seconds": round(t_pred, 4),
        "inference_throughput_flows_sec": round(inf_throughput_flows_sec, 2),
        "inference_microseconds_per_flow": round(inf_microseconds_per_flow, 4),
        "accuracy": round(acc, 6),
        "balanced_accuracy": round(bal_acc, 6),
        "macro_precision": round(macro_prec, 6),
        "macro_recall": round(macro_rec, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "per_class": per_class_metrics,
        "confusion_matrix_raw": cm_raw.tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, 6).tolist(),
        "categories": categories,
        "_predictions": y_pred,
    }

    logger.info(
        "[%s] Macro-F1=%.4f | Bal-Acc=%.4f | Acc=%.4f | Weighted-F1=%.4f | Latency=%.2fus/flow",
        model_name,
        macro_f1,
        bal_acc,
        acc,
        weighted_f1,
        inf_microseconds_per_flow,
    )
    return metrics


def analyze_multiclass_errors(
    y_val: pd.Series,
    y_pred: np.ndarray,
    val_meta: pd.DataFrame,
    model_name: str,
    categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Diagnoses inter-attack confusion, false alarms, and attacks escaping detection.

    Args:
        y_val: Ground-truth series of category strings.
        y_pred: Predicted category strings.
        val_meta: Metadata DataFrame.
        model_name: Unique model identifier.
        categories: Canonical class list.

    Returns:
        Structured error attribution dictionary.
    """
    if categories is None:
        categories = CANONICAL_CATEGORIES

    df_err = val_meta.copy().reset_index(drop=True)
    df_err["y_true"] = y_val.values
    df_err["y_pred"] = y_pred

    # 1. Attacks Misclassified as BENIGN (Escaped Intrusions / False Negatives against security)
    attacks_as_benign = df_err[(df_err["y_true"] != "BENIGN") & (df_err["y_pred"] == "BENIGN")]
    attacks_as_benign_counts = attacks_as_benign["y_true"].value_counts().to_dict()

    # Total counts per attack class
    attack_totals = df_err[df_err["y_true"] != "BENIGN"]["y_true"].value_counts().to_dict()
    escaped_rates = {}
    for cat, total in attack_totals.items():
        escaped = attacks_as_benign_counts.get(cat, 0)
        escaped_rates[cat] = round(escaped / total, 6)

    # 2. BENIGN Misclassified as Attacks (False Alarms)
    benign_as_attacks = df_err[(df_err["y_true"] == "BENIGN") & (df_err["y_pred"] != "BENIGN")]
    benign_as_attacks_counts = benign_as_attacks["y_pred"].value_counts().to_dict()

    # 3. Inter-Attack Confusion (Attacks classified as the WRONG attack family)
    inter_attack_conf = df_err[
        (df_err["y_true"] != "BENIGN")
        & (df_err["y_pred"] != "BENIGN")
        & (df_err["y_true"] != df_err["y_pred"])
    ]
    inter_attack_pairs = {}
    if len(inter_attack_conf) > 0:
        pair_counts = (
            inter_attack_conf["y_true"] + " -> " + inter_attack_conf["y_pred"]
        ).value_counts().head(10).to_dict()
        inter_attack_pairs = {k: int(v) for k, v in pair_counts.items()}

    return {
        "model_name": model_name,
        "total_validation_flows": len(df_err),
        "total_attacks_misclassified_as_benign": len(attacks_as_benign),
        "attacks_misclassified_as_benign_counts": {k: int(v) for k, v in attacks_as_benign_counts.items()},
        "attacks_misclassified_as_benign_rates": escaped_rates,
        "total_false_alarms": len(benign_as_attacks),
        "false_alarms_breakdown_by_predicted_attack": {k: int(v) for k, v in benign_as_attacks_counts.items()},
        "total_inter_attack_confusions": len(inter_attack_conf),
        "top_inter_attack_confusion_pairs": inter_attack_pairs,
    }


def generate_multiclass_plots(
    results: Dict[str, Any],
    y_val: pd.Series,
    output_dir: Union[str, Path] = "reports/figures",
    categories: Optional[List[str]] = None,
) -> List[str]:
    """Generates publication-quality diagnostic plots for multiclass attack classification.

    Creates:
    - 13_multiclass_confusion_matrices.png: 2x2 grid of 8x8 normalized confusion heatmaps.
    - 14_multiclass_per_class_f1_comparison.png: Grouped bar chart of per-class F1-scores.
    - 15_multiclass_port_vs_agnostic.png: Full vs Port-Agnostic comparison.
    - 16_multiclass_error_confusion_analysis.png: Escaped attacks & inter-attack confusion.

    Args:
        results: Benchmark results dictionary.
        y_val: Validation ground truth.
        output_dir: Target directory for figure persistence.
        categories: Canonical class labels.

    Returns:
        List of generated figure paths.
    """
    if categories is None:
        categories = CANONICAL_CATEGORIES

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_plots = []

    model_names = [m for m in results if m not in ("data", "error_analysis", "summary")]
    colors = {
        "multiclass_rf_full": "#1f77b4",
        "multiclass_rf_port_agnostic": "#ff7f0e",
        "multiclass_dt_full": "#2ca02c",
        "multiclass_dt_port_agnostic": "#d62728",
    }
    labels = {
        "multiclass_rf_full": "RF Full (63 feat)",
        "multiclass_rf_port_agnostic": "RF Port-Agnostic (62 feat)",
        "multiclass_dt_full": "DT Full (63 feat)",
        "multiclass_dt_port_agnostic": "DT Port-Agnostic (62 feat)",
    }

    # Short category labels for clean heatmap axes
    short_cats = [
        "BENIGN", "DoS/DDoS", "PortScan", "BruteForce",
        "WebAttack", "Botnet", "Infiltration", "Heartbleed"
    ]

    # --- Plot 1: 13_multiclass_confusion_matrices.png ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        ax = axes[idx]
        cm_norm = np.array(results[name]["confusion_matrix_normalized"])
        cm_raw = np.array(results[name]["confusion_matrix_raw"])

        im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
        ax.set_title(
            f"{labels.get(name, name)}\n(Macro-F1: {results[name]['macro_f1']:.4f}, Bal-Acc: {results[name]['balanced_accuracy']:.4f})",
            fontsize=12,
            fontweight="bold",
        )
        tick_marks = np.arange(len(short_cats))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(short_cats, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(short_cats, fontsize=9)
        ax.set_ylabel("True Category", fontsize=10)
        ax.set_xlabel("Predicted Category", fontsize=10)

        # Annotate percentages
        for i in range(len(short_cats)):
            for j in range(len(short_cats)):
                pct = cm_norm[i, j] * 100.0
                raw = cm_raw[i, j]
                if raw > 0:
                    text_str = f"{pct:.1f}%\n({raw})" if pct >= 1.0 else f"{raw}"
                    ax.text(
                        j, i, text_str,
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black",
                        fontsize=7,
                        fontweight="bold" if i == j else "normal",
                    )

    plt.tight_layout()
    cm_file = out_path / "13_multiclass_confusion_matrices.png"
    plt.savefig(cm_file, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots.append(str(cm_file))
    logger.info("Saved multiclass confusion matrices: %s", cm_file)

    # --- Plot 2: 14_multiclass_per_class_f1_comparison.png ---
    fig, ax_f1 = plt.subplots(figsize=(14, 6))
    x = np.arange(len(categories))
    width = 0.2

    for idx, name in enumerate(model_names):
        f1_vals = [results[name]["per_class"][cat]["f1_score"] for cat in categories]
        ax_f1.bar(x + idx * width, f1_vals, width, label=labels.get(name, name), color=colors.get(name))

    ax_f1.set_xticks(x + width * 1.5)
    ax_f1.set_xticklabels(categories, rotation=25, ha="right", fontsize=10)
    ax_f1.set_ylabel("F1-Score", fontsize=11)
    ax_f1.set_ylim(0.0, 1.05)
    ax_f1.set_title("Per-Class F1-Score Across Attack Families", fontsize=13, fontweight="bold")
    ax_f1.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax_f1.legend(loc="lower left", fontsize=10)

    plt.tight_layout()
    f1_file = out_path / "14_multiclass_per_class_f1_comparison.png"
    plt.savefig(f1_file, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots.append(str(f1_file))
    logger.info("Saved per-class F1 plot: %s", f1_file)

    # --- Plot 3: 15_multiclass_port_vs_agnostic.png ---
    fig, (ax_macro, ax_diff) = plt.subplots(1, 2, figsize=(14, 5))

    macro_metrics = ["macro_f1", "balanced_accuracy", "accuracy", "weighted_f1"]
    metric_labels = ["Macro F1", "Balanced Acc", "Accuracy", "Weighted F1"]
    xm = np.arange(len(macro_metrics))

    for idx, name in enumerate(model_names):
        vals = [results[name][m] for m in macro_metrics]
        ax_macro.bar(xm + idx * width, vals, width, label=labels.get(name, name), color=colors.get(name))

    ax_macro.set_xticks(xm + width * 1.5)
    ax_macro.set_xticklabels(metric_labels, fontsize=10)
    ax_macro.set_ylim(0.85, 1.02)
    ax_macro.set_ylabel("Score", fontsize=11)
    ax_macro.set_title("Aggregate Multiclass Performance Metrics", fontsize=12, fontweight="bold")
    ax_macro.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax_macro.legend(loc="lower right", fontsize=9)

    # F1 Difference (Full minus Port-Agnostic) for RF and DT
    rf_diffs = [
        results["multiclass_rf_full"]["per_class"][cat]["f1_score"]
        - results["multiclass_rf_port_agnostic"]["per_class"][cat]["f1_score"]
        for cat in categories
    ]
    dt_diffs = [
        results["multiclass_dt_full"]["per_class"][cat]["f1_score"]
        - results["multiclass_dt_port_agnostic"]["per_class"][cat]["f1_score"]
        for cat in categories
    ]

    xd = np.arange(len(categories))
    w_d = 0.35
    ax_diff.bar(xd - w_d/2, rf_diffs, w_d, label="RF Delta (Full - Agnostic)", color="#1f77b4")
    ax_diff.bar(xd + w_d/2, dt_diffs, w_d, label="DT Delta (Full - Agnostic)", color="#2ca02c")
    ax_diff.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax_diff.set_xticks(xd)
    ax_diff.set_xticklabels(categories, rotation=30, ha="right", fontsize=9)
    ax_diff.set_ylabel("F1 Drop When Port Removed (Delta)", fontsize=10)
    ax_diff.set_title("Impact of Removing Destination Port (Per-Class)", fontsize=12, fontweight="bold")
    ax_diff.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax_diff.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    port_comp_file = out_path / "15_multiclass_port_vs_agnostic.png"
    plt.savefig(port_comp_file, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots.append(str(port_comp_file))
    logger.info("Saved port vs agnostic comparison plot: %s", port_comp_file)

    # --- Plot 4: 16_multiclass_error_confusion_analysis.png ---
    err_analysis = results.get("error_analysis", {})
    if err_analysis:
        fig, ax_esc = plt.subplots(figsize=(10, 5))
        attack_cats = [c for c in categories if c != "BENIGN"]
        xa = np.arange(len(attack_cats))

        for idx, name in enumerate(model_names):
            esc_rates = [
                err_analysis[name]["attacks_misclassified_as_benign_rates"].get(c, 0.0) * 100.0
                for c in attack_cats
            ]
            ax_esc.bar(xa + idx * width, esc_rates, width, label=labels.get(name, name), color=colors.get(name))

        ax_esc.set_xticks(xa + width * 1.5)
        ax_esc.set_xticklabels(attack_cats, rotation=25, ha="right", fontsize=10)
        ax_esc.set_ylabel("Miss Rate / Escaped to BENIGN (%)", fontsize=11)
        ax_esc.set_title("Attacks Misclassified as BENIGN (Detection Blind Spots)", fontsize=12, fontweight="bold")
        ax_esc.grid(True, axis="y", linestyle="--", alpha=0.6)
        ax_esc.legend(loc="upper right", fontsize=10)

        plt.tight_layout()
        err_file = out_path / "16_multiclass_error_confusion_analysis.png"
        plt.savefig(err_file, dpi=150, bbox_inches="tight")
        plt.close()
        generated_plots.append(str(err_file))
        logger.info("Saved error confusion plot: %s", err_file)

    return generated_plots


def run_multiclass_evaluation_suite(
    train_results: Dict[str, Any],
    output_report_path: Union[str, Path] = "reports/multiclass_benchmark_results.json",
    figures_dir: Union[str, Path] = "reports/figures",
) -> Dict[str, Any]:
    """Runs full evaluation and error diagnostics across all trained multiclass models.

    Args:
        train_results: Output dictionary from train_multiclass.run_multiclass_training_suite.
        output_report_path: Path for saving JSON results.
        figures_dir: Directory for writing diagnostic plots.

    Returns:
        Consolidated multiclass evaluation dictionary.
    """
    X_val = train_results["data"]["X_val"]
    y_val = train_results["data"]["y_val"]
    val_meta = train_results["data"]["val_meta"]
    categories = train_results["data"].get("categories", CANONICAL_CATEGORIES)

    benchmark_summary: Dict[str, Any] = {
        "metadata": {
            "validation_samples": len(X_val),
            "training_samples": train_results["data"]["n_train"],
            "categories": categories,
            "class_distribution_val": {cat: int((y_val == cat).sum()) for cat in categories},
        },
        "models": {},
        "error_analysis": {},
    }

    eval_plots_input = {}
    model_keys = [k for k in train_results if k != "data"]

    for name in model_keys:
        pipe = train_results[name]["pipeline"]
        train_time = train_results[name]["training_time"]
        eval_metrics = evaluate_single_multiclass_model(
            pipe, X_val, y_val, name, training_time=train_time, categories=categories
        )

        # Error diagnostics
        y_pred = eval_metrics["_predictions"]
        err_diag = analyze_multiclass_errors(y_val, y_pred, val_meta, name, categories=categories)

        eval_plots_input[name] = eval_metrics
        benchmark_summary["models"][name] = {
            k: v for k, v in eval_metrics.items() if not k.startswith("_")
        }
        benchmark_summary["error_analysis"][name] = err_diag

    eval_plots_input["error_analysis"] = benchmark_summary["error_analysis"]

    # Generate figures
    plots = generate_multiclass_plots(eval_plots_input, y_val, output_dir=figures_dir, categories=categories)
    benchmark_summary["generated_figures"] = plots

    # Save machine-readable JSON
    report_file = Path(output_report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)
    logger.info("Saved multiclass benchmark results JSON: %s", report_file)

    return benchmark_summary
