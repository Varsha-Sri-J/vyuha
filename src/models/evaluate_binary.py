"""VYUHA Binary Intrusion Detection Evaluation & Diagnostics Engine.

Evaluates the four Phase 3B binary classification models on held-out validation data:
- RF Full (63 features) vs RF Port-Agnostic (62 features)
- DT Full (63 features) vs DT Port-Agnostic (62 features)

Computes:
- Precision, Recall, F1, PR-AUC, ROC-AUC, FPR, FNR, Confusion Matrices.
- Training latency and per-flow inference latency.
- In-depth False Positive & False Negative attribution across attack types and ports.
- Exports machine-readable benchmark JSON and publication-quality diagnostic plots.
"""

import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Configure non-interactive backend for headless execution
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_single_model(
    pipeline: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str,
    training_time: float = 0.0,
) -> Dict[str, Any]:
    """Computes exhaustive classification metrics and latency for a single model pipeline.

    Args:
        pipeline: Trained Scikit-Learn Pipeline.
        X_val: Validation feature matrix.
        y_val: Validation binary ground truth (0=BENIGN, 1=ATTACK).
        model_name: Unique identifier for the configuration.
        training_time: Recorded training duration in seconds.

    Returns:
        Dictionary of comprehensive evaluation metrics.
    """
    n_samples = len(X_val)
    logger.info("Evaluating model: %s on %d validation flows...", model_name, n_samples)

    # Measure inference latency with microsecond precision
    t_start = time.perf_counter()
    y_pred = pipeline.predict(X_val)
    t_pred = time.perf_counter() - t_start

    # Predict probabilities for PR-AUC and ROC-AUC
    t_prob_start = time.perf_counter()
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_val)[:, 1]
    else:
        y_prob = y_pred.astype(float)
    t_prob = time.perf_counter() - t_prob_start

    # Latency benchmarks
    inf_latency_total_ms = t_pred * 1000.0
    inf_throughput_flows_sec = n_samples / t_pred if t_pred > 0 else 0.0
    inf_microseconds_per_flow = (t_pred / n_samples) * 1e6 if n_samples > 0 else 0.0

    # Confusion matrix extraction
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Rate calculations
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Metric scores
    precision = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
    f1_binary = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
    f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
    pr_auc = average_precision_score(y_val, y_prob)
    roc_auc = roc_auc_score(y_val, y_prob)
    acc = accuracy_score(y_val, y_pred)
    bal_acc = balanced_accuracy_score(y_val, y_pred)

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
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1_binary, 6),
        "f1_macro": round(f1_macro, 6),
        "pr_auc": round(pr_auc, 6),
        "roc_auc": round(roc_auc, 6),
        "false_positive_rate": round(fpr, 6),
        "false_negative_rate": round(fnr, 6),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "_predictions": y_pred,
        "_probabilities": y_prob,
    }

    logger.info(
        "[%s] F1=%.4f | PR-AUC=%.4f | Precision=%.4f | Recall=%.4f | FPR=%.4f | Latency=%.2fus/flow",
        model_name,
        f1_binary,
        pr_auc,
        precision,
        recall,
        fpr,
        inf_microseconds_per_flow,
    )
    return metrics


def analyze_prediction_errors(
    y_val: pd.Series,
    y_pred: np.ndarray,
    val_meta: pd.DataFrame,
    model_name: str,
) -> Dict[str, Any]:
    """Dissects False Positives and False Negatives against original traffic metadata.

    Args:
        y_val: Ground truth binary labels (0=BENIGN, 1=ATTACK).
        y_pred: Model predictions.
        val_meta: DataFrame with metadata columns ('label_raw', 'label_category', 'Destination Port').
        model_name: Identifier for model.

    Returns:
        Structured breakdown of errors by attack type, port, and traffic class.
    """
    df_err = val_meta.copy().reset_index(drop=True)
    df_err["y_true"] = y_val.values
    df_err["y_pred"] = y_pred
    df_err["error_type"] = "CORRECT"
    df_err.loc[(df_err["y_true"] == 0) & (df_err["y_pred"] == 1), "error_type"] = "FALSE_POSITIVE"
    df_err.loc[(df_err["y_true"] == 1) & (df_err["y_pred"] == 0), "error_type"] = "FALSE_NEGATIVE"

    fps = df_err[df_err["error_type"] == "FALSE_POSITIVE"]
    fns = df_err[df_err["error_type"] == "FALSE_NEGATIVE"]
    attacks = df_err[df_err["y_true"] == 1]

    # 1. False Positives Breakdown by Destination Port
    fp_by_port = {}
    if "Destination Port" in fps.columns and len(fps) > 0:
        port_counts = fps["Destination Port"].value_counts().head(10).to_dict()
        fp_by_port = {str(int(k)): int(v) for k, v in port_counts.items()}

    # 2. False Negatives Breakdown by Granular Raw Attack Label
    fn_by_attack_raw = {}
    fn_rates_by_attack_raw = {}
    if "label_raw" in attacks.columns and len(attacks) > 0:
        total_by_raw = attacks["label_raw"].value_counts().to_dict()
        fn_counts = fns["label_raw"].value_counts().to_dict() if len(fns) > 0 else {}
        for attack_name, total_count in total_by_raw.items():
            missed = fn_counts.get(attack_name, 0)
            fn_by_attack_raw[attack_name] = int(missed)
            fn_rates_by_attack_raw[attack_name] = round(missed / total_count, 6)

    # 3. False Negatives Breakdown by Grouped Attack Category
    fn_by_category = {}
    fn_rates_by_category = {}
    if "label_category" in attacks.columns and len(attacks) > 0:
        total_by_cat = attacks["label_category"].value_counts().to_dict()
        fn_cat_counts = fns["label_category"].value_counts().to_dict() if len(fns) > 0 else {}
        for cat_name, total_count in total_by_cat.items():
            if cat_name == "BENIGN":
                continue
            missed = fn_cat_counts.get(cat_name, 0)
            fn_by_category[cat_name] = int(missed)
            fn_rates_by_category[cat_name] = round(missed / total_count, 6)

    return {
        "model_name": model_name,
        "total_validation_records": len(df_err),
        "total_false_positives": len(fps),
        "total_false_negatives": len(fns),
        "false_positives_top_ports": fp_by_port,
        "false_negatives_by_raw_label": fn_by_attack_raw,
        "false_negative_rate_by_raw_label": fn_rates_by_attack_raw,
        "false_negatives_by_category": fn_by_category,
        "false_negative_rate_by_category": fn_rates_by_category,
    }


def generate_evaluation_plots(
    results: Dict[str, Any],
    y_val: pd.Series,
    output_dir: Union[str, Path] = "reports/figures",
) -> List[str]:
    """Generates diagnostic visualization plots comparing the 4 model configurations.

    Creates:
    - 09_binary_confusion_matrices.png: 2x2 grid of confusion matrices.
    - 10_binary_precision_recall_curves.png: Precision-Recall curves.
    - 11_port_vs_agnostic_comparison.png: Grouped bar chart comparing metrics.
    - 12_error_analysis_breakdown.png: Missed attacks by category.

    Args:
        results: Dictionary containing evaluated model metrics and error analyses.
        y_val: Ground truth binary validation labels.
        output_dir: Target directory for generated figures.

    Returns:
        List of generated image file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_plots = []

    model_names = [m for m in results if m not in ("data", "error_analysis", "summary")]

    # --- Plot 1: 09_binary_confusion_matrices.png ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        ax = axes[idx]
        cm_data = results[name]["confusion_matrix"]
        cm_matrix = np.array([
            [cm_data["true_negatives"], cm_data["false_positives"]],
            [cm_data["false_negatives"], cm_data["true_positives"]],
        ])
        # Compute percentages
        cm_norm = cm_matrix.astype("float") / cm_matrix.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(cm_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{name.upper()}\n(F1: {results[name]['f1_score']:.4f}, FPR: {results[name]['false_positive_rate']:.4f})", fontsize=11, fontweight="bold")
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(["BENIGN", "ATTACK"])
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(["BENIGN", "ATTACK"])
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

        thresh = cm_matrix.max() / 2.0
        for i in range(2):
            for j in range(2):
                raw_val = cm_matrix[i, j]
                pct_val = cm_norm[i, j] * 100.0
                ax.text(
                    j,
                    i,
                    f"{raw_val:,}\n({pct_val:.2f}%)",
                    ha="center",
                    va="center",
                    color="white" if raw_val > thresh else "black",
                    fontweight="bold",
                    fontsize=10,
                )

    plt.tight_layout()
    cm_plot_file = out_path / "09_binary_confusion_matrices.png"
    plt.savefig(cm_plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots.append(str(cm_plot_file))
    logger.info("Saved confusion matrix plot: %s", cm_plot_file)

    # --- Plot 2: 10_binary_precision_recall_curves.png ---
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"rf_full": "#1f77b4", "rf_port_agnostic": "#ff7f0e", "dt_full": "#2ca02c", "dt_port_agnostic": "#d62728"}
    labels = {
        "rf_full": "RF Full (63 feat)",
        "rf_port_agnostic": "RF Port-Agnostic (62 feat)",
        "dt_full": "DT Full (63 feat)",
        "dt_port_agnostic": "DT Port-Agnostic (62 feat)",
    }

    for name in model_names:
        y_prob = results[name]["_probabilities"]
        # PR Curve
        prec_curve, rec_curve, _ = precision_recall_curve(y_val, y_prob)
        pr_score = results[name]["pr_auc"]
        ax_pr.plot(rec_curve, prec_curve, label=f"{labels.get(name, name)} (PR-AUC={pr_score:.4f})", color=colors.get(name), lw=2)

        # ROC Curve
        fpr_curve, tpr_curve, _ = roc_curve(y_val, y_prob)
        roc_score = results[name]["roc_auc"]
        ax_roc.plot(fpr_curve, tpr_curve, label=f"{labels.get(name, name)} (ROC-AUC={roc_score:.4f})", color=colors.get(name), lw=2)

    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.set_title("Precision-Recall Curves (Binary Intrusion)", fontsize=12, fontweight="bold")
    ax_pr.grid(True, linestyle="--", alpha=0.6)
    ax_pr.legend(loc="lower left", fontsize=10)

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax_roc.grid(True, linestyle="--", alpha=0.6)
    ax_roc.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    curve_plot_file = out_path / "10_binary_precision_recall_curves.png"
    plt.savefig(curve_plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots.append(str(curve_plot_file))
    logger.info("Saved PR/ROC curves plot: %s", curve_plot_file)

    # --- Plot 3: 11_port_vs_agnostic_comparison.png ---
    fig, (ax_bar, ax_lat) = plt.subplots(1, 2, figsize=(14, 5))

    metrics_to_plot = ["f1_score", "pr_auc", "recall", "precision"]
    x = np.arange(len(metrics_to_plot))
    width = 0.2

    for idx, name in enumerate(model_names):
        vals = [results[name][m] for m in metrics_to_plot]
        ax_bar.bar(x + idx * width, vals, width, label=labels.get(name, name), color=colors.get(name))

    ax_bar.set_xticks(x + width * 1.5)
    ax_bar.set_xticklabels(["F1-Score", "PR-AUC", "Recall", "Precision"], fontsize=11)
    ax_bar.set_ylim(0.95, 1.002)
    ax_bar.set_ylabel("Score", fontsize=11)
    ax_bar.set_title("Performance Comparison Across Configurations", fontsize=12, fontweight="bold")
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax_bar.legend(loc="lower right", fontsize=9)

    # Latency & FPR Subplot
    fpr_vals = [results[name]["false_positive_rate"] * 100.0 for name in model_names]
    lat_vals = [results[name]["inference_microseconds_per_flow"] for name in model_names]

    ax_lat.scatter(lat_vals, fpr_vals, s=150, c=[colors.get(m) for m in model_names], edgecolors="black", zorder=5)
    for name in model_names:
        ax_lat.annotate(
            labels.get(name, name),
            (results[name]["inference_microseconds_per_flow"], results[name]["false_positive_rate"] * 100.0),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9,
            fontweight="bold",
        )
    ax_lat.set_xlabel("Inference Latency (Microseconds / Flow)", fontsize=11)
    ax_lat.set_ylabel("False Positive Rate (%)", fontsize=11)
    ax_lat.set_title("Operational Trade-off: Latency vs False Alarm Rate", fontsize=12, fontweight="bold")
    ax_lat.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    comp_plot_file = out_path / "11_port_vs_agnostic_comparison.png"
    plt.savefig(comp_plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots.append(str(comp_plot_file))
    logger.info("Saved comparison bar plot: %s", comp_plot_file)

    # --- Plot 4: 12_error_analysis_breakdown.png ---
    err_analysis = results.get("error_analysis", {})
    if err_analysis:
        fig, ax_err = plt.subplots(figsize=(10, 6))
        # Compare Miss Rate by Category across models
        categories = sorted(list(err_analysis[model_names[0]]["false_negative_rate_by_category"].keys()))
        x_cats = np.arange(len(categories))
        w = 0.2
        for idx, name in enumerate(model_names):
            fn_rates = [err_analysis[name]["false_negative_rate_by_category"].get(c, 0.0) * 100.0 for c in categories]
            ax_err.bar(x_cats + idx * w, fn_rates, w, label=labels.get(name, name), color=colors.get(name))

        ax_err.set_xticks(x_cats + w * 1.5)
        ax_err.set_xticklabels(categories, rotation=30, ha="right", fontsize=10)
        ax_err.set_ylabel("Miss Rate / False Negative Rate (%)", fontsize=11)
        ax_err.set_title("Attack Category Miss Rate (False Negatives)", fontsize=12, fontweight="bold")
        ax_err.grid(True, axis="y", linestyle="--", alpha=0.6)
        ax_err.legend(loc="upper right", fontsize=10)

        plt.tight_layout()
        err_plot_file = out_path / "12_error_analysis_breakdown.png"
        plt.savefig(err_plot_file, dpi=150, bbox_inches="tight")
        plt.close()
        generated_plots.append(str(err_plot_file))
        logger.info("Saved error breakdown plot: %s", err_plot_file)

    return generated_plots


def run_evaluation_suite(
    train_results: Dict[str, Any],
    output_report_path: Union[str, Path] = "reports/binary_benchmark_results.json",
    figures_dir: Union[str, Path] = "reports/figures",
) -> Dict[str, Any]:
    """Runs full evaluation and error diagnostics across all trained binary models.

    Args:
        train_results: Dictionary output from train_binary.run_training_suite.
        output_report_path: Path for writing JSON results.
        figures_dir: Directory for writing evaluation plots.

    Returns:
        Consolidated evaluation dictionary.
    """
    X_val = train_results["data"]["X_val"]
    y_val = train_results["data"]["y_val"]
    val_meta = train_results["data"]["val_meta"]

    benchmark_summary: Dict[str, Any] = {
        "metadata": {
            "validation_samples": len(X_val),
            "training_samples": train_results["data"]["n_train"],
            "attack_count_val": int(y_val.sum()),
            "benign_count_val": int((y_val == 0).sum()),
            "attack_prevalence_pct": round(float(y_val.mean() * 100), 2),
        },
        "models": {},
        "error_analysis": {},
    }

    eval_plots_input = {}
    model_keys = [k for k in train_results if k != "data"]

    for name in model_keys:
        pipe = train_results[name]["pipeline"]
        train_time = train_results[name]["training_time"]
        eval_metrics = evaluate_single_model(pipe, X_val, y_val, name, training_time=train_time)

        # Dissect errors
        y_pred = eval_metrics["_predictions"]
        err_diag = analyze_prediction_errors(y_val, y_pred, val_meta, name)

        eval_plots_input[name] = eval_metrics
        benchmark_summary["models"][name] = {
            k: v for k, v in eval_metrics.items() if not k.startswith("_")
        }
        benchmark_summary["error_analysis"][name] = err_diag

    eval_plots_input["error_analysis"] = benchmark_summary["error_analysis"]

    # Generate figures
    plots = generate_evaluation_plots(eval_plots_input, y_val, output_dir=figures_dir)
    benchmark_summary["generated_figures"] = plots

    # Save machine-readable JSON
    report_file = Path(output_report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)
    logger.info("Saved benchmark results JSON: %s", report_file)

    return benchmark_summary
