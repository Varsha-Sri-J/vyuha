"""VYUHA Unsupervised Anomaly Detection Evaluation & Diagnostic Module (Phase 3D).

Provides comprehensive evaluation metrics, threshold benchmarks, per-attack
anomaly score analysis, latency benchmarking, and publication-quality diagnostic plots:
- Figure 17: Anomaly Score Distributions by Traffic Category
- Figure 18: ROC and Precision-Recall Curves with Calibrated Operating Thresholds
- Figure 19: Per-Attack-Family Anomaly Detection Rates Across Calibrated Thresholds
- Figure 20: Training Size (50k/100k/200k) & Port-Agnostic Comparative Analysis
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

CANONICAL_CATEGORIES = [
    "BENIGN",
    "DoS_DDoS",
    "PortScan",
    "BruteForce",
    "WebAttack",
    "Botnet",
    "Infiltration",
    "Heartbleed",
]


def compute_anomaly_scores(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    """Computes normalized anomaly scores for input feature matrix.

    Score Definition:
        scikit-learn IsolationForest score_samples(X) returns negative values,
        where lower/more negative means more anomalous.
        We return inverted score:
            A(x) = - score_samples(x)
        Higher A(x) indicates greater degree of anomaly (A(x) in [0, 1]).

    Args:
        pipeline: Fitted anomaly detection pipeline.
        X: Feature matrix.

    Returns:
        1D numpy array of anomaly scores.
    """
    detector = pipeline.named_steps["anomaly_detector"]

    # Transform through preceding steps (selector, scaler)
    if len(pipeline.steps) > 1:
        prep_pipeline = Pipeline(pipeline.steps[:-1])
        X_trans = prep_pipeline.transform(X)
    else:
        X_trans = X

    raw_scores = detector.score_samples(X_trans)
    return -raw_scores


def benchmark_anomaly_inference(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    n_warmup: int = 3,
    n_runs: int = 5,
    sample_size: int = 50000,
) -> Dict[str, float]:
    """Measures batch inference latency and throughput for anomaly scoring.

    Args:
        pipeline: Fitted pipeline.
        X_val: Validation DataFrame.
        n_warmup: Warm-up passes.
        n_runs: Measurement repetitions.
        sample_size: Number of flows used for benchmark.

    Returns:
        Dict with mean batch time, amortized microseconds per flow, and throughput.
    """
    benchmark_data = X_val.iloc[:sample_size].copy()

    # Warm-up passes
    for _ in range(n_warmup):
        _ = compute_anomaly_scores(pipeline, benchmark_data)

    # Timed runs
    durations = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = compute_anomaly_scores(pipeline, benchmark_data)
        durations.append(time.perf_counter() - t0)

    mean_duration = float(np.mean(durations))
    latency_us_per_flow = float((mean_duration / len(benchmark_data)) * 1e6)
    throughput_flows_sec = float(len(benchmark_data) / mean_duration)

    return {
        "batch_inference_seconds": round(mean_duration, 5),
        "amortized_microseconds_per_flow": round(latency_us_per_flow, 4),
        "inference_throughput_flows_sec": round(throughput_flows_sec, 2),
    }


def evaluate_anomaly_detector(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    calibrated_thresholds: Dict[str, Dict[str, float]],
    benchmark_latency: bool = True,
) -> Dict[str, Union[float, dict, list]]:
    """Evaluates an anomaly detector against mixed validation flows.

    Args:
        pipeline: Fitted anomaly pipeline.
        X_val: Validation feature matrix.
        y_val: Validation target series (containing multiclass category names).
        calibrated_thresholds: Dict of calibrated thresholds from training/calibration stage.
        benchmark_latency: Whether to run inference speed benchmarking.

    Returns:
        Comprehensive dictionary of evaluation results.
    """
    logger.info("Computing anomaly scores on %d validation flows...", len(X_val))
    scores = compute_anomaly_scores(pipeline, X_val)

    # Binary ground truth: BENIGN=0, ATTACK=1
    y_true_binary = (y_val != "BENIGN").astype(int).values

    # Continuous ranking metrics
    roc_auc = float(roc_auc_score(y_true_binary, scores))
    pr_auc = float(average_precision_score(y_true_binary, scores))

    # Curve calculation for serialization and plotting
    fpr_arr, tpr_arr, roc_thresh = roc_curve(y_true_binary, scores)
    prec_arr, rec_arr, pr_thresh = precision_recall_curve(y_true_binary, scores)

    # Subsample curve points for compact JSON storage (100 points)
    indices = np.linspace(0, len(fpr_arr) - 1, 100, dtype=int)
    compact_roc = {
        "fpr": [float(fpr_arr[i]) for i in indices],
        "tpr": [float(tpr_arr[i]) for i in indices],
    }

    pr_indices = np.linspace(0, len(prec_arr) - 1, 100, dtype=int)
    compact_pr = {
        "precision": [float(prec_arr[i]) for i in pr_indices],
        "recall": [float(rec_arr[i]) for i in pr_indices],
    }

    # Per-category score statistics
    category_score_stats = {}
    for cat in CANONICAL_CATEGORIES:
        mask = (y_val.values == cat)
        if np.sum(mask) > 0:
            cat_scores = scores[mask]
            category_score_stats[cat] = {
                "support": int(np.sum(mask)),
                "min": float(np.min(cat_scores)),
                "median": float(np.median(cat_scores)),
                "mean": float(np.mean(cat_scores)),
                "max": float(np.max(cat_scores)),
                "std": float(np.std(cat_scores)),
            }

    # Threshold evaluation
    threshold_evaluations = {}
    for key, cal_info in calibrated_thresholds.items():
        threshold = cal_info["threshold"]
        y_pred = (scores >= threshold).astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred, labels=[0, 1]).ravel()

        # Binary metrics
        p, r, f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred, average="binary", zero_division=0
        )
        benign_fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
        attack_fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0

        # Per-category recall (detection rate)
        per_class_detection_rate = {}
        for cat in CANONICAL_CATEGORIES:
            cat_mask = (y_val.values == cat)
            if np.sum(cat_mask) > 0:
                det_rate = float(np.mean(y_pred[cat_mask] == 1))
                per_class_detection_rate[cat] = {
                    "detection_rate": det_rate,
                    "detected_count": int(np.sum(y_pred[cat_mask] == 1)),
                    "total_count": int(np.sum(cat_mask)),
                }

        threshold_evaluations[key] = {
            "target_fpr": cal_info["target_fpr"],
            "threshold": threshold,
            "validation_benign_fpr": benign_fpr,
            "validation_attack_fnr": attack_fnr,
            "precision": float(p),
            "recall": float(r),
            "f1_score": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "per_class_detection_rate": per_class_detection_rate,
        }

    results = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "category_score_stats": category_score_stats,
        "threshold_evaluations": threshold_evaluations,
        "compact_roc": compact_roc,
        "compact_pr": compact_pr,
    }

    if benchmark_latency:
        latency_info = benchmark_anomaly_inference(pipeline, X_val)
        results["latency"] = latency_info

    return results


def generate_anomaly_diagnostic_plots(
    experiments_data: Dict[str, dict],
    output_dir: Union[str, Path] = "reports/figures/",
) -> List[str]:
    """Generates all publication-quality diagnostic plots for Phase 3D:

    17_anomaly_score_distributions.png
    18_anomaly_roc_pr_curves.png
    19_anomaly_per_attack_recall.png
    20_anomaly_feature_and_size_comparison.png

    Args:
        experiments_data: Dictionary containing results for all evaluated models.
        output_dir: Directory where figures will be saved.

    Returns:
        List of generated figure file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_figures = []

    # Choose primary representative model (e.g. iforest_full_100k)
    primary_key = "iforest_full_100k"
    if primary_key not in experiments_data:
        primary_key = list(experiments_data.keys())[0]
    primary = experiments_data[primary_key]

    # -------------------------------------------------------------
    # Figure 17: Anomaly Score Distributions by Traffic Category
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))
    categories = [c for c in CANONICAL_CATEGORIES if c in primary["category_score_stats"]]
    stats = [primary["category_score_stats"][c] for c in categories]

    medians = [s["median"] for s in stats]
    means = [s["mean"] for s in stats]
    mins = [s["min"] for s in stats]
    maxs = [s["max"] for s in stats]

    y_pos = np.arange(len(categories))
    colors = ["#2b5c8f" if c == "BENIGN" else "#d95f02" for c in categories]

    # Horizontal bar of median scores
    bars = ax.barh(y_pos, medians, height=0.55, color=colors, alpha=0.85, edgecolor="black")

    # Overlay min-max range error bars and mean points
    for i, (mn, mx, mdn, mean) in enumerate(zip(mins, maxs, medians, means)):
        ax.plot([mn, mx], [i, i], color="black", lw=1.2, zorder=3)
        ax.scatter([mean], [i], color="gold", edgecolor="black", s=60, zorder=4, label="Mean" if i == 0 else "")

    # Overlay 1% calibrated threshold line
    cal_1pct = primary["threshold_evaluations"].get("fpr_1.0pct", {})
    if "threshold" in cal_1pct:
        tau = cal_1pct["threshold"]
        ax.axvline(tau, color="#e41a1c", ls="--", lw=2, label=f"Calibrated 1% FPR Threshold (tau={tau:.3f})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{c} (N={s['support']:,})" for c, s in zip(categories, stats)], fontsize=10)
    ax.set_xlabel("Anomaly Score $A(x) = -\\mathrm{score\\_samples}(x)$ (Higher = More Anomalous)", fontsize=11)
    ax.set_title(
        f"Figure 17: Anomaly Score Distributions Across Network Traffic Categories\n"
        f"Isolation Forest (100k Benign Training Flows, 63 Features)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, axis="x", ls=":", alpha=0.6)
    fig.tight_layout()

    fig17_path = out_path / "17_anomaly_score_distributions.png"
    fig.savefig(fig17_path, dpi=300)
    plt.close(fig)
    generated_figures.append(str(fig17_path))

    # -------------------------------------------------------------
    # Figure 18: ROC and Precision-Recall Curves with Calibrated Thresholds
    # -------------------------------------------------------------
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curve
    roc_data = primary["compact_roc"]
    ax_roc.plot(roc_data["fpr"], roc_data["tpr"], color="#2b5c8f", lw=2.5, label=f"ROC Curve (AUC = {primary['roc_auc']:.4f})")
    ax_roc.plot([0, 1], [0, 1], color="gray", ls="--", lw=1)

    # Plot calibrated operating points on ROC
    marker_colors = {"fpr_1.0pct": "#e41a1c", "fpr_0.5pct": "#ff7f00", "fpr_0.1pct": "#984ea3", "fpr_5.0pct": "#4daf4a"}
    for k, v in primary["threshold_evaluations"].items():
        c = marker_colors.get(k, "black")
        ax_roc.scatter(v["validation_benign_fpr"], v["recall"], color=c, s=90, zorder=5, label=f"Cal {k} (FPR={v['validation_benign_fpr']*100:.2f}%, R={v['recall']*100:.1f}%)")

    ax_roc.set_xlabel("False Positive Rate (Benign Traffic)", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (Attack Recall)", fontsize=11)
    ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax_roc.grid(True, ls=":", alpha=0.6)

    # PR Curve
    pr_data = primary["compact_pr"]
    ax_pr.plot(pr_data["recall"], pr_data["precision"], color="#d95f02", lw=2.5, label=f"PR Curve (PR-AUC = {primary['pr_auc']:.4f})")
    base_rate = 68138 / 399906
    ax_pr.axhline(base_rate, color="gray", ls="--", lw=1, label=f"Prevalence Base Rate ({base_rate*100:.1f}%)")

    for k, v in primary["threshold_evaluations"].items():
        c = marker_colors.get(k, "black")
        ax_pr.scatter(v["recall"], v["precision"], color=c, s=90, zorder=5, label=f"Cal {k} (P={v['precision']*100:.1f}%, R={v['recall']*100:.1f}%)")

    ax_pr.set_xlabel("Recall (True Positive Rate)", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax_pr.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax_pr.grid(True, ls=":", alpha=0.6)

    fig.suptitle("Figure 18: Anomaly Detection Performance & Calibrated Operating Thresholds", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    fig18_path = out_path / "18_anomaly_roc_pr_curves.png"
    fig.savefig(fig18_path, dpi=300)
    plt.close(fig)
    generated_figures.append(str(fig18_path))

    # -------------------------------------------------------------
    # Figure 19: Per-Attack-Family Anomaly Detection Rates
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6.5))

    target_threshold_keys = ["fpr_1.0pct", "fpr_0.5pct", "fpr_0.1pct"]
    available_keys = [k for k in target_threshold_keys if k in primary["threshold_evaluations"]]

    attack_cats = [c for c in CANONICAL_CATEGORIES if c != "BENIGN"]
    x = np.arange(len(attack_cats))
    width = 0.25

    colors_thresh = ["#e41a1c", "#ff7f00", "#377eb8"]
    for idx, k in enumerate(available_keys):
        eval_dict = primary["threshold_evaluations"][k]
        recalls = [eval_dict["per_class_detection_rate"].get(cat, {}).get("detection_rate", 0.0) * 100 for cat in attack_cats]
        ax.bar(x + (idx - 1) * width, recalls, width, label=f"Target {k} (Tau={eval_dict['threshold']:.3f})", color=colors_thresh[idx], alpha=0.85, edgecolor="black")

    ax.set_ylabel("Detection Rate / Recall (%)", fontsize=11)
    ax.set_xlabel("Attack Category", fontsize=11)
    ax.set_title(
        "Figure 19: Per-Attack-Family Anomaly Detection Rates Across Calibrated Thresholds\n"
        "(Evaluated on 399,906 Validation Flows)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(attack_cats, fontsize=10, rotation=15)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(True, axis="y", ls=":", alpha=0.6)
    fig.tight_layout()

    fig19_path = out_path / "19_anomaly_per_attack_recall.png"
    fig.savefig(fig19_path, dpi=300)
    plt.close(fig)
    generated_figures.append(str(fig19_path))

    # -------------------------------------------------------------
    # Figure 20: Training Size (50k/100k/200k) & Port-Agnostic Comparison
    # -------------------------------------------------------------
    fig, (ax_size, ax_port) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Size comparison (50k, 100k, 200k)
    size_models = ["iforest_full_50k", "iforest_full_100k", "iforest_full_200k"]
    sizes_present = [m for m in size_models if m in experiments_data]

    if sizes_present:
        sizes_labels = [m.replace("iforest_full_", "").upper() for m in sizes_present]
        roc_scores = [experiments_data[m]["roc_auc"] for m in sizes_present]
        pr_scores = [experiments_data[m]["pr_auc"] for m in sizes_present]
        f1_1pct = [experiments_data[m]["threshold_evaluations"].get("fpr_1.0pct", {}).get("f1_score", 0.0) for m in sizes_present]

        x_s = np.arange(len(sizes_present))
        w = 0.25
        ax_size.bar(x_s - w, roc_scores, w, label="ROC-AUC", color="#2b5c8f", edgecolor="black")
        ax_size.bar(x_s, pr_scores, w, label="PR-AUC", color="#d95f02", edgecolor="black")
        ax_size.bar(x_s + w, f1_1pct, w, label="F1 (at 1% FPR)", color="#4daf4a", edgecolor="black")

        ax_size.set_xticks(x_s)
        ax_size.set_xticklabels(sizes_labels, fontsize=10)
        ax_size.set_ylabel("Metric Score", fontsize=11)
        ax_size.set_title("Training Subset Size Comparison", fontsize=12, fontweight="bold")
        ax_size.set_ylim(0, 1.0)
        ax_size.legend(loc="lower right", fontsize=9)
        ax_size.grid(True, axis="y", ls=":", alpha=0.6)

    # Feature & Scaling comparison (Full vs Port-Agnostic; Unscaled vs Scaled)
    comp_models = ["iforest_full_100k", "iforest_port_100k", "iforest_full_scaled_100k"]
    comp_present = [m for m in comp_models if m in experiments_data]

    if comp_present:
        comp_labels = ["Full (63)\nUnscaled", "Port-Agnostic (62)\nUnscaled", "Full (63)\nRobustScaler"]
        comp_labels = comp_labels[:len(comp_present)]
        roc_comp = [experiments_data[m]["roc_auc"] for m in comp_present]
        pr_comp = [experiments_data[m]["pr_auc"] for m in comp_present]

        x_c = np.arange(len(comp_present))
        w = 0.3
        ax_port.bar(x_c - w/2, roc_comp, w, label="ROC-AUC", color="#2b5c8f", edgecolor="black")
        ax_port.bar(x_c + w/2, pr_comp, w, label="PR-AUC", color="#d95f02", edgecolor="black")

        ax_port.set_xticks(x_c)
        ax_port.set_xticklabels(comp_labels, fontsize=10)
        ax_port.set_ylabel("Metric Score", fontsize=11)
        ax_port.set_title("Feature & Scaling Invariance Comparison", fontsize=12, fontweight="bold")
        ax_port.set_ylim(0, 1.0)
        ax_port.legend(loc="lower right", fontsize=9)
        ax_port.grid(True, axis="y", ls=":", alpha=0.6)

    fig.suptitle("Figure 20: Computational Scaling & Feature Architecture Diagnostics", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    fig20_path = out_path / "20_anomaly_feature_and_size_comparison.png"
    fig.savefig(fig20_path, dpi=300)
    plt.close(fig)
    generated_figures.append(str(fig20_path))

    return generated_figures
