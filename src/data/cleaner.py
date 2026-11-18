"""VYUHA Data Cleaner Module.

Provides robust functions to standardize schema, clean corrupted characters,
handle missing and infinite values, filter invalid timestamps, and encode labels.
"""

import logging
import re
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_RATE_BOUNDS,
    DUPLICATE_COLUMNS_TO_DROP,
    LABEL_CATEGORY_MAPPING,
    SENTINEL_FEATURES,
)

logger = logging.getLogger(__name__)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names by stripping whitespace and dropping redundant duplicates.

    Args:
        df: Input DataFrame with raw column headers.

    Returns:
        DataFrame with stripped column names and duplicate columns removed.
    """
    df = df.copy()
    # Strip whitespace from column names
    df.columns = [col.strip() for col in df.columns]

    # Handle duplicate columns if any exist with identical names
    # Keep the first occurrence
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Drop explicit duplicate columns like 'Fwd Header Length.1'
    for col in DUPLICATE_COLUMNS_TO_DROP:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes attack labels, handling encoding artifacts, and creates binary & category targets.

    Creates:
      - `label_raw`: Sanitized original attack string.
      - `label_binary`: 0 for BENIGN, 1 for any attack.
      - `label_category`: Consolidated threat family.

    Args:
        df: Input DataFrame containing a 'Label' column.

    Returns:
        DataFrame with 'label_raw', 'label_binary', 'label_category' added, and 'Label' dropped.
    """
    df = df.copy()
    if "Label" not in df.columns:
        raise ValueError("DataFrame does not contain a 'Label' column.")

    # Convert to string and sanitize text
    raw_labels = df["Label"].astype(str).str.strip()

    # Clean encoding artifacts (e.g., em-dash \x96 or special chars in Web Attack)
    def clean_label_text(text: str) -> str:
        # Normalize double spaces or non-breaking spaces
        text = text.replace("\xa0", " ").replace("\x96", "–")
        text = re.sub(r"\s+", " ", text).strip()
        # Fallback to ascii replacement if needed
        clean = text.encode("ascii", "ignore").decode("ascii").strip()
        clean = re.sub(r"\s+", " ", clean)
        return clean if clean else text

    sanitized_labels = raw_labels.apply(clean_label_text)

    # Map to standardized category using robust semantic matching
    def map_category(label: str) -> str:
        # 1. Direct match in dictionary
        if label in LABEL_CATEGORY_MAPPING:
            return LABEL_CATEGORY_MAPPING[label]
        
        # 2. Match after removing punctuation and lowercasing
        norm = re.sub(r"[^a-zA-Z0-9]", "", label).lower()
        if norm == "benign":
            return "BENIGN"
        if "dos" in norm or "ddos" in norm:
            return "DoS_DDoS"
        if "portscan" in norm:
            return "PortScan"
        if "patator" in norm:
            return "BruteForce"
        if "webattack" in norm or ("web" in norm and "attack" in norm):
            return "WebAttack"
        if "bot" in norm:
            return "Botnet"
        if "infiltration" in norm:
            return "Infiltration"
        if "heartbleed" in norm:
            return "Heartbleed"

        logger.warning("Encountered unknown label '%s'; categorizing as 'Other_Attack'", label)
        return "Other_Attack"

    category_labels = sanitized_labels.apply(map_category)

    # Binary label: 0 for BENIGN, 1 for ATTACK
    binary_labels = (sanitized_labels.str.upper() != "BENIGN").astype(np.int8)

    df["label_raw"] = sanitized_labels
    df["label_category"] = category_labels
    df["label_binary"] = binary_labels

    df = df.drop(columns=["Label"])
    return df


def clean_numerical_anomalies(
    df: pd.DataFrame,
    rate_bounds: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Handles missing and infinite values, negative flow durations, and preserves sentinels.

    1. Filters rows with Flow Duration < 0 (timer clock drift capture artifacts).
    2. Recalculates rate features when Flow Duration == 0 using 1 µs epsilon to avoid +inf.
    3. Imputes missing Flow Bytes/s with the median.
    4. Clips rate outliers to the 99.9th percentile.
    5. Preserves -1 sentinels in Init_Win_bytes_forward / backward.

    Args:
        df: Input DataFrame with numeric columns.
        rate_bounds: Optional pre-computed clipping thresholds (used during test evaluation).

    Returns:
        Tuple of (cleaned_df, computed_rate_bounds)
    """
    df = df.copy()
    initial_count = len(df)

    # 1. Handle negative Flow Duration
    if "Flow Duration" in df.columns:
        df["Flow Duration"] = pd.to_numeric(df["Flow Duration"], errors="coerce")
        negative_duration_mask = df["Flow Duration"] < 0
        dropped_negatives = int(negative_duration_mask.sum())
        if dropped_negatives > 0:
            logger.info("Dropping %d rows with negative Flow Duration.", dropped_negatives)
            df = df.loc[~negative_duration_mask].copy()

    # 2. Convert numeric columns and handle Flow Bytes/s and Flow Packets/s
    for col in ["Flow Bytes/s", "Flow Packets/s"]:
        if col in df.columns:
            # Coerce object strings like 'Infinity', 'inf', 'NaN'
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # If Flow Duration == 0 and rate is NaN or Inf, recalculate with 1 microsecond epsilon
    if "Flow Duration" in df.columns and "Flow Bytes/s" in df.columns and "Total Length of Fwd Packets" in df.columns:
        zero_duration_mask = (df["Flow Duration"] == 0) & (df["Flow Bytes/s"].isna() | np.isinf(df["Flow Bytes/s"]))
        if zero_duration_mask.any():
            tot_bytes = df.loc[zero_duration_mask, "Total Length of Fwd Packets"]
            if "Total Length of Bwd Packets" in df.columns:
                tot_bytes = tot_bytes + df.loc[zero_duration_mask, "Total Length of Bwd Packets"]
            # 1 microsecond = 1e-6 seconds -> 1 microsecond duration yields bytes * 1e6 bytes/s
            df.loc[zero_duration_mask, "Flow Bytes/s"] = tot_bytes * 1e6

    if "Flow Duration" in df.columns and "Flow Packets/s" in df.columns and "Total Fwd Packets" in df.columns:
        zero_duration_pkt_mask = (df["Flow Duration"] == 0) & (df["Flow Packets/s"].isna() | np.isinf(df["Flow Packets/s"]))
        if zero_duration_pkt_mask.any():
            tot_pkts = df.loc[zero_duration_pkt_mask, "Total Fwd Packets"]
            if "Total Backward Packets" in df.columns:
                tot_pkts = tot_pkts + df.loc[zero_duration_pkt_mask, "Total Backward Packets"]
            df.loc[zero_duration_pkt_mask, "Flow Packets/s"] = tot_pkts * 1e6

    # 3. Calculate or apply bounds for Flow Bytes/s and Flow Packets/s
    computed_bounds = dict(rate_bounds) if rate_bounds else dict(DEFAULT_RATE_BOUNDS)

    for col in ["Flow Bytes/s", "Flow Packets/s"]:
        if col in df.columns:
            valid_finite = df[col].replace([np.inf, -np.inf], np.nan).dropna()

            if f"{col}_upper" not in computed_bounds:
                median_val = float(valid_finite.median()) if len(valid_finite) > 0 else 0.0
                upper_bound = float(valid_finite.quantile(0.999)) if len(valid_finite) > 0 else 1e9
                computed_bounds[f"{col}_median"] = median_val
                computed_bounds[f"{col}_upper"] = upper_bound

            # Impute NaN with median
            median_val = computed_bounds.get(f"{col}_median", 0.0)
            upper_bound = computed_bounds.get(f"{col}_upper", 1e9)

            df[col] = df[col].fillna(median_val)
            # Replace +inf with upper bound, -inf with 0
            df[col] = df[col].replace(np.inf, upper_bound)
            df[col] = df[col].replace(-np.inf, 0.0)
            # Clip negative rates (if any remaining capture errors)
            df[col] = df[col].clip(lower=0.0, upper=upper_bound)

    # 4. Convert all other features to numeric, preserving sentinel -1 values
    label_cols = {"label_raw", "label_binary", "label_category"}
    for col in df.columns:
        if col in label_cols:
            continue
        if col in SENTINEL_FEATURES:
            # Ensure sentinel columns remain integer, retaining -1
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(np.int32)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)

    return df, computed_bounds
