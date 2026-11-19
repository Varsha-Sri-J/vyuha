"""VYUHA Flow Inference Preprocessor Module.

Provides strict, leakage-free flow vector normalization and validation
prior to model inference.

Enforces:
1. Exact 63-feature schema requirement for full models.
2. Helpful rejection of incomplete flow vectors with specific missing features.
3. Feature name standardization and alias mapping (e.g. snake_case to canonical).
4. Numeric validation and type coercion.
5. Training-time numerical anomaly handling (rate clipping, 1 µs zero-duration epsilon,
   sentinel -1 preservation in TCP window features).
6. Preserves exact feature ordering matching fitted Scikit-Learn transformers.
7. Zero fabrication of missing endpoint metadata (IPs, timestamps, protocols).
"""

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.config import DEFAULT_RATE_BOUNDS, SENTINEL_FEATURES

logger = logging.getLogger(__name__)

# Canonical 63 features required by full models in strict fitted order
CANONICAL_63_FEATURES: List[str] = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Fwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

# Set of canonical feature names for fast membership lookup
CANONICAL_FEATURE_SET: Set[str] = set(CANONICAL_63_FEATURES)

# Case-insensitive and alias mappings to canonical feature names
FEATURE_ALIAS_MAP: Dict[str, str] = {
    # Destination port aliases
    "destination port": "Destination Port",
    "destination_port": "Destination Port",
    "dst_port": "Destination Port",
    "dport": "Destination Port",
    # Duration aliases
    "flow duration": "Flow Duration",
    "flow_duration": "Flow Duration",
    "duration": "Flow Duration",
    # Packet count aliases
    "total fwd packets": "Total Fwd Packets",
    "total_fwd_packets": "Total Fwd Packets",
    "total_fwd_pkts": "Total Fwd Packets",
    "total backward packets": "Total Backward Packets",
    "total_backward_packets": "Total Backward Packets",
    "total_bwd_packets": "Total Backward Packets",
    "total_bwd_pkts": "Total Backward Packets",
    # Length aliases
    "total length of fwd packets": "Total Length of Fwd Packets",
    "total_length_of_fwd_packets": "Total Length of Fwd Packets",
    "total length of bwd packets": "Total Length of Bwd Packets",
    "total_length_of_bwd_packets": "Total Length of Bwd Packets",
    # Rate aliases
    "flow bytes/s": "Flow Bytes/s",
    "flow_bytes_s": "Flow Bytes/s",
    "flow_byts_s": "Flow Bytes/s",
    "flow packets/s": "Flow Packets/s",
    "flow_packets_s": "Flow Packets/s",
    "flow_pkts_s": "Flow Packets/s",
    # Flags aliases
    "fin flag count": "FIN Flag Count",
    "fin_flag_count": "FIN Flag Count",
    "syn flag count": "SYN Flag Count",
    "syn_flag_count": "SYN Flag Count",
    "rst flag count": "RST Flag Count",
    "rst_flag_count": "RST Flag Count",
    "psh flag count": "PSH Flag Count",
    "psh_flag_count": "PSH Flag Count",
    "ack flag count": "ACK Flag Count",
    "ack_flag_count": "ACK Flag Count",
    "urg flag count": "URG Flag Count",
    "urg_flag_count": "URG Flag Count",
    "cwe flag count": "CWE Flag Count",
    "cwe_flag_count": "CWE Flag Count",
    "ece flag count": "ECE Flag Count",
    "ece_flag_count": "ECE Flag Count",
    # TCP Window sentinel aliases
    "init_win_bytes_forward": "Init_Win_bytes_forward",
    "init_win_bytes_fwd": "Init_Win_bytes_forward",
    "init win bytes forward": "Init_Win_bytes_forward",
    "init_win_bytes_backward": "Init_Win_bytes_backward",
    "init_win_bytes_bwd": "Init_Win_bytes_backward",
    "init win bytes backward": "Init_Win_bytes_backward",
    # Header & Segment aliases
    "fwd header length": "Fwd Header Length",
    "fwd_header_length": "Fwd Header Length",
    "bwd header length": "Bwd Header Length",
    "bwd_header_length": "Bwd Header Length",
    "min_seg_size_forward": "min_seg_size_forward",
    "min seg size forward": "min_seg_size_forward",
    "act_data_pkt_fwd": "act_data_pkt_fwd",
    "down/up ratio": "Down/Up Ratio",
    "down_up_ratio": "Down/Up Ratio",
    "average packet size": "Average Packet Size",
    "avg_packet_size": "Average Packet Size",
}

# Add lowercase versions of all canonical features into alias map
for feat in CANONICAL_63_FEATURES:
    FEATURE_ALIAS_MAP[feat.lower()] = feat
    FEATURE_ALIAS_MAP[feat.lower().replace(" ", "_")] = feat


class PreprocessingError(Exception):
    """Base exception for inference preprocessing failures."""
    pass


class IncompleteFlowError(PreprocessingError):
    """Raised when an incoming flow record is missing required model features."""

    def __init__(self, record_index: int, missing_features: List[str]):
        self.record_index = record_index
        self.missing_features = missing_features
        msg = (
            f"Flow record at index {record_index} is missing {len(missing_features)} required features: "
            f"{missing_features[:10]}{'...' if len(missing_features) > 10 else ''}. "
            f"All {len(CANONICAL_63_FEATURES)} features are required for genuine inference."
        )
        super().__init__(msg)


class InvalidFeatureValueError(PreprocessingError):
    """Raised when a feature value cannot be coerced into valid numeric format."""

    def __init__(self, record_index: int, feature_name: str, value: Any, detail: str):
        self.record_index = record_index
        self.feature_name = feature_name
        self.value = value
        self.detail = detail
        msg = f"Invalid value for '{feature_name}' at flow index {record_index}: {value!r} ({detail})"
        super().__init__(msg)


class FlowInferencePreprocessor:
    """Prepares and validates incoming flow records for machine learning inference."""

    def __init__(
        self,
        rate_bounds: Optional[Dict[str, float]] = None,
        allow_metadata_columns: bool = True,
    ) -> None:
        """Initializes the preprocessor.

        Args:
            rate_bounds: Optional clipping thresholds for rate features (defaults to DEFAULT_RATE_BOUNDS).
            allow_metadata_columns: If True, ignores non-model metadata fields like source_ip, timestamp.
        """
        self.rate_bounds = rate_bounds or DEFAULT_RATE_BOUNDS
        self.allow_metadata = allow_metadata_columns

    def normalize_feature_name(self, raw_name: str) -> str:
        """Resolves feature aliases and formatting to canonical name."""
        clean = raw_name.strip()
        if clean in CANONICAL_FEATURE_SET:
            return clean

        lower_clean = clean.lower()
        if lower_clean in FEATURE_ALIAS_MAP:
            return FEATURE_ALIAS_MAP[lower_clean]

        lower_snake = lower_clean.replace(" ", "_").replace("-", "_")
        if lower_snake in FEATURE_ALIAS_MAP:
            return FEATURE_ALIAS_MAP[lower_snake]

        return clean

    def prepare_flow_dataframe(
        self,
        flow_records: Sequence[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Validates, sanitizes, and converts flow records into a canonical 63-feature DataFrame.

        Args:
            flow_records: Sequence of dictionaries representing network flows.

        Returns:
            pd.DataFrame containing exactly the 63 required features in strict canonical order.

        Raises:
            IncompleteFlowError: If any flow record is missing required features.
            InvalidFeatureValueError: If a value cannot be coerced to numeric.
        """
        if not flow_records:
            return pd.DataFrame(columns=CANONICAL_63_FEATURES)

        standardized_records: List[Dict[str, float]] = []

        for idx, raw_record in enumerate(flow_records):
            if not isinstance(raw_record, dict):
                raise PreprocessingError(
                    f"Flow record at index {idx} must be a dictionary, got {type(raw_record).__name__}"
                )

            # Map raw keys to canonical feature names
            mapped_record: Dict[str, Any] = {}
            for k, v in raw_record.items():
                canon_k = self.normalize_feature_name(str(k))
                mapped_record[canon_k] = v

            # Validate that all 63 required features are present
            present_features = set(mapped_record.keys())
            missing = [f for f in CANONICAL_63_FEATURES if f not in present_features]
            if missing:
                raise IncompleteFlowError(record_index=idx, missing_features=missing)

            # Sanitize and convert numeric values
            sanitized: Dict[str, float] = {}

            # 1. Flow Duration sanitization
            raw_dur = mapped_record.get("Flow Duration")
            try:
                dur = float(raw_dur) if raw_dur is not None else 0.0
            except (ValueError, TypeError) as exc:
                raise InvalidFeatureValueError(idx, "Flow Duration", raw_dur, str(exc)) from exc
            if dur < 0.0 or math.isnan(dur):
                dur = 0.0
            sanitized["Flow Duration"] = dur

            # 2. Extract packet/length sums for zero-duration rate calculations
            fwd_pkts = self._coerce_numeric(idx, "Total Fwd Packets", mapped_record.get("Total Fwd Packets", 0))
            bwd_pkts = self._coerce_numeric(idx, "Total Backward Packets", mapped_record.get("Total Backward Packets", 0))
            fwd_bytes = self._coerce_numeric(idx, "Total Length of Fwd Packets", mapped_record.get("Total Length of Fwd Packets", 0))
            bwd_bytes = self._coerce_numeric(idx, "Total Length of Bwd Packets", mapped_record.get("Total Length of Bwd Packets", 0))

            # 3. Flow Bytes/s and Flow Packets/s rate handling
            sanitized["Flow Bytes/s"] = self._sanitize_rate(
                idx, "Flow Bytes/s", mapped_record.get("Flow Bytes/s"), dur, fwd_bytes + bwd_bytes
            )
            sanitized["Flow Packets/s"] = self._sanitize_rate(
                idx, "Flow Packets/s", mapped_record.get("Flow Packets/s"), dur, fwd_pkts + bwd_pkts
            )

            # 4. Convert all other features
            for feat in CANONICAL_63_FEATURES:
                if feat in sanitized:
                    continue

                raw_val = mapped_record[feat]
                if feat in SENTINEL_FEATURES:
                    # TCP Window sentinels: preserve -1
                    sanitized[feat] = self._coerce_sentinel(idx, feat, raw_val)
                else:
                    sanitized[feat] = self._coerce_numeric(idx, feat, raw_val)

            standardized_records.append(sanitized)

        # Build DataFrame with strict canonical column ordering
        df = pd.DataFrame(standardized_records)
        df = df[CANONICAL_63_FEATURES].copy()

        # Enforce exact dtypes matching training pipeline
        for col in CANONICAL_63_FEATURES:
            if col in SENTINEL_FEATURES or col == "Destination Port":
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.float32)

        return df

    # -------------------------------------------------------------------------
    # Helper Coercion Methods
    # -------------------------------------------------------------------------

    def _coerce_numeric(self, idx: int, feat: str, val: Any) -> float:
        """Safely coerces value to finite float."""
        if val is None:
            raise InvalidFeatureValueError(idx, feat, val, "Value cannot be null/None")
        try:
            f = float(val)
        except (ValueError, TypeError) as exc:
            raise InvalidFeatureValueError(idx, feat, val, str(exc)) from exc

        if math.isnan(f):
            raise InvalidFeatureValueError(idx, feat, val, "Value is NaN")
        if math.isinf(f):
            raise InvalidFeatureValueError(idx, feat, val, "Value is infinite")
        return f

    def _coerce_sentinel(self, idx: int, feat: str, val: Any) -> int:
        """Coerces TCP window sentinel feature, strictly preserving -1."""
        if val is None:
            return -1
        try:
            i = int(float(val))
        except (ValueError, TypeError) as exc:
            raise InvalidFeatureValueError(idx, feat, val, str(exc)) from exc
        return i

    def _sanitize_rate(
        self,
        idx: int,
        feat: str,
        val: Any,
        duration: float,
        total_units: float,
    ) -> float:
        """Sanitizes rate features handling +inf, zero-duration recalculation, and upper bounds."""
        median_bound = self.rate_bounds.get(f"{feat}_median", 1000.0)
        upper_bound = self.rate_bounds.get(f"{feat}_upper", 1e8)

        # Handle zero-duration recalculation with 1 microsecond (1e-6 s) epsilon
        if duration == 0.0:
            if val is None or str(val).lower() in ("inf", "+inf", "infinity", "nan"):
                recalculated = total_units * 1e6
                return min(float(recalculated), upper_bound)

        # Coerce raw rate value
        if val is None or str(val).lower() == "nan":
            return median_bound

        if str(val).lower() in ("inf", "+inf", "infinity"):
            return upper_bound

        if str(val).lower() in ("-inf", "-infinity"):
            return 0.0

        try:
            rate_val = float(val)
        except (ValueError, TypeError) as exc:
            raise InvalidFeatureValueError(idx, feat, val, str(exc)) from exc

        if math.isnan(rate_val):
            return median_bound
        if math.isinf(rate_val):
            return upper_bound if rate_val > 0 else 0.0

        # Clip bounds
        return max(0.0, min(rate_val, upper_bound))


_CALIBRATED_PRESETS: Dict[str, Dict[str, Any]] = {
    "benign": {
        "Destination Port": 443.0,
        "Flow Duration": 163665.0,
        "Total Fwd Packets": 2.0,
        "Total Backward Packets": 2.0,
        "Total Length of Fwd Packets": 102.0,
        "Total Length of Bwd Packets": 158.0,
        "Fwd Packet Length Max": 51.0,
        "Fwd Packet Length Min": 51.0,
        "Fwd Packet Length Mean": 51.0,
        "Fwd Packet Length Std": 0.0,
        "Bwd Packet Length Max": 79.0,
        "Bwd Packet Length Min": 79.0,
        "Bwd Packet Length Mean": 79.0,
        "Bwd Packet Length Std": 0.0,
        "Flow Bytes/s": 1588.61083984375,
        "Flow Packets/s": 24.440166473388672,
        "Flow IAT Mean": 54555.0,
        "Flow IAT Std": 94446.1328125,
        "Flow IAT Max": 163612.0,
        "Flow IAT Min": 4.0,
        "Fwd IAT Total": 4.0,
        "Fwd IAT Mean": 4.0,
        "Fwd IAT Std": 0.0,
        "Fwd IAT Max": 4.0,
        "Fwd IAT Min": 4.0,
        "Bwd IAT Total": 49.0,
        "Bwd IAT Mean": 49.0,
        "Bwd IAT Std": 0.0,
        "Bwd IAT Max": 49.0,
        "Bwd IAT Min": 49.0,
        "Fwd PSH Flags": 0.0,
        "Fwd URG Flags": 0.0,
        "Fwd Header Length": 40.0,
        "Bwd Header Length": 40.0,
        "Fwd Packets/s": 12.220083236694336,
        "Bwd Packets/s": 12.220083236694336,
        "Min Packet Length": 51.0,
        "Max Packet Length": 79.0,
        "Packet Length Mean": 62.20000076293945,
        "Packet Length Std": 15.336231231689453,
        "Packet Length Variance": 235.1999969482422,
        "FIN Flag Count": 0.0,
        "SYN Flag Count": 0.0,
        "RST Flag Count": 0.0,
        "PSH Flag Count": 0.0,
        "ACK Flag Count": 0.0,
        "URG Flag Count": 0.0,
        "CWE Flag Count": 0.0,
        "ECE Flag Count": 0.0,
        "Down/Up Ratio": 1.0,
        "Average Packet Size": 77.75,
        "Init_Win_bytes_forward": -1,
        "Init_Win_bytes_backward": -1,
        "act_data_pkt_fwd": 1.0,
        "min_seg_size_forward": 20.0,
        "Active Mean": 0.0,
        "Active Std": 0.0,
        "Active Max": 0.0,
        "Active Min": 0.0,
        "Idle Mean": 0.0,
        "Idle Std": 0.0,
        "Idle Max": 0.0,
        "Idle Min": 0.0
},
    "portscan": {
        "Destination Port": 7070.0,
        "Flow Duration": 116562832.0,
        "Total Fwd Packets": 2.0,
        "Total Backward Packets": 2.0,
        "Total Length of Fwd Packets": 2.0,
        "Total Length of Bwd Packets": 12.0,
        "Fwd Packet Length Max": 2.0,
        "Fwd Packet Length Min": 0.0,
        "Fwd Packet Length Mean": 1.0,
        "Fwd Packet Length Std": 1.4142135381698608,
        "Bwd Packet Length Max": 6.0,
        "Bwd Packet Length Min": 6.0,
        "Bwd Packet Length Mean": 6.0,
        "Bwd Packet Length Std": 0.0,
        "Flow Bytes/s": 0.12010689079761505,
        "Flow Packets/s": 0.034316256642341614,
        "Flow IAT Mean": 38900000.0,
        "Flow IAT Std": 67300000.0,
        "Flow IAT Max": 117000000.0,
        "Flow IAT Min": 44.0,
        "Fwd IAT Total": 117000000.0,
        "Fwd IAT Mean": 117000000.0,
        "Fwd IAT Std": 0.0,
        "Fwd IAT Max": 117000000.0,
        "Fwd IAT Min": 117000000.0,
        "Bwd IAT Total": 117000000.0,
        "Bwd IAT Mean": 117000000.0,
        "Bwd IAT Std": 0.0,
        "Bwd IAT Max": 117000000.0,
        "Bwd IAT Min": 117000000.0,
        "Fwd PSH Flags": 0.0,
        "Fwd URG Flags": 0.0,
        "Fwd Header Length": 64.0,
        "Bwd Header Length": 40.0,
        "Fwd Packets/s": 0.017158126458525658,
        "Bwd Packets/s": 0.017158126458525658,
        "Min Packet Length": 0.0,
        "Max Packet Length": 6.0,
        "Packet Length Mean": 3.200000047683716,
        "Packet Length Std": 2.683281660079956,
        "Packet Length Variance": 7.199999809265137,
        "FIN Flag Count": 0.0,
        "SYN Flag Count": 0.0,
        "RST Flag Count": 0.0,
        "PSH Flag Count": 1.0,
        "ACK Flag Count": 0.0,
        "URG Flag Count": 0.0,
        "CWE Flag Count": 0.0,
        "ECE Flag Count": 0.0,
        "Down/Up Ratio": 1.0,
        "Average Packet Size": 4.0,
        "Init_Win_bytes_forward": 1024,
        "Init_Win_bytes_backward": -1,
        "act_data_pkt_fwd": 0.0,
        "min_seg_size_forward": 24.0,
        "Active Mean": 88.0,
        "Active Std": 0.0,
        "Active Max": 88.0,
        "Active Min": 88.0,
        "Idle Mean": 117000000.0,
        "Idle Std": 0.0,
        "Idle Max": 117000000.0,
        "Idle Min": 117000000.0
},
    "dos": {
        "Destination Port": 80.0,
        "Flow Duration": 85127248.0,
        "Total Fwd Packets": 6.0,
        "Total Backward Packets": 6.0,
        "Total Length of Fwd Packets": 362.0,
        "Total Length of Bwd Packets": 11595.0,
        "Fwd Packet Length Max": 344.0,
        "Fwd Packet Length Min": 0.0,
        "Fwd Packet Length Mean": 60.33333206176758,
        "Fwd Packet Length Std": 138.99879455566406,
        "Bwd Packet Length Max": 7240.0,
        "Bwd Packet Length Min": 0.0,
        "Bwd Packet Length Mean": 1932.5,
        "Bwd Packet Length Std": 2848.073486328125,
        "Flow Bytes/s": 140.46031188964844,
        "Flow Packets/s": 0.14096544682979584,
        "Flow IAT Mean": 7738840.5,
        "Flow IAT Std": 25600000.0,
        "Flow IAT Max": 85000000.0,
        "Flow IAT Min": 1.0,
        "Fwd IAT Total": 85000000.0,
        "Fwd IAT Mean": 17000000.0,
        "Fwd IAT Std": 38000000.0,
        "Fwd IAT Max": 85000000.0,
        "Fwd IAT Min": 1.0,
        "Bwd IAT Total": 85100000.0,
        "Bwd IAT Mean": 17000000.0,
        "Bwd IAT Std": 38000000.0,
        "Bwd IAT Max": 85000000.0,
        "Bwd IAT Min": 195.0,
        "Fwd PSH Flags": 0.0,
        "Fwd URG Flags": 0.0,
        "Fwd Header Length": 164.0,
        "Bwd Header Length": 200.0,
        "Fwd Packets/s": 0.07048272341489792,
        "Bwd Packets/s": 0.07048272341489792,
        "Min Packet Length": 0.0,
        "Max Packet Length": 7240.0,
        "Packet Length Mean": 920.2307739257812,
        "Packet Length Std": 2083.160888671875,
        "Packet Length Variance": 4339559.5,
        "FIN Flag Count": 1.0,
        "SYN Flag Count": 0.0,
        "RST Flag Count": 0.0,
        "PSH Flag Count": 0.0,
        "ACK Flag Count": 0.0,
        "URG Flag Count": 0.0,
        "CWE Flag Count": 0.0,
        "ECE Flag Count": 0.0,
        "Down/Up Ratio": 1.0,
        "Average Packet Size": 996.9166870117188,
        "Init_Win_bytes_forward": 0,
        "Init_Win_bytes_backward": 235,
        "act_data_pkt_fwd": 3.0,
        "min_seg_size_forward": 20.0,
        "Active Mean": 13990.0,
        "Active Std": 0.0,
        "Active Max": 13990.0,
        "Active Min": 13990.0,
        "Idle Mean": 85000000.0,
        "Idle Std": 0.0,
        "Idle Max": 85000000.0,
        "Idle Min": 85000000.0
},
    "anomaly": {
        "Destination Port": 443.0,
        "Flow Duration": 119944800.0,
        "Total Fwd Packets": 317.0,
        "Total Backward Packets": 520.0,
        "Total Length of Fwd Packets": 7103.0,
        "Total Length of Bwd Packets": 829107.0,
        "Fwd Packet Length Max": 654.0,
        "Fwd Packet Length Min": 0.0,
        "Fwd Packet Length Mean": 22.406940460205078,
        "Fwd Packet Length Std": 101.98660278320312,
        "Bwd Packet Length Max": 4254.0,
        "Bwd Packet Length Min": 0.0,
        "Bwd Packet Length Mean": 1594.4365234375,
        "Bwd Packet Length Std": 812.1492309570312,
        "Flow Bytes/s": 6971.6240234375,
        "Flow Packets/s": 6.978209972381592,
        "Flow IAT Mean": 143474.640625,
        "Flow IAT Std": 2370084.5,
        "Flow IAT Max": 39700000.0,
        "Flow IAT Min": -1.0,
        "Fwd IAT Total": 120000000.0,
        "Fwd IAT Mean": 379572.125,
        "Fwd IAT Std": 3854609.0,
        "Fwd IAT Max": 39700000.0,
        "Fwd IAT Min": 3.0,
        "Bwd IAT Total": 120000000.0,
        "Bwd IAT Mean": 231106.265625,
        "Bwd IAT Std": 3008581.75,
        "Bwd IAT Max": 39700000.0,
        "Bwd IAT Min": 1.0,
        "Fwd PSH Flags": 1.0,
        "Fwd URG Flags": 0.0,
        "Fwd Header Length": 10144.0,
        "Bwd Header Length": 16640.0,
        "Fwd Packets/s": 2.6428825855255127,
        "Bwd Packets/s": 4.335327625274658,
        "Min Packet Length": 0.0,
        "Max Packet Length": 4254.0,
        "Packet Length Mean": 997.9916381835938,
        "Packet Length Std": 997.6698608398438,
        "Packet Length Variance": 995345.125,
        "FIN Flag Count": 0.0,
        "SYN Flag Count": 1.0,
        "RST Flag Count": 0.0,
        "PSH Flag Count": 0.0,
        "ACK Flag Count": 1.0,
        "URG Flag Count": 0.0,
        "CWE Flag Count": 0.0,
        "ECE Flag Count": 0.0,
        "Down/Up Ratio": 1.0,
        "Average Packet Size": 999.1839599609375,
        "Init_Win_bytes_forward": 507,
        "Init_Win_bytes_backward": 665,
        "act_data_pkt_fwd": 25.0,
        "min_seg_size_forward": 32.0,
        "Active Mean": 319617.34375,
        "Active Std": 51019.75390625,
        "Active Max": 365147.0,
        "Active Min": 264475.0,
        "Idle Mean": 39600000.0,
        "Idle Std": 68041.265625,
        "Idle Max": 39700000.0,
        "Idle Min": 39500000.0
},
}

def get_sample_flow_preset(preset_type: str = "benign") -> Dict[str, Any]:
    """Returns a synthetic, validated 63-feature flow record for testing and UI demonstrations.

    Args:
        preset_type: 'benign' (normal baseline flow),
                     'portscan' (stealth SYN probe),
                     'dos' (high-volume flood),
                     'anomaly' (novel out-of-distribution flow).

    Returns:
        Complete 63-feature dictionary matching CANONICAL_63_FEATURES.
    """
    if preset_type not in _CALIBRATED_PRESETS:
        preset_type = "benign"
    return dict(_CALIBRATED_PRESETS[preset_type])
