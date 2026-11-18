"""VYUHA Configuration Module.

Defines global path resolutions, random seeds, label mappings, and pipeline constants.
"""

import os
from pathlib import Path
from typing import Dict, List

# Repository and Workspace paths
REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent

RAW_DATA_DIR = Path(os.getenv("VYUHA_RAW_DATA_DIR", WORKSPACE_ROOT / "datasets" / "raw" / "CICIDS2017"))
PROCESSED_DATA_DIR = Path(os.getenv("VYUHA_PROCESSED_DATA_DIR", WORKSPACE_ROOT / "datasets" / "processed"))

# Reproducibility and Partitioning Constants
RANDOM_SEED = 42
TEST_SIZE = 0.2
CHUNK_SIZE = 100_000
DEV_SAMPLE_SIZE = 200_000

# Canonical 8 CSV Files in CICIDS2017 MachineLearningCVE
DATASET_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

# Canonical Mapping from Original Raw Attack Labels to Standard Categories
# Handles varying character encodings (em-dash, multiple spaces, hyphens)
LABEL_CATEGORY_MAPPING: Dict[str, str] = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS_DDoS",
    "DDoS": "DoS_DDoS",
    "DoS GoldenEye": "DoS_DDoS",
    "DoS slowloris": "DoS_DDoS",
    "DoS Slowhttptest": "DoS_DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Bot": "Botnet",
    "Web Attack  Brute Force": "WebAttack",
    "Web Attack – Brute Force": "WebAttack",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack  XSS": "WebAttack",
    "Web Attack – XSS": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack  Sql Injection": "WebAttack",
    "Web Attack – Sql Injection": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}

# Key TCP Window features that use -1 as a protocol sentinel
SENTINEL_FEATURES: List[str] = [
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]

# Canonical Rate Clipping Bounds (Derived from 99.9th percentile of valid flows)
# Ensures consistent rate thresholds regardless of chunk streaming order
DEFAULT_RATE_BOUNDS: Dict[str, float] = {
    "Flow Bytes/s_median": 1951.0166,
    "Flow Bytes/s_upper": 131695652.0,  # ~131.7 MB/s (~1 Gbps line rate)
    "Flow Packets/s_median": 39.1594,
    "Flow Packets/s_upper": 2000000.0,   # 2,000,000 pkts/s (standard high-speed interface max)
}

# Duplicate column to remove (present in raw CICIDS2017)
DUPLICATE_COLUMNS_TO_DROP: List[str] = [
    "Fwd Header Length.1",
]
