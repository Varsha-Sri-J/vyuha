"""VYUHA Incident Intelligence Package (Phase 4).

Provides alert normalization, entity extraction, deduplication, and multi-engine
evidence correlation for enterprise intrusion detection.
"""

from src.intelligence.adapters import (
    AnomalyAlertAdapter,
    BinaryAlertAdapter,
    MulticlassAlertAdapter,
)
from src.intelligence.extractor import (
    EntityExtractor,
    ExtractedEntities,
)
from src.intelligence.normalizer import MultiEngineAlertNormalizer
from src.intelligence.schema import (
    Alert,
    AlertSeverity,
    DetectionEngineType,
    SEVERITY_ORDER,
    ThreatStatus,
)

__all__ = [
    "Alert",
    "AlertSeverity",
    "DetectionEngineType",
    "SEVERITY_ORDER",
    "ThreatStatus",
    "EntityExtractor",
    "ExtractedEntities",
    "BinaryAlertAdapter",
    "MulticlassAlertAdapter",
    "AnomalyAlertAdapter",
    "MultiEngineAlertNormalizer",
]
