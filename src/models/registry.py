"""VYUHA Model Registry Module.

Provides thread-safe, project-relative loading, caching, and validation
for the primary trained machine learning model artifacts:
1. Binary Intrusion Detection (Random Forest, Full 63 features)
2. Multiclass Attack Family Classification (Random Forest, Full 63 features, 8 classes)
3. Unsupervised Anomaly Detection (Isolation Forest, 100k Benign, Full 63 features, calibrated thresholds)

Guarantees:
- Models are loaded once and cached in memory (no reloading on each API request).
- No models are ever retrained at runtime.
- Model artifacts are strictly validated for expected Pipeline steps and capabilities.
- Clear exceptions raised when artifacts are missing or corrupt (no silent mock fallbacks).
"""

import logging
from pathlib import Path
import threading
from typing import Any, Dict, Optional, Tuple

import joblib
from sklearn.pipeline import Pipeline

from src.config import REPO_ROOT

logger = logging.getLogger(__name__)

# Default artifact filenames in models/ directory
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
BINARY_MODEL_FILENAME = "rf_full.joblib"
MULTICLASS_MODEL_FILENAME = "multiclass_rf_full.joblib"
ANOMALY_MODEL_FILENAME = "iforest_full_100k.joblib"

# Canonical expected class list for multiclass classification
CANONICAL_MULTICLASS_CATEGORIES = [
    "BENIGN",
    "Botnet",
    "BruteForce",
    "DoS_DDoS",
    "Heartbleed",
    "Infiltration",
    "PortScan",
    "WebAttack",
]


class ModelRegistryError(Exception):
    """Base exception for model registry failures."""
    pass


class ModelNotFoundError(ModelRegistryError):
    """Raised when a required model artifact cannot be located on disk."""
    pass


class ModelCorruptError(ModelRegistryError):
    """Raised when a model artifact fails deserialization or is malformed."""
    pass


class ModelIncompatibleError(ModelRegistryError):
    """Raised when a loaded model lacks expected steps, estimators, or capabilities."""
    pass


class ModelRegistry:
    """Thread-safe singleton registry for trained VYUHA model artifacts."""

    _instance: Optional["ModelRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls, models_dir: Optional[Path] = None) -> "ModelRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        if getattr(self, "_initialized", False):
            return

        self.models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self._binary_pipeline: Optional[Pipeline] = None
        self._multiclass_pipeline: Optional[Pipeline] = None
        self._anomaly_pipeline: Optional[Pipeline] = None
        self._anomaly_metadata: Optional[Dict[str, Any]] = None

        self._load_lock = threading.Lock()
        self._initialized = True

    @property
    def is_loaded(self) -> bool:
        """Returns True if all three primary production models are loaded and validated."""
        return (
            self._binary_pipeline is not None
            and self._multiclass_pipeline is not None
            and self._anomaly_pipeline is not None
        )

    def load_all(self) -> None:
        """Preloads all three primary models into memory and validates their capabilities."""
        with self._load_lock:
            self._load_binary_model()
            self._load_multiclass_model()
            self._load_anomaly_model()
            logger.info("All primary VYUHA model artifacts successfully loaded and validated.")

    def get_binary_model(self) -> Pipeline:
        """Returns the loaded Binary Random Forest classification pipeline."""
        if self._binary_pipeline is None:
            with self._load_lock:
                if self._binary_pipeline is None:
                    self._load_binary_model()
        return self._binary_pipeline

    def get_multiclass_model(self) -> Pipeline:
        """Returns the loaded Multiclass Random Forest classification pipeline."""
        if self._multiclass_pipeline is None:
            with self._load_lock:
                if self._multiclass_pipeline is None:
                    self._load_multiclass_model()
        return self._multiclass_pipeline

    def get_anomaly_model(self) -> Tuple[Pipeline, Dict[str, Any]]:
        """Returns a tuple of (anomaly_pipeline, metadata_dict) for Isolation Forest."""
        if self._anomaly_pipeline is None or self._anomaly_metadata is None:
            with self._load_lock:
                if self._anomaly_pipeline is None or self._anomaly_metadata is None:
                    self._load_anomaly_model()
        return self._anomaly_pipeline, self._anomaly_metadata

    def get_status(self) -> Dict[str, Any]:
        """Returns operational health and capability status of the model registry."""
        status = {
            "models_dir": str(self.models_dir),
            "is_loaded": self.is_loaded,
            "models": {},
        }
        if self._binary_pipeline is not None:
            clf = self._binary_pipeline.named_steps.get("classifier")
            selector = self._binary_pipeline.named_steps.get("selector")
            status["models"]["binary"] = {
                "name": BINARY_MODEL_FILENAME,
                "estimator": clf.__class__.__name__ if clf else None,
                "features_in": clf.n_features_in_ if hasattr(clf, "n_features_in_") else None,
                "selected_features_count": len(selector.selected_features_) if selector else None,
                "classes": [int(c) for c in clf.classes_] if hasattr(clf, "classes_") else None,
            }
        else:
            status["models"]["binary"] = {"name": BINARY_MODEL_FILENAME, "loaded": False}

        if self._multiclass_pipeline is not None:
            clf = self._multiclass_pipeline.named_steps.get("classifier")
            selector = self._multiclass_pipeline.named_steps.get("selector")
            status["models"]["multiclass"] = {
                "name": MULTICLASS_MODEL_FILENAME,
                "estimator": clf.__class__.__name__ if clf else None,
                "features_in": clf.n_features_in_ if hasattr(clf, "n_features_in_") else None,
                "selected_features_count": len(selector.selected_features_) if selector else None,
                "classes": list(clf.classes_) if hasattr(clf, "classes_") else None,
            }
        else:
            status["models"]["multiclass"] = {"name": MULTICLASS_MODEL_FILENAME, "loaded": False}

        if self._anomaly_pipeline is not None:
            detector = self._anomaly_pipeline.named_steps.get("anomaly_detector")
            selector = self._anomaly_pipeline.named_steps.get("selector")
            thresholds = self._anomaly_metadata.get("calibrated_thresholds", {}) if self._anomaly_metadata else {}
            status["models"]["anomaly"] = {
                "name": ANOMALY_MODEL_FILENAME,
                "estimator": detector.__class__.__name__ if detector else None,
                "features_in": detector.n_features_in_ if hasattr(detector, "n_features_in_") else None,
                "selected_features_count": len(selector.selected_features_) if selector else None,
                "calibrated_thresholds": list(thresholds.keys()),
            }
        else:
            status["models"]["anomaly"] = {"name": ANOMALY_MODEL_FILENAME, "loaded": False}

        return status

    # -------------------------------------------------------------------------
    # Internal Loaders & Validators
    # -------------------------------------------------------------------------

    def _resolve_artifact_path(self, filename: str) -> Path:
        """Resolves artifact path, checking models_dir and raising if missing."""
        p = self.models_dir / filename
        if p.exists():
            return p.resolve()

        # If custom directory was provided and file is missing, do not fall back
        if self.models_dir != DEFAULT_MODELS_DIR:
            raise ModelNotFoundError(
                f"Model artifact '{filename}' not found in custom directory {self.models_dir.resolve()}."
            )

        # Fallback search for default paths
        alt_p = Path("models") / filename
        if alt_p.exists():
            return alt_p.resolve()
        alt_parent = REPO_ROOT / "models" / filename
        if alt_parent.exists():
            return alt_parent.resolve()

        raise ModelNotFoundError(
            f"Model artifact '{filename}' not found at {p.resolve()}. "
            "Ensure trained models are present in the models/ directory."
        )

    def _load_binary_model(self) -> None:
        path = self._resolve_artifact_path(BINARY_MODEL_FILENAME)
        try:
            obj = joblib.load(path)
        except Exception as exc:
            raise ModelCorruptError(f"Failed to load binary model from {path}: {exc}") from exc

        if not isinstance(obj, Pipeline):
            raise ModelIncompatibleError(f"Binary model at {path} is not a Pipeline; got {type(obj)}")
        if "selector" not in obj.named_steps or "classifier" not in obj.named_steps:
            raise ModelIncompatibleError(
                f"Binary model at {path} missing required steps ('selector', 'classifier'): {list(obj.named_steps.keys())}"
            )
        clf = obj.named_steps["classifier"]
        if not hasattr(clf, "predict") or not hasattr(clf, "predict_proba"):
            raise ModelIncompatibleError(f"Binary classifier in {path} lacks predict or predict_proba")

        self._binary_pipeline = obj
        logger.info("Loaded binary intrusion detection model: %s", path.name)

    def _load_multiclass_model(self) -> None:
        path = self._resolve_artifact_path(MULTICLASS_MODEL_FILENAME)
        try:
            obj = joblib.load(path)
        except Exception as exc:
            raise ModelCorruptError(f"Failed to load multiclass model from {path}: {exc}") from exc

        if not isinstance(obj, Pipeline):
            raise ModelIncompatibleError(f"Multiclass model at {path} is not a Pipeline; got {type(obj)}")
        if "selector" not in obj.named_steps or "classifier" not in obj.named_steps:
            raise ModelIncompatibleError(
                f"Multiclass model at {path} missing required steps ('selector', 'classifier'): {list(obj.named_steps.keys())}"
            )
        clf = obj.named_steps["classifier"]
        if not hasattr(clf, "predict") or not hasattr(clf, "predict_proba"):
            raise ModelIncompatibleError(f"Multiclass classifier in {path} lacks predict or predict_proba")

        self._multiclass_pipeline = obj
        logger.info("Loaded multiclass attack family model: %s", path.name)

    def _load_anomaly_model(self) -> None:
        path = self._resolve_artifact_path(ANOMALY_MODEL_FILENAME)
        try:
            obj = joblib.load(path)
        except Exception as exc:
            raise ModelCorruptError(f"Failed to load anomaly model from {path}: {exc}") from exc

        if not isinstance(obj, dict):
            raise ModelIncompatibleError(
                f"Anomaly model at {path} expected to be serialized as dict(pipeline, metadata); got {type(obj)}"
            )
        if "pipeline" not in obj or "metadata" not in obj:
            raise ModelIncompatibleError(
                f"Anomaly model at {path} missing 'pipeline' or 'metadata' keys: {list(obj.keys())}"
            )

        pipe = obj["pipeline"]
        meta = obj["metadata"]
        if not isinstance(pipe, Pipeline):
            raise ModelIncompatibleError(f"Anomaly pipeline is not a Pipeline instance; got {type(pipe)}")
        if "anomaly_detector" not in pipe.named_steps:
            raise ModelIncompatibleError(
                f"Anomaly pipeline missing 'anomaly_detector' step: {list(pipe.named_steps.keys())}"
            )
        detector = pipe.named_steps["anomaly_detector"]
        if not hasattr(detector, "score_samples"):
            raise ModelIncompatibleError(f"Anomaly detector in {path} lacks score_samples()")

        if "calibrated_thresholds" not in meta or not meta["calibrated_thresholds"]:
            raise ModelIncompatibleError(f"Anomaly metadata in {path} missing 'calibrated_thresholds'")

        self._anomaly_pipeline = pipe
        self._anomaly_metadata = meta
        logger.info("Loaded unsupervised anomaly detection model: %s", path.name)
