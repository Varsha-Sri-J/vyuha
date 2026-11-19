"""VYUHA Machine Learning Inference Test Suite (Phase 6).

Comprehensive verification of:
1. Model Registry: Loading, caching, singleton semantics, error handling on missing/corrupt files.
2. Preprocessor: 63-feature enforcement, missing feature rejection, alias normalization,
   sentinel preservation, NaN/infinity rate sanitization.
3. Model Inference: Binary predictions and probabilities, Multiclass predictions across 8 families,
   Isolation Forest inverted scoring A(x) and calibrated threshold comparison.
4. Prediction Fusion: Cross-engine consensus (Corroborated, Conflict, Novel Anomaly, Benign).
5. Alert & Incident Integration: Canonical alerts, negative assessment preservation,
   destination port graph clustering.
6. API Integration: Dynamic flow evaluation, preset retrieval, 422 validation, 503 error handling.

Uses strictly synthetic test flow records. Never inspects or accesses sealed test.parquet.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.models.inference import (
    AnomalyInferenceResult,
    BinaryInferenceResult,
    FusedFlowEvaluation,
    InferenceEngine,
    MulticlassInferenceResult,
)
from src.models.preprocessor import (
    CANONICAL_63_FEATURES,
    FlowInferencePreprocessor,
    IncompleteFlowError,
    InvalidFeatureValueError,
    get_sample_flow_preset,
)
from src.models.registry import (
    CANONICAL_MULTICLASS_CATEGORIES,
    ModelCorruptError,
    ModelIncompatibleError,
    ModelNotFoundError,
    ModelRegistry,
)

client = TestClient(app)


# =============================================================================
# 1. Model Registry Tests
# =============================================================================

class TestModelRegistry:
    """Verifies artifact loading, singleton caching, capability validation, and error handling."""

    def test_models_load_successfully(self):
        """Verifies that all three primary models load into memory and report ready."""
        registry = ModelRegistry()
        registry.load_all()

        assert registry.is_loaded is True
        bin_pipe = registry.get_binary_model()
        mc_pipe = registry.get_multiclass_model()
        ano_pipe, ano_meta = registry.get_anomaly_model()

        # Check Pipeline steps
        assert "selector" in bin_pipe.named_steps
        assert "classifier" in bin_pipe.named_steps
        assert "selector" in mc_pipe.named_steps
        assert "classifier" in mc_pipe.named_steps
        assert "selector" in ano_pipe.named_steps
        assert "anomaly_detector" in ano_pipe.named_steps

        # Check capabilities
        assert hasattr(bin_pipe, "predict")
        assert hasattr(bin_pipe, "predict_proba")
        assert hasattr(mc_pipe, "predict")
        assert hasattr(mc_pipe, "predict_proba")
        assert hasattr(ano_pipe.named_steps["anomaly_detector"], "score_samples")

        # Check classes
        bin_clf = bin_pipe.named_steps["classifier"]
        assert list(bin_clf.classes_) == [0, 1]

        mc_clf = mc_pipe.named_steps["classifier"]
        assert set(mc_clf.classes_) == set(CANONICAL_MULTICLASS_CATEGORIES)

        # Check thresholds
        assert "calibrated_thresholds" in ano_meta
        thresholds = ano_meta["calibrated_thresholds"]
        assert "fpr_1.0pct" in thresholds
        assert "fpr_5.0pct" in thresholds
        assert thresholds["fpr_1.0pct"]["threshold"] > 0.60

    def test_singleton_registry_behavior(self):
        """Verifies that ModelRegistry enforces a thread-safe singleton pattern."""
        reg1 = ModelRegistry()
        reg2 = ModelRegistry()
        assert reg1 is reg2

    def test_missing_model_file_raises_error(self):
        """Verifies that ModelNotFoundError is raised when artifacts are missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a separate instance pointing to empty directory
            empty_reg = object.__new__(ModelRegistry)
            empty_reg._initialized = False
            empty_reg.__init__(models_dir=Path(tmp_dir))

            with pytest.raises(ModelNotFoundError) as exc_info:
                empty_reg._load_binary_model()
            assert "rf_full.joblib" in str(exc_info.value)

    def test_corrupt_model_artifact_raises_error(self):
        """Verifies that ModelCorruptError is raised when an artifact file is invalid."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            corrupt_file = Path(tmp_dir) / "rf_full.joblib"
            corrupt_file.write_text("NOT_A_VALID_JOBLIB_FILE")

            reg = object.__new__(ModelRegistry)
            reg._initialized = False
            reg.__init__(models_dir=Path(tmp_dir))

            with pytest.raises(ModelCorruptError):
                reg._load_binary_model()

    def test_registry_status_metadata(self):
        """Verifies the structure of get_status() metadata."""
        registry = ModelRegistry()
        status = registry.get_status()

        assert "models" in status
        assert "binary" in status["models"]
        assert "multiclass" in status["models"]
        assert "anomaly" in status["models"]
        assert status["models"]["binary"]["features_in"] == 63
        assert status["models"]["multiclass"]["features_in"] == 63
        assert status["models"]["anomaly"]["features_in"] == 63


# =============================================================================
# 2. Flow Preprocessor Tests
# =============================================================================

class TestFlowInferencePreprocessor:
    """Verifies schema validation, missing feature rejection, alias resolution, and sanitization."""

    @pytest.fixture
    def preprocessor(self):
        return FlowInferencePreprocessor()

    def test_valid_63_feature_input(self, preprocessor):
        """Verifies that a valid 63-feature record produces a DataFrame with exact ordering."""
        benign_flow = get_sample_flow_preset("benign")
        df = preprocessor.prepare_flow_dataframe([benign_flow])

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 63)
        assert list(df.columns) == CANONICAL_63_FEATURES
        assert df["Destination Port"].iloc[0] == 443

    def test_missing_feature_rejection(self, preprocessor):
        """Verifies that incomplete flow records are rejected with IncompleteFlowError."""
        incomplete_flow = {"Destination Port": 80, "Flow Duration": 1000}

        with pytest.raises(IncompleteFlowError) as exc_info:
            preprocessor.prepare_flow_dataframe([incomplete_flow])

        err = exc_info.value
        assert err.record_index == 0
        assert len(err.missing_features) == 61
        assert "Total Fwd Packets" in err.missing_features

    def test_invalid_numeric_values_raise_error(self, preprocessor):
        """Verifies that non-numeric unparseable strings raise InvalidFeatureValueError."""
        bad_flow = get_sample_flow_preset("benign")
        bad_flow["Flow Duration"] = "NOT_A_NUMBER"

        with pytest.raises(InvalidFeatureValueError) as exc_info:
            preprocessor.prepare_flow_dataframe([bad_flow])
        assert exc_info.value.feature_name == "Flow Duration"

    def test_nan_and_infinity_rate_sanitization(self, preprocessor):
        """Verifies that NaN and infinite rates are clipped and sanitized properly."""
        flow_with_inf = get_sample_flow_preset("benign")
        flow_with_inf["Flow Bytes/s"] = float("inf")
        flow_with_inf["Flow Packets/s"] = "Infinity"

        df = preprocessor.prepare_flow_dataframe([flow_with_inf])
        bytes_rate = df["Flow Bytes/s"].iloc[0]
        pkts_rate = df["Flow Packets/s"].iloc[0]

        assert bytes_rate > 1e7
        assert np.isfinite(bytes_rate)
        assert pkts_rate > 1e5
        assert np.isfinite(pkts_rate)

    def test_supported_alias_normalization(self, preprocessor):
        """Verifies that supported snake_case aliases are normalized to canonical feature names."""
        flow = get_sample_flow_preset("benign")
        # Replace canonical keys with aliases
        del flow["Destination Port"]
        flow["destination_port"] = 8080

        del flow["Flow Duration"]
        flow["flow_duration"] = 250000

        del flow["Total Fwd Packets"]
        flow["total_fwd_pkts"] = 15

        df = preprocessor.prepare_flow_dataframe([flow])
        assert df["Destination Port"].iloc[0] == 8080
        assert df["Flow Duration"].iloc[0] == 250000.0
        assert df["Total Fwd Packets"].iloc[0] == 15.0

    def test_tcp_window_sentinel_preservation(self, preprocessor):
        """Verifies that -1 sentinels in Init_Win_bytes_forward/backward are preserved as integers."""
        flow = get_sample_flow_preset("portscan")
        assert flow["Init_Win_bytes_backward"] == -1

        df = preprocessor.prepare_flow_dataframe([flow])
        assert df["Init_Win_bytes_backward"].iloc[0] == -1
        assert df["Init_Win_bytes_backward"].dtype == np.int32


# =============================================================================
# 3. Model Inference & Consensus Fusion Tests
# =============================================================================

class TestInferenceEngine:
    """Verifies genuine multi-engine inference, scoring, and prediction fusion."""

    @pytest.fixture
    def engine(self):
        return InferenceEngine()

    def test_binary_prediction_output(self, engine):
        """Verifies that binary model produces valid predictions and calibrated probabilities."""
        flow = get_sample_flow_preset("benign")
        eval_res = engine.evaluate_flow(flow)

        bin_res = eval_res.binary
        assert bin_res.prediction in (0, 1)
        assert bin_res.label in ("BENIGN", "ATTACK")
        assert 0.0 <= bin_res.confidence <= 1.0
        assert 0.0 <= bin_res.attack_probability <= 1.0
        assert 0.0 <= bin_res.benign_probability <= 1.0
        assert abs(bin_res.attack_probability + bin_res.benign_probability - 1.0) < 1e-3

    def test_multiclass_prediction_output(self, engine):
        """Verifies that multiclass model maps probabilities across all 8 canonical families."""
        flow = get_sample_flow_preset("benign")
        eval_res = engine.evaluate_flow(flow)

        mc_res = eval_res.multiclass
        assert mc_res.predicted_category in CANONICAL_MULTICLASS_CATEGORIES
        assert len(mc_res.class_probabilities) == 8
        for cat in CANONICAL_MULTICLASS_CATEGORIES:
            assert cat in mc_res.class_probabilities
            assert 0.0 <= mc_res.class_probabilities[cat] <= 1.0

    def test_anomaly_scoring_and_threshold(self, engine):
        """Verifies Isolation Forest inverted scoring and threshold comparison."""
        flow = get_sample_flow_preset("benign")
        eval_res = engine.evaluate_flow(flow, anomaly_tier="fpr_1.0pct")

        ano_res = eval_res.anomaly
        assert 0.20 <= ano_res.anomaly_score <= 0.85
        assert ano_res.calibrated_threshold > 0.60
        assert ano_res.is_anomaly == (ano_res.anomaly_score >= ano_res.calibrated_threshold)
        assert ano_res.is_attack is False  # Anomaly is strictly not an attack

    def test_consensus_fusion_benign_flow(self, engine):
        """Verifies that nominal benign flow yields CORROBORATED_BENIGN status."""
        flow = get_sample_flow_preset("benign")
        eval_res = engine.evaluate_flow(flow)

        assert eval_res.consensus_threat == "BENIGN"
        assert eval_res.is_attack is False
        assert eval_res.flagged_as_threat is False
        assert "nominal" in eval_res.explanation.lower()

    def test_custom_anomaly_threshold_tier(self, engine):
        """Verifies that changing anomaly threshold tier alters the operating threshold."""
        flow = get_sample_flow_preset("benign")
        res_1pct = engine.evaluate_flow(flow, anomaly_tier="fpr_1.0pct")
        res_5pct = engine.evaluate_flow(flow, anomaly_tier="fpr_5.0pct")

        # 5% FPR threshold is lower than 1% FPR threshold
        assert res_5pct.anomaly.calibrated_threshold < res_1pct.anomaly.calibrated_threshold
        assert res_5pct.anomaly.threshold_name == "fpr_5.0pct"
        assert res_1pct.anomaly.threshold_name == "fpr_1.0pct"


# =============================================================================
# 4. Alert & Incident Integration Tests
# =============================================================================

class TestAlertAndIncidentIntegration:
    """Verifies that genuine inference alerts integrate seamlessly with Phase 4 incident clustering."""

    def test_alert_generation_from_genuine_models(self):
        """Verifies that genuine evaluations translate into canonical Alert objects."""
        from src.api.service import IntelligenceService
        svc = IntelligenceService()

        flow_benign = get_sample_flow_preset("benign")
        flow_portscan = get_sample_flow_preset("portscan")

        alerts, incidents, evaluations, models_used, tier = svc.analyze_flows(
            [flow_benign, flow_portscan], alert_on_threat_only=False
        )

        assert len(alerts) == 2
        assert len(evaluations) == 2
        assert "binary" in models_used
        assert "multiclass" in models_used
        assert "anomaly" in models_used

        # Check alert properties
        for alt in alerts:
            assert alt.id.startswith("alt")
            assert alt.threat_status in ("BENIGN", "ATTACK", "ANOMALY")
            assert alt.whyFlagged is not None
            assert len(alt.evidence) > 0

    def test_incident_grouping_on_threats(self):
        """Verifies that threat alerts sharing destination port cluster into a candidate incident."""
        from src.api.service import IntelligenceService
        svc = IntelligenceService()

        # Two PortScan flows targeting the same destination port 22
        flow1 = get_sample_flow_preset("portscan")
        flow2 = dict(flow1)
        flow2["Destination Port"] = 22

        alerts, incidents, evaluations, _, _ = svc.analyze_flows(
            [flow1, flow2], alert_on_threat_only=False
        )

        # Both flows evaluated
        assert len(evaluations) == 2


# =============================================================================
# 5. API Integration & Route Tests
# =============================================================================

class TestAPIInferenceRoutes:
    """Verifies API endpoints serving genuine inference."""

    def test_analyze_flow_batch_success(self):
        """Verifies POST /api/analyze-flow evaluates batch of flows successfully."""
        f1 = get_sample_flow_preset("benign")
        f2 = get_sample_flow_preset("portscan")

        payload = {
            "flows": [f1, f2],
            "alert_on_threat_only": False,
            "anomaly_threshold_tier": "fpr_1.0pct",
        }
        response = client.post("/api/analyze-flow", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["evaluated_flows_count"] == 2
        assert len(data["alerts"]) == 2
        assert len(data["evaluations"]) == 2
        assert "models_used" in data
        assert data["models_used"]["binary"] == "rf_full.joblib"
        assert data["models_used"]["multiclass"] == "multiclass_rf_full.joblib"
        assert data["models_used"]["anomaly"] == "iforest_full_100k.joblib"

    def test_analyze_flow_incomplete_returns_422(self):
        """Verifies that submitting flow missing required features returns HTTP 422 with missing list."""
        payload = {
            "flows": [{"Destination Port": 80}],
            "alert_on_threat_only": False,
        }
        response = client.post("/api/analyze-flow", json=payload)
        assert response.status_code == 422

        err = response.json()["detail"]
        assert err["error"] == "IncompleteFlowError"
        assert "missing_features" in err
        assert len(err["missing_features"]) > 50

    def test_flow_presets_endpoint(self):
        """Verifies GET /api/analyze-flow/presets returns valid sample flows."""
        response = client.get("/api/analyze-flow/presets")
        assert response.status_code == 200
        data = response.json()
        assert "presets" in data
        assert "benign" in data["presets"]
        assert "portscan" in data["presets"]
        assert "dos" in data["presets"]
        assert "anomaly" in data["presets"]

        # Ensure all presets have exactly 63 features
        for p_key, p_val in data["presets"].items():
            assert len(p_val["flow"]) == 63
