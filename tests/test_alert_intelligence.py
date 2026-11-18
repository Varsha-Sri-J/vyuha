"""VYUHA Test Suite: Alert Normalization & Entity Extraction (Phase 4.1).

Tests:
1. Binary alert normalisation.
2. Multiclass alert normalisation.
3. Anomaly alert normalisation.
4. Missing and nullable fields.
5. Invalid IP addresses.
6. Port validation.
7. Timestamp handling.
8. Duplicate-flow handling.
9. Evidence preservation.
10. Target-column exclusion.
"""

import datetime
import json
import numpy as np
import pandas as pd
import pytest

from src.intelligence.adapters import (
    AnomalyAlertAdapter,
    BinaryAlertAdapter,
    MulticlassAlertAdapter,
)
from src.intelligence.extractor import EntityExtractor, ExtractedEntities
from src.intelligence.normalizer import MultiEngineAlertNormalizer
from src.intelligence.schema import (
    Alert,
    AlertSeverity,
    DetectionEngineType,
    ThreatStatus,
)
from src.models.feature_selector import TARGET_COLUMNS, extract_features_and_targets


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def sample_flow_df() -> pd.DataFrame:
    """Creates a sample DataFrame representative of CICIDS2017 processed flows."""
    return pd.DataFrame({
        "Destination Port": [80.0, 443.0, 22.0, 8080.0],
        "Flow Duration": [1000.0, 50000.0, 250.0, 12000.0],
        "Total Fwd Packets": [5.0, 10.0, 2.0, 8.0],
        "Total Backward Packets": [3.0, 8.0, 0.0, 6.0],
        "label_raw": ["BENIGN", "DoS Hulk", "SSH-Patator", "PortScan"],
        "label_category": ["BENIGN", "DoS_DDoS", "BruteForce", "PortScan"],
        "label_binary": [0, 1, 1, 1],
    })


@pytest.fixture
def sample_enriched_flow_df() -> pd.DataFrame:
    """Creates a sample DataFrame enriched with IP, timestamp, and flow ID."""
    return pd.DataFrame({
        "Flow ID": ["flow_alpha_01", "flow_beta_02", "flow_gamma_03"],
        "Source IP": ["192.168.1.105", "10.0.0.15", "172.16.0.5"],
        "Destination IP": ["192.168.1.1", "10.0.0.1", "172.16.0.1"],
        "Source Port": [49152, 52100, 38900],
        "Destination Port": [80, 443, 22],
        "Timestamp": ["2026-11-19T00:40:00", "2026-11-19T00:41:00", "2026-11-19T00:42:00"],
        "Flow Duration": [1200.0, 45000.0, 800.0],
        "label_category": ["BENIGN", "DoS_DDoS", "BruteForce"],
        "label_binary": [0, 1, 1],
    })


# -----------------------------------------------------------------------------
# 1. Binary Alert Normalisation
# -----------------------------------------------------------------------------
def test_binary_alert_normalisation(sample_enriched_flow_df):
    """Verifies that BinaryAlertAdapter properly normalizes binary predictions and probabilities."""
    adapter = BinaryAlertAdapter(
        model_name="rf_full",
        alert_on_attack_only=False,  # Keep benign flows for testing
    )
    predictions = np.array([0, 1, 1])
    probabilities = np.array([
        [0.98, 0.02],  # Benign
        [0.05, 0.95],  # High confidence attack
        [0.30, 0.70],  # Moderate confidence attack
    ])

    alerts = adapter.adapt(
        predictions=predictions,
        flow_records=sample_enriched_flow_df,
        probabilities=probabilities,
    )

    assert len(alerts) == 3

    # Check benign alert
    alt_benign = alerts[0]
    assert alt_benign.flow_id == "flow_alpha_01"
    assert alt_benign.detection_engine == DetectionEngineType.BINARY.value
    assert alt_benign.model_name == "rf_full"
    assert alt_benign.threat_status == ThreatStatus.BENIGN.value
    assert not alt_benign.is_attack
    assert alt_benign.prediction == "BENIGN"
    assert alt_benign.confidence == pytest.approx(0.98)
    assert alt_benign.severity == AlertSeverity.INFO.value

    # Check high-confidence attack
    alt_attack1 = alerts[1]
    assert alt_attack1.flow_id == "flow_beta_02"
    assert alt_attack1.threat_status == ThreatStatus.ATTACK.value
    assert alt_attack1.is_attack
    assert alt_attack1.prediction == "ATTACK"
    assert alt_attack1.confidence == pytest.approx(0.95)
    assert alt_attack1.severity == AlertSeverity.HIGH.value
    assert alt_attack1.evidence["attack_probability"] == pytest.approx(0.95)

    # Check moderate-confidence attack
    alt_attack2 = alerts[2]
    assert alt_attack2.threat_status == ThreatStatus.ATTACK.value
    assert alt_attack2.is_attack
    assert alt_attack2.confidence == pytest.approx(0.70)
    assert alt_attack2.severity == AlertSeverity.MEDIUM.value

    # Test probability-free prediction (should NOT fabricate confidence)
    adapter_no_prob = BinaryAlertAdapter(model_name="dt_full", alert_on_attack_only=True)
    alerts_no_prob = adapter_no_prob.adapt(
        predictions=predictions,
        flow_records=sample_enriched_flow_df,
        probabilities=None,
    )
    assert len(alerts_no_prob) == 2  # Only attacks
    for alt in alerts_no_prob:
        assert alt.confidence is None


# -----------------------------------------------------------------------------
# 2. Multiclass Alert Normalisation
# -----------------------------------------------------------------------------
def test_multiclass_alert_normalisation(sample_enriched_flow_df):
    """Verifies that MulticlassAlertAdapter preserves attack categories and class confidence."""
    classes = ["BENIGN", "DoS_DDoS", "BruteForce", "Heartbleed"]
    adapter = MulticlassAlertAdapter(
        model_name="multiclass_rf_full",
        alert_on_attack_only=False,
        classes=classes,
    )

    predictions = ["BENIGN", "DoS_DDoS", "Heartbleed"]
    probabilities = np.array([
        [0.99, 0.01, 0.00, 0.00],
        [0.02, 0.94, 0.03, 0.01],
        [0.05, 0.05, 0.10, 0.80],
    ])

    alerts = adapter.adapt(
        predictions=predictions,
        flow_records=sample_enriched_flow_df,
        probabilities=probabilities,
        classes=classes,
    )

    assert len(alerts) == 3

    # Check DoS alert
    dos_alt = alerts[1]
    assert dos_alt.threat_status == ThreatStatus.ATTACK.value
    assert dos_alt.is_attack
    assert dos_alt.attack_category == "DoS_DDoS"
    assert dos_alt.prediction == "DoS_DDoS"
    assert dos_alt.confidence == pytest.approx(0.94)
    assert dos_alt.severity == AlertSeverity.HIGH.value
    assert dos_alt.evidence["class_probabilities"]["DoS_DDoS"] == pytest.approx(0.94)

    # Check Heartbleed critical alert
    hb_alt = alerts[2]
    assert hb_alt.threat_status == ThreatStatus.ATTACK.value
    assert hb_alt.is_attack
    assert hb_alt.attack_category == "Heartbleed"
    assert hb_alt.confidence == pytest.approx(0.80)
    assert hb_alt.severity == AlertSeverity.CRITICAL.value

    # Verify filtering when alert_on_attack_only=True
    adapter_filter = MulticlassAlertAdapter(alert_on_attack_only=True)
    filtered = adapter_filter.adapt(predictions=predictions, flow_records=sample_enriched_flow_df)
    assert len(filtered) == 2
    assert all(a.is_attack for a in filtered)


# -----------------------------------------------------------------------------
# 3. Anomaly Alert Normalisation
# -----------------------------------------------------------------------------
def test_anomaly_alert_normalisation(sample_enriched_flow_df):
    """Verifies that AnomalyAlertAdapter preserves scores and does not declare confirmed attacks."""
    adapter = AnomalyAlertAdapter(
        model_name="iforest_port_100k",
        calibrated_threshold=0.6430,
        threshold_name="fpr_1.0pct",
        alert_on_anomaly_only=False,
    )

    anomaly_scores = np.array([0.3450, 0.6550, 0.7200])

    alerts = adapter.adapt(
        anomaly_scores=anomaly_scores,
        flow_records=sample_enriched_flow_df,
    )

    assert len(alerts) == 3

    # Flow 0: Normal (0.345 < 0.6430)
    normal_alt = alerts[0]
    assert normal_alt.prediction == "NORMAL"
    assert normal_alt.threat_status == ThreatStatus.BENIGN.value
    assert not normal_alt.is_attack
    assert normal_alt.attack_category is None
    assert normal_alt.confidence is None
    assert normal_alt.anomaly_score == pytest.approx(0.3450)
    assert normal_alt.anomaly_threshold == pytest.approx(0.6430)
    assert normal_alt.severity == AlertSeverity.INFO.value

    # Flow 1: Anomaly (0.6550 >= 0.6430)
    ano_alt1 = alerts[1]
    assert ano_alt1.prediction == "ANOMALOUS"
    assert ano_alt1.threat_status == ThreatStatus.ANOMALY.value
    # CRITICAL: Anomaly is NOT marked as confirmed attack
    assert not ano_alt1.is_attack
    assert ano_alt1.attack_category is None
    assert ano_alt1.confidence is None
    assert ano_alt1.anomaly_score == pytest.approx(0.6550)
    assert ano_alt1.evidence["threshold_margin"] == pytest.approx(0.6550 - 0.6430, abs=1e-5)
    assert ano_alt1.severity == AlertSeverity.MEDIUM.value

    # Flow 2: Extreme Anomaly (0.7200 >= 0.70)
    ano_alt2 = alerts[2]
    assert ano_alt2.prediction == "ANOMALOUS"
    assert ano_alt2.threat_status == ThreatStatus.ANOMALY.value
    assert not ano_alt2.is_attack
    assert ano_alt2.severity == AlertSeverity.HIGH.value


# -----------------------------------------------------------------------------
# 4. Missing and Nullable Fields
# -----------------------------------------------------------------------------
def test_missing_and_nullable_fields(sample_flow_df):
    """Verifies handling of dataset records where IP, timestamp, and protocol are absent."""
    entities = EntityExtractor.extract_batch(sample_flow_df)
    assert len(entities) == 4

    for ent in entities:
        # MachineLearningCVE fields that are missing
        assert ent.source_ip is None
        assert ent.destination_ip is None
        assert ent.source_port is None
        assert ent.protocol is None
        assert ent.timestamp is None
        # Fields that are present
        assert ent.destination_port in (80, 443, 22, 8080)
        assert ent.flow_duration > 0
        assert ent.flow_id.startswith("flow_")

    # Verify Alert schema serialization with null values
    alert = Alert(
        alert_id="alt_test_001",
        flow_id="flow_001",
        detection_engine="binary",
        model_name="rf_full",
        prediction="ATTACK",
        threat_status=ThreatStatus.ATTACK.value,
        is_attack=True,
        destination_port=80,
    )

    json_str = alert.to_json()
    assert "source_ip" in json_str
    assert "null" in json_str

    reconstructed = Alert.from_json(json_str)
    assert reconstructed.alert_id == "alt_test_001"
    assert reconstructed.source_ip is None
    assert reconstructed.destination_port == 80
    assert reconstructed.is_attack is True


# -----------------------------------------------------------------------------
# 5. Invalid IP Addresses
# -----------------------------------------------------------------------------
def test_invalid_ip_addresses():
    """Verifies IP address validation in both lenient and strict modes."""
    # Valid IPv4
    ip, err = EntityExtractor.validate_ip("192.168.1.100")
    assert ip == "192.168.1.100"
    assert err is None

    # Valid IPv6
    ip, err = EntityExtractor.validate_ip("2001:db8::1")
    assert ip == "2001:db8::1"
    assert err is None

    # Invalid IPs (Lenient)
    for bad_ip in ["999.999.999.999", "256.0.0.1", "not_an_ip", "192.168.1"]:
        ip, err = EntityExtractor.validate_ip(bad_ip, strict=False)
        assert ip is None
        assert err is not None
        assert "Invalid IP" in err

    # Invalid IPs (Strict)
    with pytest.raises(ValueError, match="Invalid IP syntax"):
        EntityExtractor.validate_ip("999.999.999.999", strict=True)


# -----------------------------------------------------------------------------
# 6. Port Validation
# -----------------------------------------------------------------------------
def test_port_validation():
    """Verifies port range validation [0, 65535] and float handling."""
    # Valid ports
    assert EntityExtractor.validate_port(80)[0] == 80
    assert EntityExtractor.validate_port(443.0)[0] == 443
    assert EntityExtractor.validate_port(0)[0] == 0
    assert EntityExtractor.validate_port(65535)[0] == 65535

    # Invalid ports (Lenient)
    for bad_port in [-1, 65536, 70000, "not_a_port", 80.5]:
        port, err = EntityExtractor.validate_port(bad_port, strict=False)
        assert port is None
        assert err is not None

    # Invalid ports (Strict)
    with pytest.raises(ValueError, match="out of bounds"):
        EntityExtractor.validate_port(70000, strict=True)
    with pytest.raises(ValueError, match="out of bounds"):
        EntityExtractor.validate_port(-1, strict=True)


# -----------------------------------------------------------------------------
# 7. Timestamp Handling
# -----------------------------------------------------------------------------
def test_timestamp_handling():
    """Verifies ISO-8601 parsing and standardization of timestamps."""
    # Datetime object
    dt = datetime.datetime(2026, 11, 19, 0, 40, 5)
    ts, err = EntityExtractor.validate_timestamp(dt)
    assert ts == "2026-11-19T00:40:05"
    assert err is None

    # Pandas Timestamp
    ts_pd, _ = EntityExtractor.validate_timestamp(pd.Timestamp("2026-11-19 00:40:05"))
    assert ts_pd == "2026-11-19T00:40:05"

    # Common string formats
    ts_str, _ = EntityExtractor.validate_timestamp("19/11/2026 00:40:05")
    assert ts_str is not None
    assert "2026" in ts_str

    # Unparseable timestamp
    ts_bad, err = EntityExtractor.validate_timestamp("unparseable_time", strict=False)
    assert ts_bad is None
    assert err is not None

    with pytest.raises(ValueError, match="Failed to parse timestamp"):
        EntityExtractor.validate_timestamp("unparseable_time", strict=True)

    # None handling
    assert EntityExtractor.validate_timestamp(None)[0] is None


# -----------------------------------------------------------------------------
# 8. Duplicate-Flow Handling
# -----------------------------------------------------------------------------
def test_duplicate_flow_handling(sample_enriched_flow_df):
    """Verifies that multiple engines analyzing the same flows produce consolidated, deduplicated alerts."""
    # Engine 1: Binary detector
    bin_adapter = BinaryAlertAdapter(model_name="rf_full", alert_on_attack_only=False)
    bin_alerts = bin_adapter.adapt(
        predictions=[0, 1, 1],
        flow_records=sample_enriched_flow_df,
        probabilities=np.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8]]),
    )

    # Engine 2: Multiclass detector
    mc_adapter = MulticlassAlertAdapter(
        model_name="multiclass_rf_full",
        alert_on_attack_only=False,
        classes=["BENIGN", "DoS_DDoS", "BruteForce"],
    )
    mc_alerts = mc_adapter.adapt(
        predictions=["BENIGN", "DoS_DDoS", "BruteForce"],
        flow_records=sample_enriched_flow_df,
        probabilities=np.array([[0.95, 0.05, 0.00], [0.05, 0.90, 0.05], [0.10, 0.10, 0.80]]),
    )

    # Engine 3: Anomaly detector
    ano_adapter = AnomalyAlertAdapter(
        model_name="iforest_port_100k",
        calibrated_threshold=0.6430,
        alert_on_anomaly_only=False,
    )
    ano_alerts = ano_adapter.adapt(
        anomaly_scores=[0.35, 0.68, 0.72],
        flow_records=sample_enriched_flow_df,
    )

    # Combine all raw alerts: 3 engines * 3 flows = 9 alerts
    all_raw_alerts = bin_alerts + mc_alerts + ano_alerts
    assert len(all_raw_alerts) == 9

    # Consolidate via MultiEngineAlertNormalizer
    consolidated = MultiEngineAlertNormalizer.consolidate_alerts(all_raw_alerts)

    # Must produce exactly 3 consolidated alerts (one per unique flow_id)
    assert len(consolidated) == 3

    flow_ids = [a.flow_id for a in consolidated]
    assert sorted(flow_ids) == ["flow_alpha_01", "flow_beta_02", "flow_gamma_03"]

    # Verify no duplicate flow IDs exist
    assert len(set(flow_ids)) == len(flow_ids)


# -----------------------------------------------------------------------------
# 9. Evidence Preservation
# -----------------------------------------------------------------------------
def test_evidence_preservation(sample_enriched_flow_df):
    """Verifies that engine-specific evidence is preserved without silent overwrite."""
    # Generate multi-engine alerts for flow_beta_02
    flow_slice = sample_enriched_flow_df.iloc[[1]].copy()

    bin_adapter = BinaryAlertAdapter(model_name="rf_full")
    bin_alerts = bin_adapter.adapt(
        predictions=[1],
        flow_records=flow_slice,
        probabilities=np.array([[0.05, 0.95]]),
    )

    mc_adapter = MulticlassAlertAdapter(
        model_name="multiclass_rf_full",
        classes=["BENIGN", "DoS_DDoS"],
    )
    mc_alerts = mc_adapter.adapt(
        predictions=["DoS_DDoS"],
        flow_records=flow_slice,
        probabilities=np.array([[0.08, 0.92]]),
    )

    ano_adapter = AnomalyAlertAdapter(
        model_name="iforest_port_100k",
        calibrated_threshold=0.6430,
    )
    ano_alerts = ano_adapter.adapt(
        anomaly_scores=[0.6750],
        flow_records=flow_slice,
    )

    consolidated = MultiEngineAlertNormalizer.consolidate_alerts(
        bin_alerts + mc_alerts + ano_alerts
    )
    assert len(consolidated) == 1
    alert = consolidated[0]

    # Verify top-level properties
    assert alert.flow_id == "flow_beta_02"
    assert alert.detection_engine == DetectionEngineType.MULTI_ENGINE.value
    assert alert.is_attack is True
    assert alert.attack_category == "DoS_DDoS"
    assert alert.confidence == pytest.approx(0.92)  # Multiclass confidence
    assert alert.anomaly_score == pytest.approx(0.6750)
    assert alert.anomaly_threshold == pytest.approx(0.6430)

    # CRITICAL: Verify evidence preserves all 3 participating engines
    evidence = alert.evidence
    assert evidence["engine_count"] == 3
    assert set(evidence["participating_engines"]) == {"binary", "multiclass", "anomaly"}
    assert "binary" in evidence["engines"]
    assert "multiclass" in evidence["engines"]
    assert "anomaly" in evidence["engines"]

    # Verify granular engine details are preserved
    assert evidence["engines"]["binary"]["model_name"] == "rf_full"
    assert evidence["engines"]["binary"]["confidence"] == pytest.approx(0.95)
    assert evidence["engines"]["multiclass"]["model_name"] == "multiclass_rf_full"
    assert evidence["engines"]["multiclass"]["attack_category"] == "DoS_DDoS"
    assert evidence["engines"]["anomaly"]["model_name"] == "iforest_port_100k"
    assert evidence["engines"]["anomaly"]["anomaly_score"] == pytest.approx(0.6750)


# -----------------------------------------------------------------------------
# 10. Target-Column Exclusion
# -----------------------------------------------------------------------------
def test_target_column_exclusion(sample_flow_df):
    """Verifies that ground-truth target columns are strictly excluded from entity extraction and features."""
    # Ensure targets exist in fixture
    assert "label_raw" in sample_flow_df.columns
    assert "label_category" in sample_flow_df.columns
    assert "label_binary" in sample_flow_df.columns

    # 1. Feature separation
    X, y_dict = extract_features_and_targets(sample_flow_df)
    for col in TARGET_COLUMNS:
        assert col not in X.columns

    # 2. Entity extraction never returns target column values in extracted fields
    ent = EntityExtractor.extract(sample_flow_df.iloc[1])
    ent_dict = ent.to_dict()
    for col in TARGET_COLUMNS:
        assert col not in ent_dict

    # 3. Deterministic flow hash excludes target columns (hash remains identical even if targets vary)
    row_a = sample_flow_df.iloc[0].to_dict()
    row_b = row_a.copy()
    row_b["label_raw"] = "ALTERED_LABEL"
    row_b["label_category"] = "ALTERED_CAT"
    row_b["label_binary"] = 99

    hash_a = EntityExtractor.generate_flow_id(row_a)
    hash_b = EntityExtractor.generate_flow_id(row_b)
    assert hash_a == hash_b, "Flow hash should be invariant to target column values!"


# -----------------------------------------------------------------------------
# 11. Threat Status Semantics Audit
# -----------------------------------------------------------------------------
def test_threat_status_semantics():
    """Audits explicit threat_status values and enforces that anomaly is never marked is_attack=True."""
    # 1. Valid threat statuses
    alt_attack = Alert(
        alert_id="alt_1",
        flow_id="flow_1",
        detection_engine="binary",
        model_name="rf_full",
        prediction="ATTACK",
        threat_status=ThreatStatus.ATTACK.value,
        is_attack=True,
    )
    assert alt_attack.threat_status == "ATTACK"
    assert alt_attack.is_attack is True

    alt_benign = Alert(
        alert_id="alt_2",
        flow_id="flow_2",
        detection_engine="binary",
        model_name="rf_full",
        prediction="BENIGN",
        threat_status=ThreatStatus.BENIGN.value,
        is_attack=False,
    )
    assert alt_benign.threat_status == "BENIGN"
    assert alt_benign.is_attack is False

    alt_anomaly = Alert(
        alert_id="alt_3",
        flow_id="flow_3",
        detection_engine="anomaly",
        model_name="iforest_port_100k",
        prediction="ANOMALOUS",
        threat_status=ThreatStatus.ANOMALY.value,
        is_attack=False,
        anomaly_score=0.68,
        anomaly_threshold=0.643,
    )
    assert alt_anomaly.threat_status == "ANOMALY"
    assert alt_anomaly.is_attack is False

    alt_unknown = Alert(
        alert_id="alt_4",
        flow_id="flow_4",
        detection_engine="binary",
        model_name="rf_full",
        prediction="UNKNOWN",
        threat_status=ThreatStatus.UNKNOWN.value,
        is_attack=False,
    )
    assert alt_unknown.threat_status == "UNKNOWN"

    # 2. Semantic violation: Anomaly cannot have is_attack=True
    with pytest.raises(ValueError, match="An anomaly alert cannot have is_attack=True"):
        Alert(
            alert_id="alt_bad_ano",
            flow_id="flow_bad",
            detection_engine="anomaly",
            model_name="iforest_port_100k",
            prediction="ANOMALOUS",
            threat_status=ThreatStatus.ANOMALY.value,
            is_attack=True,  # VIOLATION
        )

    # 3. Semantic violation: Benign cannot have is_attack=True
    with pytest.raises(ValueError, match="A benign alert cannot have is_attack=True"):
        Alert(
            alert_id="alt_bad_ben",
            flow_id="flow_bad",
            detection_engine="binary",
            model_name="rf_full",
            prediction="BENIGN",
            threat_status=ThreatStatus.BENIGN.value,
            is_attack=True,  # VIOLATION
        )

    # 4. Semantic violation: Attack must have is_attack=True
    with pytest.raises(ValueError, match="An attack alert must have is_attack=True"):
        Alert(
            alert_id="alt_bad_att",
            flow_id="flow_bad",
            detection_engine="binary",
            model_name="rf_full",
            prediction="ATTACK",
            threat_status=ThreatStatus.ATTACK.value,
            is_attack=False,  # VIOLATION
        )

    # 5. Invalid status string
    with pytest.raises(ValueError, match="Invalid threat_status"):
        Alert(
            alert_id="alt_bad_stat",
            flow_id="flow_bad",
            detection_engine="binary",
            model_name="rf_full",
            prediction="ATTACK",
            threat_status="INVALID_STATUS",
            is_attack=True,
        )


# -----------------------------------------------------------------------------
# 12. Flow Identifier Integrity on Statistically Identical Records
# -----------------------------------------------------------------------------
def test_flow_id_identical_feature_vectors_distinct_records():
    """Verifies that statistically identical flows at distinct row indices remain strictly separate."""
    # Two flows with 100% IDENTICAL feature values (e.g. identical packet bursts)
    identical_flows_df = pd.DataFrame({
        "Destination Port": [80.0, 80.0],
        "Flow Duration": [1500.0, 1500.0],
        "Total Fwd Packets": [3.0, 3.0],
        "Total Backward Packets": [2.0, 2.0],
    }, index=[101, 102])

    # 1. Entity extraction preserves distinct positional identifiers
    entities = EntityExtractor.extract_batch(identical_flows_df)
    assert len(entities) == 2
    assert entities[0].flow_id == "flow_101"
    assert entities[1].flow_id == "flow_102"
    assert entities[0].flow_id != entities[1].flow_id

    # 2. Multi-engine batch processing does NOT merge distinct original records
    alerts = MultiEngineAlertNormalizer.process_flow_batch(
        flow_records=identical_flows_df,
        binary_predictions=[1, 1],
        binary_probabilities=np.array([[0.1, 0.9], [0.1, 0.9]]),
        multiclass_predictions=["DoS_DDoS", "DoS_DDoS"],
        multiclass_probabilities=np.array([[0.1, 0.9], [0.1, 0.9]]),
        multiclass_classes=["BENIGN", "DoS_DDoS"],
        alert_on_threat_only=True,
    )

    # Must produce exactly TWO distinct alerts (one for each original flow record)
    assert len(alerts) == 2
    assert alerts[0].flow_id == "flow_101"
    assert alerts[1].flow_id == "flow_102"
    assert alerts[0].alert_id != alerts[1].alert_id


# -----------------------------------------------------------------------------
# 13. Multi-Engine Negative Assessment Preservation
# -----------------------------------------------------------------------------
def test_multi_engine_negative_assessment_preservation():
    """Verifies that non-flagging engines are preserved as negative evaluations rather than missing."""
    flow_df = pd.DataFrame({
        "Destination Port": [443.0],
        "Flow Duration": [25000.0],
        "Total Fwd Packets": [6.0],
    }, index=[500])

    # Engine 1 (Binary): ATTACK (flagged)
    # Engine 2 (Multiclass): BENIGN (not flagged as threat)
    # Engine 3 (Anomaly): NORMAL (not flagged as anomaly)
    alerts = MultiEngineAlertNormalizer.process_flow_batch(
        flow_records=flow_df,
        binary_predictions=[1],
        binary_probabilities=np.array([[0.15, 0.85]]),
        multiclass_predictions=["BENIGN"],
        multiclass_probabilities=np.array([[0.92, 0.08]]),
        multiclass_classes=["BENIGN", "DoS_DDoS"],
        anomaly_scores=[0.4200],
        anomaly_threshold=0.6430,
        alert_on_threat_only=True,  # Retain threats
    )

    assert len(alerts) == 1
    alt = alerts[0]

    # Overall consolidated threat is ATTACK (from binary)
    assert alt.flow_id == "flow_500"
    assert alt.threat_status == ThreatStatus.ATTACK.value
    assert alt.is_attack is True

    # Check evidence preserves ALL 3 engines independently
    engines = alt.evidence["engines"]
    assert len(engines) == 3
    assert set(engines.keys()) == {"binary", "multiclass", "anomaly"}

    # Binary engine: flagged
    assert engines["binary"]["flagged_as_threat"] is True
    assert engines["binary"]["prediction"] == "ATTACK"
    assert engines["binary"]["threat_status"] == "ATTACK"
    assert engines["binary"]["confidence"] == pytest.approx(0.85)

    # Multiclass engine: explicitly preserved as non-flagging (NOT missing)
    assert engines["multiclass"]["flagged_as_threat"] is False
    assert engines["multiclass"]["prediction"] == "BENIGN"
    assert engines["multiclass"]["threat_status"] == "BENIGN"
    assert engines["multiclass"]["confidence"] == pytest.approx(0.92)

    # Anomaly engine: explicitly preserved as non-flagging (NOT missing)
    assert engines["anomaly"]["flagged_as_threat"] is False
    assert engines["anomaly"]["prediction"] == "NORMAL"
    assert engines["anomaly"]["threat_status"] == "BENIGN"
    assert engines["anomaly"]["anomaly_score"] == pytest.approx(0.4200)
    assert engines["anomaly"]["anomaly_threshold"] == pytest.approx(0.6430)

    # Consensus summary reflects exact breakdown
    consensus = alt.evidence["consensus_summary"]
    assert consensus["engines_flagging_threat"] == ["binary"]
    assert set(consensus["engines_not_flagging"]) == {"multiclass", "anomaly"}
    assert consensus["all_engines_agreed_attack"] is False


# -----------------------------------------------------------------------------
# 14. Provisional Severity Semantics
# -----------------------------------------------------------------------------
def test_provisional_severity_semantics():
    """Verifies that alert severity is explicitly documented as provisional flow heuristic."""
    alt = Alert(
        alert_id="alt_prov_01",
        flow_id="flow_01",
        detection_engine="binary",
        model_name="rf_full",
        prediction="ATTACK",
        threat_status=ThreatStatus.ATTACK.value,
        is_attack=True,
        severity=AlertSeverity.LOW.value,
        metadata={"severity_semantics": "provisional_flow_heuristic"},
    )
    assert alt.severity == "LOW"
    assert alt.metadata["severity_semantics"] == "provisional_flow_heuristic"
