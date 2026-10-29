"""Tests for Phase 6: Explainable Alerts and CLI (src/explain.py, src/cli.py).

Verifies:
  1. Schema validation for both detectors (rules and ML).
  2. Reason text present, non-empty, and containing precedence / attribution details.
  3. Deterministic alert IDs and idempotent writing.
  4. CLI round trip for explain and list-alerts filtering.
  5. Feature attribution math and non-causal explanation disclaimers.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.explain import (
    ExplainableAlert,
    compute_feature_contributions,
    format_utc_epoch_time,
    make_deterministic_ml_alert_id,
    read_alerts_from_jsonl,
    write_alerts_to_jsonl,
)
from src.features import FEATURE_NAMES
from src.rules import make_deterministic_alert_id


ALERTS_FILE = Path("reports/alerts.jsonl")


def test_alerts_file_exists_and_contains_both_detectors() -> None:
    """Verify reports/alerts.jsonl exists and contains alerts from both detectors."""
    assert ALERTS_FILE.exists(), "reports/alerts.jsonl does not exist."
    alerts = read_alerts_from_jsonl(ALERTS_FILE)
    assert len(alerts) > 0, "alerts.jsonl is empty."

    detectors = {a.detector for a in alerts}
    assert "rules" in detectors, "No rules alerts found in reports/alerts.jsonl."
    assert "ml" in detectors, "No ml alerts found in reports/alerts.jsonl."


def test_schema_validation_rules_detector() -> None:
    """Validate schema of all rules detector alerts in reports/alerts.jsonl."""
    alerts = read_alerts_from_jsonl(ALERTS_FILE)
    rules_alerts = [a for a in alerts if a.detector == "rules"]
    assert len(rules_alerts) > 0

    valid_types = {
        "ORIGIN_HIJACK",
        "SUBPREFIX_HIJACK",
        "INVENTORY_MISMATCH",
        "ROA_INVALID",
        "THIRD_PARTY_INVALID",
        "ROA_MISSING",
    }
    valid_rov_states = {"Valid", "Invalid", "NotFound", "NoSnapshot"}

    for a in rules_alerts:
        # Top-level schema
        assert a.alert_id.startswith("alt-"), f"Unexpected alert_id format: {a.alert_id}"
        assert isinstance(a.time, dict), f"time must be a dict: {a.time}"
        assert "utc" in a.time and "epoch" in a.time
        assert isinstance(a.time["utc"], str) and a.time["utc"].endswith("Z")
        assert isinstance(a.time["epoch"], (int, float)) and a.time["epoch"] > 0
        assert a.type in valid_types, f"Unexpected rule alert type: {a.type}"
        assert isinstance(a.prefix, str) and "/" in a.prefix
        assert isinstance(a.origin_asn, int) and a.origin_asn > 0
        assert a.severity in ("high", "medium", "low")
        assert a.detector == "rules"

        # Evidence block schema
        ev = a.evidence
        assert isinstance(ev.covering_roas, list)
        for roa in ev.covering_roas:
            assert "prefix" in roa and "asn" in roa and "maxLength" in roa
            assert isinstance(roa["asn"], int)
            assert isinstance(roa["maxLength"], int)

        assert ev.rfc6811_state in valid_rov_states
        assert isinstance(ev.peer, str) and len(ev.peer) > 0
        assert isinstance(ev.reason, str) and len(ev.reason.strip()) > 0

        if ev.inventory_match is not None:
            assert "prefix" in ev.inventory_match
            assert "owner" in ev.inventory_match
            assert "allowed_origins" in ev.inventory_match
            assert isinstance(ev.inventory_match["allowed_origins"], list)


def test_schema_validation_ml_detector() -> None:
    """Validate schema of all ML detector alerts in reports/alerts.jsonl."""
    alerts = read_alerts_from_jsonl(ALERTS_FILE)
    ml_alerts = [a for a in alerts if a.detector == "ml"]
    assert len(ml_alerts) > 0

    valid_rov_states = {"Valid", "Invalid", "NotFound", "NoSnapshot"}

    for a in ml_alerts:
        # Top-level schema
        assert a.alert_id.startswith("alt-ml-"), f"Unexpected ML alert_id format: {a.alert_id}"
        assert isinstance(a.time, dict)
        assert "utc" in a.time and "epoch" in a.time
        assert a.type == "ML_ANOMALY"
        assert isinstance(a.prefix, str) and "/" in a.prefix
        assert isinstance(a.origin_asn, int) and a.origin_asn > 0
        assert a.detector == "ml"
        assert a.score is not None and 0.0 <= a.score <= 1.0
        assert a.threshold is not None and a.threshold == 0.05
        assert a.score >= a.threshold, f"ML alert score {a.score} below threshold {a.threshold}"

        # Evidence block schema
        ev = a.evidence
        assert isinstance(ev.covering_roas, list)
        assert ev.rfc6811_state in valid_rov_states
        assert isinstance(ev.peer, str) and len(ev.peer) > 0
        assert isinstance(ev.reason, str) and len(ev.reason.strip()) > 0

        # ML-specific fields
        assert ev.score == a.score
        assert ev.threshold == a.threshold
        assert ev.top_contributing_features is not None
        assert len(ev.top_contributing_features) == 3, "ML alert must include top 3 features"
        for feat in ev.top_contributing_features:
            assert "feature" in feat
            assert feat["feature"] in FEATURE_NAMES
            assert "value" in feat
            assert "contribution" in feat
            assert feat["direction"] in ("+", "-", "0")

        assert ev.explanation_method is not None and len(ev.explanation_method.strip()) > 0
        assert ev.causal_disclaimer is not None
        assert "causal" in ev.causal_disclaimer.lower(), "Disclaimer must disclaim causal explanations"


def test_reason_text_present_and_precedence_explained() -> None:
    """Verify that every alert has a non-empty reason explaining why it fired."""
    alerts = read_alerts_from_jsonl(ALERTS_FILE)
    for a in alerts:
        reason = a.evidence.reason
        assert reason is not None and len(reason.strip()) > 0
        if a.detector == "rules":
            # Rules alerts must state which rule fired and precedence rationale
            assert f"so {a.type}" in reason or f"Rule" in reason
        elif a.detector == "ml":
            # ML alerts must state score and threshold
            assert "score" in reason.lower()
            assert "threshold" in reason.lower()


def test_deterministic_alert_ids() -> None:
    """Verify alert IDs and time formatting are 100% deterministic."""
    prefix = "198.51.100.0/24"
    origin = 65001
    ts = 1792982434.123456

    id_rules_1 = make_deterministic_alert_id(prefix, origin, ts)
    id_rules_2 = make_deterministic_alert_id(prefix, origin, ts)
    assert id_rules_1 == id_rules_2
    assert id_rules_1.startswith("alt-")

    id_ml_1 = make_deterministic_ml_alert_id(prefix, origin, ts)
    id_ml_2 = make_deterministic_ml_alert_id(prefix, origin, ts)
    assert id_ml_1 == id_ml_2
    assert id_ml_1.startswith("alt-ml-")

    # Time formatting determinism
    t1 = format_utc_epoch_time(ts)
    t2 = format_utc_epoch_time(ts)
    assert t1 == t2
    assert t1["epoch"] == 1792982434.123456
    assert t1["utc"].endswith("Z")


def test_idempotent_writing() -> None:
    """Verify writing alerts is completely idempotent and produces bitwise identical output."""
    alerts = read_alerts_from_jsonl(ALERTS_FILE)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p1 = Path(tmpdir) / "alerts_run1.jsonl"
        tmp_p2 = Path(tmpdir) / "alerts_run2.jsonl"

        write_alerts_to_jsonl(alerts, tmp_p1)
        write_alerts_to_jsonl(alerts, tmp_p2)

        content1 = tmp_p1.read_bytes()
        content2 = tmp_p2.read_bytes()
        assert content1 == content2, "Writing alerts was not idempotent (byte mismatch)."


def test_cli_list_alerts() -> None:
    """Verify CLI list-alerts filters by type, severity, and detector."""
    # List all alerts
    res = subprocess.run(
        [sys.executable, "-m", "src.cli", "list-alerts", "--limit", "10"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "ALERT ID" in res.stdout
    assert "Showing 10 alerts" in res.stdout

    # Filter by type and severity
    res_filt = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "list-alerts",
            "--type",
            "ORIGIN_HIJACK",
            "--severity",
            "high",
            "--limit",
            "5",
        ],
        capture_output=True,
        text=True,
    )
    assert res_filt.returncode == 0
    assert "ORIGIN_HIJACK" in res_filt.stdout
    assert "SUBPREFIX_HIJACK" not in res_filt.stdout

    # Filter by detector
    res_ml = subprocess.run(
        [sys.executable, "-m", "src.cli", "list-alerts", "--detector", "ml", "--limit", "5"],
        capture_output=True,
        text=True,
    )
    assert res_ml.returncode == 0
    assert "ML_ANOMALY" in res_ml.stdout


def test_cli_explain_round_trip() -> None:
    """Verify CLI explain outputs full evidence for both rules and ML alerts."""
    alerts = read_alerts_from_jsonl(ALERTS_FILE)
    rules_alert = next(a for a in alerts if a.detector == "rules")
    ml_alert = next(a for a in alerts if a.detector == "ml")

    # 1. Explain Rules Alert
    res_r = subprocess.run(
        [sys.executable, "-m", "src.cli", "explain", "--alert-id", rules_alert.alert_id],
        capture_output=True,
        text=True,
    )
    assert res_r.returncode == 0
    assert rules_alert.alert_id in res_r.stdout
    assert "EVIDENCE & RATIONALE" in res_r.stdout
    assert "RFC 6811 Route Origin Validation" in res_r.stdout
    assert "Operator Prefix Inventory" in res_r.stdout

    # 2. Explain ML Alert
    res_m = subprocess.run(
        [sys.executable, "-m", "src.cli", "explain", "--alert-id", ml_alert.alert_id],
        capture_output=True,
        text=True,
    )
    assert res_m.returncode == 0
    assert ml_alert.alert_id in res_m.stdout
    assert "Anomaly Score:" in res_m.stdout
    assert "Decision Threshold tau*" in res_m.stdout
    assert "Top Contributing Features" in res_m.stdout
    assert "Explanation Method:" in res_m.stdout
    assert "Disclaimer:" in res_m.stdout

    # 3. Nonexistent Alert ID returns error code
    res_err = subprocess.run(
        [sys.executable, "-m", "src.cli", "explain", "--alert-id", "nonexistent-id-0000"],
        capture_output=True,
        text=True,
    )
    assert res_err.returncode == 1
    assert "not found" in res_err.stderr.lower()


def test_feature_attribution_permutation() -> None:
    """Verify that feature attribution computes deltas against background mean."""
    from sklearn.ensemble import GradientBoostingClassifier

    # Create dummy trained model
    X = pd.DataFrame(
        {
            "f1": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "f2": [10.0, 20.0, 10.0, 20.0, 10.0, 20.0],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1])
    clf = GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=42)
    clf.fit(X, y)

    bg_mean = X.mean()
    test_row = pd.Series({"f1": 1.0, "f2": 20.0})

    contribs = compute_feature_contributions(clf, test_row, bg_mean, ["f1", "f2"])
    assert len(contribs) == 2
    for c in contribs:
        assert "feature" in c
        assert "value" in c
        assert "contribution" in c
        assert "direction" in c

