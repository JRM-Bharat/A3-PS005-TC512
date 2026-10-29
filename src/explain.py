"""Phase 6: Explainable Alerts for Rules Baseline and ML Detector.

Implements rich evidence extraction and explanation generation for both detectors:
  1. Rules Alerts:
     - Full evidence: covering ROAs, snapshot time, RFC 6811 state, invalid reason,
       inventory match (prefix, owner, allowed origins), peer, and plain-English rationale.
     - Strict rule precedence rationale: states which rule fired and why earlier
       rules in the precedence table did not.
  2. ML Alerts:
     - Full evidence: covering ROAs, snapshot time, RFC 6811 state, invalid reason,
       inventory match, peer, anomaly score, operating threshold tau*, and top 3
       contributing features with values and direction.
     - Model-appropriate feature attribution via marginal permutation against empirical
       background training mean (non-causal attribution).
  3. Deterministic Alert IDs:
     - Hash-derived IDs ensuring reproducible and idempotent alert generation.
  4. JSONL Export:
     - Every alert from BOTH detectors written as one JSON line to reports/alerts.jsonl.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.features import FEATURE_NAMES, RollingHistory, extract_features_from_feed
from src.inventory import InventoryItem, PrefixInventory
from src.replay import BGPReplay, BGPUpdate
from src.roa_store import ROARecord, ROASnapshot, ROAStore
from src.rov import ROVResult, ROVState, validate_route
from src.rules import Alert, RuleBasedDetector, make_deterministic_alert_id
from src.train_ml import time_split_df


def format_utc_epoch_time(timestamp: float) -> Dict[str, Any]:
    """Format Unix timestamp into ISO 8601 UTC and epoch dictionary."""
    dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
    utc_str = dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return {
        "utc": utc_str,
        "epoch": round(float(timestamp), 6),
    }


def make_deterministic_ml_alert_id(prefix: str, origin_asn: int, timestamp: float) -> str:
    """Generate deterministic alert ID for ML anomalies."""
    payload = f"ml_{prefix}_{origin_asn}_{timestamp:.6f}"
    return f"alt-ml-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:8]}"


@dataclass
class ExplainableEvidence:
    """Structured evidence backing an anomaly alert."""

    covering_roas: List[Dict[str, Any]] = field(default_factory=list)
    roa_snapshot_time: Optional[float] = None
    rfc6811_state: str = "NotFound"
    invalid_reason: Optional[str] = None
    inventory_match: Optional[Dict[str, Any]] = None
    peer: str = "unknown"
    reason: str = ""
    # ML-specific attribution fields
    score: Optional[float] = None
    threshold: Optional[float] = None
    top_contributing_features: Optional[List[Dict[str, Any]]] = None
    explanation_method: Optional[str] = None
    causal_disclaimer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert evidence to dictionary, omitting null ML fields for rules alerts."""
        d: Dict[str, Any] = {
            "covering_roas": self.covering_roas,
            "roa_snapshot_time": self.roa_snapshot_time,
            "rfc6811_state": self.rfc6811_state,
            "invalid_reason": self.invalid_reason,
            "inventory_match": self.inventory_match,
            "peer": self.peer,
            "reason": self.reason,
        }
        if self.score is not None:
            d["score"] = self.score
        if self.threshold is not None:
            d["threshold"] = self.threshold
        if self.top_contributing_features is not None:
            d["top_contributing_features"] = self.top_contributing_features
        if self.explanation_method is not None:
            d["explanation_method"] = self.explanation_method
        if self.causal_disclaimer is not None:
            d["causal_disclaimer"] = self.causal_disclaimer
        return d


@dataclass
class ExplainableAlert:
    """Full explainable alert representing an anomaly detection event."""

    alert_id: str
    time: Dict[str, Any]
    type: str
    prefix: str
    origin_asn: int
    severity: str
    detector: str
    evidence: ExplainableEvidence
    score: Optional[float] = None
    threshold: Optional[float] = None
    suppressed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to a JSON-serializable dictionary matching Phase 6 specification."""
        d: Dict[str, Any] = {
            "alert_id": self.alert_id,
            "time": self.time,
            "type": self.type,
            "prefix": self.prefix,
            "origin_asn": self.origin_asn,
            "severity": self.severity,
            "detector": self.detector,
            "evidence": self.evidence.to_dict(),
        }
        if self.score is not None:
            d["score"] = self.score
        if self.threshold is not None:
            d["threshold"] = self.threshold
        if self.suppressed_count > 0:
            d["suppressed_count"] = self.suppressed_count
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExplainableAlert:
        """Construct ExplainableAlert from dictionary."""
        ev_data = dict(data.get("evidence", {}))
        evidence = ExplainableEvidence(
            covering_roas=ev_data.get("covering_roas", []),
            roa_snapshot_time=ev_data.get("roa_snapshot_time"),
            rfc6811_state=ev_data.get("rfc6811_state", "NotFound"),
            invalid_reason=ev_data.get("invalid_reason"),
            inventory_match=ev_data.get("inventory_match"),
            peer=ev_data.get("peer", "unknown"),
            reason=ev_data.get("reason", ""),
            score=ev_data.get("score"),
            threshold=ev_data.get("threshold"),
            top_contributing_features=ev_data.get("top_contributing_features"),
            explanation_method=ev_data.get("explanation_method"),
            causal_disclaimer=ev_data.get("causal_disclaimer"),
        )
        return cls(
            alert_id=data["alert_id"],
            time=data["time"],
            type=data["type"],
            prefix=data["prefix"],
            origin_asn=data["origin_asn"],
            severity=data["severity"],
            detector=data["detector"],
            evidence=evidence,
            score=data.get("score"),
            threshold=data.get("threshold"),
            suppressed_count=data.get("suppressed_count", 0),
        )


def generate_rule_precedence_reason(
    alert_type: str,
    prefix_str: str,
    origin_asn: int,
    rov_state: str,
    invalid_reason: Optional[str],
    inv_match: Optional[InventoryItem],
    is_more_specific: bool,
    origin_in_inventory: bool,
) -> str:
    """Generate a single plain-English sentence explaining which rule fired and why earlier rules did not."""
    if alert_type == "ORIGIN_HIJACK":
        return (
            f"Origin AS{origin_asn} not in inventory for exact prefix {prefix_str}, "
            f"ROV {rov_state}, so ORIGIN_HIJACK "
            f"(Rule 1 fired on exact match; pass-through skipped due to unauthorized origin and non-valid ROV)."
        )

    if alert_type == "SUBPREFIX_HIJACK":
        parent_info = f" under {inv_match.prefix}" if inv_match else ""
        return (
            f"Origin AS{origin_asn} not in inventory{parent_info}, "
            f"ROV {rov_state}, so SUBPREFIX_HIJACK "
            f"(Rule 2 fired on subprefix; Rule 1 skipped: announced prefix is more-specific, not exact match)."
        )

    if alert_type == "INVENTORY_MISMATCH":
        return (
            f"Origin AS{origin_asn} not in inventory, "
            f"ROV Valid, so INVENTORY_MISMATCH "
            f"(Rule 3 fired; Rules 1-2 skipped: ROV state is Valid, indicating uncoordinated ROA or stale inventory)."
        )

    if alert_type == "ROA_INVALID":
        inv_desc = invalid_reason or "invalid"
        if origin_in_inventory:
            return (
                f"Origin AS{origin_asn} authorized in inventory but ROV Invalid ({inv_desc}), "
                f"so ROA_INVALID "
                f"(Rule 4 fired; Rules 1-3 skipped: origin authorized in inventory, indicating ROA update lag)."
            )
        return (
            f"Prefix in inventory, ROV Invalid ({inv_desc}), "
            f"so ROA_INVALID (Rule 4 fired; Rules 1-3 did not fire)."
        )

    if alert_type == "THIRD_PARTY_INVALID":
        inv_desc = invalid_reason or "invalid"
        return (
            f"Prefix outside operator inventory, ROV Invalid ({inv_desc}), "
            f"so THIRD_PARTY_INVALID (Rule 5 fired; Rules 1-4 skipped: unmonitored external route)."
        )

    if alert_type == "ROA_MISSING":
        return (
            f"Origin AS{origin_asn} authorized in inventory, ROV NotFound, "
            f"so ROA_MISSING (Rule 6 fired; Rules 1-4 skipped: origin authorized and no conflicting ROA)."
        )

    return f"Rule {alert_type} fired on announcement of {prefix_str} from AS{origin_asn}."


def extract_covering_roas_and_snapshot_time(
    prefix: Any,
    timestamp: float,
    roa_store: ROAStore,
) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    """Retrieve covering ROA records and snapshot time for an announced prefix."""
    snapshot: Optional[ROASnapshot] = roa_store.get_snapshot_for_time(timestamp, fallback_to_earliest=False)
    if snapshot is None:
        return [], None

    records = snapshot.find_covering_roas(prefix)
    roas_json = [
        {
            "prefix": str(r.prefix),
            "asn": r.asn,
            "maxLength": r.max_length,
        }
        for r in records
    ]
    return roas_json, snapshot.snapshot_time


def explain_rule_alert(
    alert: Alert,
    update: BGPUpdate,
    inventory: PrefixInventory,
    roa_store: ROAStore,
) -> ExplainableAlert:
    """Build a fully explainable alert from a rule detector Alert and its BGPUpdate."""
    prefix_str = str(update.prefix)
    origin_asn = update.origin_asn
    timestamp = update.timestamp
    peer = update.peer

    # 1. Covering ROAs and Snapshot Time
    covering_roas, snap_time = extract_covering_roas_and_snapshot_time(
        update.prefix, timestamp, roa_store
    )

    # 2. ROV State & Reason
    rov_res = alert.rov_result or validate_route(update.prefix, origin_asn, timestamp, roa_store)
    rov_state_val = rov_res.state.value if rov_res else "NotFound"
    inv_reason = rov_res.invalid_reason if rov_res else None

    # 3. Inventory Match
    inv_exact = inventory.lookup_exact(update.prefix)
    inv_covering = inventory.lookup_covering(update.prefix)
    inv_item = alert.inventory_match or inv_exact or inv_covering
    is_more_specific = inv_covering is not None and update.prefix.prefixlen > inv_covering.prefix.prefixlen
    origin_in_inventory = inventory.is_origin_allowed(update.prefix, origin_asn)

    inv_match_dict: Optional[Dict[str, Any]] = None
    if inv_item is not None:
        inv_match_dict = {
            "prefix": str(inv_item.prefix),
            "owner": inv_item.owner,
            "allowed_origins": sorted(list(inv_item.expected_origin_asns)),
        }

    # 4. Strict Precedence Rationale
    reason = generate_rule_precedence_reason(
        alert_type=alert.type,
        prefix_str=prefix_str,
        origin_asn=origin_asn,
        rov_state=rov_state_val,
        invalid_reason=inv_reason,
        inv_match=inv_item,
        is_more_specific=is_more_specific,
        origin_in_inventory=origin_in_inventory,
    )

    evidence = ExplainableEvidence(
        covering_roas=covering_roas,
        roa_snapshot_time=snap_time,
        rfc6811_state=rov_state_val,
        invalid_reason=inv_reason,
        inventory_match=inv_match_dict,
        peer=peer,
        reason=reason,
    )

    return ExplainableAlert(
        alert_id=alert.alert_id,
        time=format_utc_epoch_time(timestamp),
        type=alert.type,
        prefix=prefix_str,
        origin_asn=origin_asn,
        severity=alert.severity,
        detector="rules",
        evidence=evidence,
        suppressed_count=alert.suppressed_count,
    )


def compute_feature_contributions(
    model: GradientBoostingClassifier,
    row: pd.Series,
    background_mean: pd.Series,
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    """Compute per-feature marginal contribution against empirical background training mean.

    Method:
      Evaluates delta in predicted anomaly probability when feature i is replaced with
      the empirical mean of the training background distribution:
        Delta_i = P(y=1 | x) - P(y=1 | x_{-i}, mean_i)
      Delta_i > 0 indicates observed feature value increases predicted anomaly risk (+).
      Delta_i < 0 indicates observed feature value decreases predicted anomaly risk (-).
    """
    row_df = pd.DataFrame([row[feature_names]], columns=feature_names)
    base_score = float(model.predict_proba(row_df)[0, 1])

    contributions: List[Dict[str, Any]] = []
    for f in feature_names:
        perturbed_df = row_df.copy()
        perturbed_df.iloc[0, perturbed_df.columns.get_loc(f)] = background_mean[f]
        perturbed_score = float(model.predict_proba(perturbed_df)[0, 1])
        delta = base_score - perturbed_score

        if delta > 1e-4:
            direction = "+"
        elif delta < -1e-4:
            direction = "-"
        else:
            direction = "0"

        contributions.append({
            "feature": f,
            "value": round(float(row[f]), 4),
            "contribution": round(delta, 4),
            "direction": direction,
        })

    # Rank features: prioritize highest positive delta (factors pushing score towards anomaly)
    contributions.sort(key=lambda x: x["contribution"], reverse=True)
    return contributions


def explain_ml_alert(
    update: BGPUpdate,
    feature_row: pd.Series,
    model: GradientBoostingClassifier,
    background_mean: pd.Series,
    threshold: float,
    inventory: PrefixInventory,
    roa_store: ROAStore,
) -> ExplainableAlert:
    """Build an explainable ML alert with feature contributions, covering ROAs, and non-causal attribution."""
    prefix_str = str(update.prefix)
    origin_asn = update.origin_asn
    timestamp = update.timestamp
    peer = update.peer

    # 1. Anomaly Score & Attribution
    row_df = pd.DataFrame([feature_row[FEATURE_NAMES]], columns=FEATURE_NAMES)
    score = float(model.predict_proba(row_df)[0, 1])
    contributions = compute_feature_contributions(model, feature_row, background_mean, FEATURE_NAMES)
    top3 = contributions[:3]

    # 2. Covering ROAs and Snapshot Time
    covering_roas, snap_time = extract_covering_roas_and_snapshot_time(
        update.prefix, timestamp, roa_store
    )

    # 3. RFC 6811 State
    rov_res = validate_route(update.prefix, origin_asn, timestamp, roa_store)
    rov_state_val = rov_res.state.value if rov_res else "NotFound"
    inv_reason = rov_res.invalid_reason if rov_res else None

    # 4. Inventory Match
    inv_exact = inventory.lookup_exact(update.prefix)
    inv_covering = inventory.lookup_covering(update.prefix)
    inv_item = inv_exact or inv_covering
    inv_match_dict: Optional[Dict[str, Any]] = None
    if inv_item is not None:
        inv_match_dict = {
            "prefix": str(inv_item.prefix),
            "owner": inv_item.owner,
            "allowed_origins": sorted(list(inv_item.expected_origin_asns)),
        }

    # 5. Plain-English Reason
    f_desc = ", ".join(f"{item['feature']} ({item['value']:.4g}, {item['direction']})" for item in top3)
    reason = (
        f"ML model assigned anomaly score {score:.3f} exceeding threshold {threshold:.3f}, "
        f"primarily driven by {f_desc} (non-causal statistical attribution)."
    )

    method_text = (
        "Per-feature contribution against empirical background mean (delta in model output "
        "probability when feature replaced with background baseline; non-causal attribution)"
    )
    disclaimer_text = (
        "Attribution reflects statistical association within the trained model and does not establish "
        "causal relationships."
    )

    evidence = ExplainableEvidence(
        covering_roas=covering_roas,
        roa_snapshot_time=snap_time,
        rfc6811_state=rov_state_val,
        invalid_reason=inv_reason,
        inventory_match=inv_match_dict,
        peer=peer,
        reason=reason,
        score=round(score, 4),
        threshold=round(threshold, 4),
        top_contributing_features=top3,
        explanation_method=method_text,
        causal_disclaimer=disclaimer_text,
    )

    alert_id = make_deterministic_ml_alert_id(prefix_str, origin_asn, timestamp)

    return ExplainableAlert(
        alert_id=alert_id,
        time=format_utc_epoch_time(timestamp),
        type="ML_ANOMALY",
        prefix=prefix_str,
        origin_asn=origin_asn,
        severity="high",
        detector="ml",
        evidence=evidence,
        score=round(score, 4),
        threshold=round(threshold, 4),
    )


def write_alerts_to_jsonl(alerts: Sequence[ExplainableAlert], output_path: Union[str, Path]) -> None:
    """Write explainable alerts to JSONL file idempotently and deterministically."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Sort deterministically by timestamp epoch, then alert_id
    sorted_alerts = sorted(
        alerts,
        key=lambda a: (a.time.get("epoch", 0.0), a.alert_id),
    )

    with open(out, "w", encoding="utf-8") as f:
        for alt in sorted_alerts:
            line = json.dumps(alt.to_dict()) + "\n"
            f.write(line)


def read_alerts_from_jsonl(path: Union[str, Path] = "reports/alerts.jsonl") -> List[ExplainableAlert]:
    """Read explainable alerts from a JSONL file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Alerts file '{path}' does not exist.")

    alerts: List[ExplainableAlert] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                alerts.append(ExplainableAlert.from_dict(json.loads(line)))
    return alerts


def generate_explainable_alerts(
    config_path: str = "config_ml.yaml",
    output_path: Optional[str] = None,
) -> List[ExplainableAlert]:
    """Execute pipeline for BOTH detectors and output comprehensive explainable alerts.

    Runs:
      1. Baseline Rule Detector on the replay feed with RFC 6811 validation & inventory checks.
      2. ML Anomaly Detector on the held-out test slice using the Phase 5 GradientBoosting model
         with threshold tau* = 0.05 and background-mean feature attribution.
      3. Combines all alerts and writes to reports/alerts.jsonl.
    """
    cfg_file = Path(config_path)
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    feed_path = paths.get("feed_replay", "data/ml_feed_replay.csv")
    labels_path = paths.get("labels", "data/ml_labels.csv")
    inv_path = paths.get("inventory", "data/ml_inventory.csv")
    roas_dir = paths.get("roa_snapshots_dir", "data/ml_roa_snapshots")
    out_file = Path(output_path or paths.get("alerts_jsonl", "reports/alerts.jsonl"))

    print(f"Loading environment from {cfg_file}...")
    inv = PrefixInventory.load_from_csv(inv_path)
    store = ROAStore(roas_dir)
    replay = BGPReplay.load_from_csv(feed_path)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Rule-Based Detector Alerts
    # ─────────────────────────────────────────────────────────────────────────
    print("Processing Rule-Based Detector announcements...")
    rule_detector = RuleBasedDetector(inventory=inv, roa_store=store, dedup_window_seconds=300.0)
    rule_explainable_alerts: List[ExplainableAlert] = []

    for upd in replay.stream():
        alt = rule_detector.process_update(upd)
        if alt is not None:
            exp_alt = explain_rule_alert(
                alert=alt,
                update=upd,
                inventory=inv,
                roa_store=store,
            )
            rule_explainable_alerts.append(exp_alt)

    print(f"  Emitted {len(rule_explainable_alerts)} rule-based explainable alerts.")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Machine Learning Anomaly Detector Alerts
    # ─────────────────────────────────────────────────────────────────────────
    print("Extracting features and training ML model (seed 777)...")
    feature_df = extract_features_from_feed(feed_path, labels_path, inv_path, roas_dir)
    train_df, val_df, test_df = time_split_df(feature_df, train_frac=0.60, val_frac=0.20)

    X_train = train_df[FEATURE_NAMES]
    y_train = train_df["label"]
    sw = compute_sample_weight("balanced", y_train)

    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=777,
    )
    clf.fit(X_train, y_train, sample_weight=sw)

    # Background training mean for feature attribution
    bg_mean = X_train.mean()
    tau_star = 0.0500  # Operating threshold selected on validation slice in Phase 5

    # Run ML detection on test slice
    test_start_t = float(test_df["timestamp"].min())
    test_announcements = [
        u for u in replay.updates if u.timestamp >= test_start_t and u.is_announcement
    ]
    test_features_df = test_df[FEATURE_NAMES].reset_index(drop=True)
    test_probs = clf.predict_proba(test_features_df)[:, 1]

    ml_explainable_alerts: List[ExplainableAlert] = []
    last_ml_alert_time: Dict[Tuple[str, int], float] = {}

    for idx, (upd, prob) in enumerate(zip(test_announcements, test_probs)):
        if prob >= tau_star:
            dedup_key = (str(upd.prefix), upd.origin_asn)
            if upd.timestamp - last_ml_alert_time.get(dedup_key, -1e9) > 300.0:
                feat_row = test_features_df.iloc[idx]
                exp_ml = explain_ml_alert(
                    update=upd,
                    feature_row=feat_row,
                    model=clf,
                    background_mean=bg_mean,
                    threshold=tau_star,
                    inventory=inv,
                    roa_store=store,
                )
                ml_explainable_alerts.append(exp_ml)
                last_ml_alert_time[dedup_key] = upd.timestamp

    print(f"  Emitted {len(ml_explainable_alerts)} ML explainable alerts on held-out test slice.")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Combine and Write to reports/alerts.jsonl
    # ─────────────────────────────────────────────────────────────────────────
    all_alerts = rule_explainable_alerts + ml_explainable_alerts
    write_alerts_to_jsonl(all_alerts, out_file)
    print(f"Successfully wrote {len(all_alerts)} explainable alerts to {out_file}.")
    return all_alerts


if __name__ == "__main__":
    generate_explainable_alerts("config_ml.yaml")

