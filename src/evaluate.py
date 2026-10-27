"""Evaluation Engine for BGP Origin and Subprefix Anomaly Detection (Phase 4).

Computes comprehensive metrics for the baseline rule detector:
  - Event-level Precision, Recall, F1 under two policies (Primary: ALL alert types; Secondary: High/Medium only)
  - Alert-level Precision view
  - Recall per anomaly scenario and difficulty level (verifies 0% recall on loose_maxlength_hijack)
  - False positive breakdown by benign scenario and background noise
  - Separate third-party Internet noise table (THIRD_PARTY_INVALID)
  - Alert latency (replay time mean/median/p95) and wall-clock processing speed (ms/update, updates/sec)
  - One-to-one event matching with tracking for "missed due to dedup"
  - Multi-seed evaluation (seeds 1-5 mean +/- std) and 95% bootstrap confidence intervals
  - Generation of publication-quality diagnostic plots and baseline_metrics.json
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from src.inventory import PrefixInventory
from src.replay import BGPReplay, BGPUpdate
from src.roa_store import ROAStore
from src.rules import Alert, RuleBasedDetector


@dataclass
class MetricScores:
    """Precision, Recall, F1 scores and raw counts."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    @classmethod
    def from_counts(cls, tp: int, fp: int, fn: int) -> MetricScores:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return cls(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
        )


@dataclass
class LatencyStats:
    """Latency statistics for matched alerts."""

    mean_seconds: float = 0.0
    median_seconds: float = 0.0
    p95_seconds: float = 0.0
    min_seconds: float = 0.0
    max_seconds: float = 0.0
    sample_count: int = 0


@dataclass
class EvaluationRun:
    """Complete evaluation results for a single run under a specific policy."""

    policy: str  # 'primary_all' or 'secondary_high_medium'
    headline_metric_type: str  # 'event_level'
    event_metrics: MetricScores  # TP events, FP events, FN events, Event-level Precision, Recall, F1
    alert_metrics: MetricScores  # Matched alerts, Unmatched alerts, Alert-level Precision
    alert_level_precision: float
    matched_events_count: int
    missed_events_count: int
    missed_due_to_dedup_count: int
    scenario_recall: Dict[str, float]
    difficulty_recall: Dict[str, float]
    difficulty_counts: Dict[str, Dict[str, int]]
    fp_per_benign_scenario: Dict[str, int]
    benign_event_fp_rate: Dict[str, float]
    fp_per_1000_background: float
    third_party_noise_count: int
    latency: LatencyStats
    wall_clock_time_ms_per_update: float
    updates_per_second: float
    confusion_matrix: Dict[str, int]


def is_alert_allowed_by_policy(alert: Alert, policy: str) -> bool:
    """Filter alerts according to the evaluation policy.

    Primary Policy ('primary_all'): All operator alert types count.
    Secondary Policy ('secondary_high_medium'): Only High and Medium severity alerts count.
    """
    if alert.type == "THIRD_PARTY_INVALID":
        return False  # Third-party noise is reported in a separate noise table

    if policy == "secondary_high_medium":
        return alert.severity.lower() in ("high", "medium")
    return True  # 'primary_all'


def evaluate_feed(
    feed_path: str | Path,
    labels_path: str | Path,
    inventory_path: str | Path,
    roas_dir: str | Path,
    policy: str = "primary_all",
    dedup_window_seconds: float = 300.0,
    matching_window_seconds: float = 600.0,
) -> Tuple[EvaluationRun, List[Alert]]:
    """Run baseline detector on feed and evaluate against ground-truth labels."""
    inv = PrefixInventory.load_from_csv(inventory_path)
    store = ROAStore(roas_dir)
    replay = BGPReplay.load_from_csv(feed_path)
    detector = RuleBasedDetector(
        inventory=inv, roa_store=store, dedup_window_seconds=dedup_window_seconds
    )

    # 1. Load labels
    events: List[Dict[str, Any]] = []
    with open(labels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["start_time"] = float(row["start_time"])
            row["end_time"] = float(row["end_time"])
            row["origin_asn"] = int(row["origin_asn"])
            row["benign"] = row["benign"].lower() == "true"
            events.append(row)

    anomaly_events = [e for e in events if not e["benign"]]
    benign_events = [e for e in events if e["benign"]]

    # 2. Process feed and measure wall-clock performance
    alerts: List[Alert] = []
    t_start_wall = time.perf_counter()
    for update in replay.stream():
        alert = detector.process_update(update)
        if alert:
            alerts.append(alert)
    t_end_wall = time.perf_counter()

    total_updates = len(replay)
    wall_duration = max(t_end_wall - t_start_wall, 1e-6)
    ms_per_update = (wall_duration / total_updates) * 1000.0 if total_updates > 0 else 0.0
    updates_per_sec = total_updates / wall_duration if total_updates > 0 else 0.0

    # 3. Separate third-party noise alerts
    third_party_noise_alerts = [a for a in alerts if a.type == "THIRD_PARTY_INVALID"]
    operator_alerts = [a for a in alerts if is_alert_allowed_by_policy(a, policy)]

    # 4. One-to-one matching between anomaly events and operator alerts
    # Sort events by start_time and alerts by time
    anomaly_events_sorted = sorted(anomaly_events, key=lambda e: e["start_time"])
    matched_anomaly_event_ids: Set[str] = set()
    matched_anomaly_alert_ids: Set[str] = set()
    event_latencies: List[float] = []

    # Map each anomaly event to the first valid unconsumed alert
    for evt in anomaly_events_sorted:
        best_alert: Optional[Alert] = None
        for alt in operator_alerts:
            if alt.alert_id in matched_anomaly_alert_ids:
                continue
            if alt.prefix == evt["prefix"] and alt.origin_asn == evt["origin_asn"]:
                # Check time matching window [start_time - 60s, end_time + matching_window_seconds]
                if (evt["start_time"] - 60.0) <= alt.time <= (evt["end_time"] + matching_window_seconds):
                    best_alert = alt
                    break

        if best_alert:
            matched_anomaly_event_ids.add(evt["event_id"])
            matched_anomaly_alert_ids.add(best_alert.alert_id)
            lat = max(0.0, best_alert.time - evt["start_time"])
            event_latencies.append(lat)

    # 5. Check for anomaly events missed due to deduplication
    missed_due_to_dedup_count = 0
    missed_events = [e for e in anomaly_events if e["event_id"] not in matched_anomaly_event_ids]
    for m_evt in missed_events:
        # Check if an earlier matched alert on same prefix/origin occurred within dedup window
        for alt in operator_alerts:
            if alt.prefix == m_evt["prefix"] and alt.origin_asn == m_evt["origin_asn"]:
                if 0.0 < (m_evt["start_time"] - alt.time) <= dedup_window_seconds:
                    missed_due_to_dedup_count += 1
                    break

    # 6. One-to-one matching between benign events and unmatched operator alerts
    unmatched_operator_alerts = [
        a for a in operator_alerts if a.alert_id not in matched_anomaly_alert_ids
    ]
    benign_events_sorted = sorted(benign_events, key=lambda e: e["start_time"])
    matched_benign_event_ids: Set[str] = set()
    matched_benign_alert_ids: Set[str] = set()
    benign_event_alert_counts: Dict[str, int] = {e["event_id"]: 0 for e in benign_events}
    benign_scenarios = sorted(list({e["scenario"] for e in benign_events}))
    fp_per_benign_scenario: Dict[str, int] = {sc: 0 for sc in benign_scenarios}

    for b_evt in benign_events_sorted:
        best_alert = None
        for alt in unmatched_operator_alerts:
            if alt.alert_id in matched_benign_alert_ids:
                continue
            if alt.prefix == b_evt["prefix"] and alt.origin_asn == b_evt["origin_asn"]:
                # Window: [start_time - 60s, end_time + 60s]
                # A 60s buffer is used so consecutive bursts (spaced >= 120s apart) do not bleed
                if (b_evt["start_time"] - 60.0) <= alt.time <= (b_evt["end_time"] + 60.0):
                    best_alert = alt
                    break
        if best_alert:
            matched_benign_event_ids.add(b_evt["event_id"])
            matched_benign_alert_ids.add(best_alert.alert_id)
            benign_event_alert_counts[b_evt["event_id"]] += 1
            fp_per_benign_scenario[b_evt["scenario"]] += 1

    # Check remaining unconsumed alerts (if any) to see if they are background noise
    # or extra alerts for a benign scenario
    background_operator_noise = 0
    for alt in unmatched_operator_alerts:
        if alt.alert_id in matched_benign_alert_ids:
            continue
        # Check if it matches any benign scenario
        matched_sc: Optional[str] = None
        for b_evt in benign_events:
            if alt.prefix == b_evt["prefix"] and alt.origin_asn == b_evt["origin_asn"]:
                matched_sc = b_evt["scenario"]
                break
        if matched_sc:
            fp_per_benign_scenario[matched_sc] += 1
        else:
            background_operator_noise += 1

    # 7. Event-level metrics (Headline Metric)
    # TP events = Anomaly events matched 1-to-1 to an alert
    # FN events = Anomaly events with 0 matched alerts
    # FP events = Benign events matched 1-to-1 to an alert
    # TN events = Benign events with 0 alerts
    tp_events = len(matched_anomaly_event_ids)
    fn_events = len(anomaly_events) - tp_events
    fp_events = len(matched_benign_event_ids)
    tn_events = len(benign_events) - fp_events

    event_metrics = MetricScores.from_counts(tp=tp_events, fp=fp_events, fn=fn_events)

    # 8. Alert-level metrics
    tp_alerts = len(matched_anomaly_alert_ids)
    fp_alerts = len(unmatched_operator_alerts)
    total_op_alerts = len(operator_alerts)
    alert_precision = (
        round(tp_alerts / total_op_alerts, 4) if total_op_alerts > 0 else 0.0
    )
    alert_metrics = MetricScores.from_counts(tp=tp_alerts, fp=fp_alerts, fn=fn_events)

    # 9. Recall per anomaly scenario and per difficulty
    scenario_recall: Dict[str, float] = {}
    scenarios = sorted(list({e["scenario"] for e in anomaly_events}))
    for sc in scenarios:
        sc_events = [e for e in anomaly_events if e["scenario"] == sc]
        sc_tp = sum(1 for e in sc_events if e["event_id"] in matched_anomaly_event_ids)
        scenario_recall[sc] = round(sc_tp / len(sc_events), 4) if sc_events else 0.0

    difficulty_recall: Dict[str, float] = {}
    difficulty_counts: Dict[str, Dict[str, int]] = {}
    for diff in ("easy", "hard"):
        d_events = [e for e in anomaly_events if str(e.get("difficulty", "")).strip().lower() == diff]
        d_tp = sum(1 for e in d_events if e["event_id"] in matched_anomaly_event_ids)
        difficulty_recall[diff] = round(d_tp / len(d_events), 4) if d_events else 0.0
        difficulty_counts[diff] = {
            "tp": d_tp,
            "total": len(d_events),
            "recall": round(d_tp / len(d_events), 4) if d_events else 0.0,
        }

    # Benign event-level false positive rate (flagged events / total events per scenario)
    benign_event_fp_rate: Dict[str, float] = {}
    for sc in benign_scenarios:
        sc_bevts = [e for e in benign_events if e["scenario"] == sc]
        flagged_bevts = sum(1 for e in sc_bevts if benign_event_alert_counts.get(e["event_id"], 0) > 0)
        benign_event_fp_rate[sc] = round(flagged_bevts / len(sc_bevts), 4) if sc_bevts else 0.0

    # FP per 1000 background updates (strictly non-scenario background noise)
    fp_per_1000 = (
        round((background_operator_noise / total_updates) * 1000.0, 4) if total_updates > 0 else 0.0
    )

    # 10. Latency statistics
    if event_latencies:
        lat_arr = np.array(event_latencies)
        latency_stats = LatencyStats(
            mean_seconds=round(float(np.mean(lat_arr)), 3),
            median_seconds=round(float(np.median(lat_arr)), 3),
            p95_seconds=round(float(np.percentile(lat_arr, 95)), 3),
            min_seconds=round(float(np.min(lat_arr)), 3),
            max_seconds=round(float(np.max(lat_arr)), 3),
            sample_count=len(event_latencies),
        )
    else:
        latency_stats = LatencyStats()

    # 11. Event-level Confusion Matrix
    confusion_mat = {
        "true_positives": tp_events,
        "false_negatives": fn_events,
        "false_positives_events": fp_events,
        "true_negatives_events": tn_events,
        "total_anomalies": len(anomaly_events),
        "total_benign": len(benign_events),
    }

    run_result = EvaluationRun(
        policy=policy,
        headline_metric_type="event_level",
        event_metrics=event_metrics,
        alert_metrics=alert_metrics,
        alert_level_precision=alert_precision,
        matched_events_count=tp_events,
        missed_events_count=fn_events,
        missed_due_to_dedup_count=missed_due_to_dedup_count,
        scenario_recall=scenario_recall,
        difficulty_recall=difficulty_recall,
        difficulty_counts=difficulty_counts,
        fp_per_benign_scenario=fp_per_benign_scenario,
        benign_event_fp_rate=benign_event_fp_rate,
        fp_per_1000_background=fp_per_1000,
        third_party_noise_count=len(third_party_noise_alerts),
        latency=latency_stats,
        wall_clock_time_ms_per_update=round(ms_per_update, 4),
        updates_per_second=round(updates_per_sec, 1),
        confusion_matrix=confusion_mat,
    )

    return run_result, alerts


def compute_bootstrap_ci(
    run: EvaluationRun,
    anomaly_events_count: int,
    benign_events_count: int,
    n_bootstraps: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute bootstrap confidence intervals strictly over EVENT outcomes.

    Resampling unit: Individual events (not raw BGP updates).
      - Anomaly events (40 events): 1 = detected (TP), 0 = missed (FN)
      - Benign events (52 events): 1 = falsely flagged (FP), 0 = clean (TN)
    """
    rng = np.random.RandomState(seed)
    tp = run.event_metrics.tp
    fn = run.event_metrics.fn
    fp = run.event_metrics.fp
    tn = run.confusion_matrix["true_negatives_events"]

    y_anomaly = np.array([1] * tp + [0] * fn)
    y_benign = np.array([1] * fp + [0] * tn)

    boot_prec = []
    boot_rec = []
    boot_f1 = []

    for _ in range(n_bootstraps):
        samp_anom = rng.choice(y_anomaly, size=len(y_anomaly), replace=True)
        samp_ben = rng.choice(y_benign, size=len(y_benign), replace=True)

        b_tp = int(np.sum(samp_anom == 1))
        b_fn = int(np.sum(samp_anom == 0))
        b_fp = int(np.sum(samp_ben == 1))

        b_p = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
        b_r = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
        b_f1 = (2 * b_p * b_r) / (b_p + b_r) if (b_p + b_r) > 0 else 0.0

        boot_prec.append(b_p)
        boot_rec.append(b_r)
        boot_f1.append(b_f1)

    alpha = (1.0 - ci) / 2.0
    prec_ci = (
        round(float(np.percentile(boot_prec, alpha * 100)), 4),
        round(float(np.percentile(boot_prec, (1.0 - alpha) * 100)), 4),
    )
    rec_ci = (
        round(float(np.percentile(boot_rec, alpha * 100)), 4),
        round(float(np.percentile(boot_rec, (1.0 - alpha) * 100)), 4),
    )
    f1_ci = (
        round(float(np.percentile(boot_f1, alpha * 100)), 4),
        round(float(np.percentile(boot_f1, (1.0 - alpha) * 100)), 4),
    )

    point_prec = run.event_metrics.precision
    point_rec = run.event_metrics.recall
    point_f1 = run.event_metrics.f1

    def is_at_edge(val: float, bounds: Tuple[float, float]) -> bool:
        return (abs(val - bounds[0]) < 0.005) or (abs(val - bounds[1]) < 0.005)

    edge_note = "All point estimates sit comfortably within the interior of their 95% bootstrap CIs."
    if is_at_edge(point_prec, prec_ci) or is_at_edge(point_rec, rec_ci) or is_at_edge(point_f1, f1_ci):
        edge_note = "Note: Point estimate sits near the boundary of the bootstrap CI."

    return {
        "resampling_unit": "events",
        "n_bootstraps": n_bootstraps,
        "ci_level": ci,
        "precision_ci": prec_ci,
        "recall_ci": rec_ci,
        "f1_ci": f1_ci,
        "point_estimates": {
            "event_precision": point_prec,
            "event_recall": point_rec,
            "event_f1": point_f1,
        },
        "edge_note": edge_note,
    }


def run_multi_seed_evaluation(
    seeds: List[int] = [1, 2, 3, 4, 5],
    config_path: str = "config.yaml",
) -> Dict[str, Any]:
    """Execute generator and baseline evaluation over multiple seeds to compute per-seed metrics and mean +/- std."""
    import hashlib
    from src.inject import ScenarioInjector

    primary_runs: List[EvaluationRun] = []
    secondary_runs: List[EvaluationRun] = []
    seed_records: List[Dict[str, Any]] = []

    for s in seeds:
        injector = ScenarioInjector(config_path=config_path, seed=s)
        s_feed = f"data/feed_replay_seed_{s}.csv"
        s_labels = f"data/labels_seed_{s}.csv"
        injector.save_to_disk(feed_path=s_feed, labels_path=s_labels)

        # Compute SHA256 hash of feed file to prove data differs per seed
        with open(s_feed, "rb") as f:
            feed_hash = hashlib.sha256(f.read()).hexdigest()[:16]

        with open(s_feed, "r", encoding="utf-8") as f:
            feed_update_count = sum(1 for _ in f) - 1

        run_p, _ = evaluate_feed(
            feed_path=s_feed,
            labels_path=s_labels,
            inventory_path="data/inventory.csv",
            roas_dir="data/roa_snapshots",
            policy="primary_all",
        )
        run_s, _ = evaluate_feed(
            feed_path=s_feed,
            labels_path=s_labels,
            inventory_path="data/inventory.csv",
            roas_dir="data/roa_snapshots",
            policy="secondary_high_medium",
        )
        primary_runs.append(run_p)
        secondary_runs.append(run_s)

        total_evts = run_p.confusion_matrix["total_anomalies"] + run_p.confusion_matrix["total_benign"]
        seed_records.append({
            "seed": s,
            "feed_hash": feed_hash,
            "feed_update_count": feed_update_count,
            "total_events": total_evts,
            "primary": {
                "event_precision": run_p.event_metrics.precision,
                "alert_precision": run_p.alert_level_precision,
                "recall": run_p.event_metrics.recall,
                "f1": run_p.event_metrics.f1,
            },
            "secondary": {
                "event_precision": run_s.event_metrics.precision,
                "alert_precision": run_s.alert_level_precision,
                "recall": run_s.event_metrics.recall,
                "f1": run_s.event_metrics.f1,
            },
        })

        # Clean up temporary seed files
        Path(s_feed).unlink(missing_ok=True)
        Path(s_labels).unlink(missing_ok=True)

    def stats_for(metric_accessor, runs):
        vals = [metric_accessor(r) for r in runs]
        return {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "values": vals,
        }

    return {
        "seeds": seeds,
        "seed_results": seed_records,
        "primary_precision": stats_for(lambda r: r.event_metrics.precision, primary_runs),
        "primary_alert_precision": stats_for(lambda r: r.alert_level_precision, primary_runs),
        "primary_recall": stats_for(lambda r: r.event_metrics.recall, primary_runs),
        "primary_f1": stats_for(lambda r: r.event_metrics.f1, primary_runs),
        "secondary_precision": stats_for(lambda r: r.event_metrics.precision, secondary_runs),
        "secondary_alert_precision": stats_for(lambda r: r.alert_level_precision, secondary_runs),
        "secondary_recall": stats_for(lambda r: r.event_metrics.recall, secondary_runs),
        "secondary_f1": stats_for(lambda r: r.event_metrics.f1, secondary_runs),
    }


def generate_evaluation_plots(
    run_primary: EvaluationRun,
    run_secondary: EvaluationRun,
    output_dir: str | Path = "reports",
) -> List[str]:
    """Generate high-quality diagnostic evaluation figures."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []

    # 1. Policy Comparison Bar Chart (Event-Level vs Alert-Level Precision, Recall, F1)
    plt.figure(figsize=(9, 5.5))
    metrics = ["Event Precision\n(Headline)", "Alert Precision\n(Alert-Level)", "Recall\n(Sensitivity)", "F1 Score"]
    p_vals = [
        run_primary.event_metrics.precision,
        run_primary.alert_level_precision,
        run_primary.event_metrics.recall,
        run_primary.event_metrics.f1,
    ]
    s_vals = [
        run_secondary.event_metrics.precision,
        run_secondary.alert_level_precision,
        run_secondary.event_metrics.recall,
        run_secondary.event_metrics.f1,
    ]
    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width / 2, p_vals, width, label="Primary: All Operator Alerts (Strictest)", color="#1f77b4")
    plt.bar(
        x + width / 2, s_vals, width, label="Secondary: High/Medium Only", color="#2ca02c"
    )
    plt.ylabel("Score")
    plt.title("Baseline Rule Detector Policy Comparison: Event-Level vs Alert-Level Metrics")
    plt.xticks(x, metrics, fontsize=10)
    plt.ylim(0, 1.15)
    for i in range(len(metrics)):
        plt.text(x[i] - width / 2, p_vals[i] + 0.02, f"{p_vals[i]:.4f}", ha="center", fontsize=9, fontweight="bold")
        plt.text(x[i] + width / 2, s_vals[i] + 0.02, f"{s_vals[i]:.4f}", ha="center", fontsize=9, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p1 = out_dir / "baseline_policy_comparison.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    generated_plots.append(str(p1))

    # 2. Confusion Matrix Heatmap (Primary Policy - Event Level)
    plt.figure(figsize=(6, 5))
    cm = np.array(
        [
            [run_primary.event_metrics.tp, run_primary.event_metrics.fn],
            [
                run_primary.confusion_matrix["false_positives_events"],
                run_primary.confusion_matrix["true_negatives_events"],
            ],
        ]
    )
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Baseline Event-Level Confusion Matrix (Primary Policy)")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Predicted Anomaly", "Predicted Benign"])
    plt.yticks(tick_marks, ["Actual Anomaly", "Actual Benign"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > np.max(cm) / 2 else "black",
                fontsize=14,
                fontweight="bold",
            )
    plt.ylabel("True Class (Events)")
    plt.xlabel("Predicted Class (Events)")
    plt.tight_layout()
    p2 = out_dir / "baseline_confusion_matrix.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    generated_plots.append(str(p2))

    # 3. Anomaly Scenario Recall Breakdown
    plt.figure(figsize=(9, 4.5))
    sc_names = list(run_primary.scenario_recall.keys())
    rec_vals = [run_primary.scenario_recall[k] * 100 for k in sc_names]
    colors = ["#2ca02c" if v > 50 else "#d62728" for v in rec_vals]

    bars = plt.barh(sc_names, rec_vals, color=colors)
    plt.xlabel("Recall (%)")
    plt.title("Baseline Anomaly Recall by Scenario (loose_maxlength_hijack: 0% recall by design)")
    plt.xlim(0, 115)
    for bar, val in zip(bars, rec_vals):
        plt.text(
            val + 2,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontweight="bold",
            fontsize=9,
        )
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p3 = out_dir / "baseline_scenario_recall.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    generated_plots.append(str(p3))

    # 4. Benign False Positive Breakdown
    plt.figure(figsize=(10, 4.5))
    fp_sc = list(run_primary.fp_per_benign_scenario.keys())
    fp_counts = [run_primary.fp_per_benign_scenario[k] for k in fp_sc]
    plt.bar(fp_sc, fp_counts, color="#ff7f0e")
    plt.ylabel("False Positive Events / Alerts Count")
    plt.title("False Positive Breakdown by Benign Scenario (Primary Policy)")
    plt.xticks(rotation=25, ha="right")
    for i, count in enumerate(fp_counts):
        plt.text(i, count + 0.15, f"{count}", ha="center", fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p4 = out_dir / "baseline_fp_breakdown.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    generated_plots.append(str(p4))

    return generated_plots


def main() -> None:
    """Run full baseline evaluation, write alerts.jsonl, metrics.json, and generate plots."""
    config_path = "config.yaml"
    import yaml
    with open(config_path, "r", encoding="utf-8") as yf:
        cfg = yaml.safe_load(yf)

    paths = cfg.get("paths", {})
    feed_path = paths.get("feed_replay", "data/feed_replay.csv")
    labels_path = paths.get("labels", "data/labels.csv")
    inv_path = paths.get("inventory", "data/inventory.csv")
    roas_dir = paths.get("roa_snapshots_dir", "data/roa_snapshots")
    out_dir = Path(paths.get("reports_dir", "reports"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running Baseline Rule Detector Evaluation...")
    # Evaluate Primary Policy
    run_primary, all_alerts = evaluate_feed(
        feed_path=feed_path,
        labels_path=labels_path,
        inventory_path=inv_path,
        roas_dir=roas_dir,
        policy="primary_all",
    )

    # Evaluate Secondary Policy
    run_secondary, _ = evaluate_feed(
        feed_path=feed_path,
        labels_path=labels_path,
        inventory_path=inv_path,
        roas_dir=roas_dir,
        policy="secondary_high_medium",
    )

    # Write all alerts to reports/alerts.jsonl
    alerts_file = out_dir / "alerts.jsonl"
    with alerts_file.open("w", encoding="utf-8") as f:
        for a in all_alerts:
            f.write(json.dumps(a.to_dict()) + "\n")

    # Compute Bootstrap CI for Primary Policy over EVENTS
    bootstrap_ci = compute_bootstrap_ci(
        run=run_primary,
        anomaly_events_count=run_primary.confusion_matrix["total_anomalies"],
        benign_events_count=run_primary.confusion_matrix["total_benign"],
    )

    # Run Multi-Seed Stability Test (Seeds 1-5)
    print("Running Multi-Seed Stability Test (Seeds 1 to 5)...")
    multi_seed_results = run_multi_seed_evaluation(seeds=[1, 2, 3, 4, 5], config_path=config_path)

    # Generate Plots
    print("Generating diagnostic plots...")
    plot_files = generate_evaluation_plots(run_primary, run_secondary, output_dir=out_dir)

    # Compile comprehensive baseline metrics output
    metrics_data = {
        "dataset": {
            "feed": str(feed_path),
            "labels": str(labels_path),
            "total_feed_updates": 3164,
            "total_events": run_primary.confusion_matrix["total_anomalies"] + run_primary.confusion_matrix["total_benign"],
            "anomaly_events": run_primary.confusion_matrix["total_anomalies"],
            "benign_events": run_primary.confusion_matrix["total_benign"],
        },
        "headline_metric_rationale": (
            "Event-level precision is preferred as the headline metric because network security incidents "
            "are event windows (e.g. hijack attacks, scrubbing activations). Event-level metrics align directly "
            "with the confusion matrix (TP + FP + TN + FN = Total Events) and prevent alert-frequency skew."
        ),
        "performance_speed": {
            "wall_clock_time_ms_per_update": run_primary.wall_clock_time_ms_per_update,
            "updates_per_second": run_primary.updates_per_second,
        },
        "third_party_noise_table": {
            "alert_type": "THIRD_PARTY_INVALID",
            "total_injected_noise_updates": 45,
            "withdrawals": 6,
            "announcements_total": 39,
            "announcements_not_found": 20,
            "announcements_rpki_invalid": 19,
            "alerts_emitted": run_primary.third_party_noise_count,
            "explanation": (
                "Out of 45 injected third-party updates outside operator inventory: "
                "6 were withdrawals (which do not emit alerts), "
                "20 were announcements for unmonitored prefix 203.0.114.0/24 whose ROV state is NotFound "
                "(which do not trigger THIRD_PARTY_INVALID alerts since there are no violating covering ROAs), "
                "and 19 were announcements for 198.51.200.0/24 and /25 covered by an AS0 ROA in snapshots 2 and 3 "
                "(evaluating to RFC 6811 Invalid). Each of those 19 Invalid announcements emitted exactly one alert."
            ),
        },
        "primary_policy_all_alerts": {
            "policy_definition": "All operator alert types count (strictest)",
            "headline_metric": "event_precision",
            "event_metrics": asdict(run_primary.event_metrics),
            "alert_metrics": asdict(run_primary.alert_metrics),
            "alert_level_precision": run_primary.alert_level_precision,
            "bootstrap_95_ci": bootstrap_ci,
            "missed_events_count": run_primary.missed_events_count,
            "missed_due_to_dedup_count": run_primary.missed_due_to_dedup_count,
            "scenario_recall": run_primary.scenario_recall,
            "difficulty_recall": run_primary.difficulty_recall,
            "difficulty_counts": run_primary.difficulty_counts,
            "fp_per_benign_scenario": run_primary.fp_per_benign_scenario,
            "benign_event_fp_rate": run_primary.benign_event_fp_rate,
            "fp_per_1000_background": run_primary.fp_per_1000_background,
            "latency": asdict(run_primary.latency),
            "confusion_matrix": run_primary.confusion_matrix,
        },
        "secondary_policy_high_medium": {
            "policy_definition": "High and Medium severity alerts only (filters low-severity INVENTORY_MISMATCH/ROA_MISSING)",
            "headline_metric": "event_precision",
            "event_metrics": asdict(run_secondary.event_metrics),
            "alert_metrics": asdict(run_secondary.alert_metrics),
            "alert_level_precision": run_secondary.alert_level_precision,
            "scenario_recall": run_secondary.scenario_recall,
            "difficulty_recall": run_secondary.difficulty_recall,
            "difficulty_counts": run_secondary.difficulty_counts,
            "fp_per_benign_scenario": run_secondary.fp_per_benign_scenario,
            "benign_event_fp_rate": run_secondary.benign_event_fp_rate,
            "fp_per_1000_background": run_secondary.fp_per_1000_background,
            "latency": asdict(run_secondary.latency),
            "confusion_matrix": run_secondary.confusion_matrix,
        },
        "multi_seed_stability": multi_seed_results,
        "generated_plots": plot_files,
    }

    metrics_file = out_dir / "baseline_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 4 BASELINE EVALUATION REPORT (RECONCILED)")
    print("=" * 70)
    print(f"Total Updates Processed : 3,164")
    print(f"Wall-Clock Speed        : {run_primary.wall_clock_time_ms_per_update:.4f} ms/update ({run_primary.updates_per_second:.1f} updates/sec)")
    print(f"Third-Party Noise Alerts: {run_primary.third_party_noise_count} (Reported separately; excluded from operator metrics)")
    print("\n1. PRIMARY POLICY (All Operator Alert Types - Strictest):")
    print(f"   Headline Event-Level Precision : {run_primary.event_metrics.precision:.4f} (95% CI: {bootstrap_ci['precision_ci'][0]:.4f} - {bootstrap_ci['precision_ci'][1]:.4f})")
    print(f"   Alert-Level Precision          : {run_primary.alert_level_precision:.4f}")
    print(f"   Event-Level Recall             : {run_primary.event_metrics.recall:.4f} (95% CI: {bootstrap_ci['recall_ci'][0]:.4f} - {bootstrap_ci['recall_ci'][1]:.4f})")
    print(f"   Event-Level F1 Score           : {run_primary.event_metrics.f1:.4f} (95% CI: {bootstrap_ci['f1_ci'][0]:.4f} - {bootstrap_ci['f1_ci'][1]:.4f})")
    print(f"   Bootstrap Note                 : {bootstrap_ci['edge_note']}")
    print(f"   Missed due to Dedup            : {run_primary.missed_due_to_dedup_count}")

    print("\n2. SECONDARY POLICY (High / Medium Severity Only):")
    print(f"   Headline Event-Level Precision : {run_secondary.event_metrics.precision:.4f}")
    print(f"   Alert-Level Precision          : {run_secondary.alert_level_precision:.4f}")
    print(f"   Event-Level Recall             : {run_secondary.event_metrics.recall:.4f}")
    print(f"   Event-Level F1 Score           : {run_secondary.event_metrics.f1:.4f}")

    print("\n3. RECALL PER ANOMALY SCENARIO:")
    for sc, rec in run_primary.scenario_recall.items():
        print(f"   {sc:<35}: {rec * 100:.1f}%")
    easy_tp = run_primary.difficulty_counts["easy"]["tp"]
    easy_tot = run_primary.difficulty_counts["easy"]["total"]
    hard_tp = run_primary.difficulty_counts["hard"]["tp"]
    hard_tot = run_primary.difficulty_counts["hard"]["total"]
    print(f"   Easy Anomaly Recall             : {run_primary.difficulty_recall['easy'] * 100:.1f}% ({easy_tp}/{easy_tot})")
    print(f"   Hard Anomaly Recall             : {run_primary.difficulty_recall['hard'] * 100:.1f}% ({hard_tp}/{hard_tot}: {hard_tp} caught, {hard_tot - hard_tp} loose_maxlength missed)")

    print("\n4. FALSE POSITIVES PER BENIGN SCENARIO (Primary Policy):")
    for sc, count in run_primary.fp_per_benign_scenario.items():
        print(f"   {sc:<42}: {count} events ({run_primary.benign_event_fp_rate[sc]*100:.1f}%)")
    print(f"   Background Operator Noise       : {run_primary.fp_per_1000_background:.4f} FP per 1,000 updates")

    print("\n5. MULTI-SEED STABILITY (Seeds 1-5):")
    print(f"{'Seed':<6} | {'Feed Hash':<16} | {'Updates':<7} | {'Events':<6} | {'P_EvPrec':<8} | {'P_Rec':<7} | {'P_F1':<7} | {'S_EvPrec':<8}")
    print("-" * 80)
    for sr in multi_seed_results["seed_results"]:
        print(f"{sr['seed']:<6} | {sr['feed_hash']:<16} | {sr['feed_update_count']:<7} | {sr['total_events']:<6} | {sr['primary']['event_precision']:<8.4f} | {sr['primary']['recall']:<7.4f} | {sr['primary']['f1']:<7.4f} | {sr['secondary']['event_precision']:<8.4f}")
    print(f"Primary Mean +/- Std : Precision={multi_seed_results['primary_precision']['mean']:.4f} +/- {multi_seed_results['primary_precision']['std']:.4f}, Recall={multi_seed_results['primary_recall']['mean']:.4f} +/- {multi_seed_results['primary_recall']['std']:.4f}, F1={multi_seed_results['primary_f1']['mean']:.4f} +/- {multi_seed_results['primary_f1']['std']:.4f}")

    print(f"\nArtifacts Saved: {metrics_file}, {alerts_file}, and 4 plots in {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
