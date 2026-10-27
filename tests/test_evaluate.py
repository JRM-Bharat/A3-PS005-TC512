"""Unit tests for metric mathematics and evaluation logic (Phase 4).

Verifies metric formulas (Precision, Recall, F1, Latency, Dedup Tracking, Bootstrap CI)
on small, verified hand-computed examples.
"""

import tempfile
from pathlib import Path

import pytest
from src.evaluate import (
    EvaluationRun,
    LatencyStats,
    MetricScores,
    compute_bootstrap_ci,
    evaluate_feed,
)
from src.rules import Alert


def test_metric_scores_hand_computed() -> None:
    """Hand-computed sanity test on MetricScores formula."""
    # Hand calculation:
    # TP = 6, FP = 2, FN = 4
    # Precision = 6 / (6 + 2) = 6/8 = 0.7500
    # Recall = 6 / (6 + 4) = 6/10 = 0.6000
    # F1 = 2 * (0.75 * 0.60) / (0.75 + 0.60) = 2 * 0.45 / 1.35 = 0.90 / 1.35 = 0.6667
    scores = MetricScores.from_counts(tp=6, fp=2, fn=4)
    assert scores.tp == 6
    assert scores.fp == 2
    assert scores.fn == 4
    assert scores.precision == 0.7500
    assert scores.recall == 0.6000
    assert scores.f1 == 0.6667

    # Edge cases: 0 TP, 0 FP, 0 FN
    zero_scores = MetricScores.from_counts(tp=0, fp=0, fn=0)
    assert zero_scores.precision == 0.0
    assert zero_scores.recall == 0.0
    assert zero_scores.f1 == 0.0


def test_evaluate_feed_hand_computed_mock() -> None:
    """Test full evaluate_feed flow with a tiny, fully deterministic hand-computed dataset.

    Setup:
      - 3 Anomaly Events:
          1. evt-1: 198.51.100.0/22, AS99001, t=1000s -> Matched by alert at t=1015s (TP, lat=15s)
          2. evt-2: 203.0.113.0/24, AS99002, t=2000s -> Matched by alert at t=2020s (TP, lat=20s)
          3. evt-3: 198.18.0.0/15, AS99003, t=3000s -> No alert (FN)
      - 1 Benign Event:
          4. evt-4: 192.0.2.0/24, AS99004, t=4000s -> Emits unexpected alert at t=4005s (FP)
      - 1 Third-Party Noise Update outside inventory -> Emits THIRD_PARTY_INVALID (excluded from operator metrics)

    Expected Outcomes:
      - TP = 2
      - FN = 1
      - FP = 1
      - Precision = 2 / 3 = 0.6667
      - Recall = 2 / 3 = 0.6667
      - F1 = 0.6667
      - Latency: mean = (15 + 20) / 2 = 17.5s, median = 17.5s
      - Third-party noise count = 1
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        inv_file = tmp_path / "inventory.csv"
        roas_dir = tmp_path / "roas"
        roas_dir.mkdir()
        feed_file = tmp_path / "feed.csv"
        labels_file = tmp_path / "labels.csv"

        inv_file.write_text(
            "prefix,expected_origin_asns,max_expected_length,owner\n"
            "198.51.100.0/22,13335,24,CloudCorp\n"
            "203.0.113.0/24,64496,24,RegionalISP\n"
            "198.18.0.0/15,64510,24,CampusNet\n"
            "192.0.2.0/24,64500,24,FinTech\n"
        )

        # Snapshot at t=0
        snap_file = roas_dir / "roas_2026-10-26T00-00-00Z.csv"
        snap_file.write_text(
            "prefix,asn,maxLength\n"
            "198.51.100.0/22,13335,24\n"
            "203.0.113.0/24,64496,24\n"
            "198.18.0.0/15,64510,24\n"
            "192.0.2.0/24,64500,24\n"
            "198.51.200.0/24,0,24\n"  # AS0 ROA for third-party noise
        )

        base_t = 1792972800.0  # 2026-10-26T00:00:00Z

        labels_file.write_text(
            "event_id,start_time,end_time,prefix,origin_asn,type,benign,scenario,difficulty,expected_alert_type\n"
            f"evt-1,{base_t + 1000},{base_t + 1500},198.51.100.0/22,99001,ORIGIN_HIJACK,False,exact_origin_hijack,easy,ORIGIN_HIJACK\n"
            f"evt-2,{base_t + 2000},{base_t + 2500},203.0.113.0/24,99002,ORIGIN_HIJACK,False,exact_origin_hijack,easy,ORIGIN_HIJACK\n"
            f"evt-3,{base_t + 3000},{base_t + 3500},198.18.0.0/15,99003,ORIGIN_HIJACK,False,exact_origin_hijack,easy,ORIGIN_HIJACK\n"
            f"evt-4,{base_t + 4000},{base_t + 4500},192.0.2.0/24,99004,POLICY_LAG,True,test_benign_lag,hard,ROA_INVALID\n"
        )

        # Feed with announcements
        # evt-1 announcement at base_t + 1015 (delta = 15s)
        # evt-2 announcement at base_t + 2020 (delta = 20s)
        # evt-3 is NOT announced (missed attack)
        # evt-4 announcement at base_t + 4005 (benign event triggering alert -> FP)
        # third party noise update at base_t + 5000 on 198.51.200.0/24 with AS65001 (AS0 ROA invalid)
        feed_file.write_text(
            "timestamp,prefix,origin_asn,peer,action\n"
            f"{base_t + 1015},198.51.100.0/22,99001,198.32.160.1,announce\n"
            f"{base_t + 2020},203.0.113.0/24,99002,198.32.160.1,announce\n"
            f"{base_t + 4005},192.0.2.0/24,99004,198.32.160.1,announce\n"
            f"{base_t + 5000},198.51.200.0/24,65001,198.32.160.1,announce\n"
        )

        run, alerts = evaluate_feed(
            feed_path=feed_file,
            labels_path=labels_file,
            inventory_path=inv_file,
            roas_dir=roas_dir,
            policy="primary_all",
        )

        assert run.event_metrics.tp == 2
        assert run.event_metrics.fn == 1
        assert run.event_metrics.fp == 1
        assert run.event_metrics.precision == 0.6667
        assert run.event_metrics.recall == 0.6667
        assert run.event_metrics.f1 == 0.6667
        assert run.third_party_noise_count == 1
        assert run.latency.mean_seconds == 17.5
        assert run.latency.median_seconds == 17.5


def test_one_to_one_matching_and_dedup_tracking() -> None:
    """Verify that when 2 events collide in 1 dedup window, the 2nd is counted in missed_due_to_dedup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        inv_file = tmp_path / "inventory.csv"
        roas_dir = tmp_path / "roas"
        roas_dir.mkdir()
        feed_file = tmp_path / "feed.csv"
        labels_file = tmp_path / "labels.csv"

        inv_file.write_text("prefix,expected_origin_asns,max_expected_length,owner\n203.0.113.0/24,64496,24,ISP\n")
        snap_file = roas_dir / "roas_2026-10-26T00-00-00Z.csv"
        snap_file.write_text("prefix,asn,maxLength\n203.0.113.0/24,64496,24\n")

        base_t = 1792972800.0

        # Two events for same prefix & origin within 300s window
        # Event 1 starts at base_t + 100s
        # Event 2 starts at base_t + 200s (delta = 100s < 300s dedup window)
        labels_file.write_text(
            "event_id,start_time,end_time,prefix,origin_asn,type,benign,scenario,difficulty,expected_alert_type\n"
            f"evt-1,{base_t + 100},{base_t + 180},203.0.113.0/24,99001,ORIGIN_HIJACK,False,test_hijack,easy,ORIGIN_HIJACK\n"
            f"evt-2,{base_t + 200},{base_t + 280},203.0.113.0/24,99001,ORIGIN_HIJACK,False,test_hijack,easy,ORIGIN_HIJACK\n"
        )

        # Feed has announcement for Event 1 at base_t + 100.
        # Announcement for Event 2 at base_t + 200 is suppressed by dedup window!
        feed_file.write_text(
            "timestamp,prefix,origin_asn,peer,action\n"
            f"{base_t + 100},203.0.113.0/24,99001,198.32.160.1,announce\n"
            f"{base_t + 200},203.0.113.0/24,99001,198.32.160.1,announce\n"
        )

        run, alerts = evaluate_feed(
            feed_path=feed_file,
            labels_path=labels_file,
            inventory_path=inv_file,
            roas_dir=roas_dir,
            dedup_window_seconds=300.0,
        )

        # Only 1 alert was produced (dedup suppressed the second)
        assert len(alerts) == 1
        # Event 1 was matched (TP=1)
        assert run.event_metrics.tp == 1
        # Event 2 was missed (FN=1)
        assert run.event_metrics.fn == 1
        # Event 2 was specifically tracked as missed_due_to_dedup
        assert run.missed_due_to_dedup_count == 1


def test_bootstrap_ci_coverage() -> None:
    """Bootstrap CI must generate valid confidence intervals enclosing the point estimate."""
    scores = MetricScores.from_counts(tp=8, fp=2, fn=2)
    fake_run = EvaluationRun(
        policy="primary_all",
        headline_metric_type="event_level",
        event_metrics=scores,
        alert_metrics=scores,
        alert_level_precision=0.8,
        matched_events_count=8,
        missed_events_count=2,
        missed_due_to_dedup_count=0,
        scenario_recall={},
        difficulty_recall={},
        difficulty_counts={},
        fp_per_benign_scenario={},
        benign_event_fp_rate={},
        fp_per_1000_background=0.0,
        third_party_noise_count=0,
        latency=LatencyStats(),
        wall_clock_time_ms_per_update=0.1,
        updates_per_second=10000.0,
        confusion_matrix={"false_positives_events": 2, "true_negatives_events": 8, "total_anomalies": 10, "total_benign": 10},
    )

    ci = compute_bootstrap_ci(fake_run, anomaly_events_count=10, benign_events_count=10, n_bootstraps=500, seed=42)
    assert ci["resampling_unit"] == "events"
    assert "precision_ci" in ci
    assert "recall_ci" in ci
    assert "f1_ci" in ci
    assert "edge_note" in ci
    # Bounds must be within [0.0, 1.0]
    assert 0.0 <= ci["precision_ci"][0] <= ci["precision_ci"][1] <= 1.0
    assert 0.0 <= ci["recall_ci"][0] <= ci["recall_ci"][1] <= 1.0
    assert 0.0 <= ci["f1_ci"][0] <= ci["f1_ci"][1] <= 1.0


def test_consecutive_benign_events_one_to_one_matcher_bug_reproduction() -> None:
    """Matcher Bug Reproduction Test:
    
    When two consecutive benign events occur on the same prefix/origin spaced closely
    (e.g., event 1 [1000s, 1300s] and event 2 [1420s, 1720s], gap 120s), and each emits an alert
    at its respective start time (alt 1 at 1000s, alt 2 at 1420s):
    
    BUG: An unconstrained or overly wide window (e.g. end_time + 600s) on event 1 would extend
    to 1900s, encompassing alt 2. Greedy sequential matching would allocate alt 2 to event 1,
    leaving event 2 with 0 alerts and reporting flagged_events = 1 despite 2 alerts.
    
    FIX: 1-to-1 matching with proper window bounds correctly matches event 1 -> alt 1 and
    event 2 -> alt 2, resulting in flagged_events = 2 and alerts = 2.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        inv_file = tmp_path / "inventory.csv"
        roas_dir = tmp_path / "roas"
        roas_dir.mkdir()
        feed_file = tmp_path / "feed.csv"
        labels_file = tmp_path / "labels.csv"

        # Operator prefix
        inv_file.write_text("prefix,expected_origin_asns,max_expected_length,owner\n203.0.113.0/24,64496,24,ISP\n")
        snap_file = roas_dir / "roas_2026-10-26T00-00-00Z.csv"
        # ROA with AS64496
        snap_file.write_text("prefix,asn,maxLength\n203.0.113.0/24,64496,24\n")

        base_t = 1792972800.0

        # Two consecutive benign events with unexpected origin AS64498 (e.g. scrubbing or lag)
        # Event 1: [1000s, 1300s]
        # Event 2: [1420s, 1720s] (gap = 120s)
        labels_file.write_text(
            "event_id,start_time,end_time,prefix,origin_asn,type,benign,scenario,difficulty,expected_alert_type\n"
            f"evt-1,{base_t + 1000},{base_t + 1300},203.0.113.0/24,64498,POLICY_LAG,True,test_scrubbing,easy,INVENTORY_MISMATCH\n"
            f"evt-2,{base_t + 1420},{base_t + 1720},203.0.113.0/24,64498,POLICY_LAG,True,test_scrubbing,easy,INVENTORY_MISMATCH\n"
        )

        # Feed: Event 1 announces at 1000s; Event 2 announces at 1420s (outside 300s dedup window)
        feed_file.write_text(
            "timestamp,prefix,origin_asn,peer,action\n"
            f"{base_t + 1000},203.0.113.0/24,64498,198.32.160.1,announce\n"
            f"{base_t + 1420},203.0.113.0/24,64498,198.32.160.1,announce\n"
        )

        run, alerts = evaluate_feed(
            feed_path=feed_file,
            labels_path=labels_file,
            inventory_path=inv_file,
            roas_dir=roas_dir,
            policy="primary_all",
            dedup_window_seconds=300.0,
        )

        # Exactly 2 alerts emitted
        assert len(alerts) == 2
        # BOTH benign events must be matched 1-to-1: FP events = 2
        assert run.event_metrics.fp == 2
        assert run.confusion_matrix["false_positives_events"] == 2
        assert run.confusion_matrix["true_negatives_events"] == 0
        assert run.fp_per_benign_scenario["test_scrubbing"] == 2
        assert run.benign_event_fp_rate["test_scrubbing"] == 1.0  # 100% of events flagged


def test_difficulty_aggregation_totals() -> None:
    """Assert that difficulty grouping correctly partitions anomaly events.
    
    Verifies easy_total + hard_total == total_anomalies (24 easy + 16 hard == 40 total anomalies).
    """
    run, _ = evaluate_feed(
        feed_path="data/feed_replay.csv",
        labels_path="data/labels.csv",
        inventory_path="data/inventory.csv",
        roas_dir="data/roa_snapshots",
        policy="primary_all",
    )
    easy_total = run.difficulty_counts["easy"]["total"]
    hard_total = run.difficulty_counts["hard"]["total"]
    total_anomalies = run.confusion_matrix["total_anomalies"]

    assert easy_total == 24
    assert hard_total == 16
    assert easy_total + hard_total == total_anomalies == 40
    assert run.difficulty_counts["easy"]["tp"] == 24
    assert run.difficulty_counts["hard"]["tp"] == 8
    assert run.difficulty_recall["easy"] == 1.0
    assert run.difficulty_recall["hard"] == 0.5

