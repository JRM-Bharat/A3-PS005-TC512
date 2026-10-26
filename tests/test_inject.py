"""Unit tests for Synthetic Scenario Generator and Label Injection (Phase 3)."""

import tempfile
from pathlib import Path

import pytest
from src.inject import ScenarioInjector


def test_generator_determinism() -> None:
    """The generator must be strictly deterministic: identical seed yields identical feed and labels."""
    inj1 = ScenarioInjector(config_path="config.yaml", seed=123)
    updates1, events1, manifest1 = inj1.generate()

    inj2 = ScenarioInjector(config_path="config.yaml", seed=123)
    updates2, events2, manifest2 = inj2.generate()

    assert len(updates1) == len(updates2)
    assert len(events1) == len(events2)

    for u1, u2 in zip(updates1, updates2):
        assert u1.timestamp == u2.timestamp
        assert u1.prefix == u2.prefix
        assert u1.origin_asn == u2.origin_asn
        assert u1.peer == u2.peer
        assert u1.action == u2.action

    for e1, e2 in zip(events1, events2):
        assert e1.event_id == e2.event_id
        assert e1.start_time == e2.start_time
        assert e1.prefix == e2.prefix
        assert e1.origin_asn == e2.origin_asn
        assert e1.scenario == e2.scenario
        assert e1.benign == e2.benign


def test_no_timestamp_before_first_snapshot() -> None:
    """No generated event or background update timestamp may fall before the first snapshot (00:00:00Z)."""
    injector = ScenarioInjector(config_path="config.yaml", seed=42)
    updates, events, _ = injector.generate()

    min_snapshot_time = injector.start_time  # 2026-10-26T00:00:00Z

    for u in updates:
        assert (
            u.timestamp >= min_snapshot_time
        ), f"Update {u} occurred before first snapshot at {min_snapshot_time}"

    for e in events:
        assert (
            e.start_time >= min_snapshot_time
        ), f"Event {e} started before first snapshot at {min_snapshot_time}"
        assert e.end_time >= e.start_time


def test_scenario_counts_match_config() -> None:
    """Generated scenario counts must match config.yaml specification."""
    injector = ScenarioInjector(config_path="config.yaml", seed=42)
    _, events, manifest = injector.generate()

    cfg_counts = injector.scenario_counts
    generated_counts = manifest["scenario_counts"]

    for sc_name, expected_count in cfg_counts.items():
        assert (
            generated_counts[sc_name] == expected_count
        ), f"Mismatch for scenario {sc_name}: expected {expected_count}, got {generated_counts[sc_name]}"

    # Verify event list matches total
    assert len(events) == sum(cfg_counts.values())


def test_label_feed_consistency() -> None:
    """Every labelled event must have matching feed rows within its [start_time, end_time] window."""
    injector = ScenarioInjector(config_path="config.yaml", seed=42)
    updates, events, _ = injector.generate()

    # Index updates by (prefix_str, origin_asn) -> list of updates
    update_index = {}
    for u in updates:
        key = (u.prefix_str, u.origin_asn)
        update_index.setdefault(key, []).append(u)

    for event in events:
        key = (event.prefix, event.origin_asn)
        matching = update_index.get(key, [])
        assert len(matching) > 0, f"No feed rows found for labelled event {event}"

        # Ensure at least one announcement matches near event start_time
        announces = [
            u
            for u in matching
            if u.is_announcement and (event.start_time - 1.0 <= u.timestamp <= event.end_time + 1.0)
        ]
        assert (
            len(announces) > 0
        ), f"No matching announcement found in feed for event {event.event_id} ({event.scenario})"


def test_save_to_disk() -> None:
    """Verify that saving to disk writes valid CSV and JSON manifest files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        feed_file = tmp_dir / "feed_test.csv"
        labels_file = tmp_dir / "labels_test.csv"
        manifest_file = tmp_dir / "manifest_test.json"

        injector = ScenarioInjector(config_path="config.yaml", seed=42)
        manifest = injector.save_to_disk(
            feed_path=feed_file,
            labels_path=labels_file,
            manifest_path=manifest_file,
        )

        assert feed_file.exists()
        assert labels_file.exists()
        assert manifest_file.exists()
        assert manifest["total_feed_updates"] > 0
        assert manifest["total_labelled_events"] > 0
