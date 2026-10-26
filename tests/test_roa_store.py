"""Unit tests for ROA Store and strict time-alignment (Phase 1)."""

import ipaddress
import tempfile
from pathlib import Path

import pytest
from src.roa_store import (
    ROARecord,
    ROASnapshot,
    ROAStore,
    parse_timestamp,
)


def test_parse_timestamp_formats() -> None:
    # Unix epoch float
    assert parse_timestamp(1700000000) == 1700000000.0

    # ISO string with Z
    ts_iso = parse_timestamp("2026-10-26T12:00:00Z")
    assert ts_iso > 0

    # Filename safe format with dashes
    ts_file = parse_timestamp("2026-10-26T12-00-00Z")
    assert ts_file == ts_iso


def test_roa_record_validation() -> None:
    record = ROARecord(
        prefix=ipaddress.ip_network("198.51.100.0/22"),
        asn=13335,
        max_length=24,
    )
    # Covering check
    assert record.covers(ipaddress.ip_network("198.51.100.0/22")) is True
    assert record.covers(ipaddress.ip_network("198.51.100.0/24")) is True
    assert record.covers(ipaddress.ip_network("198.51.100.0/25")) is True
    assert record.covers(ipaddress.ip_network("198.51.104.0/24")) is False  # Outside /22
    assert record.covers(ipaddress.ip_network("2001:db8::/32")) is False  # IPv6 mismatch

    # Origin and length checks
    assert record.matches_origin(13335) is True
    assert record.matches_origin(64496) is False
    assert record.matches_length(ipaddress.ip_network("198.51.100.0/24")) is True
    assert record.matches_length(ipaddress.ip_network("198.51.100.0/25")) is False  # 25 > 24


def test_time_aligned_snapshot_selection_no_future_leak() -> None:
    """MANDATORY TEST: For an update at time t, use the most recent snapshot with snapshot_time <= t."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)
        # Create three snapshots at T0 (00:00), T1 (06:00), and T2 (12:00)
        snap0 = dir_path / "roas_2026-10-26T00-00-00Z.csv"
        snap1 = dir_path / "roas_2026-10-26T06-00-00Z.csv"
        snap2 = dir_path / "roas_2026-10-26T12-00-00Z.csv"

        snap0.write_text("prefix,asn,maxLength\n198.51.100.0/22,13335,24\n203.0.113.0/24,64496,24\n")
        # In snapshot 1, 203.0.113.0/24 migrates to 64498
        snap1.write_text("prefix,asn,maxLength\n198.51.100.0/22,13335,24\n203.0.113.0/24,64498,24\n")
        # In snapshot 2, Cloudflare adds multi-origin ASN 13336
        snap2.write_text("prefix,asn,maxLength\n198.51.100.0/22,13335,24\n198.51.100.0/22,13336,24\n203.0.113.0/24,64498,24\n")

        store = ROAStore(dir_path)
        assert store.snapshot_count == 3

        # Update at 03:00 (between T0 and T1) MUST use T0
        s_0300 = store.get_snapshot_for_time("2026-10-26T03:00:00Z")
        assert s_0300 is not None
        roas_isp = s_0300.find_covering_roas("203.0.113.0/24")
        assert len(roas_isp) == 1
        assert roas_isp[0].asn == 64496  # Snapshot 0 value, not migrated 64498!

        # Update at 06:00 (exact T1) MUST use T1
        s_0600 = store.get_snapshot_for_time("2026-10-26T06:00:00Z")
        assert s_0600 is not None
        roas_isp_migrated = s_0600.find_covering_roas("203.0.113.0/24")
        assert roas_isp_migrated[0].asn == 64498

        # Update at 09:00 (between T1 and T2) MUST use T1, MUST NOT leak T2
        s_0900 = store.get_snapshot_for_time("2026-10-26T09:00:00Z")
        assert s_0900 is not None
        roas_cf = s_0900.find_covering_roas("198.51.100.0/22")
        # Only ASN 13335, ASN 13336 has NOT happened yet
        assert {r.asn for r in roas_cf} == {13335}

        # Update at 14:00 (after T2) MUST use T2
        s_1400 = store.get_snapshot_for_time("2026-10-26T14:00:00Z")
        assert s_1400 is not None
        roas_cf_multi = s_1400.find_covering_roas("198.51.100.0/22")
        assert {r.asn for r in roas_cf_multi} == {13335, 13336}


def test_covering_prefix_ipv4_and_ipv6() -> None:
    snapshot = ROASnapshot(snapshot_time=1700000000.0)
    snapshot.add_record("198.51.100.0/22", 13335, 24)
    snapshot.add_record("2001:db8:abcd::/48", 64500, 64)

    # IPv4 subprefix
    cov_v4 = snapshot.find_covering_roas("198.51.100.128/25")
    assert len(cov_v4) == 1
    assert cov_v4[0].asn == 13335

    # IPv6 subprefix
    cov_v6 = snapshot.find_covering_roas("2001:db8:abcd:1234::/64")
    assert len(cov_v6) == 1
    assert cov_v6[0].asn == 64500

    # Cross-family isolation: IPv4 cannot match IPv6 ROA
    assert len(snapshot.find_covering_roas("10.0.0.0/8")) == 0
    assert len(snapshot.find_covering_roas("2001:db8:9999::/48")) == 0
