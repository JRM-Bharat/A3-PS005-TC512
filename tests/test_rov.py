"""Unit tests for RFC 6811 Route Origin Validation (ROV)."""

import ipaddress
import tempfile
from pathlib import Path

import pytest
from src.roa_store import ROASnapshot, ROAStore
from src.rov import ROVResult, ROVState, validate_route


def test_max_length_boundary_equal_vs_plus_one() -> None:
    """Test maxLength boundary condition: equal to maxLength is Valid; +1 is Invalid."""
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_path = Path(tmpdir) / "roas_2026-10-26T00-00-00Z.csv"
        # ROA: 198.51.100.0/22, AS13335, maxLength=24
        snap_path.write_text("prefix,asn,maxLength\n198.51.100.0/22,13335,24\n")

        store = ROAStore(tmpdir)
        ts = "2026-10-26T01:00:00Z"

        # Boundary equal: /24 <= 24 -> VALID
        res_equal = validate_route("198.51.100.0/24", 13335, ts, store)
        assert res_equal.state == ROVState.VALID
        assert res_equal.matched_roa is not None
        assert res_equal.matched_roa.max_length == 24
        assert res_equal.invalid_reason is None

        # Boundary +1: /25 > 24 -> INVALID (length_exceeds_maxlength)
        res_plus_one = validate_route("198.51.100.0/25", 13335, ts, store)
        assert res_plus_one.state == ROVState.INVALID
        assert res_plus_one.matched_roa is None
        assert res_plus_one.invalid_reason == "length_exceeds_maxlength"


def test_multiple_covering_roas_one_matches() -> None:
    """Test multiple covering ROAs where one matches origin/length and others do not."""
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_path = Path(tmpdir) / "roas_2026-10-26T00-00-00Z.csv"
        # Two covering ROAs: one for AS13335 (max 24), one for AS13336 (max 23)
        snap_path.write_text(
            "prefix,asn,maxLength\n"
            "198.51.100.0/22,13335,24\n"
            "198.51.100.0/22,13336,23\n"
        )
        store = ROAStore(tmpdir)
        ts = "2026-10-26T01:00:00Z"

        # Origin AS13335 with /24: matches first ROA even though second ROA origin and maxLength mismatch
        res = validate_route("198.51.100.0/24", 13335, ts, store)
        assert res.state == ROVState.VALID
        assert res.matched_roa.asn == 13335
        assert len(res.covering_roas) == 2


def test_as0_roa_invalidation() -> None:
    """RFC 6483 / RFC 7607: AS0 ROAs can never match; routes covered only by AS0 are Invalid."""
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_path = Path(tmpdir) / "roas_2026-10-26T00-00-00Z.csv"
        snap_path.write_text("prefix,asn,maxLength\n198.51.200.0/24,0,24\n")

        store = ROAStore(tmpdir)
        ts = "2026-10-26T01:00:00Z"

        # Even if origin announces 0 or any other ASN, AS0 route is Invalid
        res_foreign = validate_route("198.51.200.0/24", 64496, ts, store)
        assert res_foreign.state == ROVState.INVALID
        assert res_foreign.invalid_reason == "as0_roa"

        res_zero = validate_route("198.51.200.0/24", 0, ts, store)
        assert res_zero.state == ROVState.INVALID
        assert res_zero.invalid_reason == "as0_roa"


def test_ipv6_validation() -> None:
    """RFC 6811 validation on IPv6 addresses."""
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_path = Path(tmpdir) / "roas_2026-10-26T00-00-00Z.csv"
        snap_path.write_text("prefix,asn,maxLength\n2001:db8:1000::/36,13335,48\n")

        store = ROAStore(tmpdir)
        ts = "2026-10-26T01:00:00Z"

        # Valid IPv6 subprefix
        res_valid = validate_route("2001:db8:1000:abcd::/48", 13335, ts, store)
        assert res_valid.state == ROVState.VALID

        # Invalid IPv6 origin
        res_inv_origin = validate_route("2001:db8:1000:abcd::/48", 64496, ts, store)
        assert res_inv_origin.state == ROVState.INVALID
        assert res_inv_origin.invalid_reason == "origin_mismatch"

        # Invalid IPv6 length
        res_inv_len = validate_route("2001:db8:1000:abcd:1::/64", 13335, ts, store)
        assert res_inv_len.state == ROVState.INVALID
        assert res_inv_len.invalid_reason == "length_exceeds_maxlength"


def test_no_snapshot_at_update_time() -> None:
    """Update time strictly before any available snapshot returns NoSnapshot."""
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_path = Path(tmpdir) / "roas_2026-10-26T06-00-00Z.csv"
        snap_path.write_text("prefix,asn,maxLength\n198.51.100.0/22,13335,24\n")

        store = ROAStore(tmpdir)
        # Update at 04:00 (before earliest snapshot at 06:00)
        res = validate_route("198.51.100.0/24", 13335, "2026-10-26T04:00:00Z", store)
        assert res.state == ROVState.NO_SNAPSHOT
        assert res.roa_snapshot_time is None
        assert res.invalid_reason == "no_snapshot_at_or_before_update_time"


def test_roa_changing_mid_stream() -> None:
    """Test same announcement is Valid before a ROA change, and Invalid after."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)
        # Snapshot 1 at 00:00:00 authorizes AS64496
        snap1 = dir_path / "roas_2026-10-26T00-00-00Z.csv"
        snap1.write_text("prefix,asn,maxLength\n203.0.113.0/24,64496,24\n")

        # Snapshot 2 at 06:00:00 revokes AS64496 and migrates to AS64498
        snap2 = dir_path / "roas_2026-10-26T06-00-00Z.csv"
        snap2.write_text("prefix,asn,maxLength\n203.0.113.0/24,64498,24\n")

        store = ROAStore(dir_path)

        # Before change at 03:00: Announcement from AS64496 is Valid
        res_before = validate_route("203.0.113.0/24", 64496, "2026-10-26T03:00:00Z", store)
        assert res_before.state == ROVState.VALID
        assert res_before.matched_roa.asn == 64496

        # After change at 08:00: Announcement from AS64496 is now Invalid!
        res_after = validate_route("203.0.113.0/24", 64496, "2026-10-26T08:00:00Z", store)
        assert res_after.state == ROVState.INVALID
        assert res_after.invalid_reason == "origin_mismatch"
