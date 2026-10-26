"""Unit tests for Rule-Based Baseline Anomaly Detector (Phase 2)."""

import ipaddress
import tempfile
from pathlib import Path

import pytest
from src.inventory import PrefixInventory
from src.replay import BGPUpdate
from src.roa_store import ROAStore
from src.rules import Alert, RuleBasedDetector


@pytest.fixture
def test_setup():
    """Create test inventory and ROA store fixture."""
    inv = PrefixInventory()
    # Cloudflare /22 prefix: allows 13335, max length 24
    inv.add("198.51.100.0/22", expected_origin_asns="13335", max_expected_length=24, owner="CloudNet")
    # Campus prefix: allows 64510
    inv.add("198.18.0.0/15", expected_origin_asns=64510, max_expected_length=24, owner="CampusNet")
    # Regional ISP: allows 64496
    inv.add("203.0.113.0/24", expected_origin_asns=64496, max_expected_length=24, owner="RegionalISP")
    # Unprotected inventory prefix (no ROA)
    inv.add("192.0.2.0/24", expected_origin_asns=64500, max_expected_length=24, owner="UnprotectedNet")

    tmpdir = tempfile.TemporaryDirectory()
    snap_path = Path(tmpdir.name) / "roas_2026-10-26T00-00-00Z.csv"
    snap_path.write_text(
        "prefix,asn,maxLength\n"
        "198.51.100.0/22,13335,24\n"
        "198.18.0.0/15,64510,24\n"
        "203.0.113.0/24,64496,24\n"
        # ROA exists for 198.51.200.0/24 with foreign ASN 65500
        "198.51.200.0/24,65500,24\n"
        # ROA exists for 198.51.100.0/24 with new origin 13336 (not in inventory)
        "198.51.100.0/24,13336,24\n"
    )
    store = ROAStore(tmpdir.name)

    detector = RuleBasedDetector(inventory=inv, roa_store=store, dedup_window_seconds=300.0)
    # Base timestamp after snapshot (snapshot is 2026-10-26T00-00-00Z)
    t0 = 1792976400.0  # 2026-10-26T01:00:00Z
    yield detector, t0, tmpdir
    tmpdir.cleanup()


def test_no_alert_for_legitimate_announcement(test_setup) -> None:
    """Legitimate exact and more-specific announcements within maxLength must not alert."""
    detector, t0, _ = test_setup

    # Legitimate exact
    u1 = BGPUpdate(t0, ipaddress.ip_network("198.51.100.0/22"), 13335)
    assert detector.process_update(u1) is None

    # Legitimate traffic-engineering more-specific within maxLength (/24 <= 24)
    u2 = BGPUpdate(t0 + 10.0, ipaddress.ip_network("198.51.100.0/24"), 13335)
    assert detector.process_update(u2) is None


def test_ignore_withdrawals(test_setup) -> None:
    """Withdrawal updates must be completely ignored for anomaly detection."""
    detector, t0, _ = test_setup
    u_withdraw = BGPUpdate(
        timestamp=t0,
        prefix=ipaddress.ip_network("198.51.100.0/22"),
        origin_asn=99999,  # Bad origin, but action is withdraw
        action="withdraw",
    )
    assert detector.process_update(u_withdraw) is None


def test_origin_hijack_alert(test_setup) -> None:
    """Exact inventory prefix announced by unauthorized foreign origin triggers ORIGIN_HIJACK."""
    detector, t0, _ = test_setup
    u_hijack = BGPUpdate(t0, ipaddress.ip_network("203.0.113.0/24"), 66666)

    alert = detector.process_update(u_hijack)
    assert alert is not None
    assert alert.type == "ORIGIN_HIJACK"
    assert alert.severity == "high"
    assert alert.origin_asn == 66666
    assert alert.prefix == "203.0.113.0/24"
    assert "Unauthorized origin" in alert.reason


def test_subprefix_hijack_alert(test_setup) -> None:
    """Subprefix of an inventory prefix announced by unauthorized origin triggers SUBPREFIX_HIJACK."""
    detector, t0, _ = test_setup
    u_sub = BGPUpdate(t0, ipaddress.ip_network("198.51.100.128/25"), 66666)

    alert = detector.process_update(u_sub)
    assert alert is not None
    assert alert.type == "SUBPREFIX_HIJACK"
    assert alert.severity == "high"
    assert alert.prefix == "198.51.100.128/25"
    assert alert.origin_asn == 66666


def test_roa_missing_alert(test_setup) -> None:
    """Inventory-owned prefix with NotFound ROV state triggers ROA_MISSING."""
    detector, t0, _ = test_setup
    # 192.0.2.0/24 is in inventory for AS64500, but has NO ROA in the snapshot
    u_missing = BGPUpdate(t0, ipaddress.ip_network("192.0.2.0/24"), 64500)

    alert = detector.process_update(u_missing)
    assert alert is not None
    assert alert.type == "ROA_MISSING"
    assert alert.severity == "low"
    assert "without RPKI ROA protection" in alert.reason


def test_edge_case_roa_lag_policy_change(test_setup) -> None:
    """Edge Case 1: Origin is in inventory's allowed set but ROV is Invalid (policy lag).

    Must emit ROA_INVALID at medium severity explaining inventory and ROA disagree.
    MUST NOT label it a hijack!
    """
    detector, t0, _ = test_setup
    # CampusNet (198.18.0.0/15) in inventory allows 64510.
    # Suppose /25 announcement from 64510 exceeds ROA maxLength 24 -> ROV Invalid!
    u_lag = BGPUpdate(t0, ipaddress.ip_network("198.18.1.0/25"), 64510)

    alert = detector.process_update(u_lag)
    assert alert is not None
    assert alert.type == "ROA_INVALID"
    assert alert.severity == "medium"
    assert "ROA lags behind policy change" in alert.reason
    assert "HIJACK" not in alert.type


def test_edge_case_stale_inventory_rov_valid(test_setup) -> None:
    """Edge Case 2: ROV is Valid but origin NOT in operator inventory (inventory stale).

    Must emit low-severity alert explaining that ROV is Valid but inventory lacks origin.
    """
    detector, t0, _ = test_setup
    # In ROA snapshot: 198.51.100.0/24 is authorized for AS13336.
    # In inventory: 198.51.100.0/22 only has AS13335 listed!
    u_stale = BGPUpdate(t0, ipaddress.ip_network("198.51.100.0/24"), 13336)

    alert = detector.process_update(u_stale)
    assert alert is not None
    assert alert.type == "INVENTORY_MISMATCH"
    assert alert.severity == "low"
    assert "HIJACK" not in alert.type
    assert "stale inventory" in alert.reason.lower()


def test_deduplication_new_origin_never_suppressed(test_setup) -> None:
    """A NEW origin on the same prefix must NEVER be suppressed by an earlier alert for a different origin."""
    detector, t0, _ = test_setup

    # First hijack from AS66661
    u1 = BGPUpdate(t0, ipaddress.ip_network("203.0.113.0/24"), 66661)
    a1 = detector.process_update(u1)
    assert a1 is not None
    assert a1.origin_asn == 66661

    # Second hijack on the same prefix from a DIFFERENT origin AS66662 at t=t0+10s (within 300s window)
    u2 = BGPUpdate(t0 + 10.0, ipaddress.ip_network("203.0.113.0/24"), 66662)
    a2 = detector.process_update(u2)
    assert a2 is not None, "New origin on same prefix must NOT be suppressed!"
    assert a2.origin_asn == 66662


def test_deduplication_same_origin_suppressed_and_tracks_count(test_setup) -> None:
    """Duplicate updates for identical (prefix, origin_asn) within 300s window are suppressed."""
    detector, t0, _ = test_setup

    # First occurrence at t0
    u1 = BGPUpdate(t0, ipaddress.ip_network("203.0.113.0/24"), 66661)
    a1 = detector.process_update(u1)
    assert a1 is not None
    assert a1.time == t0

    # Duplicate at t0 + 50s (suppressed)
    u2 = BGPUpdate(t0 + 50.0, ipaddress.ip_network("203.0.113.0/24"), 66661)
    assert detector.process_update(u2) is None

    # Duplicate at t0 + 100s (suppressed)
    u3 = BGPUpdate(t0 + 100.0, ipaddress.ip_network("203.0.113.0/24"), 66661)
    assert detector.process_update(u3) is None

    # Check that suppressed count on the original alert incremented to 2
    assert a1.suppressed_count == 2


def test_deduplication_window_expiry(test_setup) -> None:
    """After the deduplication window expires (> 300s), a new alert is generated."""
    detector, t0, _ = test_setup

    # First occurrence at t0
    u1 = BGPUpdate(t0, ipaddress.ip_network("203.0.113.0/24"), 66661)
    a1 = detector.process_update(u1)
    assert a1 is not None

    # Duplicate after window expires at t0 + 350s (delta = 350s > 300s)
    u2 = BGPUpdate(t0 + 350.0, ipaddress.ip_network("203.0.113.0/24"), 66661)
    a2 = detector.process_update(u2)
    assert a2 is not None
    assert a2.time == t0 + 350.0


def test_third_party_invalid_alert(test_setup) -> None:
    """Routes outside operator inventory that are RFC 6811 Invalid emit THIRD_PARTY_INVALID at low severity."""
    detector, t0, tmpdir = test_setup
    # In test_setup, 198.51.200.0/24 has ROA for AS65500 with max 24
    # Announce with AS99999 (invalid origin) on unmonitored prefix
    u_tp = BGPUpdate(t0, ipaddress.ip_network("198.51.200.0/24"), 99999)
    alert = detector.process_update(u_tp)
    assert alert is not None
    assert alert.type == "THIRD_PARTY_INVALID"
    assert alert.severity == "low"
    assert "Third-party" in alert.reason
    assert alert.inventory_match is None
