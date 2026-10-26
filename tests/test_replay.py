"""Unit tests for BGP Feed Replay Engine (Phase 1)."""

import ipaddress
import tempfile
from pathlib import Path

import pytest
from src.replay import BGPReplay, BGPUpdate


def test_bgp_update_properties() -> None:
    u = BGPUpdate(
        timestamp=1700000000.0,
        prefix=ipaddress.ip_network("198.51.100.0/24"),
        origin_asn=13335,
        peer="192.0.2.1",
        action="announce",
    )
    assert u.is_announcement is True
    assert u.is_withdrawal is False
    assert u.prefix_str == "198.51.100.0/24"


def test_time_ordering_guarantee() -> None:
    # Intentionally out-of-order updates
    u1 = BGPUpdate(100.0, ipaddress.ip_network("10.0.0.0/24"), 65001)
    u2 = BGPUpdate(50.0, ipaddress.ip_network("10.0.1.0/24"), 65002)
    u3 = BGPUpdate(150.0, ipaddress.ip_network("10.0.2.0/24"), 65003)

    replay = BGPReplay([u1, u2, u3])
    ordered = list(replay.stream())

    assert [u.timestamp for u in ordered] == [50.0, 100.0, 150.0]
    assert [u.origin_asn for u in ordered] == [65002, 65001, 65003]


def test_replay_time_filtering() -> None:
    u1 = BGPUpdate(100.0, ipaddress.ip_network("10.0.0.0/24"), 65001)
    u2 = BGPUpdate(200.0, ipaddress.ip_network("10.0.1.0/24"), 65002)
    u3 = BGPUpdate(300.0, ipaddress.ip_network("10.0.2.0/24"), 65003)

    replay = BGPReplay([u1, u2, u3])
    filtered = list(replay.stream(start_time=150.0, end_time=250.0))

    assert len(filtered) == 1
    assert filtered[0].timestamp == 200.0


def test_replay_announce_only_filter() -> None:
    u1 = BGPUpdate(100.0, ipaddress.ip_network("10.0.0.0/24"), 65001, action="announce")
    u2 = BGPUpdate(110.0, ipaddress.ip_network("10.0.0.0/24"), 65001, action="withdraw")

    replay = BGPReplay([u1, u2])
    all_updates = list(replay.stream(announce_only=False))
    announces_only = list(replay.stream(announce_only=True))

    assert len(all_updates) == 2
    assert len(announces_only) == 1
    assert announces_only[0].action == "announce"


def test_load_from_csv() -> None:
    csv_data = """timestamp,prefix,origin_asn,peer,action
2026-10-26T02:00:00Z,203.0.113.0/24,64496,192.0.2.1,announce
2026-10-26T01:00:00Z,198.51.100.0/24,13335,192.0.2.1,announce
2026-10-26T03:00:00Z,2001:db8::/32,64500,192.0.2.2,announce
"""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp.write(csv_data)
        tmp_path = Path(tmp.name)

    try:
        replay = BGPReplay.load_from_csv(tmp_path)
        assert len(replay) == 3
        # Should be sorted chronologically: 01:00, then 02:00, then 03:00
        items = list(replay.stream())
        assert items[0].origin_asn == 13335
        assert items[1].origin_asn == 64496
        assert items[2].origin_asn == 64500
        assert items[2].prefix == ipaddress.ip_network("2001:db8::/32")
    finally:
        tmp_path.unlink(missing_ok=True)


def test_load_from_raw_ris() -> None:
    ris_data = """# Timestamp|Prefix|Peer|AS_Path|Origin_ASN
2026-10-26T01:05:00Z|198.51.100.0/24|192.0.2.1|65001 174 13335|13335
2026-10-26T01:10:00Z|2001:db8:1000::/36|192.0.2.2|65002 6939 13335|13335
"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        tmp.write(ris_data)
        tmp_path = Path(tmp.name)

    try:
        replay = BGPReplay.load_from_raw_ris(tmp_path)
        assert len(replay) == 2
        items = list(replay.stream())
        assert items[0].origin_asn == 13335
        assert items[1].prefix == ipaddress.ip_network("2001:db8:1000::/36")
    finally:
        tmp_path.unlink(missing_ok=True)
