"""Unit tests for Prefix Inventory management (Phase 1)."""

import ipaddress
import tempfile
from pathlib import Path

import pytest
from src.inventory import InventoryItem, PrefixInventory, parse_network, parse_origin_asns


def test_parse_network() -> None:
    net_v4 = parse_network("198.51.100.0/22")
    assert isinstance(net_v4, ipaddress.IPv4Network)
    assert net_v4.prefixlen == 22

    net_v6 = parse_network("2001:db8::/32")
    assert isinstance(net_v6, ipaddress.IPv6Network)
    assert net_v6.prefixlen == 32


def test_parse_origin_asns() -> None:
    assert parse_origin_asns(13335) == {13335}
    assert parse_origin_asns("13335") == {13335}
    assert parse_origin_asns("13335, 13336") == {13335, 13336}
    assert parse_origin_asns("64500; 64501; 64502") == {64500, 64501, 64502}
    assert parse_origin_asns([64496, 64497]) == {64496, 64497}


def test_inventory_exact_and_covering_lookup() -> None:
    inv = PrefixInventory()
    inv.add("198.51.100.0/22", expected_origin_asns="13335, 13336", max_expected_length=24, owner="CloudNet")
    inv.add("198.51.100.0/24", expected_origin_asns=13335, max_expected_length=24, owner="CloudNet-Specific")
    inv.add("2001:db8:1000::/36", expected_origin_asns=13335, max_expected_length=48, owner="CloudNet-v6")

    # Exact match on /22
    item_22 = inv.lookup_exact("198.51.100.0/22")
    assert item_22 is not None
    assert item_22.owner == "CloudNet"
    assert item_22.expected_origin_asns == {13335, 13336}

    # Exact match on /24
    item_24 = inv.lookup_exact("198.51.100.0/24")
    assert item_24 is not None
    assert item_24.owner == "CloudNet-Specific"

    # Covering match: 198.51.101.0/24 is not exact in inventory, but covered by /22
    assert inv.lookup_exact("198.51.101.0/24") is None
    cov_item = inv.lookup_covering("198.51.101.0/24")
    assert cov_item is not None
    assert cov_item.prefix == ipaddress.ip_network("198.51.100.0/22")
    assert cov_item.owner == "CloudNet"

    # IPv6 covering match
    cov_v6 = inv.lookup_covering("2001:db8:1000:5::/64")
    assert cov_v6 is not None
    assert cov_v6.prefix == ipaddress.ip_network("2001:db8:1000::/36")

    # Outside inventory returns None
    assert inv.lookup_covering("8.8.8.0/24") is None
    assert inv.lookup_covering("2606:4700::/32") is None


def test_inventory_authorization_checks() -> None:
    inv = PrefixInventory()
    inv.add("192.0.2.0/24", expected_origin_asns="64500, 64501", max_expected_length=24, owner="FinTech")

    assert inv.is_origin_allowed("192.0.2.0/24", 64500) is True
    assert inv.is_origin_allowed("192.0.2.0/24", 64501) is True
    assert inv.is_origin_allowed("192.0.2.0/24", 64599) is False


def test_inventory_load_from_csv() -> None:
    csv_content = """prefix,expected_origin_asns,max_expected_length,owner
198.51.100.0/22,"13335, 13336",24,Cloudflare-Sample
203.0.113.0/24,64496,24,Example-Net
2001:db8:feed::/48,64520,48,SecureCorp-v6
"""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp.write(csv_content)
        tmp_path = Path(tmp.name)

    try:
        inv = PrefixInventory.load_from_csv(tmp_path)
        assert len(inv) == 3
        assert inv.is_origin_allowed("203.0.113.0/24", 64496) is True
        assert inv.is_origin_allowed("203.0.113.0/24", 99999) is False
        assert inv.lookup_exact("2001:db8:feed::/48") is not None
    finally:
        tmp_path.unlink(missing_ok=True)
