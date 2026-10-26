"""Operator Prefix Inventory Management.

Provides exact and covering-prefix (longest prefix match) lookups for operator-owned
prefixes and their authorized origin Autonomous System Numbers (ASNs).
"""

from __future__ import annotations

import csv
import ipaddress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Set, Union

IPNetworkType = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]


def parse_network(prefix_str: str | IPNetworkType) -> IPNetworkType:
    """Safely parse an IPv4 or IPv6 prefix string into an ipaddress network object."""
    if isinstance(prefix_str, (ipaddress.IPv4Network, ipaddress.IPv6Network)):
        return prefix_str
    return ipaddress.ip_network(prefix_str.strip(), strict=False)


def parse_origin_asns(val: Union[int, str, Iterable[int]]) -> Set[int]:
    """Parse origin ASN(s) from integer, list, or delimited string (e.g., '13335, 13336' or '13335;13336')."""
    if isinstance(val, int):
        return {val}
    if isinstance(val, (set, list, tuple)):
        return {int(x) for x in val}
    if isinstance(val, str):
        # Handle delimiters like comma, semicolon, space, or JSON/Python list brackets
        cleaned = val.replace("[", "").replace("]", "").replace(";", ",")
        asns = set()
        for part in cleaned.split(","):
            part = part.strip()
            if part:
                asns.add(int(part))
        return asns
    raise ValueError(f"Unsupported ASN representation: {val} (type {type(val)})")


@dataclass(frozen=True)
class InventoryItem:
    """Represents an operator-owned prefix inventory record."""

    prefix: IPNetworkType
    expected_origin_asns: Set[int] = field(default_factory=set)
    max_expected_length: int = 0
    owner: str = "Unknown"

    def __post_init__(self) -> None:
        if self.max_expected_length < self.prefix.prefixlen:
            # Default max_expected_length to at least the prefix length itself
            object.__setattr__(self, "max_expected_length", self.prefix.prefixlen)

    def is_asn_authorized(self, origin_asn: int) -> bool:
        """Check if an origin ASN is in the authorized set."""
        return origin_asn in self.expected_origin_asns

    def is_length_authorized(self, prefix_len: int) -> bool:
        """Check if a subprefix length is within max_expected_length."""
        return prefix_len <= self.max_expected_length


class PrefixInventory:
    """Stores and queries an operator's IP prefix inventory."""

    def __init__(self) -> None:
        self._items: dict[IPNetworkType, InventoryItem] = {}

    def add(
        self,
        prefix: str | IPNetworkType,
        expected_origin_asns: int | str | Iterable[int],
        max_expected_length: Optional[int] = None,
        owner: str = "Unknown",
    ) -> InventoryItem:
        """Add or update an inventory entry."""
        net = parse_network(prefix)
        asns = parse_origin_asns(expected_origin_asns)
        max_len = max_expected_length if max_expected_length is not None else net.prefixlen
        item = InventoryItem(
            prefix=net,
            expected_origin_asns=asns,
            max_expected_length=max_len,
            owner=owner.strip(),
        )
        self._items[net] = item
        return item

    def lookup_exact(self, prefix: str | IPNetworkType) -> Optional[InventoryItem]:
        """Perform an exact match lookup for a prefix."""
        net = parse_network(prefix)
        return self._items.get(net)

    def lookup_covering(self, prefix: str | IPNetworkType) -> Optional[InventoryItem]:
        """Find the longest covering prefix (Longest Prefix Match) in inventory.

        If the prefix itself is in inventory, it returns that exact match.
        Otherwise, returns the most specific parent prefix that encloses it.
        """
        net = parse_network(prefix)
        best_match: Optional[InventoryItem] = None

        for item_net, item in self._items.items():
            if net.version != item_net.version:
                continue
            # Check if net is equal to or a subnet of item_net
            if net.subnet_of(item_net):
                if best_match is None or item_net.prefixlen > best_match.prefix.prefixlen:
                    best_match = item

        return best_match

    def is_origin_allowed(self, prefix: str | IPNetworkType, origin_asn: int) -> bool:
        """Check if the origin ASN is authorized under exact or covering inventory entry."""
        item = self.lookup_exact(prefix) or self.lookup_covering(prefix)
        if item is None:
            return False
        return item.is_asn_authorized(origin_asn)

    def __len__(self) -> int:
        return len(self._items)

    def all_items(self) -> list[InventoryItem]:
        """Return all inventory items sorted by prefix."""
        return sorted(self._items.values(), key=lambda item: (item.prefix.version, item.prefix))

    @classmethod
    def load_from_csv(cls, filepath: str | Path) -> PrefixInventory:
        """Load inventory records from a CSV file.

        Expected CSV headers: prefix, expected_origin_asns, max_expected_length, owner
        """
        inv = cls()
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Inventory file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prefix = row["prefix"].strip()
                asns = row["expected_origin_asns"].strip()
                max_len_val = row.get("max_expected_length", "").strip()
                max_len = int(max_len_val) if max_len_val else None
                owner = row.get("owner", "Unknown").strip()
                inv.add(prefix, asns, max_len, owner)

        return inv
