"""RPKI ROA Time-Aligned Store and Snapshot Manager.

Loads periodic ROA snapshots from disk and resolves covering ROAs for any prefix
at any given timestamp using strict time-alignment:
For an update at time t, select the latest snapshot with snapshot_time <= t.
"""

from __future__ import annotations

import csv
import datetime
import ipaddress
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

IPNetworkType = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]


def parse_timestamp(ts: Union[str, int, float, datetime.datetime]) -> float:
    """Parse various timestamp representations into UTC Unix epoch seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, datetime.datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        return ts.timestamp()

    if isinstance(ts, str):
        ts_clean = ts.strip()
        # Check if it's purely a numeric string (epoch)
        try:
            return float(ts_clean)
        except ValueError:
            pass

        # Replace dashes or underscores in time portion if needed
        # Standardize ISO-8601 strings
        iso_str = ts_clean.replace("Z", "+00:00")
        # Handle filenames like 2026-10-26T00-00-00+00-00
        # If timestamp has '-' instead of ':' in time component (e.g., in Windows-safe filenames)
        match = re.match(r"^(\d{4}-\d{2}-\d{2})[T_](\d{2})[-:](\d{2})[-:](\d{2})(.*)$", ts_clean)
        if match:
            date_part, hh, mm, ss, tz_part = match.groups()
            normalized_tz = "+00:00" if tz_part in ("", "Z") else tz_part.replace("-", ":")
            iso_str = f"{date_part}T{hh}:{mm}:{ss}{normalized_tz}"

        try:
            dt = datetime.datetime.fromisoformat(iso_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt.timestamp()
        except Exception as exc:
            raise ValueError(f"Could not parse timestamp '{ts}': {exc}") from exc

    raise TypeError(f"Unsupported timestamp type: {type(ts)}")


@dataclass(frozen=True)
class ROARecord:
    """Represents a single Route Origin Authorization (ROA) entry."""

    prefix: IPNetworkType
    asn: int
    max_length: int

    def covers(self, announced_prefix: IPNetworkType) -> bool:
        """Check if this ROA covers the announced prefix (i.e. announced is equal or more specific)."""
        if self.prefix.version != announced_prefix.version:
            return False
        return announced_prefix.subnet_of(self.prefix)

    def matches_origin(self, origin_asn: int) -> bool:
        """Check if origin ASN matches the ROA authorized ASN."""
        return self.asn == origin_asn

    def matches_length(self, announced_prefix: IPNetworkType) -> bool:
        """Check if announced prefix length is within the authorized maxLength."""
        return announced_prefix.prefixlen <= self.max_length


class ROASnapshot:
    """An immutable snapshot of ROAs valid at a specific point in time."""

    def __init__(self, snapshot_time: float, source_file: Optional[Path] = None) -> None:
        self.snapshot_time: float = snapshot_time
        self.source_file: Optional[Path] = source_file
        self.records: List[ROARecord] = []
        self._v4_records: List[ROARecord] = []
        self._v6_records: List[ROARecord] = []

    def add_record(self, prefix: str | IPNetworkType, asn: int, max_length: Optional[int] = None) -> ROARecord:
        """Add a ROA record to this snapshot."""
        if not isinstance(prefix, (ipaddress.IPv4Network, ipaddress.IPv6Network)):
            net = ipaddress.ip_network(str(prefix).strip(), strict=False)
        else:
            net = prefix

        max_len = int(max_length) if max_length is not None else net.prefixlen
        record = ROARecord(prefix=net, asn=int(asn), max_length=max_len)
        self.records.append(record)
        if net.version == 4:
            self._v4_records.append(record)
        else:
            self._v6_records.append(record)
        return record

    def find_covering_roas(self, prefix: str | IPNetworkType) -> List[ROARecord]:
        """Return all ROAs covering the announced prefix."""
        if not isinstance(prefix, (ipaddress.IPv4Network, ipaddress.IPv6Network)):
            net = ipaddress.ip_network(str(prefix).strip(), strict=False)
        else:
            net = prefix

        target_records = self._v4_records if net.version == 4 else self._v6_records
        return [r for r in target_records if r.covers(net)]

    def __len__(self) -> int:
        return len(self.records)

    @classmethod
    def load_from_csv(cls, filepath: str | Path, snapshot_time: Optional[float] = None) -> ROASnapshot:
        """Load ROA records from a CSV file.

        Expected headers: prefix, asn, maxLength
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"ROA snapshot file not found: {path}")

        if snapshot_time is None:
            # Infer timestamp from filename: roas_<timestamp>.csv
            stem = path.stem
            prefix_tag = "roas_"
            if stem.startswith(prefix_tag):
                ts_str = stem[len(prefix_tag):]
                snapshot_time = parse_timestamp(ts_str)
            else:
                snapshot_time = path.stat().st_mtime

        snapshot = cls(snapshot_time=snapshot_time, source_file=path)
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p_val = row["prefix"].strip()
                asn_val = int(row["asn"].strip())
                max_len_val = row.get("maxLength", row.get("max_length", "")).strip()
                max_len = int(max_len_val) if max_len_val else None
                snapshot.add_record(p_val, asn_val, max_len)

        return snapshot


class ROAStore:
    """Manages multiple historical ROA snapshots and performs time-aligned lookups."""

    def __init__(self, snapshots_dir: Optional[str | Path] = None) -> None:
        self.snapshots_dir: Optional[Path] = Path(snapshots_dir) if snapshots_dir else None
        # List of tuples: (snapshot_time: float, path: Path)
        self._index: List[Tuple[float, Path]] = []
        # In-memory cache for loaded ROASnapshot objects
        self._cache: Dict[Path, ROASnapshot] = {}

        if self.snapshots_dir and self.snapshots_dir.exists():
            self.refresh_index()

    def register_snapshot(self, filepath: str | Path, snapshot_time: Optional[float] = None) -> float:
        """Register a single snapshot file into the store."""
        path = Path(filepath)
        if snapshot_time is None:
            stem = path.stem
            prefix_tag = "roas_"
            if stem.startswith(prefix_tag):
                ts_str = stem[len(prefix_tag):]
                snapshot_time = parse_timestamp(ts_str)
            else:
                snapshot_time = path.stat().st_mtime

        self._index.append((snapshot_time, path))
        self._index.sort(key=lambda item: item[0])
        return snapshot_time

    def refresh_index(self) -> None:
        """Scan snapshots_dir for roas_*.csv files and re-index them chronologically."""
        if not self.snapshots_dir or not self.snapshots_dir.exists():
            return

        self._index.clear()
        for p in self.snapshots_dir.glob("roas_*.csv"):
            stem = p.stem
            ts_str = stem[len("roas_"):]
            try:
                t = parse_timestamp(ts_str)
                self._index.append((t, p))
            except Exception:
                # If timestamp parsing fails, skip or use modification time
                self._index.append((p.stat().st_mtime, p))

        self._index.sort(key=lambda item: item[0])

    def get_snapshot_for_time(
        self,
        update_time: Union[str, int, float, datetime.datetime],
        fallback_to_earliest: bool = False,
    ) -> Optional[ROASnapshot]:
        """Find the latest snapshot with snapshot_time <= update_time.

        If update_time is strictly earlier than all snapshots:
          - returns None if fallback_to_earliest is False
          - returns the earliest snapshot if fallback_to_earliest is True
        """
        t = parse_timestamp(update_time)
        if not self._index:
            return None

        # Binary search / linear scan for largest snapshot_time <= t
        candidate: Optional[Tuple[float, Path]] = None
        for snap_time, snap_path in self._index:
            if snap_time <= t:
                candidate = (snap_time, snap_path)
            else:
                break

        if candidate is None:
            if fallback_to_earliest and self._index:
                candidate = self._index[0]
            else:
                return None

        snap_path = candidate[1]
        if snap_path not in self._cache:
            self._cache[snap_path] = ROASnapshot.load_from_csv(snap_path, candidate[0])

        return self._cache[snap_path]

    def find_covering_roas(
        self,
        prefix: str | IPNetworkType,
        update_time: Union[str, int, float, datetime.datetime],
    ) -> List[ROARecord]:
        """Retrieve covering ROAs for a prefix using the time-aligned snapshot."""
        snapshot = self.get_snapshot_for_time(update_time)
        if snapshot is None:
            return []
        return snapshot.find_covering_roas(prefix)

    @property
    def snapshot_count(self) -> int:
        return len(self._index)

    @property
    def snapshot_timestamps(self) -> List[float]:
        return [ts for ts, _ in self._index]
