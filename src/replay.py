"""Time-Ordered BGP Update Feed Replay Engine.

Normalizes updates from CSV feeds and collector dumps into a canonical schema:
  - timestamp (float, epoch seconds)
  - prefix (ipaddress network object)
  - origin_asn (int)
  - peer (str)
  - action (str: 'announce' or 'withdraw')

Ensures strictly time-ordered playback for online or batch anomaly detection.
"""

from __future__ import annotations

import csv
import datetime
import ipaddress
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from src.roa_store import parse_timestamp

IPNetworkType = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]


@dataclass(frozen=True)
class BGPUpdate:
    """Canonical representation of a single BGP update."""

    timestamp: float
    prefix: IPNetworkType
    origin_asn: int
    peer: str = "unknown"
    action: str = "announce"
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_announcement(self) -> bool:
        return self.action.lower() in ("announce", "a", "update", "announcement")

    @property
    def is_withdrawal(self) -> bool:
        return self.action.lower() in ("withdraw", "w", "withdrawal")

    @property
    def prefix_str(self) -> str:
        return str(self.prefix)

    @property
    def datetime_utc(self) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(self.timestamp, tz=datetime.timezone.utc)

    def __lt__(self, other: BGPUpdate) -> bool:
        return self.timestamp < other.timestamp


class BGPReplay:
    """Loads, validates, sorts, and streams BGP updates in strict chronological order."""

    def __init__(self, updates: Optional[List[BGPUpdate]] = None) -> None:
        self.updates: List[BGPUpdate] = updates or []
        self._is_sorted: bool = False

    def add_update(self, update: BGPUpdate) -> None:
        """Add a single update to the replay buffer."""
        self.updates.append(update)
        self._is_sorted = False

    def sort_by_time(self) -> None:
        """Ensure all updates in the buffer are strictly ordered by timestamp."""
        self.updates.sort(key=lambda u: u.timestamp)
        self._is_sorted = True

    def __len__(self) -> int:
        return len(self.updates)

    def stream(
        self,
        start_time: Optional[Union[float, str]] = None,
        end_time: Optional[Union[float, str]] = None,
        announce_only: bool = False,
        speed_multiplier: float = 0.0,
    ) -> Iterator[BGPUpdate]:
        """Stream updates in chronological order.

        Args:
            start_time: Earliest timestamp to yield.
            end_time: Latest timestamp to yield.
            announce_only: If True, skip withdrawals.
            speed_multiplier: If > 0, sleep proportionally to simulate real-time playback.
                              If 0.0, yields instantly (batch mode).
        """
        if not self._is_sorted:
            self.sort_by_time()

        t_start = parse_timestamp(start_time) if start_time is not None else None
        t_end = parse_timestamp(end_time) if end_time is not None else None

        last_ts: Optional[float] = None

        for update in self.updates:
            if t_start is not None and update.timestamp < t_start:
                continue
            if t_end is not None and update.timestamp > t_end:
                break
            if announce_only and not update.is_announcement:
                continue

            if speed_multiplier > 0 and last_ts is not None:
                delta = max(0.0, update.timestamp - last_ts)
                sleep_duration = delta / speed_multiplier
                # Cap sleep to avoid hanging on long gaps
                time.sleep(min(sleep_duration, 1.0))

            last_ts = update.timestamp
            yield update

    @classmethod
    def load_from_csv(
        cls,
        filepath: str | Path,
        sort: bool = True,
    ) -> BGPReplay:
        """Load normalized BGP updates from a CSV file.

        Expected CSV columns:
          timestamp,prefix,origin_asn,peer,action
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Feed file not found: {path}")

        updates: List[BGPUpdate] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_val = parse_timestamp(row["timestamp"].strip())
                prefix_val = ipaddress.ip_network(row["prefix"].strip(), strict=False)
                origin_asn = int(row["origin_asn"].strip())
                peer_val = row.get("peer", "unknown").strip()
                action_val = row.get("action", "announce").strip().lower()

                update = BGPUpdate(
                    timestamp=ts_val,
                    prefix=prefix_val,
                    origin_asn=origin_asn,
                    peer=peer_val,
                    action=action_val,
                    raw_data=dict(row),
                )
                updates.append(update)

        replay = cls(updates)
        if sort:
            replay.sort_by_time()
        return replay

    @classmethod
    def load_from_raw_ris(
        cls,
        filepath: str | Path,
        sort: bool = True,
    ) -> BGPReplay:
        """Load updates from a RIPE RIS dump format or tab/pipe-delimited text.

        Accepts standard RIS output formats (e.g. TABLE_DUMP or BGP4MP lines):
        timestamp|peer_ip|peer_as|prefix|as_path|origin_as
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Raw RIS dump file not found: {path}")

        updates: List[BGPUpdate] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Support pipe, tab, or comma delimiter
                if "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                elif "\t" in line:
                    parts = [p.strip() for p in line.split("\t")]
                else:
                    parts = [p.strip() for p in line.split(",")]

                if len(parts) >= 4:
                    try:
                        ts_val = parse_timestamp(parts[0])
                        prefix_val = ipaddress.ip_network(parts[1], strict=False)
                        # If AS path is in field 2 and origin in field 3 or 4:
                        # Extract last AS from path or direct origin
                        asn_str = parts[-1]
                        if " " in asn_str:
                            asn_str = asn_str.split()[-1]
                        origin_asn = int(asn_str)
                        peer_val = parts[2] if len(parts) > 3 else "unknown"

                        updates.append(
                            BGPUpdate(
                                timestamp=ts_val,
                                prefix=prefix_val,
                                origin_asn=origin_asn,
                                peer=peer_val,
                                action="announce",
                            )
                        )
                    except Exception:
                        continue

        replay = cls(updates)
        if sort:
            replay.sort_by_time()
        return replay
