"""Feature engineering for BGP anomaly ML model (Phase 5).

All features are computed from PAST data only, using rolling per-prefix and
per-peer histories. No raw prefix strings, ASNs, or event IDs are used as features.

Allowed features (per spec):
  1.  rov_state_encoded       – ROV state ordinal (Valid=0, NotFound=1, Invalid=2, NoSnapshot=3)
  2.  origin_in_inventory     – bool(0/1): origin ASN authorized in operator inventory
  3.  prefix_len_delta        – (announced_prefix_len - covering_parent_len); 0 if no parent
  4.  origin_new_for_prefix   – bool: first time this origin seen for this prefix in past window
  5.  peer_new_for_prefix_origin – bool: first time peer seen for this (prefix, origin) pair
  6.  more_specific_first_seen – bool: first time this prefix (more-specific) seen at all
  7.  prefix_age_hours        – hours since this prefix was first ever observed
  8.  origin_rarity           – 1 / (1 + count of updates with this origin in past 6h)
  9.  update_rate_30m         – count of updates for this prefix in the past 30 minutes
  10. hour_of_day             – UTC hour of the announcement (0-23)

Leakage safeguards:
  - Features only look BACKWARD in time.
  - No rule-engine output used.
  - No raw prefix/ASN string encoded.
  - Labels come strictly from labels.csv, never from rule alerts.
"""

from __future__ import annotations

import ipaddress
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.inventory import PrefixInventory
from src.replay import BGPUpdate
from src.roa_store import ROAStore
from src.rov import ROVState, validate_route

IPNetworkType = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ROV_ENCODING: Dict[ROVState, int] = {
    ROVState.VALID: 0,
    ROVState.NOT_FOUND: 1,
    ROVState.INVALID: 2,
    ROVState.NO_SNAPSHOT: 3,
}

FEATURE_NAMES: List[str] = [
    "rov_state_encoded",
    "origin_in_inventory",
    "prefix_len_delta",
    "origin_new_for_prefix",
    "peer_new_for_prefix_origin",
    "more_specific_first_seen",
    "prefix_age_hours",
    "origin_rarity",
    "update_rate_30m",
    "hour_of_day",
]


# ─────────────────────────────────────────────────────────────────────────────
# Rolling history state (per-update, online, past-only)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RollingHistory:
    """Online rolling feature state.  All lookups are PAST-DATA-ONLY."""

    # First time a prefix was observed (epoch seconds)
    prefix_first_seen: Dict[str, float] = field(default_factory=dict)
    # Set of origins seen for each prefix
    prefix_origins: DefaultDict[str, set] = field(
        default_factory=lambda: defaultdict(set)
    )
    # Set of peers seen for each (prefix, origin) pair
    prefix_origin_peers: DefaultDict[Tuple[str, int], set] = field(
        default_factory=lambda: defaultdict(set)
    )
    # Timestamps of recent updates per prefix (for update_rate_30m)
    prefix_update_times: DefaultDict[str, Deque[float]] = field(
        default_factory=lambda: defaultdict(deque)
    )
    # Timestamps of recent updates per origin ASN (for origin_rarity over 6h)
    origin_update_times: DefaultDict[int, Deque[float]] = field(
        default_factory=lambda: defaultdict(deque)
    )

    WINDOW_30M: float = 1800.0   # 30 minutes in seconds
    WINDOW_6H: float = 21600.0   # 6 hours in seconds

    def _prune(self, dq: Deque[float], cutoff: float) -> None:
        while dq and dq[0] < cutoff:
            dq.popleft()

    def compute_features(
        self,
        update: BGPUpdate,
        inventory: PrefixInventory,
        roa_store: ROAStore,
    ) -> Dict[str, float]:
        """Compute features for a single update using ONLY past state, then update state."""
        t = update.timestamp
        prefix_str = str(update.prefix)
        origin = update.origin_asn
        peer = update.peer

        # ── Feature 1: ROV state ─────────────────────────────────────────────
        rov_res = validate_route(update.prefix, origin, t, roa_store)
        rov_encoded = float(ROV_ENCODING.get(rov_res.state, 3))

        # ── Feature 2: origin_in_inventory ──────────────────────────────────
        origin_in_inv = float(inventory.is_origin_allowed(update.prefix, origin))

        # ── Feature 3: prefix_len_delta vs covering parent ───────────────────
        covering = inventory.lookup_covering(update.prefix)
        if covering is not None and update.prefix.prefixlen > covering.prefix.prefixlen:
            len_delta = float(update.prefix.prefixlen - covering.prefix.prefixlen)
        else:
            len_delta = 0.0

        # ── Feature 4: origin_new_for_prefix (past state) ────────────────────
        origin_new = float(origin not in self.prefix_origins[prefix_str])

        # ── Feature 5: peer_new_for_prefix_origin (past state) ───────────────
        pair_key = (prefix_str, origin)
        peer_new = float(peer not in self.prefix_origin_peers[pair_key])

        # ── Feature 6: more_specific_first_seen (past state) ─────────────────
        ms_new = float(prefix_str not in self.prefix_first_seen)

        # ── Feature 7: prefix_age_hours (past state) ─────────────────────────
        if prefix_str in self.prefix_first_seen:
            age_hours = (t - self.prefix_first_seen[prefix_str]) / 3600.0
        else:
            age_hours = 0.0

        # ── Feature 8: origin_rarity in past 6h ─────────────────────────────
        cutoff_6h = t - self.WINDOW_6H
        self._prune(self.origin_update_times[origin], cutoff_6h)
        origin_count_6h = len(self.origin_update_times[origin])
        origin_rarity = 1.0 / (1.0 + origin_count_6h)

        # ── Feature 9: update_rate_30m ────────────────────────────────────────
        cutoff_30m = t - self.WINDOW_30M
        self._prune(self.prefix_update_times[prefix_str], cutoff_30m)
        update_rate_30m = float(len(self.prefix_update_times[prefix_str]))

        # ── Feature 10: hour_of_day ───────────────────────────────────────────
        import datetime
        dt_utc = datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc)
        hour_of_day = float(dt_utc.hour)

        feats: Dict[str, float] = {
            "rov_state_encoded": rov_encoded,
            "origin_in_inventory": origin_in_inv,
            "prefix_len_delta": len_delta,
            "origin_new_for_prefix": origin_new,
            "peer_new_for_prefix_origin": peer_new,
            "more_specific_first_seen": ms_new,
            "prefix_age_hours": age_hours,
            "origin_rarity": origin_rarity,
            "update_rate_30m": update_rate_30m,
            "hour_of_day": hour_of_day,
        }

        # ── Update rolling state AFTER computing features (past-only guarantee) ─
        if prefix_str not in self.prefix_first_seen:
            self.prefix_first_seen[prefix_str] = t
        self.prefix_origins[prefix_str].add(origin)
        self.prefix_origin_peers[pair_key].add(peer)
        self.prefix_update_times[prefix_str].append(t)
        self.origin_update_times[origin].append(t)

        return feats


# ─────────────────────────────────────────────────────────────────────────────
# Bulk feature extraction for a full feed
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_from_feed(
    feed_path: str,
    labels_path: str,
    inventory_path: str,
    roas_dir: str,
    announcements_only: bool = True,
) -> pd.DataFrame:
    """Process the entire feed chronologically and return a feature+label DataFrame.

    Only BGP announcements are processed (withdrawals have no ROV state).
    Labels come exclusively from labels.csv matched by (prefix, origin_asn, time window).
    Returns a DataFrame with columns = FEATURE_NAMES + ['label', 'timestamp',
    'scenario', 'benign_event'].
    """
    import csv as _csv

    from src.replay import BGPReplay
    from src.inventory import PrefixInventory
    from src.roa_store import ROAStore

    inv = PrefixInventory.load_from_csv(inventory_path)
    store = ROAStore(roas_dir)
    replay = BGPReplay.load_from_csv(feed_path)

    # ── Load labels (ground truth from labels.csv ONLY) ──────────────────────
    labels: List[Dict[str, Any]] = []
    with open(labels_path, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            labels.append({
                "start_time": float(row["start_time"]),
                "end_time": float(row["end_time"]),
                "prefix": row["prefix"].strip(),
                "origin_asn": int(row["origin_asn"]),
                "benign": row["benign"].lower() == "true",
                "scenario": row["scenario"].strip(),
            })

    # Separate anomaly events (positive class) and benign events (negative class)
    anomaly_labels = [lb for lb in labels if not lb["benign"]]

    def _is_anomaly(t: float, pstr: str, asn: int) -> Tuple[bool, str]:
        """Return (is_anomaly, scenario) for this update."""
        for lb in anomaly_labels:
            if lb["prefix"] == pstr and lb["origin_asn"] == asn:
                if lb["start_time"] - 5.0 <= t <= lb["end_time"] + 5.0:
                    return True, lb["scenario"]
        return False, ""

    history = RollingHistory()
    rows: List[Dict[str, Any]] = []

    for update in replay.stream():
        if announcements_only and not update.is_announcement:
            continue

        prefix_str = str(update.prefix)
        feats = history.compute_features(update, inv, store)

        is_anom, scenario = _is_anomaly(update.timestamp, prefix_str, update.origin_asn)

        row = dict(feats)
        row["label"] = 1 if is_anom else 0
        row["timestamp"] = update.timestamp
        row["scenario"] = scenario if is_anom else "benign"
        rows.append(row)

    df = pd.DataFrame(rows, columns=FEATURE_NAMES + ["label", "timestamp", "scenario"])
    return df
