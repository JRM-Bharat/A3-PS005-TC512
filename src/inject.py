"""Synthetic BGP Anomaly and Benign Scenario Generator (v2).

Generates realistic background traffic and injected labeled anomaly and benign scenarios:
  - data/feed_replay.csv: Normalized chronological BGP update feed
  - data/labels.csv: Ground-truth event labels with scenario tags, difficulty, and expected alert types
  - data/manifest.json: Run configuration, seed, changelog, and summary statistics

Changelog (v2):
  - Made loose_maxlength_hijack a genuine hard anomaly: attacker forges legitimate origin ASN
    on a more-specific within ROA maxLength, causing ROV to be Valid and origin in inventory.
    The rule baseline is expected to miss these.
  - Split policy lag into two distinct scenarios:
      * policy_change_roa_lag_inventory_updated (origin in inventory, ROV Invalid -> expected ROA_INVALID)
      * policy_change_roa_lag_inventory_stale (origin not in inventory, ROV Invalid -> known FP ORIGIN_HIJACK)
  - Added ~1.5% realistic third-party RFC 6811 Invalid background routes outside operator inventory
    tagged as THIRD_PARTY_INVALID.
  - Balanced peer novelty across benign TE and forged anomalies to prevent trivial feature leakage.
  - Multi-slice time-spread support for ML scale datasets: spreads events across Train (60%),
    Val (20%), and Test (20%) boundaries with zero feature leakage.
"""

from __future__ import annotations

import csv
import datetime
import ipaddress
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.inventory import PrefixInventory
from src.replay import BGPUpdate
from src.roa_store import ROAStore, parse_timestamp


@dataclass
class LabelledEvent:
    """Ground-truth label for a simulated routing event."""

    event_id: str
    start_time: float
    end_time: float
    prefix: str
    origin_asn: int
    type: str
    benign: bool
    scenario: str
    difficulty: str  # 'easy' or 'hard'
    expected_alert_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ScenarioInjector:
    """Generates synthetic background traffic and injected scenarios with deterministic seeds."""

    def __init__(
        self,
        config_path: str | Path = "config.yaml",
        seed: Optional[int] = None,
    ) -> None:
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        sim_cfg = self.config.get("simulation", {})
        self.seed = seed if seed is not None else sim_cfg.get("random_seed", 42)
        self.rng = random.Random(self.seed)

        start_str = sim_cfg.get("start_time", "2026-10-26T00:00:00Z")
        self.start_time = parse_timestamp(start_str)
        duration_hrs = sim_cfg.get("duration_hours", 24)
        self.end_time = self.start_time + duration_hrs * 3600.0

        self.background_count = sim_cfg.get("background_count", 3000)
        self.known_peers = sim_cfg.get(
            "peers",
            ["198.32.160.1", "198.32.176.1", "198.32.124.1", "206.108.34.1"],
        )
        self.novel_peer = "198.32.240.1"  # Novel peer observed for feature analysis
        self.all_peers = self.known_peers + [self.novel_peer]

        self.scenario_counts = sim_cfg.get("scenario_counts", {})
        self.spread_full_24h = sim_cfg.get("spread_forged_origin_full_24h", False)

        # Snapshot epoch timestamps
        self.snap1_time = self.start_time                    # 00:00:00Z
        self.snap2_time = self.start_time + 6 * 3600.0       # 06:00:00Z
        self.snap3_time = self.start_time + 12 * 3600.0      # 12:00:00Z
        self.snap4_time = self.start_time + 18 * 3600.0      # 18:00:00Z (Val boundary)
        self.snap5_time = self.start_time + 21 * 3600.0      # 21:00:00Z (Test boundary)

        # Slice boundaries
        self.t_val_start = self.start_time + 14.4 * 3600.0
        self.t_test_start = self.start_time + 19.2 * 3600.0

        # Load inventory to know monitored prefixes
        inv_path = self.config.get("paths", {}).get("inventory", "data/inventory.csv")
        self.inventory = PrefixInventory.load_from_csv(inv_path)

        # Background external Internet prefixes (benign non-inventory)
        self.external_prefixes = [
            "1.1.1.0/24",
            "8.8.8.0/24",
            "93.184.216.0/24",
            "151.101.0.0/16",
            "204.79.197.0/24",
            "140.82.112.0/20",
            "104.16.0.0/12",
            "185.199.108.0/22",
            "2606:4700::/32",
            "2001:4860:4860::/48",
        ]
        self.external_asns = [13335, 15169, 16509, 32934, 2914, 3356, 6939, 54113]

        # Dynamic background prefixes pool for scale runs (models realistic Internet route churn)
        self.dynamic_bg_prefixes: List[Tuple[ipaddress.IPv4Network, float, int]] = []
        if self.spread_full_24h:
            for k in range(250):
                p = ipaddress.ip_network(f"199.{k // 256}.{k % 256}.0/24")
                t_act = self.rng.uniform(self.start_time + 300.0, self.end_time - 1800.0)
                asn = 20000 + (k % 80)
                self.dynamic_bg_prefixes.append((p, t_act, asn))
            self.tail_asns = [20000 + k for k in range(100)]
        else:
            self.tail_asns = []

    def _random_time_in_range(self, t_min: float, t_max: float) -> float:
        return round(self.rng.uniform(t_min, t_max), 2)

    def _random_peer(self, novel_prob: float = 0.20) -> str:
        """Randomly select a peer with novel_prob probability of picking novel_peer."""
        if self.rng.random() < novel_prob:
            return self.novel_peer
        return self.rng.choice(self.known_peers)

    def _distribute_across_slices(self, count: int) -> Tuple[int, int, int]:
        """Distribute count proportionally: 60% Train, 20% Val, 20% Test."""
        n_train = int(round(count * 0.60))
        n_val = int(round(count * 0.20))
        n_test = count - n_train - n_val
        return n_train, n_val, n_test

    def generate(self) -> Tuple[List[BGPUpdate], List[LabelledEvent], Dict[str, Any]]:
        """Generate complete dataset with background traffic and all injected scenarios."""
        updates: List[BGPUpdate] = []
        events: List[LabelledEvent] = []
        scenario_stats: Dict[str, int] = {}

        event_counter = 1

        # ---------------------------------------------------------
        # 1. Anomaly Scenarios (benign = False)
        # ---------------------------------------------------------

        # 1.1 exact_origin_hijack
        count = self.scenario_counts.get("exact_origin_hijack", 12)
        scenario_stats["exact_origin_hijack"] = count
        inv_items = self.inventory.all_items()
        for i in range(count):
            item = inv_items[i % len(inv_items)]
            if self.spread_full_24h:
                t_start = self.start_time + 1800.0 + (i * ((self.end_time - self.start_time - 3600.0) / count))
            else:
                t_start = self._random_time_in_range(self.start_time + 1800, self.end_time - 3600)
            duration = self.rng.randint(600, 2400)
            t_end = t_start + duration
            hijack_asn = 65500 + self.rng.randint(1, 999)
            while item.is_asn_authorized(hijack_asn):
                hijack_asn += 1

            eid = f"evt-{event_counter:04d}"
            event_counter += 1
            evt = LabelledEvent(
                event_id=eid,
                start_time=t_start,
                end_time=t_end,
                prefix=str(item.prefix),
                origin_asn=hijack_asn,
                type="ORIGIN_HIJACK",
                benign=False,
                scenario="exact_origin_hijack",
                difficulty="easy",
                expected_alert_type="ORIGIN_HIJACK",
            )
            events.append(evt)

            peer = self._random_peer(0.30) if self.spread_full_24h else self.rng.choice(self.known_peers)
            peer2 = self._random_peer(0.30) if self.spread_full_24h else self.rng.choice(self.known_peers)
            updates.append(BGPUpdate(t_start, item.prefix, hijack_asn, peer, "announce"))
            updates.append(BGPUpdate(t_start + 15.0, item.prefix, hijack_asn, peer2, "announce"))
            updates.append(BGPUpdate(t_end, item.prefix, hijack_asn, peer, "withdraw"))

        # 1.2 subprefix_hijack
        count = self.scenario_counts.get("subprefix_hijack", 12)
        scenario_stats["subprefix_hijack"] = count
        for i in range(count):
            if self.spread_full_24h:
                t_start = self.start_time + 1800.0 + (i * ((self.end_time - self.start_time - 3600.0) / count))
            else:
                t_start = self._random_time_in_range(self.start_time + 1800, self.end_time - 3600)
            duration = self.rng.randint(600, 1800)
            t_end = t_start + duration
            hijack_asn = 65600 + self.rng.randint(1, 999)

            if i % 2 == 0:
                sub_net = ipaddress.ip_network(f"198.51.100.{128 if (i % 4 == 0) else 0}/25")
            else:
                sub_net = ipaddress.ip_network(f"198.18.{i + 10}.0/24")

            eid = f"evt-{event_counter:04d}"
            event_counter += 1
            evt = LabelledEvent(
                event_id=eid,
                start_time=t_start,
                end_time=t_end,
                prefix=str(sub_net),
                origin_asn=hijack_asn,
                type="SUBPREFIX_HIJACK",
                benign=False,
                scenario="subprefix_hijack",
                difficulty="easy",
                expected_alert_type="SUBPREFIX_HIJACK",
            )
            events.append(evt)

            peer = self._random_peer(0.30) if self.spread_full_24h else self.rng.choice(self.known_peers)
            peer2 = self._random_peer(0.30) if self.spread_full_24h else self.rng.choice(self.known_peers)
            updates.append(BGPUpdate(t_start, sub_net, hijack_asn, peer, "announce"))
            updates.append(BGPUpdate(t_start + 20.0, sub_net, hijack_asn, peer2, "announce"))
            updates.append(BGPUpdate(t_end, sub_net, hijack_asn, peer, "withdraw"))

        # 1.3 hijack_after_roa_change (tests snapshot time alignment boundary)
        count = self.scenario_counts.get("hijack_after_roa_change", 8)
        scenario_stats["hijack_after_roa_change"] = count
        if self.spread_full_24h:
            n_tr, n_va, n_te = self._distribute_across_slices(count)
            boundaries = (
                [(self.snap2_time if k % 2 == 0 else self.snap3_time) for k in range(n_tr)] +
                [self.snap4_time for _ in range(n_va)] +
                [self.snap5_time for _ in range(n_te)]
            )
            for i, boundary in enumerate(boundaries):
                t_start = boundary + self.rng.randint(120, 600) + (i % 5) * 60.0
                t_end = t_start + self.rng.randint(600, 1500)
                hijack_asn = 65700 + i
                target_prefix = ipaddress.ip_network("203.0.113.0/24" if (i % 2 == 0) else "198.51.100.0/22")

                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                evt = LabelledEvent(
                    event_id=eid,
                    start_time=t_start,
                    end_time=t_end,
                    prefix=str(target_prefix),
                    origin_asn=hijack_asn,
                    type="ORIGIN_HIJACK",
                    benign=False,
                    scenario="hijack_after_roa_change",
                    difficulty="hard",
                    expected_alert_type="ORIGIN_HIJACK",
                )
                events.append(evt)

                peer = self._random_peer(0.30)
                updates.append(BGPUpdate(t_start, target_prefix, hijack_asn, peer, "announce"))
                updates.append(BGPUpdate(t_end, target_prefix, hijack_asn, peer, "withdraw"))
        else:
            for i in range(count):
                boundary = self.snap2_time if (i % 2 == 0) else self.snap3_time
                t_start = boundary + self.rng.randint(120, 600)
                t_end = t_start + self.rng.randint(600, 1800)
                hijack_asn = 65700 + i
                target_prefix = ipaddress.ip_network("203.0.113.0/24" if (i % 2 == 0) else "198.51.100.0/22")

                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                evt = LabelledEvent(
                    event_id=eid,
                    start_time=t_start,
                    end_time=t_end,
                    prefix=str(target_prefix),
                    origin_asn=hijack_asn,
                    type="ORIGIN_HIJACK",
                    benign=False,
                    scenario="hijack_after_roa_change",
                    difficulty="hard",
                    expected_alert_type="ORIGIN_HIJACK",
                )
                events.append(evt)

                peer = self.rng.choice(self.known_peers)
                updates.append(BGPUpdate(t_start, target_prefix, hijack_asn, peer, "announce"))
                updates.append(BGPUpdate(t_end, target_prefix, hijack_asn, peer, "withdraw"))

        # 1.4 loose_maxlength_hijack (Genuine hard anomaly: forged legitimate origin within maxLength)
        count = self.scenario_counts.get("loose_maxlength_hijack", 8)
        scenario_stats["loose_maxlength_hijack"] = count
        if self.spread_full_24h:
            t_step = (self.end_time - self.start_time - 3600.0) / count
            for i in range(count):
                t_start = self.start_time + 1800.0 + i * t_step + self.rng.uniform(-t_step * 0.15, t_step * 0.15)
                t_end = t_start + self.rng.randint(900, 2400)
                sub_prefix = ipaddress.ip_network(f"198.19.{i % 64}.0/24")
                forged_origin = 64510

                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                evt = LabelledEvent(
                    event_id=eid,
                    start_time=t_start,
                    end_time=t_end,
                    prefix=str(sub_prefix),
                    origin_asn=forged_origin,
                    type="FORGED_ORIGIN_SUBPREFIX",
                    benign=False,
                    scenario="loose_maxlength_hijack",
                    difficulty="hard",
                    expected_alert_type="NONE",
                )
                events.append(evt)

                # ~40-50% from KNOWN peers, ~50-60% from NOVEL peer (randomly drawn per update!)
                peer = self._random_peer(novel_prob=0.55)
                updates.append(BGPUpdate(t_start, sub_prefix, forged_origin, peer, "announce"))
                updates.append(BGPUpdate(t_end, sub_prefix, forged_origin, peer, "withdraw"))
        else:
            for i in range(count):
                t_start = self._random_time_in_range(self.start_time + 3600, self.end_time - 3600)
                t_end = t_start + self.rng.randint(900, 2400)
                sub_prefix = ipaddress.ip_network(f"198.18.{200 + i}.0/24")
                forged_origin = 64510

                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                evt = LabelledEvent(
                    event_id=eid,
                    start_time=t_start,
                    end_time=t_end,
                    prefix=str(sub_prefix),
                    origin_asn=forged_origin,
                    type="FORGED_ORIGIN_SUBPREFIX",
                    benign=False,
                    scenario="loose_maxlength_hijack",
                    difficulty="hard",
                    expected_alert_type="NONE",
                )
                events.append(evt)

                peer = self.novel_peer if (i % 2 == 0) else self.rng.choice(self.known_peers)
                updates.append(BGPUpdate(t_start, sub_prefix, forged_origin, peer, "announce"))
                updates.append(BGPUpdate(t_end, sub_prefix, forged_origin, peer, "withdraw"))

        # ---------------------------------------------------------
        # 2. Benign Scenarios (benign = True)
        # ---------------------------------------------------------

        # 2.1 multi_origin_anycast: 2+ legitimate origins in inventory & ROA
        count = self.scenario_counts.get("multi_origin_anycast", 10)
        scenario_stats["multi_origin_anycast"] = count
        if self.spread_full_24h:
            t_step = (self.end_time - self.snap2_time - 1200.0) / count
            for i in range(count):
                t_start = self.snap2_time + 600.0 + (i * t_step)
                t_end = t_start + 1800.0
                origin = 64500 if (i % 2 == 0) else 64501
                net = ipaddress.ip_network("192.0.2.0/24")

                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                evt = LabelledEvent(
                    event_id=eid,
                    start_time=t_start,
                    end_time=t_end,
                    prefix=str(net),
                    origin_asn=origin,
                    type="MULTI_ORIGIN",
                    benign=True,
                    scenario="multi_origin_anycast",
                    difficulty="easy",
                    expected_alert_type="NONE",
                )
                events.append(evt)

                peer = self._random_peer(0.25)
                updates.append(BGPUpdate(t_start, net, origin, peer, "announce"))
        else:
            for i in range(count):
                t_start = self._random_time_in_range(self.snap2_time + 1200, self.end_time - 1800)
                t_end = t_start + 1800
                origin = 64500 if (i % 2 == 0) else 64501
                net = ipaddress.ip_network("192.0.2.0/24")

                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                evt = LabelledEvent(
                    event_id=eid,
                    start_time=t_start,
                    end_time=t_end,
                    prefix=str(net),
                    origin_asn=origin,
                    type="MULTI_ORIGIN",
                    benign=True,
                    scenario="multi_origin_anycast",
                    difficulty="easy",
                    expected_alert_type="NONE",
                )
                events.append(evt)

                peer = self.rng.choice(self.known_peers)
                updates.append(BGPUpdate(t_start, net, origin, peer, "announce"))

        # 2.2 multi_origin_ddos_scrubbing: provider ASN announces during mitigation window
        count = self.scenario_counts.get("multi_origin_ddos_scrubbing", 8)
        scenario_stats["multi_origin_ddos_scrubbing"] = count
        if self.spread_full_24h:
            n_tr, n_va, n_te = self._distribute_across_slices(count)
            net_ddos = ipaddress.ip_network("203.0.113.0/24")
            t_tr_step = (self.t_val_start - (self.snap2_time + 1200.0)) / max(n_tr, 1)
            for i in range(n_tr):
                t_start = self.snap2_time + 1200.0 + i * t_tr_step
                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                events.append(LabelledEvent(
                    eid, t_start, t_start + 300.0, str(net_ddos), 64498,
                    "DDOS_SCRUBBING", True, "multi_origin_ddos_scrubbing", "hard", "INVENTORY_MISMATCH"
                ))
                peer = self._random_peer(0.25)
                updates.append(BGPUpdate(t_start, net_ddos, 64498, peer, "announce"))
                updates.append(BGPUpdate(t_start + 300.0, net_ddos, 64498, peer, "withdraw"))

            t_va_step = (self.t_test_start - (self.snap4_time + 300.0)) / max(n_va, 1)
            for i in range(n_va):
                t_start = self.snap4_time + 300.0 + i * t_va_step
                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                events.append(LabelledEvent(
                    eid, t_start, t_start + 300.0, str(net_ddos), 64499,
                    "DDOS_SCRUBBING", True, "multi_origin_ddos_scrubbing", "hard", "INVENTORY_MISMATCH"
                ))
                peer = self._random_peer(0.25)
                updates.append(BGPUpdate(t_start, net_ddos, 64499, peer, "announce"))
                updates.append(BGPUpdate(t_start + 300.0, net_ddos, 64499, peer, "withdraw"))

            t_te_step = (self.end_time - (self.snap5_time + 300.0)) / max(n_te, 1)
            for i in range(n_te):
                t_start = self.snap5_time + 300.0 + i * t_te_step
                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                events.append(LabelledEvent(
                    eid, t_start, t_start + 300.0, str(net_ddos), 64497,
                    "DDOS_SCRUBBING", True, "multi_origin_ddos_scrubbing", "hard", "INVENTORY_MISMATCH"
                ))
                peer = self._random_peer(0.25)
                updates.append(BGPUpdate(t_start, net_ddos, 64497, peer, "announce"))
                updates.append(BGPUpdate(t_start + 300.0, net_ddos, 64497, peer, "withdraw"))
        else:
            t_step = 420.0
            ddos_base_time = self.snap2_time + 1200.0
            for i in range(count):
                t_start = ddos_base_time + (i * t_step)
                t_end = t_start + 300.0
                net = ipaddress.ip_network("203.0.113.0/24")
                scrubber_asn = 64498

                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                evt = LabelledEvent(
                    event_id=eid,
                    start_time=t_start,
                    end_time=t_end,
                    prefix=str(net),
                    origin_asn=scrubber_asn,
                    type="DDOS_SCRUBBING",
                    benign=True,
                    scenario="multi_origin_ddos_scrubbing",
                    difficulty="hard",
                    expected_alert_type="INVENTORY_MISMATCH",
                )
                events.append(evt)

                peer = self.rng.choice(self.known_peers)
                updates.append(BGPUpdate(t_start, net, scrubber_asn, peer, "announce"))
                updates.append(BGPUpdate(t_end, net, scrubber_asn, peer, "withdraw"))

        # 2.3 policy_change_roa_first: ROA updated BEFORE new origin announces
        count = self.scenario_counts.get("policy_change_roa_first", 8)
        scenario_stats["policy_change_roa_first"] = count
        if self.spread_full_24h:
            n_tr, n_va, n_te = self._distribute_across_slices(count)
            net_pol = ipaddress.ip_network("198.51.100.0/22")
            t_tr_step = (self.t_val_start - (self.snap3_time + 600.0)) / max(n_tr, 1)
            for i in range(n_tr):
                t_start = self.snap3_time + 600.0 + i * t_tr_step
                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                events.append(LabelledEvent(
                    eid, t_start, t_start + 1800.0, str(net_pol), 13336,
                    "POLICY_CHANGE", True, "policy_change_roa_first", "easy", "NONE"
                ))
                peer = self._random_peer(0.25)
                updates.append(BGPUpdate(t_start, net_pol, 13336, peer, "announce"))

            t_va_step = (self.t_test_start - (self.snap4_time + 600.0)) / max(n_va, 1)
            for i in range(n_va):
                t_start = self.snap4_time + 600.0 + i * t_va_step
                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                events.append(LabelledEvent(
                    eid, t_start, t_start + 1800.0, str(net_pol), 13337,
                    "POLICY_CHANGE", True, "policy_change_roa_first", "easy", "NONE"
                ))
                peer = self._random_peer(0.25)
                updates.append(BGPUpdate(t_start, net_pol, 13337, peer, "announce"))

            t_te_step = (self.end_time - (self.snap5_time + 600.0)) / max(n_te, 1)
            for i in range(n_te):
                t_start = self.snap5_time + 600.0 + i * t_te_step
                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                events.append(LabelledEvent(
                    eid, t_start, t_start + 1800.0, str(net_pol), 13338,
                    "POLICY_CHANGE", True, "policy_change_roa_first", "easy", "NONE"
                ))
                peer = self._random_peer(0.25)
                updates.append(BGPUpdate(t_start, net_pol, 13338, peer, "announce"))
        else:
            for i in range(count):
                t_start = self.snap3_time + 3600 + (i * 900)
                t_end = t_start + 1800
                net = ipaddress.ip_network("198.51.100.0/22")
                origin = 13336

                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                evt = LabelledEvent(
                    event_id=eid,
                    start_time=t_start,
                    end_time=t_end,
                    prefix=str(net),
                    origin_asn=origin,
                    type="POLICY_CHANGE",
                    benign=True,
                    scenario="policy_change_roa_first",
                    difficulty="easy",
                    expected_alert_type="NONE",
                )
                events.append(evt)

                peer = self.rng.choice(self.known_peers)
                updates.append(BGPUpdate(t_start, net, origin, peer, "announce"))

        # 2.4A policy_change_roa_lag_inventory_updated:
        count_lag_inv = self.scenario_counts.get("policy_change_roa_lag_inventory_updated", 6)
        scenario_stats["policy_change_roa_lag_inventory_updated"] = count_lag_inv
        if self.spread_full_24h:
            n_tr, n_va, n_te = self._distribute_across_slices(count_lag_inv)
            net_pol = ipaddress.ip_network("198.51.100.0/22")
            lag_tr_base = self.snap3_time - (n_tr * 360.0) - 300.0
            for i in range(n_tr):
                t_start = lag_tr_base + (i * 360.0)
                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                events.append(LabelledEvent(
                    eid, t_start, t_start + 300.0, str(net_pol), 13336,
                    "POLICY_LAG_UPDATED_INV", True, "policy_change_roa_lag_inventory_updated", "hard", "ROA_INVALID"
                ))
                peer = self._random_peer(0.25)
                updates.append(BGPUpdate(t_start, net_pol, 13336, peer, "announce"))

            lag_va_base = self.snap4_time - (n_va * 360.0) - 300.0
            for i in range(n_va):
                t_start = lag_va_base + (i * 360.0)
                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                events.append(LabelledEvent(
                    eid, t_start, t_start + 300.0, str(net_pol), 13337,
                    "POLICY_LAG_UPDATED_INV", True, "policy_change_roa_lag_inventory_updated", "hard", "ROA_INVALID"
                ))
                peer = self._random_peer(0.25)
                updates.append(BGPUpdate(t_start, net_pol, 13337, peer, "announce"))

            lag_te_base = self.snap5_time - (n_te * 360.0) - 300.0
            for i in range(n_te):
                t_start = lag_te_base + (i * 360.0)
                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                events.append(LabelledEvent(
                    eid, t_start, t_start + 300.0, str(net_pol), 13338,
                    "POLICY_LAG_UPDATED_INV", True, "policy_change_roa_lag_inventory_updated", "hard", "ROA_INVALID"
                ))
                peer = self._random_peer(0.25)
                updates.append(BGPUpdate(t_start, net_pol, 13338, peer, "announce"))
        else:
            lag_inv_base = self.snap3_time - 2400.0
            for i in range(count_lag_inv):
                t_start = lag_inv_base + (i * 360.0)
                t_end = t_start + 300.0
                net = ipaddress.ip_network("198.51.100.0/22")
                origin = 13336

                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                evt = LabelledEvent(
                    event_id=eid,
                    start_time=t_start,
                    end_time=t_end,
                    prefix=str(net),
                    origin_asn=origin,
                    type="POLICY_LAG_UPDATED_INV",
                    benign=True,
                    scenario="policy_change_roa_lag_inventory_updated",
                    difficulty="hard",
                    expected_alert_type="ROA_INVALID",
                )
                events.append(evt)

                peer = self.rng.choice(self.known_peers)
                updates.append(BGPUpdate(t_start, net, origin, peer, "announce"))

        # 2.4B policy_change_roa_lag_inventory_stale:
        count_lag_stale = self.scenario_counts.get("policy_change_roa_lag_inventory_stale", 6)
        scenario_stats["policy_change_roa_lag_inventory_stale"] = count_lag_stale
        if self.spread_full_24h:
            n_tr, n_va, n_te = self._distribute_across_slices(count_lag_stale)
            net_stale = ipaddress.ip_network("203.0.113.0/24")
            stale_tr_base = self.snap2_time - (n_tr * 360.0) - 300.0
            for i in range(n_tr):
                t_start = stale_tr_base + (i * 360.0)
                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                events.append(LabelledEvent(
                    eid, t_start, t_start + 300.0, str(net_stale), 64498,
                    "POLICY_LAG_STALE_INV", True, "policy_change_roa_lag_inventory_stale", "hard", "ORIGIN_HIJACK"
                ))
                peer = self._random_peer(0.25)
                updates.append(BGPUpdate(t_start, net_stale, 64498, peer, "announce"))

            stale_va_base = self.snap4_time - (n_va * 360.0) - 300.0
            for i in range(n_va):
                t_start = stale_va_base + (i * 360.0)
                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                events.append(LabelledEvent(
                    eid, t_start, t_start + 300.0, str(net_stale), 64499,
                    "POLICY_LAG_STALE_INV", True, "policy_change_roa_lag_inventory_stale", "hard", "ORIGIN_HIJACK"
                ))
                peer = self._random_peer(0.25)
                updates.append(BGPUpdate(t_start, net_stale, 64499, peer, "announce"))

            stale_te_base = self.snap5_time - (n_te * 360.0) - 300.0
            for i in range(n_te):
                t_start = stale_te_base + (i * 360.0)
                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                events.append(LabelledEvent(
                    eid, t_start, t_start + 300.0, str(net_stale), 64497,
                    "POLICY_LAG_STALE_INV", True, "policy_change_roa_lag_inventory_stale", "hard", "ORIGIN_HIJACK"
                ))
                peer = self._random_peer(0.25)
                updates.append(BGPUpdate(t_start, net_stale, 64497, peer, "announce"))
        else:
            lag_stale_base = self.snap2_time - 2400.0
            for i in range(count_lag_stale):
                t_start = lag_stale_base + (i * 360.0)
                t_end = t_start + 300.0
                net = ipaddress.ip_network("203.0.113.0/24")
                origin = 64498

                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                evt = LabelledEvent(
                    event_id=eid,
                    start_time=t_start,
                    end_time=t_end,
                    prefix=str(net),
                    origin_asn=origin,
                    type="POLICY_LAG_STALE_INV",
                    benign=True,
                    scenario="policy_change_roa_lag_inventory_stale",
                    difficulty="hard",
                    expected_alert_type="ORIGIN_HIJACK",
                )
                events.append(evt)

                peer = self.rng.choice(self.known_peers)
                updates.append(BGPUpdate(t_start, net, origin, peer, "announce"))

        # 2.5 te_more_specific: legitimate origin announces more-specific within maxLength
        count = self.scenario_counts.get("te_more_specific", 14)
        scenario_stats["te_more_specific"] = count
        if self.spread_full_24h:
            t_step = (self.end_time - self.start_time - 3600.0) / count
            for i in range(count):
                t_start = self.start_time + 1800.0 + i * t_step + self.rng.uniform(-t_step * 0.1, t_step * 0.1)
                t_end = t_start + 1800.0
                if i % 2 == 0:
                    net = ipaddress.ip_network(f"198.51.10{i % 4}.0/24")
                    origin = 13335
                else:
                    net = ipaddress.ip_network(f"198.18.{i % 32 + 50}.0/24")
                    origin = 64510

                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                evt = LabelledEvent(
                    event_id=eid,
                    start_time=t_start,
                    end_time=t_end,
                    prefix=str(net),
                    origin_asn=origin,
                    type="TRAFFIC_ENGINEERING",
                    benign=True,
                    scenario="te_more_specific",
                    difficulty="easy",
                    expected_alert_type="NONE",
                )
                events.append(evt)

                # ~40-50% from NOVEL peer (randomly drawn per update!)
                peer = self._random_peer(novel_prob=0.45)
                updates.append(BGPUpdate(t_start, net, origin, peer, "announce"))
        else:
            for i in range(count):
                t_start = self._random_time_in_range(self.start_time + 600, self.end_time - 1200)
                t_end = t_start + 1800
                sub_id = i % 4
                net = ipaddress.ip_network(f"198.51.10{sub_id}.0/24")
                origin = 13335

                eid = f"evt-{event_counter:04d}"
                event_counter += 1
                evt = LabelledEvent(
                    event_id=eid,
                    start_time=t_start,
                    end_time=t_end,
                    prefix=str(net),
                    origin_asn=origin,
                    type="TRAFFIC_ENGINEERING",
                    benign=True,
                    scenario="te_more_specific",
                    difficulty="easy",
                    expected_alert_type="NONE",
                )
                events.append(evt)

                peer = self.novel_peer if (i % 3 == 0) else self.rng.choice(self.known_peers)
                updates.append(BGPUpdate(t_start, net, origin, peer, "announce"))

        # ---------------------------------------------------------
        # 3. Benign Background Traffic Generation
        # ---------------------------------------------------------
        bg_prefixes = [ipaddress.ip_network(p) for p in self.external_prefixes]
        for inv_it in self.inventory.all_items():
            if str(inv_it.prefix) != "203.0.113.0/24":
                bg_prefixes.append(inv_it.prefix)

        if self.spread_full_24h:
            # Include carrier subprefixes under 198.18.0.0/15 announced by AS64510
            for k in range(16):
                bg_prefixes.append(ipaddress.ip_network(f"198.18.{k}.0/24"))

        normal_bg_count = self.background_count - int(self.background_count * 0.015)
        for _ in range(normal_bg_count):
            t = self._random_time_in_range(self.start_time + 10, self.end_time - 10)
            
            if self.spread_full_24h and self.rng.random() < 0.30:
                active_dyn = [item for item in self.dynamic_bg_prefixes if item[1] <= t]
                if active_dyn:
                    dyn_p, _, dyn_asn = self.rng.choice(active_dyn)
                    prefix = dyn_p
                    origin = dyn_asn
                else:
                    prefix = self.rng.choice(bg_prefixes)
                    origin = self.rng.choice(self.external_asns)
            else:
                prefix = self.rng.choice(bg_prefixes)
                inv_exact = self.inventory.lookup_exact(prefix)
                p_str = str(prefix)
                if p_str.startswith("198.18."):
                    origin = 64510
                elif inv_exact:
                    if p_str == "198.51.100.0/22":
                        origin = 13335
                    elif p_str == "192.0.2.0/24":
                        origin = 64500 if t < self.snap2_time else (64500 if self.rng.random() < 0.5 else 64501)
                    else:
                        origin = min(inv_exact.expected_origin_asns)
                else:
                    if self.spread_full_24h and self.tail_asns and self.rng.random() < 0.30:
                        origin = self.rng.choice(self.tail_asns)
                    else:
                        origin = self.rng.choice(self.external_asns)

            peer = self._random_peer(0.20) if self.spread_full_24h else self.rng.choice(self.known_peers)
            action = "withdraw" if self.rng.random() < 0.15 else "announce"
            updates.append(BGPUpdate(t, prefix, origin, peer, action))

        # 3.2 Realistic Third-Party Noise (~1.5% ROV Invalid outside operator inventory)
        noise_count = int(self.background_count * 0.015)
        third_party_invalid_prefixes = [
            (ipaddress.ip_network("198.51.200.0/24"), 65001),
            (ipaddress.ip_network("198.51.200.0/25"), 65002),
            (ipaddress.ip_network("203.0.114.0/24"), 65003),
        ]
        for _ in range(noise_count):
            t = self._random_time_in_range(self.snap2_time + 300, self.end_time - 300)
            prefix, origin = self.rng.choice(third_party_invalid_prefixes)
            peer = self._random_peer(0.20) if self.spread_full_24h else self.rng.choice(self.all_peers)
            action = "withdraw" if self.rng.random() < 0.10 else "announce"
            updates.append(BGPUpdate(t, prefix, origin, peer, action))

        # Sort updates chronologically
        updates.sort(key=lambda u: u.timestamp)
        events.sort(key=lambda e: e.start_time)

        # Build manifest v2
        manifest = {
            "version": "generator v2",
            "random_seed": self.seed,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "start_time": datetime.datetime.fromtimestamp(self.start_time, tz=datetime.timezone.utc).isoformat(),
            "end_time": datetime.datetime.fromtimestamp(self.end_time, tz=datetime.timezone.utc).isoformat(),
            "duration_hours": 24,
            "total_feed_updates": len(updates),
            "total_announcements": sum(1 for u in updates if u.is_announcement),
            "total_withdrawals": sum(1 for u in updates if u.is_withdrawal),
            "total_labelled_events": len(events),
            "anomaly_events_count": sum(1 for e in events if not e.benign),
            "benign_events_count": sum(1 for e in events if e.benign),
            "scenario_counts": scenario_stats,
            "changelog": [
                "v2: Made loose_maxlength_hijack a forged-origin more-specific (ROV Valid, origin in inventory; baseline misses)",
                "v2: Split policy_change_roa_lag into policy_change_roa_lag_inventory_updated and policy_change_roa_lag_inventory_stale",
                "v2: Added ~1.5% third-party RFC 6811 Invalid background routes tagged as THIRD_PARTY_INVALID",
                "v2: Balanced peer novelty across benign TE and forged anomalies to prevent trivial feature leakage",
                "v2: Multi-slice time-spread support for ML scale datasets across Train, Val, and Test boundaries",
            ],
        }

        return updates, events, manifest

    def save_to_disk(
        self,
        feed_path: Optional[str | Path] = None,
        labels_path: Optional[str | Path] = None,
        manifest_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Generate and write feed_replay.csv, labels.csv, and manifest.json."""
        feed_file = Path(feed_path or self.config["paths"]["feed_replay"])
        labels_file = Path(labels_path or self.config["paths"]["labels"])
        manifest_file = Path(manifest_path or self.config.get("paths", {}).get("manifest", "data/manifest.json"))

        feed_file.parent.mkdir(parents=True, exist_ok=True)
        labels_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.parent.mkdir(parents=True, exist_ok=True)

        updates, events, manifest = self.generate()

        # Write labels.csv
        with labels_file.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "event_id",
                "start_time",
                "end_time",
                "prefix",
                "origin_asn",
                "type",
                "benign",
                "scenario",
                "difficulty",
                "expected_alert_type",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for evt in events:
                writer.writerow(evt.to_dict())

        # Write feed_replay.csv
        with feed_file.open("w", encoding="utf-8", newline="") as f:
            fieldnames = ["timestamp", "prefix", "origin_asn", "peer", "action"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for u in updates:
                writer.writerow(
                    {
                        "timestamp": datetime.datetime.fromtimestamp(
                            u.timestamp, tz=datetime.timezone.utc
                        ).isoformat(),
                        "prefix": u.prefix_str,
                        "origin_asn": u.origin_asn,
                        "peer": u.peer,
                        "action": u.action,
                    }
                )

        # Write manifest.json
        with manifest_file.open("w", encoding="utf-8", newline="") as f:
            json.dump(manifest, f, indent=2)

        return manifest


def generate_synthetic_data(config_path: str = "config.yaml", seed: Optional[int] = None) -> Dict[str, Any]:
    """Helper entry point for dataset generation."""
    injector = ScenarioInjector(config_path=config_path, seed=seed)
    return injector.save_to_disk()


if __name__ == "__main__":
    manifest = generate_synthetic_data()
    print("Synthetic dataset successfully generated (v2):")
    print(json.dumps(manifest, indent=2))
