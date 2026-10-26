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

        # Snapshot epoch timestamps
        self.snap1_time = self.start_time  # 00:00:00Z
        self.snap2_time = self.start_time + 6 * 3600.0  # 06:00:00Z
        self.snap3_time = self.start_time + 12 * 3600.0  # 12:00:00Z

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

    def _random_time_in_range(self, t_min: float, t_max: float) -> float:
        return round(self.rng.uniform(t_min, t_max), 2)

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

            peer = self.rng.choice(self.known_peers)
            updates.append(BGPUpdate(t_start, item.prefix, hijack_asn, peer, "announce"))
            updates.append(BGPUpdate(t_start + 15.0, item.prefix, hijack_asn, self.rng.choice(self.known_peers), "announce"))
            updates.append(BGPUpdate(t_end, item.prefix, hijack_asn, peer, "withdraw"))

        # 1.2 subprefix_hijack
        count = self.scenario_counts.get("subprefix_hijack", 12)
        scenario_stats["subprefix_hijack"] = count
        for i in range(count):
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

            peer = self.rng.choice(self.known_peers)
            updates.append(BGPUpdate(t_start, sub_net, hijack_asn, peer, "announce"))
            updates.append(BGPUpdate(t_start + 20.0, sub_net, hijack_asn, self.rng.choice(self.known_peers), "announce"))
            updates.append(BGPUpdate(t_end, sub_net, hijack_asn, peer, "withdraw"))

        # 1.3 hijack_after_roa_change (tests snapshot time alignment boundary)
        count = self.scenario_counts.get("hijack_after_roa_change", 8)
        scenario_stats["hijack_after_roa_change"] = count
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
        # Attacker announces a more-specific /24 under 198.18.0.0/15 forging legitimate origin AS64510!
        # Result: ROV is VALID and origin is in inventory -> Baseline rule misses it (NO ALERT).
        count = self.scenario_counts.get("loose_maxlength_hijack", 8)
        scenario_stats["loose_maxlength_hijack"] = count
        for i in range(count):
            t_start = self._random_time_in_range(self.start_time + 3600, self.end_time - 3600)
            t_end = t_start + self.rng.randint(900, 2400)
            sub_prefix = ipaddress.ip_network(f"198.18.{200 + i}.0/24")
            # Forged legitimate origin: AS64510
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
                expected_alert_type="NONE",  # Expected to be missed by baseline rules!
            )
            events.append(evt)

            # Half from novel peer, half from known peer to prevent trivial leakage
            peer = self.novel_peer if (i % 2 == 0) else self.rng.choice(self.known_peers)
            updates.append(BGPUpdate(t_start, sub_prefix, forged_origin, peer, "announce"))
            updates.append(BGPUpdate(t_end, sub_prefix, forged_origin, peer, "withdraw"))

        # ---------------------------------------------------------
        # 2. Benign Scenarios (benign = True)
        # ---------------------------------------------------------

        # 2.1 multi_origin_anycast: 2+ legitimate origins in inventory & ROA
        count = self.scenario_counts.get("multi_origin_anycast", 10)
        scenario_stats["multi_origin_anycast"] = count
        for i in range(count):
            # Both 64500 and 64501 are authorized in ROA snapshot 2 and 3 (from 06:00:00Z onwards)
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
        # Authorized by ROA but not in inventory -> triggers INVENTORY_MISMATCH (benign FP)
        count = self.scenario_counts.get("multi_origin_ddos_scrubbing", 8)
        scenario_stats["multi_origin_ddos_scrubbing"] = count
        # Space them by at least 400s so each event triggers a distinct alert without dedup suppression
        ddos_base_time = self.snap2_time + 1200.0
        for i in range(count):
            t_start = ddos_base_time + (i * 420.0)
            t_end = t_start + 300.0
            net = ipaddress.ip_network("203.0.113.0/24")
            scrubber_asn = 64498  # Valid in ROA after 06:00, but outside initial inventory

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
        # Origin IS in inventory, but ROA lags -> ROV Invalid -> expected ROA_INVALID (medium severity)
        count_lag_inv = self.scenario_counts.get("policy_change_roa_lag_inventory_updated", 6)
        scenario_stats["policy_change_roa_lag_inventory_updated"] = count_lag_inv
        lag_inv_base = self.snap3_time - 2400.0  # Before 12:00:00Z snapshot
        for i in range(count_lag_inv):
            t_start = lag_inv_base + (i * 360.0)  # Spaced > 300s to avoid dedup suppression
            t_end = t_start + 300.0
            # 198.51.100.0/22 has AS13336 in inventory, but ROA adds it only at 12:00:00Z
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
        # Origin NOT in inventory AND ROA lags -> ROV Invalid -> triggers ORIGIN_HIJACK (known FP source)
        count_lag_stale = self.scenario_counts.get("policy_change_roa_lag_inventory_stale", 6)
        scenario_stats["policy_change_roa_lag_inventory_stale"] = count_lag_stale
        lag_stale_base = self.snap2_time - 2400.0  # Before 06:00:00Z snapshot
        for i in range(count_lag_stale):
            t_start = lag_stale_base + (i * 360.0)  # Spaced > 300s
            t_end = t_start + 300.0
            # 203.0.113.0/24 has neither inventory nor ROA for AS64498 before 06:00:00Z
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

            # Mix of known and novel peers (so peer novelty is NOT a 100% separator)
            peer = self.novel_peer if (i % 3 == 0) else self.rng.choice(self.known_peers)
            updates.append(BGPUpdate(t_start, net, origin, peer, "announce"))

        # ---------------------------------------------------------
        # 3. Benign Background Traffic Generation
        # ---------------------------------------------------------
        bg_prefixes = [ipaddress.ip_network(p) for p in self.external_prefixes]
        for inv_it in self.inventory.all_items():
            if str(inv_it.prefix) != "203.0.113.0/24":
                bg_prefixes.append(inv_it.prefix)

        # 3.1 Normal Background (~98.5% of updates: Valid or NotFound)
        normal_bg_count = self.background_count - int(self.background_count * 0.015)
        for _ in range(normal_bg_count):
            t = self._random_time_in_range(self.start_time + 10, self.end_time - 10)
            prefix = self.rng.choice(bg_prefixes)

            inv_exact = self.inventory.lookup_exact(prefix)
            if inv_exact:
                if str(prefix) == "198.51.100.0/22":
                    origin = 13335
                elif str(prefix) == "192.0.2.0/24":
                    origin = 64500 if t < self.snap2_time else (64500 if self.rng.random() < 0.5 else 64501)
                else:
                    origin = min(inv_exact.expected_origin_asns)
            else:
                origin = self.rng.choice(self.external_asns)

            peer = self.rng.choice(self.known_peers)
            action = "withdraw" if self.rng.random() < 0.15 else "announce"
            updates.append(BGPUpdate(t, prefix, origin, peer, action))

        # 3.2 Realistic Third-Party Noise (~1.5% ROV Invalid outside operator inventory)
        # Models real Internet routing hygiene issues (AS0 ROA routes, maxLength violations on public prefixes)
        noise_count = int(self.background_count * 0.015)
        third_party_invalid_prefixes = [
            (ipaddress.ip_network("198.51.200.0/24"), 65001),  # Covered by AS0 ROA in snapshot 2/3
            (ipaddress.ip_network("198.51.200.0/25"), 65002),  # AS0 + subprefix
            (ipaddress.ip_network("203.0.114.0/24"), 65003),   # External prefix adjacent to inventory
        ]
        for _ in range(noise_count):
            # Place after snapshot 2 (06:00) when AS0 ROA is published
            t = self._random_time_in_range(self.snap2_time + 300, self.end_time - 300)
            prefix, origin = self.rng.choice(third_party_invalid_prefixes)
            peer = self.rng.choice(self.all_peers)
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
                "v2: Spaced recurring scenario events beyond 300s dedup window to allow consistent event-level evaluation",
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
