"""Rule-Based BGP Anomaly Detector (Baseline).

Implements RFC 6811 validation combined with operator prefix inventory analysis.
Emits evidence-based alerts for unexpected origins and subprefix anomalies.

Strict Rule Precedence:
======================
1. Ignore withdrawals: Only announcements ('announce') are evaluated.
2. Legitimate pass-through (NO ALERT):
   - Announcement for inventory prefix where origin is authorized in inventory AND
     ROV state is Valid (within ROA maxLength).
   - Benign external routes outside operator inventory with Valid or NotFound ROV state.
3. Precedence 1: ORIGIN_HIJACK (High Severity)
   - Condition: Exact match on inventory prefix, origin ASN is NOT in inventory allowed set,
     AND ROV state is Invalid/NotFound (or NoSnapshot).
   - Genuine unauthorized origin hijack of an operator-owned prefix.
4. Precedence 2: SUBPREFIX_HIJACK (High Severity)
   - Condition: Announced prefix is a more-specific (subprefix) of an inventory or ROA prefix,
     origin ASN is NOT in inventory allowed set, AND ROV state is Invalid/NotFound.
   - Genuine unauthorized more-specific hijack.
5. Precedence 3: INVENTORY_MISMATCH (Low Severity)
   - Condition: Prefix is in operator inventory (exact or covering), origin ASN is NOT registered
     in inventory, BUT ROV state is Valid.
   - Cryptographically valid per RPKI ROA, but origin ASN missing from operator inventory
     (stale inventory documentation or uncoordinated ROA creation).
6. Precedence 4: ROA_INVALID (Medium / High Severity)
   - Condition: RFC 6811 ROV state is Invalid and not categorized as a hijack.
   - Sub-case A (Policy Lag): Origin IS authorized in inventory, but ROA is Invalid.
     Emits ROA_INVALID at medium severity (inventory and ROA disagree; ROA update lag).
   - Sub-case B: General RFC 6811 Invalid (AS0 ROA, length exceeds maxLength).
7. Precedence 5: ROA_MISSING (Low Severity)
   - Condition: Operator inventory prefix announced by authorized origin, but ROV state is NotFound.
   - Informational warning indicating lack of RPKI ROA coverage.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

from src.inventory import InventoryItem, PrefixInventory
from src.replay import BGPUpdate
from src.roa_store import ROAStore
from src.rov import ROVResult, ROVState, validate_route


@dataclass
class Alert:
    """Represents an anomaly detection alert emitted by the monitor."""

    alert_id: str
    time: float
    type: str  # ORIGIN_HIJACK, SUBPREFIX_HIJACK, INVENTORY_MISMATCH, ROA_INVALID, ROA_MISSING
    prefix: str
    origin_asn: int
    severity: str  # high, medium, low
    detector: str = "rules"
    rov_result: Optional[ROVResult] = None
    inventory_match: Optional[InventoryItem] = None
    reason: str = ""
    suppressed_count: int = 0

    def to_dict(self) -> dict:
        """Convert alert to a dictionary for reporting and JSON serialization."""
        return {
            "alert_id": self.alert_id,
            "time": self.time,
            "type": self.type,
            "prefix": self.prefix,
            "origin_asn": self.origin_asn,
            "severity": self.severity,
            "detector": self.detector,
            "rov_state": self.rov_result.state.value if self.rov_result else None,
            "rov_invalid_reason": self.rov_result.invalid_reason if self.rov_result else None,
            "inventory_owner": self.inventory_match.owner if self.inventory_match else None,
            "reason": self.reason,
            "suppressed_count": self.suppressed_count,
        }


class RuleBasedDetector:
    """Baseline rule-based BGP origin and subprefix anomaly detector."""

    def __init__(
        self,
        inventory: PrefixInventory,
        roa_store: ROAStore,
        dedup_window_seconds: float = 300.0,
    ) -> None:
        self.inventory: PrefixInventory = inventory
        self.roa_store: ROAStore = roa_store
        self.dedup_window_seconds: float = dedup_window_seconds
        # Deduplication cache: key = (prefix_str, origin_asn) -> Alert
        self._recent_alerts: Dict[Tuple[str, int], Alert] = {}

    def reset(self) -> None:
        """Reset deduplication state for fresh test or replay runs."""
        self._recent_alerts.clear()

    def process_update(self, update: BGPUpdate) -> Optional[Alert]:
        """Process a single BGP update and return an Alert if an anomaly is detected."""
        # 1. Ignore withdrawals: Process announcements only
        if not update.is_announcement:
            return None

        prefix = update.prefix
        prefix_str = str(prefix)
        origin_asn = update.origin_asn
        timestamp = update.timestamp

        # 2. Perform RFC 6811 ROV validation
        rov_res = validate_route(prefix, origin_asn, timestamp, self.roa_store)

        # 3. Perform Inventory Lookups (Exact and Covering)
        inv_exact = self.inventory.lookup_exact(prefix)
        inv_covering = self.inventory.lookup_covering(prefix)
        inv_item = inv_exact or inv_covering
        is_more_specific = inv_covering is not None and prefix.prefixlen > inv_covering.prefix.prefixlen

        origin_in_inventory = self.inventory.is_origin_allowed(prefix, origin_asn)

        # --- Rule Evaluation: Legitimate Pass-Through ---
        # Legitimate traffic: in inventory, origin allowed, ROV Valid -> DO NOT ALERT
        if origin_in_inventory and rov_res.is_valid:
            return None

        # Background Internet route: not in inventory, ROV Valid or NotFound -> DO NOT ALERT
        if inv_item is None and (rov_res.is_valid or rov_res.is_not_found):
            return None

        candidate_alert: Optional[Alert] = None

        # --- Strict Rule Precedence ---

        # 1. ORIGIN_HIJACK: Exact inventory prefix announced by unauthorized origin with Invalid/NotFound ROV
        if inv_exact is not None and not origin_in_inventory and not rov_res.is_valid:
            candidate_alert = Alert(
                alert_id=f"alt-{uuid.uuid4().hex[:8]}",
                time=timestamp,
                type="ORIGIN_HIJACK",
                prefix=prefix_str,
                origin_asn=origin_asn,
                severity="high",
                detector="rules",
                rov_result=rov_res,
                inventory_match=inv_exact,
                reason=f"Unauthorized origin AS{origin_asn} announced exact inventory prefix (expected ASNs: {sorted(inv_exact.expected_origin_asns)})",
            )

        # 2. SUBPREFIX_HIJACK: More-specific of inventory prefix announced by unauthorized origin with Invalid/NotFound ROV
        elif (is_more_specific or (rov_res.covering_roas and not origin_in_inventory)) and not origin_in_inventory and not rov_res.is_valid:
            candidate_alert = Alert(
                alert_id=f"alt-{uuid.uuid4().hex[:8]}",
                time=timestamp,
                type="SUBPREFIX_HIJACK",
                prefix=prefix_str,
                origin_asn=origin_asn,
                severity="high",
                detector="rules",
                rov_result=rov_res,
                inventory_match=inv_covering,
                reason=f"Subprefix /{prefix.prefixlen} hijacked by unauthorized AS{origin_asn} under parent prefix",
            )

        # 3. INVENTORY_MISMATCH: ROV Valid per RPKI ROA but origin is NOT registered in operator inventory
        elif inv_item is not None and not origin_in_inventory and rov_res.is_valid:
            candidate_alert = Alert(
                alert_id=f"alt-{uuid.uuid4().hex[:8]}",
                time=timestamp,
                type="INVENTORY_MISMATCH",
                prefix=prefix_str,
                origin_asn=origin_asn,
                severity="low",
                detector="rules",
                rov_result=rov_res,
                inventory_match=inv_item,
                reason=f"ROV Valid per RPKI ROA but origin AS{origin_asn} is not registered in operator inventory (potential stale inventory or uncoordinated ROA)",
            )

        # 4. ROA_INVALID: RFC 6811 Invalid not covered by previous hijack rules
        elif rov_res.is_invalid:
            if origin_in_inventory:
                # Edge Case 1: Origin is in inventory allowed set, but ROV is Invalid (ROA lags policy change)
                # Emit ROA_INVALID at medium severity; do not label as a hijack!
                candidate_alert = Alert(
                    alert_id=f"alt-{uuid.uuid4().hex[:8]}",
                    time=timestamp,
                    type="ROA_INVALID",
                    prefix=prefix_str,
                    origin_asn=origin_asn,
                    severity="medium",
                    detector="rules",
                    rov_result=rov_res,
                    inventory_match=inv_item,
                    reason="Origin authorized in operator inventory but ROA state is Invalid (ROA lags behind policy change; inventory and ROA disagree)",
                )
            else:
                # General ROA Invalid (e.g. AS0 ROA, length exceeds maxLength on unowned route)
                severity = "high" if inv_item is not None else "medium"
                candidate_alert = Alert(
                    alert_id=f"alt-{uuid.uuid4().hex[:8]}",
                    time=timestamp,
                    type="ROA_INVALID",
                    prefix=prefix_str,
                    origin_asn=origin_asn,
                    severity=severity,
                    detector="rules",
                    rov_result=rov_res,
                    inventory_match=inv_item,
                    reason=f"RFC 6811 Invalid: {rov_res.invalid_reason}",
                )

        # 5. ROA_MISSING: Inventory-owned prefix announced with NotFound ROV state
        elif inv_item is not None and rov_res.is_not_found:
            candidate_alert = Alert(
                alert_id=f"alt-{uuid.uuid4().hex[:8]}",
                time=timestamp,
                type="ROA_MISSING",
                prefix=prefix_str,
                origin_asn=origin_asn,
                severity="low",
                detector="rules",
                rov_result=rov_res,
                inventory_match=inv_item,
                reason="Operator inventory prefix announced without RPKI ROA protection (NotFound)",
            )

        if candidate_alert is None:
            return None

        # --- Deduplication Engine ---
        # Key is strictly (prefix_str, origin_asn)
        # A NEW origin on the same prefix has a different key and is NEVER suppressed!
        dedup_key = (prefix_str, origin_asn)
        if dedup_key in self._recent_alerts:
            existing = self._recent_alerts[dedup_key]
            if (timestamp - existing.time) <= self.dedup_window_seconds:
                # Within deduplication window: suppress duplicate and update counter
                existing.suppressed_count += 1
                return None

        # First occurrence or window expired: emit alert and record in active window
        self._recent_alerts[dedup_key] = candidate_alert
        return candidate_alert
