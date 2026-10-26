"""RFC 6811 Route Origin Validation (ROV).

Strictly implements RFC 6811:
- A ROA is "covering" if the announced prefix is equal to or inside the ROA prefix.
- A covering ROA "matches" iff origin ASN == ROA ASN AND route prefix length <= maxLength.
- Valid = at least one covering ROA matches.
- Invalid = covering ROAs exist but none match.
- NotFound = no covering ROA exists.
- AS0 ROAs can never match; routes covered only by AS0 ROAs are Invalid ("as0_roa").
- If no ROA snapshot exists at or before update time, returns "NoSnapshot".
"""

from __future__ import annotations

import enum
import ipaddress
from dataclasses import dataclass, field
from typing import List, Optional, Union

from src.roa_store import ROARecord, ROASnapshot, ROAStore, parse_timestamp

IPNetworkType = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]


class ROVState(str, enum.Enum):
    """RFC 6811 Route Origin Validation States."""

    VALID = "Valid"
    INVALID = "Invalid"
    NOT_FOUND = "NotFound"
    NO_SNAPSHOT = "NoSnapshot"


@dataclass(frozen=True)
class ROVResult:
    """Rich RFC 6811 validation result."""

    state: ROVState
    covering_roas: List[ROARecord] = field(default_factory=list)
    matched_roa: Optional[ROARecord] = None
    invalid_reason: Optional[str] = None
    roa_snapshot_time: Optional[float] = None

    @property
    def is_valid(self) -> bool:
        return self.state == ROVState.VALID

    @property
    def is_invalid(self) -> bool:
        return self.state == ROVState.INVALID

    @property
    def is_not_found(self) -> bool:
        return self.state == ROVState.NOT_FOUND

    @property
    def is_no_snapshot(self) -> bool:
        return self.state == ROVState.NO_SNAPSHOT


def validate_route(
    prefix: Union[str, IPNetworkType],
    origin_asn: int,
    timestamp: Union[str, int, float],
    roa_store: ROAStore,
) -> ROVResult:
    """Validate a BGP announcement strictly per RFC 6811 using a time-aligned ROA snapshot.

    Args:
        prefix: Announced IPv4 or IPv6 prefix.
        origin_asn: Origin Autonomous System Number (integer).
        timestamp: Time of the announcement.
        roa_store: ROAStore providing historical snapshots.

    Returns:
        ROVResult containing validation state, covering ROAs, matching ROA, and specific reasons.
    """
    if not isinstance(prefix, (ipaddress.IPv4Network, ipaddress.IPv6Network)):
        net = ipaddress.ip_network(str(prefix).strip(), strict=False)
    else:
        net = prefix

    t_sec = parse_timestamp(timestamp)
    snapshot: Optional[ROASnapshot] = roa_store.get_snapshot_for_time(t_sec, fallback_to_earliest=False)

    if snapshot is None:
        return ROVResult(
            state=ROVState.NO_SNAPSHOT,
            covering_roas=[],
            matched_roa=None,
            invalid_reason="no_snapshot_at_or_before_update_time",
            roa_snapshot_time=None,
        )

    covering_roas = snapshot.find_covering_roas(net)
    snap_time = snapshot.snapshot_time

    if not covering_roas:
        return ROVResult(
            state=ROVState.NOT_FOUND,
            covering_roas=[],
            matched_roa=None,
            invalid_reason=None,
            roa_snapshot_time=snap_time,
        )

    # Check for matching ROAs
    # RFC 6811: Origin match AND prefix length <= maxLength. AS0 can NEVER match.
    for roa in covering_roas:
        if roa.asn != 0 and roa.asn == origin_asn and net.prefixlen <= roa.max_length:
            return ROVResult(
                state=ROVState.VALID,
                covering_roas=covering_roas,
                matched_roa=roa,
                invalid_reason=None,
                roa_snapshot_time=snap_time,
            )

    # If no ROA matched, state is INVALID. Determine specific RFC 6811 invalidity cause.
    has_as0 = any(r.asn == 0 for r in covering_roas)
    has_origin_match = any(r.asn == origin_asn and r.asn != 0 for r in covering_roas)
    has_length_match = any(net.prefixlen <= r.max_length for r in covering_roas if r.asn != 0)

    # Check if only AS0 covers this route
    if all(r.asn == 0 for r in covering_roas):
        reason = "as0_roa"
    elif has_as0 and not has_origin_match:
        reason = "as0_roa"
    elif has_origin_match and not has_length_match:
        reason = "length_exceeds_maxlength"
    elif has_length_match and not has_origin_match:
        reason = "origin_mismatch"
    else:
        # Neither origin matched nor length was within allowed bounds
        reason = "origin_and_length_mismatch"

    return ROVResult(
        state=ROVState.INVALID,
        covering_roas=covering_roas,
        matched_roa=None,
        invalid_reason=reason,
        roa_snapshot_time=snap_time,
    )
