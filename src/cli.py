"""Command-Line Interface (CLI) for BGP Anomaly Monitor and Explainable Alerts.

Commands:
  1. explain: Pretty-print full evidence and rationales for a specific alert.
     Usage: python -m src.cli explain --alert-id <ALERT_ID> [--alerts-file <PATH>]
  2. list-alerts: Filter and display anomaly detection alerts.
     Usage: python -m src.cli list-alerts [--type <TYPE>] [--severity <SEVERITY>] [--detector <DETECTOR>]
  3. generate: Regenerate explainable alerts JSONL for both detectors.
     Usage: python -m src.cli generate [--config <CONFIG>] [--output <OUTPUT>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.explain import (
    ExplainableAlert,
    generate_explainable_alerts,
    read_alerts_from_jsonl,
)


def format_cli_time(time_val: Any) -> str:
    """Safely format time value into human-readable string."""
    if isinstance(time_val, dict):
        utc = time_val.get("utc", "")
        epoch = time_val.get("epoch", "")
        return f"{utc} ({epoch})"
    return str(time_val)


def pretty_print_evidence(alert: ExplainableAlert, raw_json: bool = False) -> None:
    """Pretty-print the alert metadata and its evidence block."""
    if raw_json:
        print(json.dumps(alert.to_dict(), indent=2))
        return

    ev = alert.evidence
    print("=" * 80)
    print(" BGP ANOMALY MONITOR: EXPLAINABLE ALERT REPORT")
    print("=" * 80)
    print(f"Alert ID:     {alert.alert_id}")
    print(f"Time (UTC):   {alert.time.get('utc', 'unknown')} (epoch: {alert.time.get('epoch', 'unknown')})")
    print(f"Detector:     {alert.detector.upper()}")
    print(f"Alert Type:   {alert.type}")
    print(f"Severity:     {alert.severity.upper()}")
    print(f"Prefix:       {alert.prefix}")
    print(f"Origin ASN:   AS{alert.origin_asn}")
    print(f"Peer:         {ev.peer}")

    if alert.score is not None:
        tau_str = f"{alert.threshold:.4f}" if alert.threshold is not None else "0.0500"
        print(f"Anomaly Score:{alert.score:.4f} (Decision Threshold tau* = {tau_str})")

    print("-" * 80)
    print("EVIDENCE & RATIONALE:")
    print("-" * 80)
    print(f"Reason:\n  {ev.reason}\n")

    # ML Feature Contributions
    if ev.top_contributing_features:
        print("Top Contributing Features (Top 3):")
        for idx, feat in enumerate(ev.top_contributing_features, 1):
            name = feat.get("feature", "unknown")
            val = feat.get("value", 0.0)
            contrib = feat.get("contribution", 0.0)
            direction = feat.get("direction", "0")
            print(f"  {idx}. {name:<26} value={val:<10.4g} contribution={contrib:+.4f} (direction: {direction})")
        print()

    if ev.explanation_method:
        print(f"Explanation Method:\n  {ev.explanation_method}\n")
    if ev.causal_disclaimer:
        print(f"Disclaimer:\n  {ev.causal_disclaimer}\n")

    # RFC 6811 Validation
    print("RFC 6811 Route Origin Validation (ROV):")
    print(f"  ROV State:      {ev.rfc6811_state}")
    if ev.invalid_reason:
        print(f"  Invalid Reason: {ev.invalid_reason}")
    if ev.roa_snapshot_time is not None:
        print(f"  Snapshot Time:  {ev.roa_snapshot_time}")

    if ev.covering_roas:
        print(f"  Covering ROAs ({len(ev.covering_roas)}):")
        for roa in ev.covering_roas:
            print(f"    - Prefix: {roa.get('prefix')} | ASN: AS{roa.get('asn')} | MaxLength: /{roa.get('maxLength')}")
    else:
        print("  Covering ROAs:  None (No covering ROA exists in snapshot)")
    print()

    # Inventory Match
    print("Operator Prefix Inventory:")
    if ev.inventory_match:
        inv = ev.inventory_match
        allowed = ", ".join(f"AS{asn}" for asn in inv.get("allowed_origins", []))
        print(f"  Status:          Matched ({inv.get('prefix')})")
        print(f"  Owner:           {inv.get('owner', 'Unknown')}")
        print(f"  Allowed Origins: [{allowed}]")
    else:
        print("  Status:          Unmatched (Prefix is outside operator inventory)")
    print("=" * 80)


def cmd_explain(args: argparse.Namespace) -> int:
    """Execute the 'explain' CLI command."""
    alerts_file = Path(args.alerts_file)
    if not alerts_file.exists():
        print(f"Error: Alerts file '{alerts_file}' does not exist.", file=sys.stderr)
        return 1

    alerts = read_alerts_from_jsonl(alerts_file)
    target_id = args.alert_id.strip()

    match: Optional[ExplainableAlert] = None
    for alt in alerts:
        if alt.alert_id == target_id:
            match = alt
            break

    if match is None:
        print(f"Error: Alert ID '{target_id}' not found in '{alerts_file}'.", file=sys.stderr)
        return 1

    pretty_print_evidence(match, raw_json=args.json)
    return 0


def cmd_list_alerts(args: argparse.Namespace) -> int:
    """Execute the 'list-alerts' CLI command."""
    alerts_file = Path(args.alerts_file)
    if not alerts_file.exists():
        print(f"Error: Alerts file '{alerts_file}' does not exist.", file=sys.stderr)
        return 1

    alerts = read_alerts_from_jsonl(alerts_file)
    total_count = len(alerts)

    # Filter alerts
    filtered = alerts
    if args.type:
        req_type = args.type.strip().upper()
        filtered = [a for a in filtered if a.type.upper() == req_type]

    if args.severity:
        req_sev = args.severity.strip().lower()
        filtered = [a for a in filtered if a.severity.lower() == req_sev]

    if args.detector:
        req_det = args.detector.strip().lower()
        filtered = [a for a in filtered if a.detector.lower() == req_det]

    if args.limit and args.limit > 0:
        filtered = filtered[: args.limit]

    if args.json:
        for alt in filtered:
            print(json.dumps(alt.to_dict()))
        return 0

    # Formatted Table View
    print(f"\n{'ALERT ID':<16} | {'DET':<5} | {'SEV':<6} | {'TYPE':<20} | {'PREFIX':<20} | {'ORIGIN':<8} | {'TIME (UTC)'}")
    print("-" * 105)
    for alt in filtered:
        time_str = alt.time.get("utc", "") if isinstance(alt.time, dict) else str(alt.time)
        print(
            f"{alt.alert_id:<16} | {alt.detector:<5} | {alt.severity:<6} | {alt.type:<20} | "
            f"{alt.prefix:<20} | AS{alt.origin_asn:<6} | {time_str}"
        )
    print("-" * 105)
    print(f"Showing {len(filtered)} alerts (filtered from {total_count} total alerts).\n")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Execute the 'generate' CLI command."""
    config_path = args.config
    output_path = args.output
    print(f"Generating explainable alerts using config: {config_path}...")
    generate_explainable_alerts(config_path=config_path, output_path=output_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="BGP Anomaly Monitor: Explainable Alerts & Inspection CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: explain
    parser_explain = subparsers.add_parser("explain", help="Pretty-print evidence for a specific alert")
    parser_explain.add_argument("--alert-id", required=True, help="Alert ID to explain (e.g. alt-600d62e5)")
    parser_explain.add_argument(
        "--alerts-file", default="reports/alerts.jsonl", help="Path to alerts.jsonl file (default: reports/alerts.jsonl)"
    )
    parser_explain.add_argument("--json", action="store_true", help="Print raw JSON format instead of report")

    # Command: list-alerts
    parser_list = subparsers.add_parser("list-alerts", help="Filter and list alerts")
    parser_list.add_argument("--type", "-t", default=None, help="Filter by alert type (e.g. ORIGIN_HIJACK, ML_ANOMALY)")
    parser_list.add_argument("--severity", "-s", default=None, help="Filter by severity (high, medium, low)")
    parser_list.add_argument("--detector", "-d", default=None, help="Filter by detector (rules, ml)")
    parser_list.add_argument("--limit", "-n", type=int, default=None, help="Limit number of output rows")
    parser_list.add_argument(
        "--alerts-file", default="reports/alerts.jsonl", help="Path to alerts.jsonl file (default: reports/alerts.jsonl)"
    )
    parser_list.add_argument("--json", action="store_true", help="Print matching alerts as JSON lines")

    # Command: generate
    parser_gen = subparsers.add_parser("generate", help="Regenerate explainable alerts for both detectors")
    parser_gen.add_argument("--config", default="config_ml.yaml", help="Path to configuration YAML")
    parser_gen.add_argument("--output", default="reports/alerts.jsonl", help="Path to output alerts JSONL file")

    return parser


def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "explain":
        return cmd_explain(args)
    if args.command == "list-alerts":
        return cmd_list_alerts(args)
    if args.command == "generate":
        return cmd_generate(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

