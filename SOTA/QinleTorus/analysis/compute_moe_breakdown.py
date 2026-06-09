#!/usr/bin/env python3
"""Compute A2A end-to-end breakdown for the MoE-layer experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def raw_d2d_bytes_per_cycle(config: dict[str, Any], clock_ghz: float) -> float:
    noini = config["NoINI"]
    raw_tbps = (
        float(noini["D2DLanes"])
        * float(noini["D2DDataRateGTpsPerLane"])
        / 8.0
        / 1000.0
    )
    return raw_tbps * 1000.0 / clock_ghz


def compute_per_node_payload(rows: list[dict[str, str]], key: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for row in rows:
        node = int(row[key])
        out[node] = out.get(node, 0) + int(row["bytes_per_chunk"])
    return out


def int_field(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    return int(value) if value not in ("", None) else default


def compute_time_cycles(rows: list[dict[str, str]]) -> int:
    if "layer_compute_done_cycle" in rows[0]:
        return max(int_field(row, "layer_compute_done_cycle") for row in rows)
    return max(int(row["compute_done_cycle"]) for row in rows)


def unique_compute_cycles(rows: list[dict[str, str]]) -> tuple[int, dict[int, int]]:
    total = 0
    by_node: dict[int, int] = {}
    seen: set[str] = set()

    for row in rows:
        key = row.get("compute_accounting_key") or f"event:{row.get('event_id', len(seen))}"
        if key in seen:
            continue
        seen.add(key)

        cycles = int_field(row, "compute_cycle")
        node = int_field(row, "compute_node", int(row["src"]))
        total += cycles
        by_node[node] = by_node.get(node, 0) + cycles

    return total, by_node


def events_by_phase(rows: list[dict[str, str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        phase = row.get("phase", "a2a")
        out[phase] = out.get(phase, 0) + 1
    return out


def max_dim_busy_cycles(metrics: dict[str, Any]) -> float:
    busy = metrics["metrics"].get("per_dim_busy_cycles", {})
    if not busy:
        return 0.0
    return max(float(value) for value in busy.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="MoE A2A breakdown")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--clock-ghz", type=float, default=1.0)
    parser.add_argument("--noi-wait-source", choices=["average-delay", "a2a-window"], default="average-delay")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = load_csv_rows(Path(args.csv))
    metrics = json.loads(Path(args.metrics).read_text())
    hardware = load_yaml(Path(args.hardware_config))

    compute_time = compute_time_cycles(rows)
    total_compute_cycles, compute_cycles_by_source = unique_compute_cycles(rows)

    noi = hardware["NoI"]
    runtime_reconfigurable = noi["interconnectType"] == "runtimeReconfigurableCrossbar"
    reconfig_wait = (
        int(noi["topologyGenerationLatencyNs"])
        + int(noi["crossbarConfigurationLatencyNs"])
        if runtime_reconfigurable else 0
    )
    barrier_wait = int(noi["syncBarrierLatencyNs"])

    bytes_per_cycle = raw_d2d_bytes_per_cycle(hardware, args.clock_ghz)
    send_bytes = compute_per_node_payload(rows, "src")
    recv_bytes = compute_per_node_payload(rows, "dst")
    injection_wait = math.ceil(max(send_bytes.values()) / bytes_per_cycle)
    ejection_wait = math.ceil(max(recv_bytes.values()) / bytes_per_cycle)

    # This MoE run exercises compute-side A2A. HBM-side staging is not used by
    # the generated trace, so the measured HBM-side wait is zero for this run.
    hbm_side_wait = 0

    if args.noi_wait_source == "average-delay":
        network_window_time = float(metrics["popnet"]["average_delay"])
        network_service_time = network_window_time
    else:
        network_window_time = float(metrics["metrics"]["a2a_latency_cycles"])
        network_service_time = max_dim_busy_cycles(metrics)

    congestion_extra = max(0.0, network_window_time - network_service_time)
    communication_time = (
        injection_wait
        + network_service_time
        + hbm_side_wait
        + ejection_wait
    )
    congestion_time = reconfig_wait + barrier_wait + congestion_extra
    total_time = compute_time + communication_time + congestion_time

    breakdown = {
        "experiment": {
            "csv": str(Path(args.csv)),
            "metrics": str(Path(args.metrics)),
            "hardware_config": str(Path(args.hardware_config)),
            "clock_ghz": args.clock_ghz,
        },
        "moe": {
            "events": len(rows),
            "events_by_phase": events_by_phase(rows),
            "compute_cycles_by_source": compute_cycles_by_source,
            "total_compute_kernel_cycles": total_compute_cycles,
            "compute_time_cycles": compute_time,
            "send_bytes_by_source": send_bytes,
            "recv_bytes_by_destination": recv_bytes,
            "d2d_bytes_per_cycle_per_compute_chiplet": bytes_per_cycle,
            "noi_wait_source": args.noi_wait_source,
        },
        "popnet": metrics["popnet"],
        "detailed_components_cycles": {
            "Reconfiguration Control": reconfig_wait,
            "Injection Transfer": injection_wait,
            "Network Service": network_service_time,
            "HBM-Side Transfer": hbm_side_wait,
            "Ejection Transfer": ejection_wait,
            "Barrier Control": barrier_wait,
            "A2A Network Window": network_window_time,
            "Congestion Extra": congestion_extra,
        },
        "breakdown_cycles": {
            "A2A End-to-End Time": total_time,
            "Compute Time": compute_time,
            "Communication Time": communication_time,
            "Congestion Time": congestion_time,
        },
        "breakdown_us": {
            key: value / (args.clock_ghz * 1000.0)
            for key, value in {
                "A2A End-to-End Time": total_time,
                "Compute Time": compute_time,
                "Communication Time": communication_time,
                "Congestion Time": congestion_time,
            }.items()
        },
        "detailed_components_us": {
            key: value / (args.clock_ghz * 1000.0)
            for key, value in {
                "Reconfiguration Control": reconfig_wait,
                "Injection Transfer": injection_wait,
                "Network Service": network_service_time,
                "HBM-Side Transfer": hbm_side_wait,
                "Ejection Transfer": ejection_wait,
                "Barrier Control": barrier_wait,
                "A2A Network Window": network_window_time,
                "Congestion Extra": congestion_extra,
            }.items()
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(breakdown, indent=2))

    c = breakdown["breakdown_cycles"]
    u = breakdown["breakdown_us"]
    print(f"A2A End-to-End Time: {c['A2A End-to-End Time']:.0f} cycles ~= {u['A2A End-to-End Time']:.3f} us")
    print(f"├── Compute Time: {c['Compute Time']:.0f} cycles ~= {u['Compute Time']:.3f} us")
    print(f"├── Communication Time: {c['Communication Time']:.0f} cycles ~= {u['Communication Time']:.3f} us")
    print(f"└── Congestion Time: {c['Congestion Time']:.0f} cycles ~= {u['Congestion Time']:.3f} us")
    print(f"wrote breakdown -> {out_path}")


if __name__ == "__main__":
    main()
