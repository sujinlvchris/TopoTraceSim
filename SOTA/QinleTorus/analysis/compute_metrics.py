#!/usr/bin/env python3
"""Compute QinleTorus metrics for one PopNet run and dump them to JSON.

Inputs:
    --bench         path to the popnet bench (we re-read it to know min/max
                    inject_cycle, total flits, per-dim coverage)
    --csv           optional chunked PyTorchSim CSV; when present, payload
                    bytes are computed from unique chunk rows instead of
                    bench flit-hop volume
    --stdout        path to the captured popnet stdout
    --log           path to popnet.log
    --dims, --ary   torus geometry
    --label         free-form label that ends up in the JSON (e.g. dimrotation)
    --flit-size     bytes per flit, used for throughput_GBps
    --clock-ghz     simulated clock, used for throughput_GBps
    --out           output JSON path
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# allow `python analysis/compute_metrics.py` from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from popnet_io.parse_log import LogSummary, parse_log, parse_stdout  # noqa: E402


def parse_bench(bench_path: Path, dims: int):
    """Returns (min_inject, max_inject, total_packets, total_flits)."""
    min_t = None
    max_t = None
    total_packets = 0
    total_flits = 0
    with Path(bench_path).open() as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            expected = 1 + dims + dims + 1
            if len(parts) != expected:
                raise ValueError(
                    f"bench line has {len(parts)} fields, expected {expected}: {line!r}"
                )
            t = int(parts[0])
            flits = int(parts[-1])
            min_t = t if min_t is None else min(min_t, t)
            max_t = t if max_t is None else max(max_t, t)
            total_packets += 1
            total_flits += flits
    return min_t or 0, max_t or 0, total_packets, total_flits


def parse_payload_bytes(csv_path: Path | None, fallback_total_flits: int, flit_size: int) -> int:
    if csv_path is None or not Path(csv_path).is_file():
        return fallback_total_flits * flit_size
    total = 0
    seen = set()
    with Path(csv_path).open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row.get("pair_id"), row.get("src"), row.get("dst"), row.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            total += int(row["bytes_per_chunk"])
    return total


def compute_metrics(
    bench_path: Path,
    csv_path: Path | None,
    stdout_path: Path,
    log_path: Path,
    dims: int,
    ary: int,
    label: str,
    flit_size: int,
    clock_ghz: float,
):
    min_inj, max_inj, total_packets, total_flits = parse_bench(bench_path, dims)
    total_bytes = parse_payload_bytes(csv_path, total_flits, flit_size)
    stdout_sum = parse_stdout(stdout_path)
    log_sum: LogSummary = parse_log(log_path)

    finish_time = stdout_sum.finish_time or float(max_inj)
    a2a_latency = float(finish_time) - float(min_inj)

    per_dim_events = log_sum.per_dim_events(dims)
    total_events = log_sum.total_events

    # event-balance: how evenly distributed across dims (1.0 = perfectly balanced)
    if total_events > 0 and dims > 0:
        ideal = total_events / dims
        max_dev = max(abs(per_dim_events[d] - ideal) for d in per_dim_events)
        balance = 1.0 - (max_dev / ideal if ideal else 0.0)
    else:
        balance = 0.0

    # PopNet's stock popnet.log has no per-wire cycle stamps, so per-dim busy
    # cycles are approximated by spreading wire/credit events over 2*ary links.
    per_dim_busy_cycles = {
        d: (per_dim_events[d] / (2 * ary) if ary > 0 else 0.0)
        for d in range(dims)
    }
    if total_events > 0 and dims > 0 and a2a_latency > 0:
        max_dim_busy = max(per_dim_busy_cycles.values())
        bubble_cycles = max(0.0, a2a_latency - max_dim_busy)
    else:
        bubble_cycles = 0.0
    per_dim_utilization = {
        d: (per_dim_busy_cycles[d] / a2a_latency if a2a_latency > 0 else 0.0)
        for d in range(dims)
    }
    throughput_gbps = (
        (total_bytes / a2a_latency) * clock_ghz
        if a2a_latency > 0 else 0.0
    )

    metrics = {
        "label": label,
        "geom": {"dims": dims, "ary": ary, "nodes": ary ** dims},
        "bench": {
            "path": str(bench_path),
            "packets": total_packets,
            "total_flits": total_flits,
            "payload_bytes": total_bytes,
            "min_inject_cycle": min_inj,
            "max_inject_cycle": max_inj,
        },
        "popnet": {
            "stdout_path": str(stdout_path),
            "log_path": str(log_path),
            "packet_count_reported": stdout_sum.packet_count,
            "total_finished": stdout_sum.total_finished,
            "all_finished": stdout_sum.all_finished,
            "average_delay": stdout_sum.average_delay,
            "final_current_time": stdout_sum.final_time,
            "finish_time": finish_time,
        },
        "metrics": {
            "a2a_latency_cycles": a2a_latency,
            "per_dim_events": per_dim_events,
            "per_dim_busy_cycles": per_dim_busy_cycles,
            "per_dim_utilization": per_dim_utilization,
            "total_wire_events": total_events,
            "dim_load_balance": balance,
            "pipeline_bubble_cycles": bubble_cycles,
            "throughput_flits_per_cycle": (
                total_flits / a2a_latency if a2a_latency > 0 else 0.0
            ),
            "throughput_GBps": throughput_gbps,
            "total_bytes": total_bytes,
        },
    }
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--csv", default="")
    ap.add_argument("--stdout", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--dims", type=int, required=True)
    ap.add_argument("--ary", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--flit-size", type=int, default=64)
    ap.add_argument("--clock-ghz", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    metrics = compute_metrics(
        Path(args.bench), Path(args.csv) if args.csv else None, Path(args.stdout), Path(args.log),
        args.dims, args.ary, args.label, args.flit_size, args.clock_ghz,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"wrote metrics -> {out_path}")
    print(json.dumps(metrics["metrics"], indent=2))


if __name__ == "__main__":
    main()
