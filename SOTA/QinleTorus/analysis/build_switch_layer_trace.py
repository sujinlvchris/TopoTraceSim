#!/usr/bin/env python3
"""Build a layer-level Switch MoE trace from real router and expert cycles.

Inputs:
  * expert CSV from ``run_switch_moe_layer.py``.  Its ``compute_cycle`` column
    contains real BackendSim cycles for the Switch expert FFN chunk.
  * router JSON from ``run_switch_router_cycles.py``.  It contains real
    BackendSim cycles for the Switch router projection per source chiplet.

Output:
  * a CSV that contains both dispatch and return packets.  ``compute_done_cycle``
    is the injection-ready cycle for that packet, which lets the existing
    DimRotation scheduler and PopNet pipeline run unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scheduler.timing import TimingParams, estimate_hop_cycles, estimate_packet_cycles  # noqa: E402
from scheduler.topology import TorusGeom, dim_rotation, expand_dim_order_path  # noqa: E402


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_router_cycles(path: Path) -> dict[int, int]:
    data = json.loads(path.read_text())
    return {int(k): int(v) for k, v in data["router_cycles_by_source"].items()}


def dimrotation_transfer_cycles(
    geom: TorusGeom,
    src_node: int,
    dst_node: int,
    chunk_id: int,
    flits: int,
    params: TimingParams,
) -> int:
    src = geom.node_to_coord(src_node)
    dst = geom.node_to_coord(dst_node)
    path = expand_dim_order_path(src, dst, dim_rotation(chunk_id, geom.dims))
    total = 0
    for hop in range(geom.dims):
        hop_src = path[hop]
        hop_dst = path[hop + 1]
        if hop_src == hop_dst:
            continue
        total += estimate_hop_cycles(geom, hop_src, hop_dst, flits, params)
    return total


def transfer_cycles(
    scheduler: str,
    geom: TorusGeom,
    src_node: int,
    dst_node: int,
    chunk_id: int,
    flits: int,
    params: TimingParams,
) -> int:
    if scheduler == "dimrotation":
        return dimrotation_transfer_cycles(geom, src_node, dst_node, chunk_id, flits, params)
    if scheduler == "direct":
        return estimate_packet_cycles(
            geom,
            geom.node_to_coord(src_node),
            geom.node_to_coord(dst_node),
            flits,
            params,
        )
    raise ValueError(f"unsupported scheduler: {scheduler}")


def base_output_row(
    row: dict[str, str],
    phase: str,
    src: int,
    dst: int,
    inject_cycle: int,
    compute_cycle: int,
    compute_node: int,
    compute_key: str,
    layer_compute_done: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "event_id": 0,
        "pair_id": row["pair_id"],
        "moe_layer_id": row.get("moe_layer_id", 0),
        "benchmark": "google/switch-base-8/layer-router-dispatch-expert-return",
        "phase": phase,
        "src": src,
        "dst": dst,
        "original_src": row["src"],
        "expert_chiplet": row["dst"],
        "expert_id": row["expert_id"],
        "chunk_id": row["chunk_id"],
        "dim_order": row["dim_order"],
        "tokens_per_source": row["tokens_per_source"],
        "tokens_per_expert": row["tokens_per_expert"],
        "tokens_per_chunk": row["tokens_per_chunk"],
        "hidden_size": row["hidden_size"],
        "intermediate_size": row["intermediate_size"],
        "expert_capacity": row["expert_capacity"],
        "num_experts": row["num_experts"],
        "bytes_per_chunk": row["bytes_per_chunk"],
        "flits_per_chunk": row["flits_per_chunk"],
        "compute_cycle": compute_cycle,
        "compute_done_cycle": inject_cycle,
        "source_ready_cycle": inject_cycle,
        "layer_compute_done_cycle": layer_compute_done,
        "compute_node": compute_node,
        "compute_accounting_key": compute_key,
        "sim_backend": row["sim_backend"],
        "kernel": row["kernel"],
        "backend_result_files": row["backend_result_files"],
    }
    out.update(extra)
    return out


def build_layer_rows(
    expert_rows: list[dict[str, str]],
    router_cycles: dict[int, int],
    geom: TorusGeom,
    scheduler: str,
    params: TimingParams,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expert_available: dict[int, int] = defaultdict(int)
    accounted_router: set[int] = set()

    sorted_rows = sorted(
        expert_rows,
        key=lambda r: (int(r["src"]), int(r["pair_id"]), int(r["chunk_id"])),
    )

    for expert_row in sorted_rows:
        original_src = int(expert_row["src"])
        expert_chiplet = int(expert_row["dst"])
        chunk_id = int(expert_row["chunk_id"])
        flits = int(expert_row["flits_per_chunk"])
        expert_cycle = int(expert_row["compute_cycle"])

        router_cycle = int(router_cycles[original_src])
        router_done = router_cycle
        dispatch_cycles = transfer_cycles(
            scheduler,
            geom,
            original_src,
            expert_chiplet,
            chunk_id,
            flits,
            params,
        )
        expert_start = max(expert_available[expert_chiplet], router_done + dispatch_cycles)
        expert_done = expert_start + expert_cycle
        expert_available[expert_chiplet] = expert_done

        router_compute_cycle = 0
        if original_src not in accounted_router:
            router_compute_cycle = router_cycle
            accounted_router.add(original_src)

        rows.append(base_output_row(
            expert_row,
            phase="dispatch",
            src=original_src,
            dst=expert_chiplet,
            inject_cycle=router_done,
            compute_cycle=router_compute_cycle,
            compute_node=original_src,
            compute_key=f"router:{original_src}",
            layer_compute_done=router_done,
            extra={
                "router_cycle": router_cycle,
                "router_done_cycle": router_done,
                "dispatch_estimated_cycles": dispatch_cycles,
                "expert_start_cycle": expert_start,
                "expert_cycle": expert_cycle,
                "expert_done_cycle": expert_done,
            },
        ))

        rows.append(base_output_row(
            expert_row,
            phase="return",
            src=expert_chiplet,
            dst=original_src,
            inject_cycle=expert_done,
            compute_cycle=expert_cycle,
            compute_node=expert_chiplet,
            compute_key=f"expert:{expert_row['pair_id']}:{chunk_id}",
            layer_compute_done=expert_done,
            extra={
                "router_cycle": router_cycle,
                "router_done_cycle": router_done,
                "dispatch_estimated_cycles": dispatch_cycles,
                "expert_start_cycle": expert_start,
                "expert_cycle": expert_cycle,
                "expert_done_cycle": expert_done,
            },
        ))

    for event_id, row in enumerate(rows):
        row["event_id"] = event_id
    return rows


def write_rows(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Switch layer-level A2A trace.")
    parser.add_argument("--expert-csv", required=True)
    parser.add_argument("--router-json", required=True)
    parser.add_argument("--dims", type=int, required=True)
    parser.add_argument("--ary", type=int, required=True)
    parser.add_argument("--scheduler", choices=["direct", "dimrotation"], default="dimrotation")
    parser.add_argument("--slack", type=float, default=0.10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    geom = TorusGeom(dims=args.dims, ary=args.ary)
    params = TimingParams(slack=args.slack)
    expert_rows = load_rows(Path(args.expert_csv))
    router_cycles = load_router_cycles(Path(args.router_json))
    rows = build_layer_rows(expert_rows, router_cycles, geom, args.scheduler, params)
    write_rows(rows, Path(args.out))

    phase_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        phase_counts[str(row["phase"])] += 1
    print(f"wrote Switch layer trace -> {args.out}")
    print(f"  events={len(rows)} phases={dict(phase_counts)}")
    print(f"  layer_compute_time_cycles={max(int(r['layer_compute_done_cycle']) for r in rows)}")


if __name__ == "__main__":
    main()
