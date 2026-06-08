#!/usr/bin/env python3
"""Build a Switch MoE A2A-only trace.

This trace is for the communication phase, not the full expert-compute layer.
It models top-1 routed remote expert traffic after router projection:

  dispatch: source chiplet -> expert chiplet
  return:   expert chiplet -> original source chiplet

The trace intentionally keeps expert FFN compute off the A2A critical path so
the PopNet run measures the MoE all-to-all pressure directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def load_router_cycles(path: Path) -> dict[int, int]:
    data = json.loads(path.read_text())
    return {int(k): int(v) for k, v in data["router_cycles_by_source"].items()}


def dim_order_label(chunk_id: int, dims: int) -> str:
    rotation = [(chunk_id + i) % dims for i in range(dims)]
    if dims <= 3:
        names = "XYZ"
        return "".join(names[d] for d in rotation)
    return "".join(str(d) for d in rotation)


def build_rows(
    router_cycles: dict[int, int],
    nodes: int,
    dims: int,
    tokens_per_source: int,
    d_model: int,
    num_experts: int,
    expert_capacity: int,
    flit_size: int,
    bytes_per_element: int,
    phases: list[str],
) -> list[dict[str, Any]]:
    if num_experts % nodes != 0:
        raise ValueError("num_experts must be divisible by nodes")
    if tokens_per_source % num_experts != 0:
        raise ValueError("tokens_per_source must be divisible by num_experts")

    tokens_per_expert = tokens_per_source // num_experts
    if tokens_per_expert > expert_capacity:
        raise ValueError("tokens_per_expert exceeds expert_capacity")
    if tokens_per_expert % dims != 0:
        raise ValueError("tokens_per_expert must be divisible by dims")

    experts_per_node = num_experts // nodes
    chunk_tokens = tokens_per_expert // dims
    bytes_per_chunk = chunk_tokens * d_model * bytes_per_element
    flits_per_chunk = math.ceil(bytes_per_chunk / flit_size)

    rows: list[dict[str, Any]] = []
    accounted_router: set[int] = set()
    event_id = 0

    for original_src in range(nodes):
        router_done = int(router_cycles[original_src])
        for expert_id in range(num_experts):
            expert_chiplet = expert_id // experts_per_node
            if expert_chiplet == original_src:
                continue

            for chunk_id in range(dims):
                for phase in phases:
                    if phase == "dispatch":
                        src = original_src
                        dst = expert_chiplet
                    elif phase == "return":
                        src = expert_chiplet
                        dst = original_src
                    else:
                        raise ValueError(f"unsupported phase: {phase}")

                    compute_cycle = 0
                    compute_key = f"none:{event_id}"
                    if original_src not in accounted_router and phase == "dispatch":
                        compute_cycle = router_done
                        compute_key = f"router:{original_src}"
                        accounted_router.add(original_src)

                    rows.append({
                        "event_id": event_id,
                        "pair_id": original_src * num_experts + expert_id,
                        "moe_layer_id": 0,
                        "benchmark": "google/switch-base-8/a2a-only-dispatch-return",
                        "phase": phase,
                        "src": src,
                        "dst": dst,
                        "original_src": original_src,
                        "expert_chiplet": expert_chiplet,
                        "expert_id": expert_id,
                        "chunk_id": chunk_id,
                        "dim_order": dim_order_label(chunk_id, dims),
                        "tokens_per_source": tokens_per_source,
                        "tokens_per_expert": tokens_per_expert,
                        "tokens_per_chunk": chunk_tokens,
                        "hidden_size": d_model,
                        "intermediate_size": "",
                        "expert_capacity": expert_capacity,
                        "num_experts": num_experts,
                        "bytes_per_chunk": bytes_per_chunk,
                        "flits_per_chunk": flits_per_chunk,
                        "compute_cycle": compute_cycle,
                        "compute_done_cycle": router_done,
                        "source_ready_cycle": router_done,
                        "layer_compute_done_cycle": router_done,
                        "compute_node": original_src,
                        "compute_accounting_key": compute_key,
                        "sim_backend": "BackendSim",
                        "kernel": "switch.router_top1_a2a_payload",
                        "backend_result_files": "",
                        "router_cycle": router_done,
                        "router_done_cycle": router_done,
                    })
                    event_id += 1

    return rows


def write_rows(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Switch MoE A2A-only trace.")
    parser.add_argument("--router-json", required=True)
    parser.add_argument("--dims", type=int, required=True)
    parser.add_argument("--ary", type=int, required=True)
    parser.add_argument("--tokens-per-source", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--expert-capacity", type=int, default=64)
    parser.add_argument("--flit-size", type=int, default=64)
    parser.add_argument("--bytes-per-element", type=int, default=4)
    parser.add_argument("--phases", default="dispatch,return")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    phases = [phase.strip() for phase in args.phases.split(",") if phase.strip()]
    rows = build_rows(
        router_cycles=load_router_cycles(Path(args.router_json)),
        nodes=args.ary ** args.dims,
        dims=args.dims,
        tokens_per_source=args.tokens_per_source,
        d_model=args.d_model,
        num_experts=args.num_experts,
        expert_capacity=args.expert_capacity,
        flit_size=args.flit_size,
        bytes_per_element=args.bytes_per_element,
        phases=phases,
    )
    write_rows(rows, Path(args.out))
    print(f"wrote Switch A2A-only trace -> {args.out}")
    print(f"  events={len(rows)} phases={phases}")
    print(f"  bytes_per_chunk={rows[0]['bytes_per_chunk']} flits_per_chunk={rows[0]['flits_per_chunk']}")


if __name__ == "__main__":
    main()
