#!/usr/bin/env python3
"""Generate a routed Switch-style MoE A2A trace with real top-1 routing.

This benchmark differs from the uniform A2A generator: tokens are routed by a
Switch-style router projection, then top-1 expert selection and expert capacity
are applied before generating dispatch/return traffic.  The resulting A2A trace
therefore carries the router's actual expert-load imbalance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from run_switch_moe_layer import (
    TORCHSIM_DIR,
    backend_result_files,
    cycles_from_new_backend_results,
    import_pytorchsim,
)


def dim_order_label(chunk_id: int, dims: int) -> str:
    rotation = [(chunk_id + i) % dims for i in range(dims)]
    if dims <= 3:
        names = "XYZ"
        return "".join(names[d] for d in rotation)
    return "".join(str(d) for d in rotation)


def zipf_probs(num_experts: int, skew: float) -> list[float]:
    weights = [1.0 / ((idx + 1) ** skew) for idx in range(num_experts)]
    total = sum(weights)
    return [value / total for value in weights]


def split_tokens(tokens: int, chunks: int) -> list[int]:
    base = tokens // chunks
    rem = tokens % chunks
    return [base + (1 if idx < rem else 0) for idx in range(chunks)]


def make_router_inputs(
    torch,
    nodes: int,
    tokens_per_source: int,
    d_model: int,
    num_experts: int,
    skew: float,
    signal: float,
    noise: float,
    seed: int,
):
    gen = torch.Generator().manual_seed(seed)
    probs = torch.tensor(zipf_probs(num_experts, skew), dtype=torch.float32)
    preferred = torch.multinomial(
        probs,
        nodes * tokens_per_source,
        replacement=True,
        generator=gen,
    ).reshape(nodes, tokens_per_source)

    tokens = noise * torch.randn(nodes, tokens_per_source, d_model, generator=gen)
    router_weight = 0.01 * torch.randn(d_model, num_experts, generator=gen)

    for expert_id in range(num_experts):
        router_weight[expert_id, expert_id] += signal

    for src in range(nodes):
        for token_id in range(tokens_per_source):
            expert_id = int(preferred[src, token_id])
            tokens[src, token_id, expert_id] += signal

    return tokens, router_weight, preferred, probs


def apply_capacity(assignments, nodes: int, tokens_per_source: int, num_experts: int, expert_capacity: int):
    raw_counts = [[0 for _ in range(num_experts)] for _ in range(nodes)]
    kept_counts = [[0 for _ in range(num_experts)] for _ in range(nodes)]
    dropped_counts = [[0 for _ in range(num_experts)] for _ in range(nodes)]
    expert_fill = [0 for _ in range(num_experts)]

    # Interleave sources to avoid giving the first source chiplet all capacity
    # for hot experts.
    for token_id in range(tokens_per_source):
        for src in range(nodes):
            expert_id = int(assignments[src, token_id])
            raw_counts[src][expert_id] += 1
            if expert_fill[expert_id] < expert_capacity:
                expert_fill[expert_id] += 1
                kept_counts[src][expert_id] += 1
            else:
                dropped_counts[src][expert_id] += 1

    return raw_counts, kept_counts, dropped_counts, expert_fill


def build_trace_rows(
    router_cycles: dict[int, int],
    kept_counts: list[list[int]],
    dropped_counts: list[list[int]],
    nodes: int,
    dims: int,
    d_model: int,
    d_ff: int,
    num_experts: int,
    expert_capacity: int,
    flit_size: int,
    bytes_per_element: int,
    phases: list[str],
) -> list[dict[str, Any]]:
    experts_per_node = num_experts // nodes
    rows: list[dict[str, Any]] = []
    event_id = 0
    accounted_router: set[int] = set()

    for original_src in range(nodes):
        router_done = int(router_cycles[original_src])
        for expert_id in range(num_experts):
            expert_chiplet = expert_id // experts_per_node
            token_count = int(kept_counts[original_src][expert_id])
            if token_count <= 0 or expert_chiplet == original_src:
                continue

            for chunk_id, chunk_tokens in enumerate(split_tokens(token_count, dims)):
                if chunk_tokens <= 0:
                    continue
                bytes_per_chunk = chunk_tokens * d_model * bytes_per_element
                flits_per_chunk = math.ceil(bytes_per_chunk / flit_size)

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
                        "benchmark": "google/switch-base-8/routed-top1-a2a",
                        "phase": phase,
                        "src": src,
                        "dst": dst,
                        "original_src": original_src,
                        "expert_chiplet": expert_chiplet,
                        "expert_id": expert_id,
                        "chunk_id": chunk_id,
                        "dim_order": dim_order_label(chunk_id, dims),
                        "tokens_per_source": "",
                        "tokens_per_expert": token_count,
                        "tokens_per_chunk": chunk_tokens,
                        "hidden_size": d_model,
                        "intermediate_size": d_ff,
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
                        "kernel": "switch.top1_router_capacity_dispatch_return",
                        "backend_result_files": "",
                        "router_cycle": router_done,
                        "router_done_cycle": router_done,
                        "raw_tokens_for_expert": token_count + int(dropped_counts[original_src][expert_id]),
                        "dropped_tokens_for_expert": int(dropped_counts[original_src][expert_id]),
                    })
                    event_id += 1

    return rows


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run routed Switch-style MoE A2A trace generation.")
    parser.add_argument("--dims", type=int, default=2)
    parser.add_argument("--ary", type=int, default=2)
    parser.add_argument("--tokens-per-source", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--expert-capacity", type=int, default=64)
    parser.add_argument("--flit-size", type=int, default=64)
    parser.add_argument("--bytes-per-element", type=int, default=4)
    parser.add_argument("--router-skew", type=float, default=0.5)
    parser.add_argument("--router-signal", type=float, default=3.0)
    parser.add_argument("--router-noise", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--phases", default="dispatch,return")
    parser.add_argument("--out", required=True)
    parser.add_argument("--router-out", required=True)
    args = parser.parse_args()

    nodes = args.ary ** args.dims
    if args.num_experts % nodes != 0:
        raise ValueError("num_experts must be divisible by nodes")

    torch, device, _tog_simulator_cls, sim_backend = import_pytorchsim()

    tokens, router_weight, preferred, probs = make_router_inputs(
        torch=torch,
        nodes=nodes,
        tokens_per_source=args.tokens_per_source,
        d_model=args.d_model,
        num_experts=args.num_experts,
        skew=args.router_skew,
        signal=args.router_signal,
        noise=args.router_noise,
        seed=args.seed,
    )

    @torch.compile(dynamic=False)
    def switch_router_projection(tokens_in, router_weight_in):
        return torch.matmul(tokens_in, router_weight_in)

    router_cycles: dict[int, int] = {}
    result_files: dict[int, list[str]] = {}
    assignments = []

    print("Routed Switch-style MoE layer trace:")
    print(f"  TORCHSIM_DIR        : {TORCHSIM_DIR}")
    print(f"  device              : {device}")
    print(f"  sim backend         : {sim_backend}")
    print(f"  nodes               : {nodes}")
    print(f"  tokens/source       : {args.tokens_per_source}")
    print(f"  experts/capacity    : {args.num_experts}/{args.expert_capacity}")
    print(f"  router skew/signal  : {args.router_skew}/{args.router_signal}")

    for src in range(nodes):
        source_tokens_cpu = tokens[src]
        before = backend_result_files()
        logits_dev = switch_router_projection(
            source_tokens_cpu.to(device=device),
            router_weight.to(device=device),
        )
        if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()
        cycles, files = cycles_from_new_backend_results(before)
        if cycles <= 0:
            raise RuntimeError(f"Unable to parse router BackendSim cycles for source {src}")

        logits_cpu = torch.matmul(source_tokens_cpu, router_weight)
        top1 = torch.argmax(logits_cpu, dim=-1)
        router_cycles[src] = cycles
        result_files[src] = files
        assignments.append(top1)
        print(f"  source {src}: router_cycles={cycles}")

    assignments_tensor = torch.stack(assignments, dim=0)
    raw_counts, kept_counts, dropped_counts, expert_fill = apply_capacity(
        assignments_tensor,
        nodes=nodes,
        tokens_per_source=args.tokens_per_source,
        num_experts=args.num_experts,
        expert_capacity=args.expert_capacity,
    )

    phases = [phase.strip() for phase in args.phases.split(",") if phase.strip()]
    rows = build_trace_rows(
        router_cycles=router_cycles,
        kept_counts=kept_counts,
        dropped_counts=dropped_counts,
        nodes=nodes,
        dims=args.dims,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_experts=args.num_experts,
        expert_capacity=args.expert_capacity,
        flit_size=args.flit_size,
        bytes_per_element=args.bytes_per_element,
        phases=phases,
    )

    if not rows:
        raise RuntimeError("routed MoE trace has no remote A2A events")

    write_csv(rows, Path(args.out))

    route_summary = {
        "benchmark": "google/switch-base-8/routed-top1-a2a",
        "nodes": nodes,
        "tokens_per_source": args.tokens_per_source,
        "global_tokens": nodes * args.tokens_per_source,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_experts": args.num_experts,
        "expert_capacity": args.expert_capacity,
        "router_skew": args.router_skew,
        "router_signal": args.router_signal,
        "router_noise": args.router_noise,
        "seed": args.seed,
        "preferred_expert_probabilities": [float(v) for v in probs.tolist()],
        "raw_counts_by_source_expert": raw_counts,
        "kept_counts_by_source_expert": kept_counts,
        "dropped_counts_by_source_expert": dropped_counts,
        "kept_counts_by_expert": expert_fill,
        "total_kept_tokens": int(sum(expert_fill)),
        "total_dropped_tokens": int(sum(sum(row) for row in dropped_counts)),
        "router_cycles_by_source": {str(k): v for k, v in router_cycles.items()},
        "backend_result_files": {str(k): v for k, v in result_files.items()},
        "sim_backend": sim_backend,
        "trace_csv": str(Path(args.out)),
    }

    router_out = Path(args.router_out)
    router_out.parent.mkdir(parents=True, exist_ok=True)
    router_out.write_text(json.dumps(route_summary, indent=2))

    print(f"trace saved        : {args.out}")
    print(f"route summary saved: {args.router_out}")
    print(f"  events={len(rows)}")
    print(f"  raw_counts_by_source_expert={raw_counts}")
    print(f"  kept_counts_by_source_expert={kept_counts}")
    print(f"  dropped_tokens={route_summary['total_dropped_tokens']}")


if __name__ == "__main__":
    main()
