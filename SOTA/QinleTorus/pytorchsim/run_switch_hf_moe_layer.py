#!/usr/bin/env python3
"""Generate a real Switch HF MoE A2A trace.

The routing source is a real ``google/switch-base-8`` checkpoint:

* hidden states are captured from a HuggingFace encoder forward pass;
* router weights are loaded from ``pytorch_model.bin``;
* top-1 expert assignment and expert capacity are applied before dispatch and
  return A2A traffic is emitted.
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
from run_switch_routed_moe_layer import apply_capacity, dim_order_label, split_tokens


def torch_load(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def load_checkpoint_state(torch: Any, model_dir: Path, checkpoint_file: str) -> dict[str, Any]:
    checkpoint_path = model_dir / checkpoint_file
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    state = torch_load(torch, checkpoint_path)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint must be a state_dict, got {type(state)!r}")
    return state


def load_hidden_states(torch: Any, path: Path, nodes: int, tokens_per_source: int, d_model: int) -> tuple[Any, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing hidden-state artifact: {path}")
    payload = torch_load(torch, path)
    if isinstance(payload, dict):
        hidden = payload["hidden_states"]
        meta = {key: value for key, value in payload.items() if key != "hidden_states"}
    else:
        hidden = payload
        meta = {}

    if hidden.dim() != 3:
        raise ValueError(f"hidden_states must have shape [nodes,tokens,d_model], got {tuple(hidden.shape)}")
    if hidden.shape[0] < nodes:
        raise ValueError(f"hidden_states only has {hidden.shape[0]} sources, need {nodes}")
    if hidden.shape[1] < tokens_per_source:
        raise ValueError(f"hidden_states only has {hidden.shape[1]} tokens/source, need {tokens_per_source}")
    if hidden.shape[2] != d_model:
        raise ValueError(f"hidden d_model={hidden.shape[2]} does not match expected {d_model}")

    return hidden[:nodes, :tokens_per_source, :].float().contiguous(), meta


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
                        "benchmark": "google/switch-base-8/hf-checkpoint-top1-a2a",
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
                        "kernel": "switch.hf_checkpoint_router_top1_dispatch_return",
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
    parser = argparse.ArgumentParser(description="Run real HF Switch MoE A2A trace generation.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--checkpoint-file", default="pytorch_model.bin")
    parser.add_argument("--hidden-states", required=True)
    parser.add_argument(
        "--router-key",
        default="encoder.block.1.layer.1.mlp.router.classifier.weight",
    )
    parser.add_argument("--dims", type=int, default=2)
    parser.add_argument("--ary", type=int, default=2)
    parser.add_argument("--tokens-per-source", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--expert-capacity", type=int, default=64)
    parser.add_argument("--flit-size", type=int, default=64)
    parser.add_argument("--bytes-per-element", type=int, default=4)
    parser.add_argument("--phases", default="dispatch,return")
    parser.add_argument("--out", required=True)
    parser.add_argument("--router-out", required=True)
    args = parser.parse_args()

    nodes = args.ary ** args.dims
    if args.num_experts % nodes != 0:
        raise ValueError("num_experts must be divisible by nodes")

    torch, device, _tog_simulator_cls, sim_backend = import_pytorchsim()
    model_dir = Path(args.model_dir)
    state = load_checkpoint_state(torch, model_dir, args.checkpoint_file)
    if args.router_key not in state:
        candidates = [key for key in state if key.endswith("router.classifier.weight")]
        raise KeyError(f"missing router key {args.router_key}; candidates={candidates}")

    router_weight = state[args.router_key].float().contiguous()
    if tuple(router_weight.shape) != (args.num_experts, args.d_model):
        raise ValueError(
            f"router weight shape {tuple(router_weight.shape)} does not match "
            f"({args.num_experts}, {args.d_model})"
        )
    router_weight_t = router_weight.t().contiguous()

    hidden, hidden_meta = load_hidden_states(
        torch,
        Path(args.hidden_states),
        nodes=nodes,
        tokens_per_source=args.tokens_per_source,
        d_model=args.d_model,
    )

    @torch.compile(dynamic=False)
    def switch_router_projection(tokens_in, router_weight_in):
        return torch.matmul(tokens_in, router_weight_in)

    router_cycles: dict[int, int] = {}
    result_files: dict[int, list[str]] = {}
    assignments = []

    print("Real HF Switch MoE layer trace:")
    print(f"  TORCHSIM_DIR      : {TORCHSIM_DIR}")
    print(f"  device            : {device}")
    print(f"  sim backend       : {sim_backend}")
    print(f"  model dir         : {model_dir}")
    print(f"  hidden states     : {args.hidden_states}")
    print(f"  router key        : {args.router_key}")
    print(f"  nodes             : {nodes}")
    print(f"  tokens/source     : {args.tokens_per_source}")
    print(f"  experts/capacity  : {args.num_experts}/{args.expert_capacity}")

    for src in range(nodes):
        source_tokens_cpu = hidden[src]
        before = backend_result_files()
        switch_router_projection(
            source_tokens_cpu.to(device=device),
            router_weight_t.to(device=device),
        )
        if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()

        cycles, files = cycles_from_new_backend_results(before)
        if cycles <= 0:
            raise RuntimeError(f"Unable to parse router BackendSim cycles for source {src}")

        logits_cpu = torch.matmul(source_tokens_cpu, router_weight_t)
        top1 = torch.argmax(logits_cpu, dim=-1)
        router_cycles[src] = cycles
        result_files[src] = files
        assignments.append(top1)
        print(f"  source {src}: router_cycles={cycles} top1={torch.bincount(top1, minlength=args.num_experts).tolist()}")

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
        raise RuntimeError("real HF MoE trace has no remote A2A events")

    write_csv(rows, Path(args.out))

    route_summary = {
        "benchmark": "google/switch-base-8/hf-checkpoint-top1-a2a",
        "routing_source": "real_huggingface_checkpoint_hidden_states",
        "model_dir": str(model_dir),
        "checkpoint_file": args.checkpoint_file,
        "hidden_states": args.hidden_states,
        "router_key": args.router_key,
        "hidden_meta": json_safe(hidden_meta),
        "nodes": nodes,
        "tokens_per_source": args.tokens_per_source,
        "global_tokens": nodes * args.tokens_per_source,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_experts": args.num_experts,
        "expert_capacity": args.expert_capacity,
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
