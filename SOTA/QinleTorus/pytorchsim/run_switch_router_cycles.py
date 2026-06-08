#!/usr/bin/env python3
"""Measure Switch top-1 router projection cycles with PyTorchSim BackendSim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_switch_moe_layer import (
    TORCHSIM_DIR,
    backend_result_files,
    cycles_from_new_backend_results,
    import_pytorchsim,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Switch router matmul cycles.")
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--tokens-per-source", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch, device, _tog_simulator_cls, sim_backend = import_pytorchsim()

    @torch.compile(dynamic=False)
    def switch_router_projection(tokens, router_weight):
        return torch.matmul(tokens, router_weight)

    print("Switch top-1 router projection PyTorchSim benchmark:")
    print(f"  TORCHSIM_DIR        : {TORCHSIM_DIR}")
    print(f"  device              : {device}")
    print(f"  sim backend         : {sim_backend}")
    print(f"  nodes               : {args.nodes}")
    print(f"  tokens per source   : {args.tokens_per_source}")
    print(f"  d_model             : {args.d_model}")
    print(f"  num_experts         : {args.num_experts}")

    router_cycles: dict[str, int] = {}
    result_files: dict[str, list[str]] = {}

    for src in range(args.nodes):
        torch.manual_seed(40_000 + src)
        tokens = torch.randn(args.tokens_per_source, args.d_model).to(device=device)
        router_weight = torch.randn(args.d_model, args.num_experts).to(device=device)

        before = backend_result_files()
        switch_router_projection(tokens, router_weight)
        if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()

        cycles, files = cycles_from_new_backend_results(before)
        if cycles <= 0:
            raise RuntimeError(f"Unable to parse router BackendSim cycles for source {src}")

        router_cycles[str(src)] = cycles
        result_files[str(src)] = files
        print(f"  source {src}: router_projection_cycles={cycles}")

    out = {
        "benchmark": "google/switch-base-8/router-projection",
        "nodes": args.nodes,
        "tokens_per_source": args.tokens_per_source,
        "d_model": args.d_model,
        "num_experts": args.num_experts,
        "sim_backend": sim_backend,
        "router_cycles_by_source": router_cycles,
        "backend_result_files": result_files,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"router cycles saved : {out_path}")


if __name__ == "__main__":
    main()
