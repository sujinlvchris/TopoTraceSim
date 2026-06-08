#!/usr/bin/env python3
"""Run a Switch Transformer expert-FFN plus routed-A2A benchmark through PyTorchSim.

Benchmark source:
    google/switch-base-8 configuration:
      d_model=768, d_ff=3072, num_experts=8, expert_capacity=64.

The experiment maps 8 experts to 4 compute chiplets, two experts per chiplet.
For every source chiplet, tokens routed to experts on other chiplets become
A2A traffic.  Each expert payload is split into ``dims`` chunks, and every
chunk runs the real Switch expert FFN kernel shape:

    x @ W_in -> relu -> hidden @ W_out

This is not a complete Switch MoE layer forward pass.  A complete layer also
includes router/gate compute, dispatch, expert execution, and combine/return.
BackendSim cycle counts are parsed before the Docker container exits and stored
as ``compute_cycle`` / ``compute_done_cycle`` in the output CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path


TORCHSIM_DIR = Path(os.environ.get("TORCHSIM_DIR", "/workspace/PyTorchSim"))
DEFAULT_TRACE_OUT = Path("traces/switch_base8_moe_a2a.csv")
BACKEND_RESULT_RE = re.compile(r"Total execution cycles:\s*(\d+)|Total cycle\s+(\d+)|Total_cycles:\s*(\d+)")


def import_pytorchsim():
    ts_dir = str(TORCHSIM_DIR)
    if ts_dir not in sys.path:
        sys.path.append(ts_dir)

    import torch

    tog_simulator_cls = None

    try:
        from Scheduler.scheduler import ExecutionEngine

        if hasattr(ExecutionEngine, "setup_device"):
            module = ExecutionEngine.setup_device()
            device = module.custom_device()
            return torch, device, None, "BackendSim"
    except (ImportError, AttributeError):
        pass

    device_pkg = TORCHSIM_DIR / "PyTorchSimDevice"
    if device_pkg.is_dir():
        pkg_path = str(device_pkg)
        if pkg_path not in sys.path:
            sys.path.insert(0, pkg_path)
        try:
            import torch_openreg  # noqa: F401
        except ImportError:
            pass

    try:
        device = torch.device("npu:0")
    except RuntimeError as exc:
        raise RuntimeError("PyTorchSim device backend is not loaded.") from exc

    try:
        from Simulator.simulator import TOGSimulator

        tog_simulator_cls = TOGSimulator
    except ImportError:
        pass

    backend = "TOGSim" if tog_simulator_cls is not None else "BackendSim"
    return torch, device, tog_simulator_cls, backend


def resolve_config_path(config_arg: str) -> Path | None:
    candidates = [
        Path(config_arg),
        TORCHSIM_DIR / "configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml",
        TORCHSIM_DIR / "PyTorchSimBackend/configs/systolic_ws_128x128_c2_simple_noc_tpuv3.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def dim_order_label(chunk_id: int, dims: int) -> str:
    rotation = [(chunk_id + i) % dims for i in range(dims)]
    if dims <= 3:
        names = "XYZ"
        return "".join(names[d] for d in rotation)
    return "".join(str(d) for d in rotation)


def backend_result_files() -> set[Path]:
    root = Path("/tmp/torchinductor")
    if not root.is_dir():
        return set()
    return {
        p
        for p in root.rglob("*")
        if p.is_file() and "backendsim_result" in p.parts
    }


def parse_backend_cycles(path: Path) -> int:
    text = path.read_text(errors="ignore")
    cycles: list[int] = []
    for match in BACKEND_RESULT_RE.finditer(text):
        token = next(group for group in match.groups() if group is not None)
        cycles.append(int(token))
    return max(cycles, default=0)


def cycles_from_new_backend_results(before: set[Path]) -> tuple[int, list[str]]:
    after = backend_result_files()
    new_files = sorted(after - before)
    cycles = [parse_backend_cycles(path) for path in new_files]
    cycles = [cycle for cycle in cycles if cycle > 0]
    return sum(cycles), [str(path) for path in new_files]


def run_switch_moe(
    nodes: int,
    dims: int,
    tokens_per_source: int,
    d_model: int,
    d_ff: int,
    num_experts: int,
    expert_capacity: int,
    flit_size: int,
    config_path: Path | None,
    smoke: bool,
) -> list[dict[str, object]]:
    torch, device, tog_simulator_cls, sim_backend = import_pytorchsim()

    if num_experts % nodes != 0:
        raise ValueError("num_experts must be divisible by nodes")
    if tokens_per_source % num_experts != 0:
        raise ValueError("tokens_per_source must be divisible by num_experts")
    tokens_per_expert = tokens_per_source // num_experts
    if tokens_per_expert > expert_capacity:
        raise ValueError("tokens_per_expert exceeds Switch expert_capacity")
    if tokens_per_expert % dims != 0:
        raise ValueError("tokens_per_expert must be divisible by dims")

    experts_per_node = num_experts // nodes
    chunk_tokens = tokens_per_expert // dims
    bytes_per_chunk = chunk_tokens * d_model * 4
    flits_per_chunk = math.ceil(bytes_per_chunk / flit_size)

    @torch.compile(dynamic=False)
    def switch_expert_ffn(tokens, w_in, w_out):
        hidden = torch.relu(torch.matmul(tokens, w_in))
        return torch.matmul(hidden, w_out)

    print("Switch-base-8 expert-FFN + routed-A2A PyTorchSim benchmark:")
    print(f"  TORCHSIM_DIR        : {TORCHSIM_DIR}")
    print(f"  device              : {device}")
    print(f"  sim backend         : {sim_backend}")
    print(f"  nodes               : {nodes}")
    print(f"  d_model / d_ff      : {d_model} / {d_ff}")
    print(f"  num_experts         : {num_experts}")
    print(f"  experts per chiplet : {experts_per_node}")
    print(f"  expert_capacity     : {expert_capacity}")
    print(f"  tokens per source   : {tokens_per_source}")
    print(f"  tokens per expert   : {tokens_per_expert}")
    print(f"  chunks per payload  : {dims}")
    print(f"  chunk tokens/bytes  : {chunk_tokens} / {bytes_per_chunk}")
    print(f"  flits per chunk     : {flits_per_chunk}")

    routed = []
    for src in range(nodes):
        for expert_id in range(num_experts):
            dst = expert_id // experts_per_node
            if dst == src:
                continue
            routed.append((src, dst, expert_id))
    if smoke:
        routed = routed[:1]
        print("  smoke mode: only first remote expert route")

    source_ready_cycle = {src: 0 for src in range(nodes)}
    rows: list[dict[str, object]] = []
    event_id = 0

    def run_all() -> None:
        nonlocal event_id
        for pair_id, (src, dst, expert_id) in enumerate(routed):
            for chunk_id in range(dims):
                torch.manual_seed(30_000 * pair_id + chunk_id)
                tokens = torch.randn(chunk_tokens, d_model).to(device=device)
                w_in = torch.randn(d_model, d_ff).to(device=device)
                w_out = torch.randn(d_ff, d_model).to(device=device)

                before = backend_result_files()
                switch_expert_ffn(tokens, w_in, w_out)
                if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
                    torch.npu.synchronize()

                compute_cycle, result_files = cycles_from_new_backend_results(before)
                if compute_cycle <= 0:
                    raise RuntimeError(
                        f"Unable to parse BackendSim cycles for src={src}, expert={expert_id}, chunk={chunk_id}"
                    )

                source_ready_cycle[src] += compute_cycle
                compute_done = source_ready_cycle[src]

                rows.append({
                    "event_id": event_id,
                    "pair_id": pair_id,
                    "moe_layer_id": 0,
                    "benchmark": "google/switch-base-8/expert-ffn-routed-a2a",
                    "src": src,
                    "dst": dst,
                    "expert_id": expert_id,
                    "chunk_id": chunk_id,
                    "dim_order": dim_order_label(chunk_id, dims),
                    "tokens_per_source": tokens_per_source,
                    "tokens_per_expert": tokens_per_expert,
                    "tokens_per_chunk": chunk_tokens,
                    "hidden_size": d_model,
                    "intermediate_size": d_ff,
                    "expert_capacity": expert_capacity,
                    "num_experts": num_experts,
                    "bytes_per_chunk": bytes_per_chunk,
                    "flits_per_chunk": flits_per_chunk,
                    "compute_cycle": compute_cycle,
                    "compute_done_cycle": compute_done,
                    "source_ready_cycle": compute_done,
                    "sim_backend": sim_backend,
                    "kernel": "switch.expert_ffn.relu_matmul",
                    "backend_result_files": ";".join(result_files),
                })
                event_id += 1
            print(f"  [{pair_id + 1}/{len(routed)}] src={src} -> expert={expert_id} on chiplet={dst} done")

    if tog_simulator_cls is not None:
        with tog_simulator_cls(config_path=str(config_path)):
            run_all()
    else:
        print("  using per-kernel BackendSim (no TOGSim session)")
        run_all()

    return rows


def write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def check_trace(rows: list[dict[str, object]], nodes: int, dims: int, num_experts: int) -> None:
    expected_events = nodes * (num_experts - num_experts // nodes) * dims
    assert len(rows) == expected_events, f"{len(rows)} vs {expected_events}"
    for row in rows:
        assert int(row["compute_cycle"]) > 0
        assert int(row["compute_done_cycle"]) > 0
    compute_time = max(int(row["compute_done_cycle"]) for row in rows)
    print("Switch expert-FFN routed-A2A trace check passed.")
    print(f"  events={len(rows)} compute_time_cycles={compute_time}")
    print(f"  total_payload_bytes={sum(int(row['bytes_per_chunk']) for row in rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run google/switch-base-8 expert-FFN routed-A2A benchmark via PyTorchSim.")
    parser.add_argument("--dims", type=int, default=2)
    parser.add_argument("--ary", type=int, default=2)
    parser.add_argument("--tokens-per-source", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--expert-capacity", type=int, default=64)
    parser.add_argument("--flit-size", type=int, default=64)
    parser.add_argument("--config", type=str,
                        default=str(TORCHSIM_DIR / "configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml"))
    parser.add_argument("--out", type=str, default=str(DEFAULT_TRACE_OUT))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    nodes = args.ary ** args.dims
    config_path = resolve_config_path(args.config)
    if config_path:
        print(f"simulator config: {config_path}")
    else:
        print("simulator config: (none, BackendSim standalone mode)")

    rows = run_switch_moe(
        nodes=nodes,
        dims=args.dims,
        tokens_per_source=args.tokens_per_source,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_experts=args.num_experts,
        expert_capacity=args.expert_capacity,
        flit_size=args.flit_size,
        config_path=config_path,
        smoke=args.smoke,
    )
    out_path = Path(args.out)
    write_csv(rows, out_path)
    if not args.smoke:
        check_trace(rows, nodes, args.dims, args.num_experts)
    print(f"trace saved : {out_path}")


if __name__ == "__main__":
    main()
