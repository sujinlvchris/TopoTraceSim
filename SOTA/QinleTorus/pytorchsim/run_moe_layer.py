#!/usr/bin/env python3
"""Run a minimal routed MoE layer through PyTorchSim and emit chunked A2A CSV.

The model is intentionally small and deterministic so it can be used as a
repeatable experiment:

* one compute chiplet per expert;
* top-1 balanced routing from every source chiplet to every remote expert;
* each routed payload is split into ``dims`` chunks;
* every chunk runs one expert projection on the PyTorchSim device.

The driver parses BackendSim result files before the Docker container exits and
stores a real ``compute_cycle`` per chunk.  ``compute_done_cycle`` is modeled as
the per-source critical path: chunks from the same source chiplet execute
sequentially, while different source chiplets may execute in parallel.
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
DEFAULT_TRACE_OUT = Path("traces/moe_layer_a2a.csv")
BACKEND_RESULT_RE = re.compile(r"Total execution cycles:\s*(\d+)|Total cycle\s+(\d+)|Total_cycles:\s*(\d+)")


def parse_size(size_str: str) -> int:
    s = size_str.strip().upper()
    if s.endswith("KB"):
        return int(s[:-2]) * 1024
    if s.endswith("MB"):
        return int(s[:-2]) * 1024 * 1024
    if s.endswith("B"):
        return int(s[:-1])
    return int(s)


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
        raise RuntimeError(
            "PyTorchSim device backend is not loaded. Run inside torchsim-ci docker."
        ) from exc

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
    if not cycles:
        return 0
    return max(cycles)


def cycles_from_new_backend_results(before: set[Path]) -> tuple[int, list[str]]:
    after = backend_result_files()
    new_files = sorted(after - before)
    cycles = [parse_backend_cycles(path) for path in new_files]
    cycles = [cycle for cycle in cycles if cycle > 0]
    return sum(cycles), [str(path) for path in new_files]


def run_moe_layer(
    nodes: int,
    dims: int,
    tokens_per_remote_expert: int,
    hidden_size: int,
    flit_size: int,
    config_path: Path | None,
    smoke: bool,
) -> list[dict[str, object]]:
    torch, device, tog_simulator_cls, sim_backend = import_pytorchsim()

    if tog_simulator_cls is not None and config_path is None:
        raise FileNotFoundError("TOGSimulator available but no config found. Pass --config.")

    chunk_tokens = tokens_per_remote_expert // dims
    if tokens_per_remote_expert % dims != 0:
        raise ValueError("tokens_per_remote_expert must be divisible by dims")
    bytes_per_chunk = chunk_tokens * hidden_size * 4
    flits_per_chunk = math.ceil(bytes_per_chunk / flit_size)

    @torch.compile(dynamic=False)
    def expert_projection(tokens, weight):
        return torch.matmul(tokens, weight)

    print("MoE layer PyTorchSim stack:")
    print(f"  TORCHSIM_DIR              : {TORCHSIM_DIR}")
    print(f"  device                    : {device}")
    print(f"  sim backend               : {sim_backend}")
    print(f"  TOGSim session            : {tog_simulator_cls is not None}")
    print(f"  nodes / experts           : {nodes}")
    print(f"  dims / chunks             : {dims}")
    print(f"  tokens per remote expert  : {tokens_per_remote_expert}")
    print(f"  hidden size               : {hidden_size}")
    print(f"  chunk tokens              : {chunk_tokens}")
    print(f"  bytes per chunk           : {bytes_per_chunk}")
    print(f"  flits per chunk           : {flits_per_chunk}")

    pairs = [(src, dst) for src in range(nodes) for dst in range(nodes) if src != dst]
    if smoke:
        pairs = pairs[:1]
        print("  smoke mode: only first routed pair")

    source_ready_cycle = {src: 0 for src in range(nodes)}
    rows: list[dict[str, object]] = []
    event_id = 0

    def run_all() -> None:
        nonlocal event_id
        for pair_id, (src, dst) in enumerate(pairs):
            for chunk_id in range(dims):
                torch.manual_seed(20_000 * pair_id + chunk_id)
                tokens = torch.randn(chunk_tokens, hidden_size).to(device=device)
                weight = torch.randn(hidden_size, hidden_size).to(device=device)

                before = backend_result_files()
                expert_projection(tokens, weight)
                if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
                    torch.npu.synchronize()

                compute_cycle, result_files = cycles_from_new_backend_results(before)
                if compute_cycle <= 0:
                    raise RuntimeError(
                        "Unable to parse BackendSim compute cycles for "
                        f"src={src} dst={dst} chunk={chunk_id}"
                    )

                source_ready_cycle[src] += compute_cycle
                compute_done = source_ready_cycle[src]

                rows.append({
                    "event_id": event_id,
                    "pair_id": pair_id,
                    "moe_layer_id": 0,
                    "src": src,
                    "dst": dst,
                    "expert_id": dst,
                    "chunk_id": chunk_id,
                    "dim_order": dim_order_label(chunk_id, dims),
                    "tokens_per_chunk": chunk_tokens,
                    "hidden_size": hidden_size,
                    "bytes_per_chunk": bytes_per_chunk,
                    "flits_per_chunk": flits_per_chunk,
                    "compute_cycle": compute_cycle,
                    "compute_done_cycle": compute_done,
                    "source_ready_cycle": compute_done,
                    "sim_backend": sim_backend,
                    "kernel": "moe.expert_projection.matmul",
                    "backend_result_files": ";".join(result_files),
                })
                event_id += 1
            print(f"  [{pair_id + 1}/{len(pairs)}] routed pair src={src} -> expert={dst} done")

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


def check_trace(rows: list[dict[str, object]], nodes: int, dims: int, tokens: int, hidden: int) -> None:
    expected_pairs = nodes * (nodes - 1)
    expected_events = expected_pairs * dims
    assert len(rows) == expected_events, f"{len(rows)} vs expected {expected_events}"
    per_node_bytes = {node: 0 for node in range(nodes)}
    for row in rows:
        per_node_bytes[int(row["src"])] += int(row["bytes_per_chunk"])
        assert int(row["compute_cycle"]) > 0
        assert int(row["compute_done_cycle"]) > 0
    expected_per_node = (nodes - 1) * tokens * hidden * 4
    for node, byte_count in per_node_bytes.items():
        assert byte_count == expected_per_node, f"node {node} bytes {byte_count} vs {expected_per_node}"
    compute_time = max(int(row["compute_done_cycle"]) for row in rows)
    print("MoE layer trace check passed.")
    print(f"  nodes={nodes} routed_pairs={expected_pairs} chunks={dims} events={len(rows)}")
    print(f"  compute_time_cycles={compute_time}")
    print(f"  total_payload_bytes={sum(int(row['bytes_per_chunk']) for row in rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal routed MoE layer via PyTorchSim.")
    parser.add_argument("--dims", type=int, default=2)
    parser.add_argument("--ary", type=int, default=2)
    parser.add_argument("--tokens-per-remote-expert", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--flit-size", type=int, default=64)
    parser.add_argument("--config", type=str,
                        default=str(TORCHSIM_DIR / "configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml"))
    parser.add_argument("--out", type=str, default=str(DEFAULT_TRACE_OUT))
    parser.add_argument("--log-path", type=str, default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.log_path:
        os.environ["TORCHSIM_LOG_PATH"] = args.log_path
    else:
        os.environ.setdefault("TORCHSIM_LOG_PATH", str(TORCHSIM_DIR / "togsim_results"))

    nodes = args.ary ** args.dims
    config_path = resolve_config_path(args.config)
    if config_path:
        print(f"simulator config: {config_path}")
    else:
        print("simulator config: (none, BackendSim standalone mode)")

    rows = run_moe_layer(
        nodes=nodes,
        dims=args.dims,
        tokens_per_remote_expert=args.tokens_per_remote_expert,
        hidden_size=args.hidden_size,
        flit_size=args.flit_size,
        config_path=config_path,
        smoke=args.smoke,
    )

    out_path = Path(args.out)
    write_csv(rows, out_path)
    if not args.smoke:
        check_trace(rows, nodes, args.dims, args.tokens_per_remote_expert, args.hidden_size)
    print(f"trace saved : {out_path}")
    print(f"TOGSim logs : {os.environ.get('TORCHSIM_LOG_PATH', '(unset)')}")


if __name__ == "__main__":
    main()
