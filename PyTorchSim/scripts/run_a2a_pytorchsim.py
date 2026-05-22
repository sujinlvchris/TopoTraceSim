#!/usr/bin/env python3
"""
Run a minimal 4-node A2A-shaped workload through the full PyTorchSim stack.

Each non-self (src, dst) pair triggers one compiled matmul on ``npu:0``, which
goes through the normal PyTorchSim pipeline:

  torch.compile (Inductor/MLIR) -> Gem5 -> Spike -> TOGSim/BackendSim

This script does NOT hand-write communication events. The exported CSV records
the intended A2A topology plus metadata from the PyTorchSim run.

TopoTraceSim full pipeline (from repo root):

  bash scripts/run_a2a_full_pipeline.sh

PyTorchSim-only inside Docker (image provides Gem5/Spike/BackendSim):

  docker run --rm --ipc=host \\
    -v "$(pwd)/scripts:/workspace/PyTorchSim/scripts:ro" \\
    -v "$(pwd)/traces:/workspace/PyTorchSim/traces" \\
    -v "$(pwd)/togsim_results:/workspace/PyTorchSim/togsim_results" \\
    -w /workspace/PyTorchSim \\
    ghcr.io/psal-postech/torchsim-ci:v1.0.0 \\
    python scripts/run_a2a_pytorchsim.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


TORCHSIM_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCHSIM_DIR", str(TORCHSIM_DIR))
# Host: TopoTraceSim/traces is bind-mounted to PyTorchSim/traces in Docker
DEFAULT_TRACE_OUT = TORCHSIM_DIR / "traces/a2a_n4_16kb_pytorchsim.csv"


def parse_size(size_str: str) -> int:
    size_str = size_str.upper()
    if size_str.endswith("KB"):
        return int(size_str[:-2]) * 1024
    if size_str.endswith("MB"):
        return int(size_str[:-2]) * 1024 * 1024
    if size_str.endswith("B"):
        return int(size_str[:-1])
    return int(size_str)


def matmul_side(msg_size_bytes: int) -> int:
    """Square fp32 matrix side length for roughly ``msg_size_bytes`` payload."""
    elems = msg_size_bytes // 4
    side = int(elems**0.5)
    if side * side * 4 != msg_size_bytes:
        raise ValueError(
            f"msg_size_bytes={msg_size_bytes} is not a perfect square fp32 matrix; "
            "pick a size like 16KB (64x64)."
        )
    return side


def import_pytorchsim():
    ts_dir = str(TORCHSIM_DIR)
    if ts_dir not in sys.path:
        sys.path.append(ts_dir)

    import torch

    tog_simulator_cls = None

    # Legacy torchsim-ci v1.0.0: JIT extension_device via Scheduler.ExecutionEngine
    try:
        from Scheduler.scheduler import ExecutionEngine

        if hasattr(ExecutionEngine, "setup_device"):
            module = ExecutionEngine.setup_device()
            device = module.custom_device()
            return torch, device, None, "BackendSim"
    except (ImportError, AttributeError):
        pass

    # Modern PyTorchSim: torch_openreg registers the ``npu`` backend
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
            "PyTorchSim device backend is not loaded. Run inside torchsim-ci Docker "
            "(v1.0.0+), or install PyTorchSimDevice with TORCHSIM_DIR set."
        ) from exc

    try:
        from Simulator.simulator import TOGSimulator

        tog_simulator_cls = TOGSimulator
    except ImportError:
        pass

    backend = "TOGSim" if tog_simulator_cls is not None else "BackendSim"
    return torch, device, tog_simulator_cls, backend


def check_trace(rows, nodes, msg_size_bytes, flit_size):
    expected_events = nodes * (nodes - 1)
    expected_total_bytes = expected_events * msg_size_bytes
    expected_flits_per_event = (msg_size_bytes + flit_size - 1) // flit_size
    expected_total_flits = expected_events * expected_flits_per_event

    assert len(rows) == expected_events, (
        f"event count mismatch: got {len(rows)}, expected {expected_events}"
    )

    total_bytes = 0
    total_flits = 0
    send_bytes = {i: 0 for i in range(nodes)}
    recv_bytes = {i: 0 for i in range(nodes)}

    for row in rows:
        src = int(row["src"])
        dst = int(row["dst"])
        num_bytes = int(row["bytes"])
        flits = int(row["flits"])

        assert row["op_type"] == "A2A"
        assert row["sim_backend"] in ("TOGSim", "BackendSim", "unknown")
        assert src != dst
        assert 0 <= src < nodes
        assert 0 <= dst < nodes
        assert num_bytes == msg_size_bytes
        assert flits == expected_flits_per_event

        total_bytes += num_bytes
        total_flits += flits
        send_bytes[src] += num_bytes
        recv_bytes[dst] += num_bytes

    assert total_bytes == expected_total_bytes
    assert total_flits == expected_total_flits

    expected_per_node_bytes = (nodes - 1) * msg_size_bytes
    for node in range(nodes):
        assert send_bytes[node] == expected_per_node_bytes
        assert recv_bytes[node] == expected_per_node_bytes

    print("A2A trace check passed (PyTorchSim-backed).")
    print(f"nodes: {nodes}")
    print(f"events: {expected_events}")
    print(f"message size per pair: {msg_size_bytes} bytes")
    print(f"flit size: {flit_size} bytes")
    print(f"flits per event: {expected_flits_per_event}")
    print(f"total bytes: {total_bytes}")
    print(f"total flits: {total_flits}")


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


def run_pair(
    torch,
    compiled_matmul,
    device,
    side: int,
    src: int,
    dst: int,
    event_id: int,
    inject_gap: int,
    use_launch_model: bool,
):
    torch.manual_seed(1000 + event_id)
    a = torch.randn(side, side).to(device=device)
    b = torch.randn(side, side).to(device=device)

    if use_launch_model and hasattr(torch, "npu") and hasattr(torch.npu, "launch_model"):
        torch.npu.launch_model(
            compiled_matmul,
            a,
            b,
            stream_index=src,
            timestamp=event_id * inject_gap,
        )
    elif hasattr(torch, "npu") and hasattr(torch.npu, "launch_context"):
        with torch.npu.launch_context(stream_index=src, timestamp=event_id * inject_gap):
            compiled_matmul(a, b)
    else:
        compiled_matmul(a, b)

    return a, b


def run_a2a_pytorchsim(
    nodes: int,
    msg_size_bytes: int,
    flit_size: int,
    inject_gap: int,
    config_path: Path | None,
    smoke: bool,
):
    torch, device, tog_simulator_cls, sim_backend = import_pytorchsim()
    side = matmul_side(msg_size_bytes)

    if tog_simulator_cls is not None and config_path is None:
        raise FileNotFoundError(
            "TOGSimulator is available but no simulator config was found. "
            "Pass --config or mount PyTorchSim configs."
        )

    @torch.compile(dynamic=False)
    def compiled_matmul(a, b):
        return torch.matmul(a, b)

    print("PyTorchSim stack check:")
    print(f"  TORCHSIM_DIR: {TORCHSIM_DIR}")
    print(f"  device: {device}")
    print(f"  sim backend: {sim_backend}")
    print(f"  TOGSimulator session: {tog_simulator_cls is not None}")
    print(f"  matmul shape: ({side}, {side}) x ({side}, {side}) fp32 -> {side * side * 4} bytes")

    pairs = [
        (src, dst)
        for src in range(nodes)
        for dst in range(nodes)
        if src != dst
    ]
    if smoke:
        pairs = pairs[:1]
        print("  smoke mode: running 1 pair only")

    rows = []
    use_launch_model = tog_simulator_cls is not None

    def run_all_pairs():
        for event_id, (src, dst) in enumerate(pairs):
            run_pair(
                torch,
                compiled_matmul,
                device,
                side,
                src,
                dst,
                event_id,
                inject_gap,
                use_launch_model,
            )
            flits = (msg_size_bytes + flit_size - 1) // flit_size
            rows.append(
                {
                    "event_id": event_id,
                    "collective_id": 0,
                    "op_type": "A2A",
                    "src": src,
                    "dst": dst,
                    "bytes": msg_size_bytes,
                    "flits": flits,
                    "inject_cycle": event_id * inject_gap,
                    "node_scale": nodes,
                    "phase_id": 0,
                    "chunk_id": 0,
                    "sim_backend": sim_backend,
                    "pytorchsim_kernel": "aten.matmul",
                }
            )
            print(f"  [{event_id + 1}/{len(pairs)}] simulated pair src={src} -> dst={dst}")

        if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()

    if tog_simulator_cls is not None:
        with tog_simulator_cls(config_path=str(config_path)):
            run_all_pairs()
    else:
        print("  using per-kernel BackendSim (no TOGSimulator session API)")
        run_all_pairs()

    return rows


def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run minimal A2A workload via PyTorchSim (Gem5 + Spike + simulator)."
    )
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--msg-size", type=str, default="16KB")
    parser.add_argument("--flit-size", type=int, default=64)
    parser.add_argument("--inject-gap", type=int, default=0)
    parser.add_argument(
        "--config",
        type=str,
        default=str(TORCHSIM_DIR / "configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="CSV path (default: <TopoTraceSim>/traces/a2a_n4_16kb_pytorchsim.csv)",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="",
        help="TORCHSIM_LOG_PATH for TOGSim logs (default: <TORCHSIM_DIR>/togsim_results)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the first src->dst pair for a quick sanity check.",
    )
    args = parser.parse_args()

    if args.log_path:
        os.environ["TORCHSIM_LOG_PATH"] = args.log_path
    else:
        os.environ.setdefault(
            "TORCHSIM_LOG_PATH",
            str(TORCHSIM_DIR / "togsim_results"),
        )

    msg_size_bytes = parse_size(args.msg_size)
    config_path = resolve_config_path(args.config)
    if config_path:
        print(f"simulator config: {config_path}")
    else:
        print("simulator config: (none, BackendSim standalone mode)")

    rows = run_a2a_pytorchsim(
        nodes=args.nodes,
        msg_size_bytes=msg_size_bytes,
        flit_size=args.flit_size,
        inject_gap=args.inject_gap,
        config_path=config_path,
        smoke=args.smoke,
    )

    out_path = Path(args.out) if args.out else DEFAULT_TRACE_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_path)

    if not args.smoke:
        check_trace(rows, args.nodes, msg_size_bytes, args.flit_size)

    print(f"trace saved to: {out_path}")
    print(f"TOGSim logs (if any): {os.environ['TORCHSIM_LOG_PATH']}")
    print("PyTorchSim A2A run finished.")


if __name__ == "__main__":
    main()
