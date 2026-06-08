#!/usr/bin/env python3
"""QinleTorus per-chunk A2A driver on top of PyTorchSim.

For every (src, dst) pair with src != dst we run ``chunks`` small matmuls
(``chunk_bytes = msg_bytes / chunks``) inside the torchsim-ci docker image.
Each matmul represents one chunk's compute and contributes one CSV row.

The CSV is the canonical interface between PyTorchSim and the QinleTorus
scheduler:

    event_id, pair_id, src, dst, chunk_id, dim_order,
    bytes_per_chunk, flits_per_chunk, compute_done_cycle,
    sim_backend, kernel

``compute_done_cycle`` is currently a deterministic schedule:
    compute_done_cycle = pair_id * inject_gap + chunk_id * per_chunk_gap
which matches the semantics of the existing TopoTraceSim driver while still
giving the scheduler a per-chunk timestamp.  Hooking the real TOGSim cycle
output is a TODO (see plan section 10).

Run inside docker:

  docker run --rm --ipc=host \\
    -v "$(pwd)/scripts:/workspace/PyTorchSim/scripts:ro" \\
    -v "$(pwd)/traces:/workspace/PyTorchSim/traces" \\
    -w /workspace/PyTorchSim \\
    ghcr.io/psal-postech/torchsim-ci:v1.0.0 \\
    python scripts/run_chunked_a2a.py --ary 4 --dims 2 --msg-size 16KB
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path


TORCHSIM_DIR = Path(os.environ.get("TORCHSIM_DIR", "/workspace/PyTorchSim"))
DEFAULT_TRACE_OUT = Path("traces/chunked_a2a.csv")


def parse_size(size_str: str) -> int:
    s = size_str.strip().upper()
    if s.endswith("KB"):
        return int(s[:-2]) * 1024
    if s.endswith("MB"):
        return int(s[:-2]) * 1024 * 1024
    if s.endswith("B"):
        return int(s[:-1])
    return int(s)


def matmul_shape(chunk_bytes: int) -> tuple[int, int]:
    """Pick an (m, n) fp32 matrix shape with m*n*4 == chunk_bytes.

    Prefer near-square shapes; fall back to (1, elems) only as a last resort.
    """
    if chunk_bytes % 4 != 0:
        raise ValueError(f"chunk_bytes={chunk_bytes} not divisible by sizeof(fp32)")
    elems = chunk_bytes // 4
    side = int(math.isqrt(elems))
    for m in range(side, 0, -1):
        if elems % m == 0:
            return (m, elems // m)
    return (1, elems)


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
    """For 2D returns 'XY' or 'YX'; ND returns rotated dim indices like '012'."""
    rotation = [(chunk_id + i) % dims for i in range(dims)]
    if dims <= 3:
        names = "XYZ"
        return "".join(names[d] for d in rotation)
    return "".join(str(d) for d in rotation)


def run_chunk(
    torch,
    compiled_matmul,
    device,
    shape: tuple[int, int],
    src: int,
    pair_id: int,
    chunk_id: int,
    compute_done_cycle: int,
    use_launch_model: bool,
):
    m, n = shape
    torch.manual_seed(10_000 * pair_id + chunk_id)
    a = torch.randn(m, n).to(device=device)
    b = torch.randn(n, m).to(device=device)

    if use_launch_model and hasattr(torch, "npu") and hasattr(torch.npu, "launch_model"):
        torch.npu.launch_model(
            compiled_matmul, a, b,
            stream_index=src, timestamp=compute_done_cycle,
        )
    elif hasattr(torch, "npu") and hasattr(torch.npu, "launch_context"):
        with torch.npu.launch_context(stream_index=src, timestamp=compute_done_cycle):
            compiled_matmul(a, b)
    else:
        compiled_matmul(a, b)


def run_chunked_a2a(
    nodes: int,
    dims: int,
    msg_bytes: int,
    flit_size: int,
    chunks: int,
    inject_gap: int,
    per_chunk_gap: int,
    config_path: Path | None,
    smoke: bool,
):
    torch, device, tog_simulator_cls, sim_backend = import_pytorchsim()

    if msg_bytes % chunks != 0:
        raise ValueError(f"msg_bytes={msg_bytes} not divisible by chunks={chunks}")
    chunk_bytes = msg_bytes // chunks
    shape = matmul_shape(chunk_bytes)

    flits_per_chunk = (chunk_bytes + flit_size - 1) // flit_size

    if tog_simulator_cls is not None and config_path is None:
        raise FileNotFoundError(
            "TOGSimulator available but no config found. Pass --config."
        )

    @torch.compile(dynamic=False)
    def compiled_matmul(a, b):
        return torch.matmul(a, b)

    print(f"PyTorchSim stack:")
    print(f"  TORCHSIM_DIR     : {TORCHSIM_DIR}")
    print(f"  device           : {device}")
    print(f"  sim backend      : {sim_backend}")
    print(f"  TOGSim session   : {tog_simulator_cls is not None}")
    print(f"  topology         : {dims}D, ary-per-dim implied by nodes={nodes}")
    print(f"  msg per pair     : {msg_bytes} B")
    print(f"  chunks per pair  : {chunks} ({chunk_bytes} B each = matmul {shape})")
    print(f"  flits per chunk  : {flits_per_chunk} ({flit_size}B/flit)")

    pairs = [(s, d) for s in range(nodes) for d in range(nodes) if s != d]
    if smoke:
        pairs = pairs[:1]
        print("  smoke mode: only first pair")

    rows = []
    use_launch_model = tog_simulator_cls is not None
    event_id = 0

    def run_all():
        nonlocal event_id
        for pair_id, (src, dst) in enumerate(pairs):
            pair_t0 = pair_id * inject_gap
            for chunk_id in range(chunks):
                compute_done = pair_t0 + chunk_id * per_chunk_gap
                run_chunk(
                    torch, compiled_matmul, device, shape,
                    src, pair_id, chunk_id, compute_done, use_launch_model,
                )
                rows.append({
                    "event_id": event_id,
                    "pair_id": pair_id,
                    "src": src,
                    "dst": dst,
                    "chunk_id": chunk_id,
                    "dim_order": dim_order_label(chunk_id, dims),
                    "bytes_per_chunk": chunk_bytes,
                    "flits_per_chunk": flits_per_chunk,
                    "compute_done_cycle": compute_done,
                    "sim_backend": sim_backend,
                    "kernel": "aten.matmul",
                })
                event_id += 1
            if (pair_id + 1) % max(1, len(pairs) // 10) == 0:
                print(f"  [{pair_id + 1}/{len(pairs)}] pair src={src} dst={dst} done")

        if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()

    if tog_simulator_cls is not None:
        with tog_simulator_cls(config_path=str(config_path)):
            run_all()
    else:
        print("  using per-kernel BackendSim (no TOGSim session)")
        run_all()

    return rows


def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def check_trace(rows, nodes, chunks, msg_bytes, flit_size):
    expected_pairs = nodes * (nodes - 1)
    expected_events = expected_pairs * chunks
    assert len(rows) == expected_events, (
        f"event count mismatch: {len(rows)} vs expected {expected_events}"
    )
    chunk_bytes = msg_bytes // chunks
    flits_per_chunk = (chunk_bytes + flit_size - 1) // flit_size

    send_bytes = {i: 0 for i in range(nodes)}
    recv_bytes = {i: 0 for i in range(nodes)}
    for row in rows:
        assert int(row["bytes_per_chunk"]) == chunk_bytes
        assert int(row["flits_per_chunk"]) == flits_per_chunk
        send_bytes[int(row["src"])] += chunk_bytes
        recv_bytes[int(row["dst"])] += chunk_bytes

    expected_per_node = (nodes - 1) * msg_bytes
    for n in range(nodes):
        assert send_bytes[n] == expected_per_node, f"node {n} send mismatch"
        assert recv_bytes[n] == expected_per_node, f"node {n} recv mismatch"

    print("chunked A2A trace check passed.")
    print(f"  nodes={nodes} chunks={chunks} events={len(rows)}")
    print(f"  total bytes per node (send/recv) = {expected_per_node}")


def main():
    parser = argparse.ArgumentParser(description="QinleTorus chunked A2A via PyTorchSim")
    parser.add_argument("--dims", type=int, default=2, help="Torus dimension count")
    parser.add_argument("--ary", type=int, default=4, help="routers per dim")
    parser.add_argument("--chunks", type=int, default=0, help="0 -> uses --dims")
    parser.add_argument("--msg-size", type=str, default="16KB")
    parser.add_argument("--flit-size", type=int, default=64)
    parser.add_argument("--inject-gap", type=int, default=0,
                        help="cycles between consecutive pairs' first chunks")
    parser.add_argument("--per-chunk-gap", type=int, default=0,
                        help="cycles between chunks within the same pair")
    parser.add_argument("--config", type=str,
                        default=str(TORCHSIM_DIR / "configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml"))
    parser.add_argument("--out", type=str, default=str(DEFAULT_TRACE_OUT))
    parser.add_argument("--log-path", type=str, default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    chunks = args.chunks if args.chunks > 0 else args.dims
    nodes = args.ary ** args.dims
    msg_bytes = parse_size(args.msg_size)

    if args.log_path:
        os.environ["TORCHSIM_LOG_PATH"] = args.log_path
    else:
        os.environ.setdefault("TORCHSIM_LOG_PATH", str(TORCHSIM_DIR / "togsim_results"))

    config_path = resolve_config_path(args.config)
    if config_path:
        print(f"simulator config: {config_path}")
    else:
        print("simulator config: (none, BackendSim standalone mode)")

    rows = run_chunked_a2a(
        nodes=nodes,
        dims=args.dims,
        msg_bytes=msg_bytes,
        flit_size=args.flit_size,
        chunks=chunks,
        inject_gap=args.inject_gap,
        per_chunk_gap=args.per_chunk_gap,
        config_path=config_path,
        smoke=args.smoke,
    )

    out_path = Path(args.out)
    write_csv(rows, out_path)

    if not args.smoke:
        check_trace(rows, nodes, chunks, msg_bytes, args.flit_size)

    print(f"trace saved : {out_path}")
    print(f"TOGSim logs : {os.environ.get('TORCHSIM_LOG_PATH', '(unset)')}")


if __name__ == "__main__":
    main()
