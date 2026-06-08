#!/usr/bin/env python3
"""Convert PyTorchSim A2A CSV trace to PopNet bench format (T sx sy dx dy n)."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

TOPOTRACE_ROOT = Path(__file__).resolve().parents[1]
TORCHSIM_DIR = TOPOTRACE_ROOT / "PyTorchSim"
DEFAULT_CSV = TOPOTRACE_ROOT / "traces/a2a_n4_16kb_pytorchsim.csv"
DEFAULT_OUT = TOPOTRACE_ROOT / "popnet_exp/traces/a2a_2x2"


def infer_nodes(rows):
    node_scale_values = {
        int(row["node_scale"])
        for row in rows
        if row.get("node_scale") not in (None, "")
    }
    if len(node_scale_values) == 1:
        return node_scale_values.pop()
    max_node = max(
        max(int(row["src"]), int(row["dst"]))
        for row in rows
    )
    return max_node + 1


def build_node_to_xy(nodes: int, mesh_ary: int | None = None):
    if mesh_ary is None:
        root = int(math.sqrt(nodes))
        if root * root != nodes:
            raise ValueError(
                "nodes must be a perfect square for automatic 2D mesh mapping; "
                f"got {nodes}. Pass --mesh-ary explicitly if needed."
            )
        mesh_ary = root
    if mesh_ary <= 0:
        raise ValueError(f"mesh_ary must be positive, got {mesh_ary}")
    if mesh_ary * mesh_ary < nodes:
        raise ValueError(
            f"mesh_ary={mesh_ary} cannot cover nodes={nodes}; "
            "need mesh_ary * mesh_ary >= nodes"
        )
    return {node: (node // mesh_ary, node % mesh_ary) for node in range(nodes)}


def convert_row(row, node_to_xy):
    src = int(row["src"])
    dst = int(row["dst"])
    sx, sy = node_to_xy[src]
    dx, dy = node_to_xy[dst]
    t = int(float(row.get("inject_cycle", 0)))
    n = int(row["flits"])
    return t, sx, sy, dx, dy, n


def write_popnet_traces(rows, out_dir: Path, node_to_xy, write_per_router: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_path = out_dir / "bench"
    per_router = {xy: [] for xy in node_to_xy.values()}

    with bench_path.open("w") as bench_f:
        for row in rows:
            t, sx, sy, dx, dy, n = convert_row(row, node_to_xy)
            line = f"{t} {sx} {sy} {dx} {dy} {n}\n"
            bench_f.write(line)
            per_router[(sx, sy)].append(line)

    if write_per_router:
        for (x, y), lines in per_router.items():
            (out_dir / f"bench.{x}.{y}").write_text("".join(lines))

    return bench_path


def main():
    parser = argparse.ArgumentParser(
        description="TopoTraceSim: PyTorchSim CSV -> PopNet bench"
    )
    parser.add_argument("--in", dest="in_csv", default=str(DEFAULT_CSV))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--nodes",
        type=int,
        default=0,
        help="Number of compute-side nodes. Default: infer from CSV metadata.",
    )
    parser.add_argument(
        "--mesh-ary",
        type=int,
        default=0,
        help="2D mesh ary used for node->(x,y). Default: sqrt(nodes).",
    )
    parser.add_argument("--no-per-router", action="store_true")
    args = parser.parse_args()

    in_csv = Path(args.in_csv)
    if not in_csv.is_file():
        print(f"ERROR: CSV not found: {in_csv}", file=sys.stderr)
        print(f"  TopoTraceSim root: {TOPOTRACE_ROOT}", file=sys.stderr)
        sys.exit(1)

    with in_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"ERROR: CSV is empty: {in_csv}", file=sys.stderr)
        sys.exit(1)

    nodes = args.nodes or infer_nodes(rows)
    node_to_xy = build_node_to_xy(nodes, args.mesh_ary or None)
    bench_path = write_popnet_traces(
        rows,
        Path(args.out_dir),
        node_to_xy,
        write_per_router=not args.no_per_router,
    )
    print(f"TopoTraceSim: {TOPOTRACE_ROOT}")
    print(f"node mapping: {nodes} nodes on {args.mesh_ary or int(math.sqrt(nodes))}x{args.mesh_ary or int(math.sqrt(nodes))} mesh")
    print(f"converted {len(rows)} events -> {bench_path}")


if __name__ == "__main__":
    main()
