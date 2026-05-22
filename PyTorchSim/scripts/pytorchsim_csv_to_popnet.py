#!/usr/bin/env python3
"""Convert PyTorchSim A2A CSV trace to PopNet bench format (T sx sy dx dy n)."""

import argparse
import csv
import sys
from pathlib import Path

TORCHSIM_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = TORCHSIM_DIR / "traces/a2a_n4_16kb_pytorchsim.csv"
DEFAULT_OUT = TORCHSIM_DIR / "popnet_exp/traces/a2a_2x2"

# 4-node -> 2x2 mesh coordinates (fixed baseline)
NODE_TO_XY = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 0),
    3: (1, 1),
}


def convert_row(row):
    src = int(row["src"])
    dst = int(row["dst"])
    sx, sy = NODE_TO_XY[src]
    dx, dy = NODE_TO_XY[dst]
    t = int(float(row.get("inject_cycle", 0)))
    n = int(row["flits"])
    return t, sx, sy, dx, dy, n


def write_popnet_traces(rows, out_dir: Path, write_per_router: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_path = out_dir / "bench"
    per_router = {xy: [] for xy in NODE_TO_XY.values()}

    with bench_path.open("w") as bench_f:
        for row in rows:
            t, sx, sy, dx, dy, n = convert_row(row)
            line = f"{t} {sx} {sy} {dx} {dy} {n}\n"
            bench_f.write(line)
            per_router[(sx, sy)].append(line)

    if write_per_router:
        for (x, y), lines in per_router.items():
            path = out_dir / f"bench.{x}.{y}"
            path.write_text("".join(lines))

    return bench_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorchSim A2A CSV to PopNet bench (run from any directory)."
    )
    parser.add_argument(
        "--in",
        dest="in_csv",
        default=str(DEFAULT_CSV),
        help=f"PyTorchSim CSV trace (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT),
        help=f"Output directory for bench (+ bench.x.y) (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--no-per-router",
        action="store_true",
        help="Only write bench (skip bench.x.y)",
    )
    args = parser.parse_args()

    in_csv = Path(args.in_csv)
    if not in_csv.is_file():
        print(f"ERROR: CSV not found: {in_csv}", file=sys.stderr)
        print(f"  Run PyTorchSim first, or: cd {TORCHSIM_DIR}", file=sys.stderr)
        sys.exit(1)

    with in_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"ERROR: CSV is empty: {in_csv}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    bench_path = write_popnet_traces(
        rows, out_dir, write_per_router=not args.no_per_router
    )

    print(f"TORCHSIM_DIR: {TORCHSIM_DIR}")
    print(f"converted {len(rows)} events -> {bench_path}")
    print(f"per-router files: {not args.no_per_router}")
    print("sample:")
    for line in bench_path.read_text().splitlines()[:5]:
        print(f"  {line}")


if __name__ == "__main__":
    main()
