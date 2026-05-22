#!/usr/bin/env python3
"""Convert PyTorchSim A2A CSV trace to PopNet bench format (T sx sy dx dy n)."""

import argparse
import csv
import sys
from pathlib import Path

TOPOTRACE_ROOT = Path(__file__).resolve().parents[1]
TORCHSIM_DIR = TOPOTRACE_ROOT / "PyTorchSim"
DEFAULT_CSV = TOPOTRACE_ROOT / "traces/a2a_n4_16kb_pytorchsim.csv"
DEFAULT_OUT = TOPOTRACE_ROOT / "popnet_exp/traces/a2a_2x2"

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
            (out_dir / f"bench.{x}.{y}").write_text("".join(lines))

    return bench_path


def main():
    parser = argparse.ArgumentParser(
        description="TopoTraceSim: PyTorchSim CSV -> PopNet bench"
    )
    parser.add_argument("--in", dest="in_csv", default=str(DEFAULT_CSV))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
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

    bench_path = write_popnet_traces(
        rows, Path(args.out_dir), write_per_router=not args.no_per_router
    )
    print(f"TopoTraceSim: {TOPOTRACE_ROOT}")
    print(f"converted {len(rows)} events -> {bench_path}")


if __name__ == "__main__":
    main()
