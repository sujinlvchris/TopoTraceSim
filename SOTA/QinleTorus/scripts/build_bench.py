#!/usr/bin/env python3
"""CLI wrapper around the schedulers.

Reads the chunked CSV and writes a popnet bench using either ``direct`` or
``dimrotation``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scheduler.direct import schedule_direct  # noqa: E402
from scheduler.dimrotation import schedule_dimrotation  # noqa: E402
from scheduler.timing import TimingParams  # noqa: E402
from scheduler.topology import TorusGeom  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="QinleTorus bench builder")
    ap.add_argument("--csv", required=True, help="chunked A2A CSV (PyTorchSim output)")
    ap.add_argument("--dims", type=int, required=True)
    ap.add_argument("--ary", type=int, required=True)
    ap.add_argument("--scheduler", required=True, choices=["direct", "dimrotation"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--slack", type=float, default=0.10,
                    help="DimRotation hop-time slack (ignored for direct)")
    args = ap.parse_args()

    geom = TorusGeom(dims=args.dims, ary=args.ary)
    out_dir = Path(args.out_dir)

    if args.scheduler == "direct":
        bench, n = schedule_direct(Path(args.csv), geom, out_dir)
    else:
        params = TimingParams(slack=args.slack)
        bench, n = schedule_dimrotation(Path(args.csv), geom, out_dir, params)

    print(f"{args.scheduler}: wrote {n} packets to {bench}")


if __name__ == "__main__":
    main()
