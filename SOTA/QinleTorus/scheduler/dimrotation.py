"""DimRotation scheduler.

For each chunk in the chunked CSV we split the (src, dst) message into D
single-dim hops.  Each hop is its own popnet packet (single coord differs
between src and dst of that hop), so PopNet's TXY routing transports it on
the matching dim only.

chunk_id k uses dim rotation (k, k+1, ..., k+D-1) mod D, so different chunks
visit dims in different orders.  As a result, at any given phase the chunks
collectively cover all dims:

  phase 1:  chunk 0 -> dim 0,  chunk 1 -> dim 1,  ...
  phase 2:  chunk 0 -> dim 1,  chunk 1 -> dim 2,  ...

Inject cycle for hop k of chunk c is:
    compute_done_cycle(c) + sum_{i<k} estimate_hop_cycles(hop_i)
"""

from __future__ import annotations

import csv
from pathlib import Path

from popnet_io.bench_writer import BenchPacket, write_bench

from .timing import TimingParams, estimate_hop_cycles
from .topology import TorusGeom, dim_rotation, expand_dim_order_path


def schedule_dimrotation(
    csv_path: Path,
    geom: TorusGeom,
    out_dir: Path,
    params: TimingParams = TimingParams(),
) -> tuple[Path, int]:
    """Returns (bench_path, packet_count)."""
    packets: list[BenchPacket] = []

    with Path(csv_path).open(newline="") as f:
        for row in csv.DictReader(f):
            src_node = int(row["src"])
            dst_node = int(row["dst"])
            chunk_id = int(row["chunk_id"])
            flits = int(row["flits_per_chunk"])
            t0 = int(row["compute_done_cycle"])

            src_c = geom.node_to_coord(src_node)
            dst_c = geom.node_to_coord(dst_node)
            order = dim_rotation(chunk_id, geom.dims)
            path = expand_dim_order_path(src_c, dst_c, order)

            t = t0
            for hop in range(geom.dims):
                hop_src = path[hop]
                hop_dst = path[hop + 1]
                if hop_src == hop_dst:
                    continue  # this dim already aligned
                packets.append(BenchPacket(
                    inject_cycle=t,
                    src=hop_src,
                    dst=hop_dst,
                    flits=flits,
                ))
                t += estimate_hop_cycles(geom, hop_src, hop_dst, flits, params)

    if not packets:
        raise RuntimeError(f"DimRotation produced 0 packets from {csv_path}")

    bench_path = write_bench(packets, out_dir)
    return bench_path, len(packets)
