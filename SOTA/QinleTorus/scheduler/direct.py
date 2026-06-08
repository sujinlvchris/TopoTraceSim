"""Classic direct A2A scheduler.

For every (src, dst) pair we emit ONE PopNet packet carrying the full message
(``D * flits_per_chunk`` flits, where D is the chunk count in the CSV).  All
packets inject at the pair's earliest ``compute_done_cycle`` (chunk 0).
PopNet's TXY routing then handles the path; with all packets injected close
together they compete on links and exhibit pipeline-style bubbles.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from popnet_io.bench_writer import BenchPacket, write_bench

from .topology import TorusGeom


def schedule_direct(csv_path: Path, geom: TorusGeom, out_dir: Path) -> tuple[Path, int]:
    """Read chunked CSV, fold chunks back into one packet per (src,dst), and
    write a popnet bench.  Returns (bench_path, packet_count)."""
    pair_rows: dict[tuple[int, int], list[dict]] = defaultdict(list)
    with Path(csv_path).open(newline="") as f:
        for row in csv.DictReader(f):
            pair_rows[(int(row["src"]), int(row["dst"]))].append(row)

    packets = []
    for (src_node, dst_node), rows in pair_rows.items():
        rows.sort(key=lambda r: int(r["chunk_id"]))
        total_flits = sum(int(r["flits_per_chunk"]) for r in rows)
        inject_cycle = min(int(r["compute_done_cycle"]) for r in rows)
        packets.append(BenchPacket(
            inject_cycle=inject_cycle,
            src=geom.node_to_coord(src_node),
            dst=geom.node_to_coord(dst_node),
            flits=total_flits,
        ))

    bench_path = write_bench(packets, out_dir)
    return bench_path, len(packets)
