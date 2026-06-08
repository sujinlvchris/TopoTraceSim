"""Write PopNet bench files for ND-Torus traces.

bench line format (PopNet, ND-aware via sim_foundation::readAddress):
    T  sx_0 sx_1 ... sx_{D-1}  dx_0 dx_1 ... dx_{D-1}  n

Records are sorted by ascending injection time T (PopNet expects increasing
order).  We also write per-router ``bench.x.y...`` shards for compatibility
with the random_trace layout, although the main ``-I`` file is what popnet
reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class BenchPacket:
    inject_cycle: int
    src: tuple[int, ...]
    dst: tuple[int, ...]
    flits: int

    def format(self) -> str:
        coords = " ".join(str(c) for c in self.src) + " " + " ".join(str(c) for c in self.dst)
        return f"{self.inject_cycle} {coords} {self.flits}"


def write_bench(
    packets: Iterable[BenchPacket],
    out_dir: Path,
    write_per_router: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkts = sorted(packets, key=lambda p: p.inject_cycle)
    if not pkts:
        raise ValueError("write_bench: no packets to emit")

    bench_path = out_dir / "bench"
    with bench_path.open("w") as f:
        for p in pkts:
            f.write(p.format() + "\n")

    if write_per_router:
        per_src: dict[tuple[int, ...], list[BenchPacket]] = {}
        for p in pkts:
            per_src.setdefault(p.src, []).append(p)
        for src, ps in per_src.items():
            shard_name = "bench." + ".".join(str(c) for c in src)
            (out_dir / shard_name).write_text(
                "\n".join(p.format() for p in ps) + "\n"
            )

    return bench_path


def packet_count(out_dir: Path) -> int:
    p = Path(out_dir) / "bench"
    if not p.is_file():
        return 0
    with p.open() as f:
        return sum(1 for _ in f if _.strip())


def coord_to_str(coord: Sequence[int]) -> str:
    return "(" + ",".join(str(c) for c in coord) + ")"
