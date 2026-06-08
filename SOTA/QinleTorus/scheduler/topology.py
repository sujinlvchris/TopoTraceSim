"""ND-torus topology utilities.

Node id <-> coordinate conversion uses the SAME row-major layout PopNet uses
in ``sim_foundation.cc``::

    long i = (* first); first++;
    for(; first!= last; first++) {
        i = i * ary_size_ + (*first);
    }

So the most-significant coord is index 0.  For a 4x4 torus
(``ary=4, dims=2``):

    node 0  -> (0, 0)        node 5  -> (1, 1)
    node 1  -> (0, 1)        node 15 -> (3, 3)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TorusGeom:
    dims: int
    ary: int

    @property
    def num_nodes(self) -> int:
        return self.ary ** self.dims

    def node_to_coord(self, node: int) -> tuple[int, ...]:
        if not 0 <= node < self.num_nodes:
            raise ValueError(f"node {node} out of range [0, {self.num_nodes})")
        coord = []
        x = node
        for _ in range(self.dims):
            coord.append(x % self.ary)
            x //= self.ary
        return tuple(reversed(coord))

    def coord_to_node(self, coord: Iterable[int]) -> int:
        node = 0
        for c in coord:
            if not 0 <= c < self.ary:
                raise ValueError(f"coord {coord} has out-of-range {c}")
            node = node * self.ary + c
        return node

    def torus_hops_in_dim(self, src_c: int, dst_c: int) -> int:
        """Minimum hops in a single torus dimension between two coords."""
        diff = abs(dst_c - src_c)
        return min(diff, self.ary - diff)

    def torus_distance(self, src: tuple[int, ...], dst: tuple[int, ...]) -> int:
        if len(src) != self.dims or len(dst) != self.dims:
            raise ValueError("coord dim mismatch")
        return sum(self.torus_hops_in_dim(s, d) for s, d in zip(src, dst))


def dim_rotation(chunk_id: int, dims: int) -> tuple[int, ...]:
    """Cyclic dim rotation order for chunk_id in a D-dim torus.

    chunk 0 -> (0, 1, ..., D-1)
    chunk 1 -> (1, 2, ..., D-1, 0)
    chunk k -> (k, k+1, ..., k-1) mod D
    """
    return tuple((chunk_id + i) % dims for i in range(dims))


def expand_dim_order_path(
    src: tuple[int, ...],
    dst: tuple[int, ...],
    dim_order: Iterable[int],
) -> list[tuple[int, ...]]:
    """Return the (D+1)-length list of intermediate coordinates that one
    chunk visits as it traverses one dim at a time in ``dim_order``.

    Each consecutive pair differs in exactly one coordinate, so PopNet's
    TXY/XY routing handles each hop as a pure single-dim transmission.
    """
    path = [src]
    current = list(src)
    for d in dim_order:
        current[d] = dst[d]
        path.append(tuple(current))
    return path
